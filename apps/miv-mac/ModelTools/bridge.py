"""Authenticated loopback translations for clients that use Responses or Messages."""
import json
import secrets
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from aiohttp import ClientSession, ClientTimeout, web
from pydantic import ValidationError
from omlx.api import anthropic_utils as a, responses_utils as r
from omlx.api.anthropic_models import MessagesRequest
from omlx.api.openai_models import ToolCall
from omlx.api.responses_models import ResponsesRequest


def uid(prefix):
    return prefix + "_" + uuid.uuid4().hex


def sse(event, **fields):
    return r.format_sse_event(event, {"type": event, **fields}).encode()


class Relay:
    def __init__(self, base, upstream_key, model, aliases=()):
        self.aliases = dict(aliases)
        self.base, self.upstream_key, self.model = base.rstrip("/"), upstream_key, model
        self.available_models = {model}
        self.key = secrets.token_urlsafe(32)
        self.previous = OrderedDict()
        self.runner = None

    async def start(self):
        app = web.Application(client_max_size=16 * 1024**2)
        app.router.add_route("*", "/{path:.*}", self.handle)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.port = self.runner.addresses[0][1]

    async def close(self):
        if self.runner: await self.runner.cleanup()

    async def handle(self, req):
        supplied = req.headers.get("x-api-key") or req.headers.get("Authorization","").removeprefix("Bearer ")
        if not secrets.compare_digest(supplied.encode(), self.key.encode()):
            raise web.HTTPUnauthorized()
        path = req.path
        headers = {"Authorization": "Bearer " + self.upstream_key} if self.upstream_key else {}
        async with ClientSession(headers=headers, timeout=ClientTimeout(total=None,sock_connect=10)) as client:
            if path in ("/v1/models","/v1/chat/completions","/v1/completions","/health"):
                async with client.request(req.method,self.base+path,data=await req.read(),
                                          headers={"Content-Type":"application/json"},allow_redirects=False) as upstream:
                    if path == "/v1/models" and self.aliases and upstream.status == 200:
                        catalog = await upstream.json()
                        physical = {item["id"]: item for item in catalog["data"]}
                        for alias, target in self.aliases.items():
                            if target not in physical or alias in physical:
                                continue
                            entry = dict(physical[target])
                            entry.update(id=alias, display_name=target, anthropic_family_tier=alias.removeprefix("claude-"),
                                         is_family_default=True, created_at=datetime.fromtimestamp(entry.get("created", 0), timezone.utc).isoformat())
                            catalog["data"].append(entry)
                        return web.json_response(catalog)
                    response = web.StreamResponse(status=upstream.status,headers={"Content-Type":upstream.headers.get("Content-Type","application/json")})
                    await response.prepare(req)
                    async for part in upstream.content.iter_any(): await response.write(part)
                    return response
            if req.method != "POST" or path not in ("/v1/messages","/v1/responses"): raise web.HTTPNotFound()
            responses = path == "/v1/responses"
            try:
                body = await req.json()
                aliases = {}
                if responses:
                    request = ResponsesRequest.model_validate(body)
                    prev = self.previous.get(request.previous_response_id)
                    if request.previous_response_id and prev is None:
                        return web.json_response({"error":{"message":"Unknown previous_response_id","code":"previous_response_not_found"}}, status=404)
                    messages = r.convert_responses_input_to_messages(request.input, request.instructions, prev)
                    tools = r.convert_responses_tools(request.tools, aliases)
                    r.apply_namespace_tool_aliases(messages, aliases)
                    limit = request.max_output_tokens
                else:
                    request = MessagesRequest.model_validate(body)
                    messages = a.convert_anthropic_to_internal(request, native_reasoning_content=True)
                    tools = a.convert_anthropic_tools_to_internal(request.tools)
                    limit = request.max_tokens
                selected = self.aliases.get(request.model, request.model)
                if not responses and selected not in self.available_models:
                    selected = self.model
                chat = {"model": selected, "messages": messages, "stream": bool(body.get("stream"))}
                if tools: chat["tools"] = tools
                if limit: chat["max_completion_tokens"] = limit
                for field in ("temperature","top_p","top_k"):
                    if body.get(field) is not None: chat[field] = body[field]
                if body.get("stop_sequences"): chat["stop"] = body["stop_sequences"]
                if body.get("tool_choice") == "none" or body.get("tool_choice") == {"type":"none"}:
                    chat["tool_choice"] = "none"
                if chat["stream"]: chat["stream_options"] = {"include_usage":True}
            except (ValueError, ValidationError, TypeError) as exc:
                return web.json_response({"error":{"type":"invalid_request_error","message":str(exc)}},status=400)
            async with client.post(self.base+"/v1/chat/completions",json=chat,allow_redirects=False) as upstream:
                if upstream.status != 200:
                    return web.Response(status=upstream.status,body=await upstream.read(),content_type="application/json")
                if not chat["stream"]:
                    result = await upstream.json()
                    choice = result["choices"][0]
                    translator = Translation(self, responses, messages, body, aliases, selected)
                    translator.add(choice["message"])
                    return web.json_response(translator.finish(choice["finish_reason"],result.get("usage",{})))
                output = web.StreamResponse(headers={"Content-Type":"text/event-stream; charset=utf-8","Cache-Control":"no-cache"})
                await output.prepare(req)
                t = Translation(self, responses, messages, body, aliases, selected)
                await output.write(t.start())
                usage, finish, done = {}, "stop", False
                try:
                    async for raw in upstream.content:
                        if not raw.startswith(b"data:"): continue
                        data = raw[5:].strip()
                        if data == b"[DONE]": done=True; break
                        chunk = json.loads(data)
                        if "error" in chunk: raise ValueError(chunk["error"].get("message","Generation failed"))
                        if chunk.get("usage"): usage=chunk["usage"]
                        if chunk.get("choices"):
                            ch=chunk["choices"][0]
                            if ch.get("finish_reason"): finish=ch["finish_reason"]
                            for event in t.add(ch.get("delta",{})): await output.write(event)
                    if not done: raise ValueError("MIV stream ended before [DONE]")
                    for event in t.end(finish,usage): await output.write(event)
                except (ValueError, KeyError) as exc:
                    await output.write(sse("error",error={"type":"api_error","message":str(exc)}))
                return output


class Translation:
    def __init__(self, relay, responses, history, body, aliases, selected):
        self.model = selected
        self.relay,self.responses,self.history,self.body,self.aliases = relay,responses,history,body,aliases
        self.id=uid("resp" if responses else "msg")
        self.parts={}
        self.seq=0

    def event(self, kind, **kwargs):
        event=sse(kind,sequence_number=self.seq,**kwargs)
        self.seq+=1
        return event

    def response(self, status, items, usage=None):
        import time
        result={"id":self.id,"object":"response","created_at":int(time.time()),"status":status,
                "model":self.model,"output":items,"error":None,"incomplete_details":None}
        if usage is not None:
            result["usage"]=r.build_response_usage(usage.get("prompt_tokens",0),usage.get("completion_tokens",0),
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens",0)).model_dump()
        if status=="incomplete": result["incomplete_details"]={"reason":"max_output_tokens"}
        return result

    def start(self):
        if self.responses:
            obj=self.response("in_progress",[])
            return self.event("response.created",response=obj)+self.event("response.in_progress",response=obj)
        return a.create_message_start_event(self.id, self.model).encode()

    def item(self, part, status):
        value=part["value"]
        if part["type"]=="text":
            obj=r.build_message_output_item(value,part["id"],status)
        elif part["type"]=="thinking":
            obj=r.build_reasoning_output_item(value,part["id"],status)
        else:
            namespace,name=r.split_namespace_tool_name(part["name"],self.aliases)
            obj=r.build_function_call_output_item(name,value,part["call"],part["id"],status,namespace)
        return obj.model_dump(exclude_none=True)

    def add(self, delta):
        changes=[]
        for key, field in [("thinking","reasoning_content"),("text","content")]:
            if delta.get(field): changes.append((key,key,delta[field],None))
        for i,call in enumerate(delta.get("tool_calls") or []):
            changes.append(("tool"+str(call.get("index",i)), "tool_use",(call.get("function") or {}).get("arguments",""),call))
        events=[]
        for key,typ,fragment,call in changes:
            if key not in self.parts:
                part={"id":uid({"text":"msg","thinking":"rs","tool_use":"fc"}[typ]),"index":len(self.parts),
                      "type":typ,"value":"","name":"","call":uid("call")}
                self.parts[key]=part
                if call:
                    part["name"]=call["function"].get("name","")
                    part["call"]=call.get("id") or part["call"]
                ix=part["index"]
                if self.responses:
                    item=self.item(part,"in_progress")
                    if typ=="text": item["content"]=[]
                    events.append(self.event("response.output_item.added",output_index=ix,item=item))
                    if typ=="text":
                        events.append(self.event("response.content_part.added",output_index=ix,item_id=part["id"],
                                                content_index=0,part={"type":"output_text","text":"","annotations":[]}))
                else:
                    if ix: events.append(a.create_content_block_stop_event(ix-1).encode())
                    events.append(a.create_content_block_start_event(ix,typ,id=part["call"],name=part["name"]).encode())
            part=self.parts[key]
            part["value"]+=fragment
            ix=part["index"]
            if self.responses:
                name={"text":"output_text","thinking":"reasoning_summary_text","tool_use":"function_call_arguments"}[typ]
                fields={("summary_index" if typ=="thinking" else "content_index"):0} if typ!="tool_use" else {}
                events.append(self.event("response."+name+".delta",output_index=ix,item_id=part["id"],delta=fragment,**fields))
            else:
                writer={"text":a.create_text_delta_event,"thinking":a.create_thinking_delta_event,"tool_use":a.create_input_json_delta_event}[typ]
                events.append(writer(ix,fragment).encode())
        return events

    def finish(self, reason, usage):
        content=self.parts.get("text",{}).get("value","")
        thinking=self.parts.get("thinking",{}).get("value","")
        calls=[{"id":p["call"],"type":"function","function":{"name":p["name"],"arguments":p["value"]}}
               for p in self.parts.values() if p["type"]=="tool_use"]
        if self.responses:
            if self.body.get("store",True):
                assistant={"role":"assistant","content":content}
                if calls: assistant["tool_calls"]=calls
                if thinking: assistant["reasoning_content"]=thinking
                self.relay.previous[self.id]=self.history+[assistant]
                if len(self.relay.previous)>64: self.relay.previous.popitem(last=False)
            return self.response("incomplete" if reason=="length" else "completed",
                                 [self.item(p,"completed") for p in self.parts.values()],usage)
        obj=a.convert_internal_to_anthropic_response(content,self.model,usage.get("prompt_tokens",0),
                   usage.get("completion_tokens",0),reason,[ToolCall(**c) for c in calls],thinking=thinking)
        return obj.model_dump(exclude_none=True)

    def end(self, reason, usage):
        result=self.finish(reason,usage)
        if not self.responses:
            if self.parts: yield a.create_content_block_stop_event(len(self.parts)-1).encode()
            yield a.create_message_delta_event(result["stop_reason"],usage.get("completion_tokens",0),
                                               input_tokens=usage.get("prompt_tokens",0)).encode()
            yield a.create_message_stop_event().encode()
            return
        for ix,item in enumerate(result["output"]):
            common={"output_index":ix,"item_id":item["id"]}
            if item["type"]=="message":
                part=item["content"][0]
                yield self.event("response.output_text.done",**common,content_index=0,text=part["text"])
                yield self.event("response.content_part.done",**common,content_index=0,part=part)
            elif item["type"]=="function_call":
                yield self.event("response.function_call_arguments.done",**common,arguments=item["arguments"])
            elif item["type"]=="reasoning":
                yield self.event("response.reasoning_summary_text.done",**common,summary_index=0,text=item["summary"][0]["text"])
            yield self.event("response.output_item.done",output_index=ix,item=item)
        yield self.event("response."+result["status"],response=result)

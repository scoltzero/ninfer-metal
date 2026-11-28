"""Exercise the two client protocols and real SSE frames against a local chat server.
Run with the app's packaged Python and ModelTools on PYTHONPATH.
"""
import json
import unittest
from aiohttp import web, ClientSession
from bridge import Relay


class BridgeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.inputs = []
        async def chat(req):
            self.assertEqual(req.headers["Authorization"], "Bearer fixture-upstream")
            body = await req.json()
            self.inputs.append(body)
            tool = {"id":"call_1","type":"function","function":{"name":"weather","arguments":'{"city":"北京"}'}}
            usage = {"prompt_tokens":13,"completion_tokens":9}
            if not body.get("stream"):
                return web.json_response({"choices":[{"message":{"role":"assistant","content":None,"tool_calls":[tool]},
                                                       "finish_reason":"tool_calls"}],"usage":usage})
            res = web.StreamResponse(headers={"Content-Type":"text/event-stream"})
            await res.prepare(req)
            pieces = [{"delta":{"reasoning_content":"先查询"}}, {"delta":{"content":"查天气。"}},
                      {"delta":{"tool_calls":[{"index":0,**tool}]}}, {"delta":{},"finish_reason":"tool_calls"}]
            for piece in pieces:
                await res.write(("data: "+json.dumps({"choices":[piece]})+"\n\n").encode())
            await res.write(("data: "+json.dumps({"choices":[],"usage":usage})+"\n\ndata: [DONE]\n\n").encode())
            return res
        app = web.Application()
        app.router.add_post("/v1/chat/completions",chat)
        async def models(_):
            return web.json_response({"data":[{"id":"tiny"}]})
        app.router.add_get("/v1/models",models)
        self.upstream = web.AppRunner(app)
        await self.upstream.setup()
        await web.TCPSite(self.upstream,"127.0.0.1",0).start()
        self.relay = Relay("http://127.0.0.1:"+str(self.upstream.addresses[0][1]),"fixture-upstream","tiny")
        await self.relay.start()
        self.url = "http://127.0.0.1:"+str(self.relay.port)
        self.client = ClientSession(headers={"Authorization":"Bearer "+self.relay.key})

    async def asyncTearDown(self):
        await self.client.close()
        await self.relay.close()
        await self.upstream.cleanup()

    async def test_auth_and_chat(self):
        async with ClientSession() as guest, guest.get(self.url+"/v1/models") as denied:
            self.assertEqual(denied.status,401)
        async with self.client.get(self.url+"/v1/models") as res:
            self.assertEqual((await res.json())["data"][0]["id"],"tiny")
        async with self.client.post(self.url+"/v1/chat/completions",json={"model":"tiny","messages":[]}) as res:
            self.assertEqual((await res.json())["choices"][0]["message"]["tool_calls"][0]["function"]["name"],"weather")

    async def test_responses(self):
        body={"model":"tiny","input":"天气？","tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]}
        async with self.client.post(self.url+"/v1/responses",json=body) as res:
            obj=await res.json()
            self.assertEqual(obj["output"][0]["name"],"weather")
            self.assertEqual(obj["usage"]["input_tokens"],13)
        async with self.client.post(self.url+"/v1/responses",json={
            **body,"stream":True,"previous_response_id":obj["id"],
            "input":[{"type":"function_call_output","call_id":"call_1","output":"晴"}]}) as res:
            chunks=[json.loads(line[6:]) for line in (await res.text()).splitlines() if line.startswith("data: ")]
            self.assertEqual(chunks[-1]["type"],"response.completed")
            self.assertEqual(chunks[-1]["response"]["output"][-1]["arguments"],'{"city":"北京"}')
            self.assertEqual([c["sequence_number"] for c in chunks],list(range(len(chunks))))
            self.assertIn("response.output_text.delta",{c["type"] for c in chunks})
            self.assertIn("response.function_call_arguments.delta",{c["type"] for c in chunks})
        self.assertEqual(self.inputs[-1]["messages"][-1]["role"],"tool")
        self.assertEqual(self.inputs[-1]["messages"][-1]["tool_call_id"],"call_1")

    async def test_claude_desktop_tiers(self):
        self.relay.aliases = {"claude-opus":"tiny"}
        async with self.client.get(self.url+"/v1/models") as res:
            alias = (await res.json())["data"][-1]
            self.assertEqual(alias["id"],"claude-opus")
            self.assertEqual(alias["display_name"],"tiny")
            self.assertTrue(alias["is_family_default"])
        body = {"model":"claude-opus","messages":[{"role":"user","content":"天气？"}],"max_tokens":32}
        async with self.client.post(self.url+"/v1/messages",json=body) as res:
            self.assertEqual(res.status,200)
            self.assertEqual(self.inputs[-1]["model"],"tiny")

    async def test_messages(self):
        body={"model":"tiny","messages":[{"role":"user","content":"天气？"}],"max_tokens":30,
              "tools":[{"name":"weather","description":"forecast","input_schema":{"type":"object"}}]}
        async with self.client.post(self.url+"/v1/messages",json=body) as res:
            obj=await res.json()
            self.assertEqual(obj["stop_reason"],"tool_use")
            self.assertEqual(obj["content"][-1]["input"],{"city":"北京"})
        async with self.client.post(self.url+"/v1/messages",json={**body,"stream":True}) as res:
            frames=[json.loads(line[6:]) for line in (await res.text()).splitlines() if line.startswith("data: ")]
            self.assertEqual(frames[-1]["type"],"message_stop")
            self.assertEqual(frames[-2]["delta"]["stop_reason"],"tool_use")
            self.assertEqual(frames[-2]["usage"]["output_tokens"],9)
            blocks=[f["content_block"]["type"] for f in frames if f["type"]=="content_block_start"]
            self.assertEqual(blocks,["thinking","text","tool_use"])
        self.assertEqual(self.inputs[-1]["tools"][0]["function"]["name"],"weather")

if __name__ == "__main__":
    unittest.main()

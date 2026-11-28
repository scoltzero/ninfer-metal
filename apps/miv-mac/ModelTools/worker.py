"""MIV model tools. Requests and responses travel over the parent app's pipes."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

# Model tools can write progress to stdout. Reserve the original descriptor
# for JSON so Python and native extension diagnostics cannot corrupt a response.
wire = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
from omlx.admin.hf_downloader import HFDownloader
from omlx.admin.ms_downloader import MSDownloader, MS_SDK_AVAILABLE
from omlx.settings import get_system_memory


class Workspace:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        support = Path.home() / "Library/Application Support/MIV/ToolJobs"
        jobs = support / hashlib.sha256(str(self.root).encode()).hexdigest()[:24]
        jobs.mkdir(parents=True, exist_ok=True)
        self.hf = HFDownloader(str(self.root), tasks_file=jobs / "hf.json")
        self.ms = MSDownloader(str(self.root), tasks_file=jobs / "ms.json")
        self.roots = [str(self.root)]
        self.quantizer = None
        self.uploader = None
        self.relays = {}
        self.profiles = tempfile.TemporaryDirectory(prefix="miv-connections-")

    async def call(self, action, params):
        group, op, *tail = action.strip("/").split("/")
        if group == "integrations" and op == "availability":
            from omlx.integrations import INTEGRATIONS
            return {key: {"installed": value.is_installed(), "hint": value.install_hint}
                    for key, value in INTEGRATIONS.items()}
        if group == "integrations" and op == "prepare":
            from bridge import Relay
            from omlx.integrations import get_integration
            tool, model = params["tool"], params["model"]
            if get_integration(tool) is None or not model: raise ValueError("请选择客户端和模型。")
            aliases = tuple(("claude-"+tier, params[tier+"_model"]) for tier in ("opus","sonnet","haiku")
                            if tool == "claude_desktop" and params.get(tier+"_model"))
            handle = (params["address"], params.get("api_key", ""), model, aliases)
            if handle not in self.relays:
                relay = Relay(*handle)
                await relay.start()
                self.relays[handle] = relay
            relay = self.relays[handle]
            relay.available_models = set(params.get("catalog", [model]))
            # Connection profile contains only a random per-session relay credential.
            identity = hashlib.sha256((relay.key + json.dumps({k:v for k,v in params.items() if k != "api_key"},sort_keys=True)).encode()).hexdigest()[:16]
            profile = Path(self.profiles.name) / (tool + "-" + identity + ".json")
            fields = {k:v for k,v in params.items() if k in {"tools_profile","context_window","opus_model","sonnet_model","haiku_model"} and v}
            fields["models_status_map"] = {name: {"model_type":"llm"} for name in relay.available_models}
            fields.update(tool=tool,host="127.0.0.1",port=relay.port,api_key=relay.key,model=model,default_model=model)
            fd, scratch = tempfile.mkstemp(dir=self.profiles.name)
            with os.fdopen(fd,"w") as out: json.dump(fields,out)
            os.replace(scratch,profile)
            return {"profile":str(profile)}
        if group in {"hf", "ms"}:
            manager = self.hf if group == "hf" else self.ms
            if op == "status":
                return {"available": MS_SDK_AVAILABLE}
            if op == "recommended":
                flags = ({"model_format": params.get("model_format", "mlx")} if group == "hf"
                         else {"mlx_only": params.get("mlx_only", "true") == "true"})
                return await manager.get_recommended_models(max_memory_bytes=get_system_memory(), result_limit=50, **flags)
            if op == "search":
                flags = ({"model_format": params.get("model_format", "mlx")} if group == "hf"
                         else {"mlx_only": params.get("mlx_only", "true") == "true"})
                return await manager.search_models(query=params["q"], sort=params.get("sort","trending"),
                                                   limit=int(params.get("limit",20)), **flags)
            if op == "model-info":
                if group == "ms":
                    return await manager.get_model_info(params["model_id"])
                repo = params["repo_id"]
                local = (self.root / repo).resolve()
                if not local.is_relative_to(self.root):
                    raise ValueError("无效仓库名称")
                info = await manager.get_model_info(repo, local_dir=local)
                info["disk_free"] = shutil.disk_usage(self.root).free
                if manager.has_unfinished_download(repo):
                    info["update_files"] = []
                return info
            if op == "download":
                task = await manager.start_download(**params)
                return {"success": True, "task": task.to_dict()}
            if op == "retry":
                task = await manager.retry_download(tail[0])
                if not task:
                    raise ValueError("此任务无法重试")
                return {"success": True, "task": task.to_dict()}
            if op == "cancel" and await manager.cancel_download(tail[0]):
                return {"success": True}
        elif group == "oq":
            if self.quantizer is None:
                from omlx.admin.oq_manager import OQManager
                self.quantizer = OQManager(self.roots)
            manager = self.quantizer
            if op == "models":
                models, all_models = await manager.list_quantizable_models()
                return {"models": models, "all_models": all_models}
            if op == "estimate":
                from omlx.oq import estimate_bpw_and_size
                for key in ("preserve_mtp", "preserve_ngram_table", "text_only"):
                    params[key] = params.get(key) == "true"
                params["oq_level"] = float(params["oq_level"])
                return await asyncio.to_thread(estimate_bpw_and_size, **params)
            if op == "start":
                task = await manager.start_quantization(**params)
                return {"success": True, "task": task.to_dict()}
            if op == "cancel" and await manager.cancel_quantization(tail[0]):
                return {"success": True}
        elif group == "upload":
            if self.uploader is None:
                from omlx.admin.hf_uploader import HFUploader
                self.uploader = HFUploader(self.roots)
            manager = self.uploader
            if op == "validate-token":
                return await manager.validate_token(params["hf_token"])
            if op == "oq-models":
                return {"oq_models": await manager.list_oq_models(), "all_models": await manager.list_all_models()}
            if op == "start":
                params["token"] = params.pop("hf_token")
                task = await manager.start_upload(**params)
                return {"success": True, "task": task.to_dict()}
            if op == "cancel" and await manager.cancel_upload(tail[0]):
                return {"success": True}
        else:
            raise ValueError("未知模型工具")
        if op == "tasks":
            return {"tasks": manager.get_tasks()}
        if op == "task" and manager.remove_task(tail[0]):
            return {"success": True}
        raise ValueError("任务不存在或当前状态不允许此操作。")

    async def close(self):
        await asyncio.gather(*(relay.close() for relay in self.relays.values()))
        self.profiles.cleanup()
        await asyncio.gather(*(m.shutdown() for m in (self.hf,self.ms,self.quantizer,self.uploader) if m),
                             return_exceptions=True)


async def main():
    workspace = Workspace(sys.argv[1])
    await workspace.hf.restore_tasks()
    await workspace.ms.restore_tasks()
    tasks = set()
    reader = asyncio.StreamReader(limit=4 * 1024 * 1024)
    await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    async def handle(request):
        rid = request["id"]
        try:
            managed = {"http_proxy","https_proxy","all_proxy","no_proxy","hf_endpoint","modelscope_domain","hf_token","modelscope_api_token","ssl_cert_file","requests_ca_bundle","curl_ca_bundle"}
            network = {k: v for k,v in request.get("network", {}).items() if k.lower() in managed}
            changed = any(os.environ.get(k) != v for k,v in network.items())
            for key in list(os.environ):
                if key.lower() in managed and key not in network:
                    del os.environ[key]
                    changed = True
            if changed:
                os.environ.update(network)
                from huggingface_hub.utils import close_session
                close_session()
            roots = list(dict.fromkeys([str(workspace.root), *request.get("roots", [])]))
            if roots != workspace.roots:
                workspace.roots = roots
                for manager in (workspace.quantizer,workspace.uploader):
                    if manager is not None: manager.update_model_dirs(roots)
            result = await workspace.call(request["path"], request.get("params", {}))
            response = {"id": rid, "result": result}
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            for key in ("hf_token", "ms_token", "token"):
                token = request.get("params", {}).get(key)
                if token: message = message.replace(token, "***")
            response = {"id": rid, "error": message}
        wire.write(json.dumps(response, ensure_ascii=True) + "\n")
    try:
        while line := await reader.readline():
            task = asyncio.create_task(handle(json.loads(line)))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    finally:
        for task in tasks: task.cancel()
        await workspace.close()

if __name__ == "__main__":
    asyncio.run(main())

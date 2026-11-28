"""Run a MIV benchmark against the selected OpenAI-compatible endpoint.

Input is a single JSON object on stdin (including the session credential).
Output is line-delimited progress and a final result, consumed by the Swift UI.
"""
import asyncio
from dataclasses import asdict
from pathlib import Path
import os
import json
import math
import random
import sys
import time
from types import SimpleNamespace

import httpx
from miv_eval import BENCHMARKS


def emit(**fields):
    print(json.dumps(fields, ensure_ascii=False, allow_nan=False), flush=True)


class Engine:
    def __init__(self, client, model, context):
        self.client = client
        self.model = model
        self.context = context
        self.first_error = None

    async def chat(self, messages, **kwargs):
        kwargs["max_tokens"] = min(kwargs["max_tokens"], max(1, self.context // 2))
        try:
            response = await self.client.post("v1/chat/completions", json={
                "messages": messages, "model": self.model, **kwargs})
            response.raise_for_status()
            obj = response.json()
            choice = obj["choices"][0]
            message = choice["message"]
            reasoning = message.get("reasoning_content")
            text = message.get("content") or ""
            if reasoning:
                text = "<think>" + reasoning + "</think>" + text
            return SimpleNamespace(text=text, finish_reason=choice.get("finish_reason"),
                                   **obj.get("usage", {}))
        except Exception as error:
            if self.first_error is None:
                self.first_error = error
            raise

    async def completion(self, length, output):
        # IDs in the ASCII range exist in MIV's byte-level BPE models. Vary
        # the beginning on each request so repeated tests do not reuse a prefix.
        prompt = random.SystemRandom().choices(range(32, 94), k=length)
        start = time.monotonic()
        first = None
        usage = None
        finished = False
        async with self.client.stream("POST", "v1/completions", json={
            "model": self.model, "prompt": prompt, "max_tokens": output, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}) as response:
            if response.is_error:
                await response.aread()
                response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError(event["error"].get("message", "流式请求失败"))
                if event.get("choices") and event["choices"][0].get("text") and first is None:
                    first = time.monotonic()
                usage = event.get("usage") or usage
        end = time.monotonic()
        if not finished or not usage or usage.get("prompt_tokens") != length:
            raise ValueError("核心返回的计量信息不完整，无法验证输入长度。")
        if first is None and usage["completion_tokens"] > 0:
            # A single EOS has no printable piece; its arrival is bounded by the final chunk.
            first = end
        return length, usage["completion_tokens"], (first or end) - start, end - start


async def probe(engine, length, output, batch):
    begin = time.monotonic()
    samples = await asyncio.gather(*(engine.completion(length, output) for _ in range(batch)))
    elapsed = time.monotonic() - begin
    total_out = sum(item[1] for item in samples)
    ttft = sum(item[2] for item in samples) / len(samples)
    return dict(input=length, batch=batch, output=total_out, seconds=round(elapsed, 3),
                ttft_ms=round(ttft * 1000, 1),
                prefill_tps=round(length / max(ttft, 0.000001), 1),
                output_tps=round(total_out / elapsed, 1),
                decode_tps=round(sum(max(0, s[1] - 1) for s in samples) /
                                 max(1e-6, max(s[3] - s[2] for s in samples)), 1))


async def run(job):
    timeout = httpx.Timeout(600, connect=10)
    async with httpx.AsyncClient(base_url=job["base"].rstrip("/") + "/", timeout=timeout,
                                 trust_env=False, follow_redirects=False,
                                 headers={"Authorization": "Bearer " + job["key"]} if job["key"] else {}) as client:
        response = await client.get("v1/models")
        response.raise_for_status()
        model = next((m for m in response.json()["data"] if m["id"] == job["model"]), None)
        if not model:
            raise ValueError("此模型尚未加载。")
        context = model.get("max_model_len") or model.get("max_context")
        if not isinstance(context, int) or context < 2:
            raise ValueError("服务没有报告可用上下文。")
        engine = Engine(client, job["model"], context)
        batch = max(1, min(job["batch"], model.get("max_concurrency") or 1))
        if job["kind"] == "accuracy":
            scores = []
            for suite in job["suites"]:
                bench = BENCHMARKS[suite]()
                items = await bench.load_dataset(job.get("sampleCounts", {}).get(suite, job["samples"]))
                async def progress(current, total):
                    emit(progress=f"{suite} · {current} / {total}")
                result = await bench.run(engine, items, on_progress=progress, batch_size=batch,
                                         enable_thinking=job["thinking"])
                if engine.first_error:
                    raise engine.first_error
                if job.get("detailPath"):
                    # Only the desktop parent supplies this destination. Append per suite so
                    # a full run never sends the question-level report through UserDefaults.
                    detail = Path(job["detailPath"])
                    fd = os.open(detail, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                    with os.fdopen(fd, "a", encoding="utf-8") as report:
                        for question in result.question_results:
                            report.write(json.dumps({"suite": suite, **asdict(question)}, ensure_ascii=False) + "\n")
                scores.append(dict(suite=suite, correct=result.correct_count, total=result.total_questions,
                                   accuracy=result.accuracy, seconds=result.time_seconds,
                                   categories=result.category_scores))
            return dict(scores=scores)
        if job["kind"] == "throughput":
            rows = []
            for length in job["lengths"]:
                if length + job["output"] > context:
                    raise ValueError(f"输入 {length} 加输出 {job['output']} 超过当前上下文 {context}。")
                for concurrency in sorted({1, batch}):
                    emit(progress=f"{length} tokens · 并发 {concurrency}")
                    rows.append(await probe(engine, length, job["output"], concurrency))
            return dict(measurements=rows)
        # A context result is a witnessed successful prefill, bounded by the
        # instance's configured limit. Do not reinterpret network/auth errors as capacity.
        high = min(context, job["limit"])
        low = 0
        while low < high:
            candidate = high if low == 0 else math.ceil((low + high) / 2)
            emit(progress=f"验证 {candidate} tokens")
            try:
                row = await probe(engine, candidate - 1, 1, 1)
                low = candidate
            except httpx.HTTPStatusError as e:
                detail = e.response.text.lower()
                if e.response.status_code not in (400, 413, 507) or not any(
                        word in detail for word in ("context", "memory", "budget", "too long")):
                    raise
                if low == 0:
                    # Start with a known small witness, then binary search up.
                    below = min(256, candidate // 2)
                    if below < 2:
                        raise
                    row = await probe(engine, below - 1, 1, 1)
                    low = below
                high = candidate - 1
        if low == 0:
            raise ValueError("未验证到可用上下文。")
        return dict(verified_context=low, configured_limit=context, measurement=row)


if __name__ == "__main__":
    try:
        emit(result=asyncio.run(run(json.load(sys.stdin))))
    except Exception as exc:
        emit(error=str(exc) or type(exc).__name__)
        sys.exit(1)

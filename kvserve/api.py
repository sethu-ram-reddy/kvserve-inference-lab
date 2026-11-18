"""Local HTTP/SSE adapter backed by one shared, cooperatively driven scheduler."""
import asyncio
import codecs
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .engine import Engine, EngineConfig
from .model import AttentionDecoder, ByteTokenizer
from .sampling import Sampling


class CompletionInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_tokens: int = Field(default=32, ge=0, le=512)
    temperature: float = Field(default=0, ge=0, le=5)
    top_k: int | None = Field(default=None, ge=1, le=258)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0, le=10)
    stream: bool = False


def create_app(model=None, config=None):
    tokenizer = ByteTokenizer()
    model = model or AttentionDecoder()
    config = config or EngineConfig(total_slots=4096, max_pending=64)

    @asynccontextmanager
    async def lifespan(app):
        engine = Engine(model, config, retain_events=False)
        channels = {}
        counters = {"accepted": 0, "finished": 0, "cancelled": 0, "failed": 0}
        app.state.engine, app.state.channels, app.state.counters = engine, channels, counters
        wake = asyncio.Event()
        app.state.wake = wake

        async def worker():
            while True:
                await wake.wait()
                try:
                    if engine.pending:
                        for event in engine.step():
                            channel = channels.get(event["request_id"])
                            if channel is not None and event["event"] in ("token", "finish"):
                                channel.put_nowait(event)
                            if event["event"] == "finish":
                                counters["finished"] += 1
                    if not engine.pending:
                        wake.clear()
                except Exception:
                    # A worker failure must release ownership and wake every affected client.
                    for rid, channel in list(channels.items()):
                        engine.cancel(rid)
                        engine.completed.pop(rid, None)
                        while not channel.empty():
                            channel.get_nowait()
                        channel.put_nowait({"event": "error", "request_id": rid})
                        counters["failed"] += 1
                    wake.clear()
                await asyncio.sleep(0)

        task = asyncio.create_task(worker())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            for rid in list(channels):
                engine.cancel(rid)
            engine.cache.check()

    app = FastAPI(title="KVServe Inference Lab", version="0.1.0", lifespan=lifespan,
                  description="Untrained CPU attention decoder. A serving mechanics demonstration.")

    @app.get("/health")
    async def health():
        return {"status": "ready", "model": "seeded-attention-decoder", "trained": False}

    @app.get("/metrics")
    async def metrics():
        engine = app.state.engine
        return {**app.state.counters, "running": len(engine.running), "waiting": len(engine.waiting),
                "assigned_kv_slots": engine.cache.assigned_slots, "reserved_kv_slots": engine.reserved,
                "pool_slots": config.total_slots, "scheduler_ticks": engine.tick,
                "uptime_s": perf_counter() - engine.started_at}

    @app.post("/v1/completions")
    async def completions(body: CompletionInput, request: HTTPRequest):
        engine, channels = app.state.engine, app.state.channels
        if len(channels) >= config.max_pending:
            raise HTTPException(429, "Request queue is full")
        prompt = tokenizer.encode(body.prompt)
        try:
            rid = engine.submit(prompt, body.max_tokens,
                                Sampling(body.temperature, body.top_k, body.top_p), body.seed, body.priority)
        except OverflowError as error:
            raise HTTPException(429, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        channel = asyncio.Queue(maxsize=body.max_tokens + 2)
        channels[rid] = channel
        app.state.counters["accepted"] += 1
        if rid in engine.completed:  # zero-token request: no worker iteration required
            app.state.counters["finished"] += 1
            channel.put_nowait({"event": "finish", "request_id": rid, "finish_reason": "length"})
        app.state.wake.set()

        async def events():
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event = await asyncio.wait_for(channel.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    if event["event"] == "error":
                        raise RuntimeError("Inference worker failed")
                    yield event
                    if event["event"] == "finish":
                        break
            finally:
                if engine.cancel(rid):
                    app.state.counters["cancelled"] += 1
                engine.completed.pop(rid, None)
                channels.pop(rid, None)

        if body.stream:
            async def sse():
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                async for event in events():
                    if event["event"] == "token":
                        token = event["token_id"]
                        event = {**event, "text": decoder.decode(bytes([token]) if token < 256 else b"")}
                    else:
                        event = {**event, "text": decoder.decode(b"", final=True)}
                    yield "data: " + json.dumps(event) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        ids, reason = [], "length"
        async for event in events():
            if event["event"] == "token":
                ids.append(event["token_id"])
            elif event["event"] == "finish":
                reason = event["finish_reason"]
        return {"id": rid, "model": "seeded-attention-decoder", "text": tokenizer.decode(ids),
                "output_ids": ids, "finish_reason": reason,
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids)}}

    return app


app = create_app()

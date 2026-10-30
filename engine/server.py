"""An OpenAI-compatible HTTP server around the scheduler.

Two worlds meet here. The scheduler is blocking and uses the GPU, so it gets its own thread,
the engine loop, and nothing else ever touches it. The web server is asyncio and must never
block. They talk through queues:

    handler -> command queue -> engine loop -> scheduler
    handler <- per-request asyncio queue <- engine loop

The engine loop only ever drops events into a request's queue, which cannot block. A slow
client just lets its own queue grow, up to at most max_tokens events, and the model never
waits for anyone.
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from engine.sampling import SamplingParams
from engine.scheduler import Event, InvalidRequest, PromptTooLong, QueueFull, Scheduler
from engine.text import IncrementalDecoder

logger = logging.getLogger("llm_engine")

DISCONNECT_CHECK_SECONDS = 0.25


class EngineLoop:
    """Owns the scheduler and runs it on one background thread."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.sinks: dict[int, Callable[[Event], None]] = {}
        self.thread = threading.Thread(target=self._run, name="engine-loop", daemon=True)
        self.running = False
        self.crashed_steps = 0

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.commands.put(lambda: None)
        self.thread.join(timeout=10)

    @property
    def alive(self) -> bool:
        return self.running and self.thread.is_alive()

    def submit(
        self, prompt: torch.Tensor, params: SamplingParams, sink: Callable[[Event], None]
    ) -> concurrent.futures.Future:
        """Ask the loop to queue a request. The future gives its id, or the refusal."""
        future: concurrent.futures.Future = concurrent.futures.Future()

        def command() -> None:
            try:
                request_id = self.scheduler.submit(prompt, params)
            except (InvalidRequest, QueueFull) as error:
                future.set_exception(error)
                return
            self.sinks[request_id] = sink
            future.set_result(request_id)

        self.commands.put(command)
        return future

    def cancel(self, request_id: int) -> None:
        self.commands.put(lambda: self.scheduler.cancel(request_id))

    def _run(self) -> None:
        while self.running:
            scheduler = self.scheduler
            idle = not (scheduler.has_work or scheduler._ready)
            try:
                # Sleep when there is nothing to do. Otherwise take what has arrived.
                while True:
                    self.commands.get(block=idle)()
                    idle = False
            except queue.Empty:
                pass
            if self.running:
                self._step()

    def _step(self) -> None:
        try:
            events = self.scheduler.step()
        except Exception as error:
            # Failures inside a request are handled by the scheduler. Reaching here means
            # something else broke, and the loop must survive it.
            logger.exception("scheduler step failed")
            self.crashed_steps += 1
            events = self._abandon_everything(error)
        for event in events:
            sink = self.sinks.get(event.request_id)
            if sink is not None:
                sink(event)
                if event.finish_reason is not None:
                    del self.sinks[event.request_id]

    def _abandon_everything(self, error: Exception) -> list[Event]:
        message = f"{type(error).__name__}: {error}"
        events = []
        for request_id in list(self.scheduler.requests):
            self.scheduler.cancel(request_id)
            events.append(Event(request_id, finish_reason="error", error=message))
        self.scheduler._ready.clear()
        return events


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content)


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    """The request body. Fields not listed here are ignored, as OpenAI's clients add many."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # Not in OpenAI's API, but vLLM accepts them too, which load tests rely on.
    top_k: int = 0
    repetition_penalty: float = 1.0
    ignore_eos: bool = False


def error_response(status: int, message: str, kind: str, code: str | None = None) -> JSONResponse:
    body = {"error": {"message": message, "type": kind, "param": None, "code": code}}
    return JSONResponse(status_code=status, content=body)


def create_app(
    scheduler: Scheduler,
    tokenizer,
    model_name: str,
    stop_tokens: tuple[int, ...] = (),
    default_max_tokens: int = 256,
) -> FastAPI:
    engine = EngineLoop(scheduler)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="llm-engine", lifespan=lifespan)
    app.state.engine = engine

    @app.exception_handler(RequestValidationError)
    async def invalid_body(_: Request, error: RequestValidationError) -> JSONResponse:
        return error_response(400, str(error.errors()[0]["msg"]), "invalid_request_error")

    def to_sampling(body: ChatRequest) -> SamplingParams:
        if body.n != 1:
            raise InvalidRequest("only n=1 is supported")
        if body.stop:
            raise InvalidRequest("stop strings are not supported")
        if body.presence_penalty or body.frequency_penalty:
            raise InvalidRequest("presence_penalty and frequency_penalty are not supported")
        max_tokens = body.max_completion_tokens or body.max_tokens or default_max_tokens
        try:
            return SamplingParams(
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                repetition_penalty=body.repetition_penalty,
                seed=body.seed,
                stop_tokens=() if body.ignore_eos else stop_tokens,
                max_tokens=max_tokens,
            )
        except ValueError as error:
            raise InvalidRequest(str(error)) from error

    def encode(body: ChatRequest) -> torch.Tensor:
        messages = [{"role": m.role, "content": m.text()} for m in body.messages]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return torch.tensor(ids, dtype=torch.long)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request):
        try:
            params = to_sampling(body)
            prompt = encode(body)
            loop = asyncio.get_running_loop()
            events: asyncio.Queue[Event] = asyncio.Queue()

            def sink(event: Event) -> None:
                # Runs on the engine thread. This call only schedules work, so it never blocks.
                loop.call_soon_threadsafe(events.put_nowait, event)

            request_id = await asyncio.wrap_future(engine.submit(prompt, params, sink))
        except PromptTooLong as error:
            return error_response(
                400, str(error), "invalid_request_error", "context_length_exceeded"
            )
        except InvalidRequest as error:
            return error_response(400, str(error), "invalid_request_error")
        except QueueFull as error:
            return error_response(429, str(error), "rate_limit_error", "queue_full")

        name = body.model or model_name
        reply = Reply(request_id, events, IncrementalDecoder(tokenizer), prompt.numel(), name)
        if body.stream:
            include_usage = bool(body.stream_options and body.stream_options.include_usage)
            return StreamingResponse(
                stream(reply, request, engine, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await complete(reply, request, engine)

    @app.get("/health")
    async def health() -> JSONResponse:
        ok = engine.alive
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "status": "ok" if ok else "engine loop stopped",
                "active": len(scheduler.requests) - len(scheduler.waiting),
                "waiting": len(scheduler.waiting),
                "free_slots": len(scheduler.free_slots),
                "total_slots": scheduler.cache.num_slots,
            },
        )

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(render_metrics(engine), media_type="text/plain; version=0.0.4")

    return app


class Reply:
    """One request's view of its events, turned into text."""

    def __init__(
        self,
        request_id: int,
        events: "asyncio.Queue[Event]",
        decoder: IncrementalDecoder,
        prompt_tokens: int,
        model: str,
    ) -> None:
        self.request_id = request_id
        self.events = events
        self.decoder = decoder
        self.prompt_tokens = prompt_tokens
        self.model = model
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.completion_tokens = 0
        self.finish_reason: str | None = None
        self.error: str | None = None
        self.checked_at = time.monotonic()

    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def absorb(self, event: Event) -> str:
        """Take in one event, and return any text it completes."""
        text = ""
        if event.token is not None:
            self.completion_tokens += 1
            # A stop token belongs to the protocol, not to the answer.
            if event.finish_reason != "stop":
                text = self.decoder.push(event.token)
        if event.finish_reason is not None:
            self.finish_reason = event.finish_reason
            self.error = event.error
            text += self.decoder.flush()
        return text

    def _event(self, choices: list[dict], **extra: Any) -> str:
        body = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
            **extra,
        }
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    def chunk(self, delta: dict, finish_reason: str | None = None) -> str:
        return self._event([{"index": 0, "delta": delta, "finish_reason": finish_reason}])

    def usage_chunk(self) -> str:
        return self._event([], usage=self.usage())


async def next_event(reply: Reply, request: Request) -> Event | None:
    """Wait for the next event. Returns None if the client has gone away.

    The check runs on a timer, because a busy request always has an event ready and
    would otherwise never look.
    """
    while True:
        now = time.monotonic()
        if now - reply.checked_at > DISCONNECT_CHECK_SECONDS:
            reply.checked_at = now
            if await request.is_disconnected():
                return None
        try:
            return await asyncio.wait_for(reply.events.get(), DISCONNECT_CHECK_SECONDS)
        except TimeoutError:
            continue


async def complete(reply: Reply, request: Request, engine: EngineLoop) -> JSONResponse:
    pieces = []
    while reply.finish_reason is None:
        event = await next_event(reply, request)
        if event is None:
            engine.cancel(reply.request_id)
            return error_response(499, "client disconnected", "client_error")
        pieces.append(reply.absorb(event))

    if reply.finish_reason == "error":
        return error_response(500, reply.error or "request failed", "server_error")
    return JSONResponse(
        {
            "id": reply.id,
            "object": "chat.completion",
            "created": reply.created,
            "model": reply.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(pieces)},
                    "finish_reason": reply.finish_reason,
                }
            ],
            "usage": reply.usage(),
        }
    )


async def stream(
    reply: Reply, request: Request, engine: EngineLoop, include_usage: bool
) -> AsyncIterator[str]:
    finished = False
    try:
        yield reply.chunk({"role": "assistant", "content": ""})
        while reply.finish_reason is None:
            event = await next_event(reply, request)
            if event is None:
                return
            text = reply.absorb(event)
            if text:
                yield reply.chunk({"content": text})

        finished = True
        if reply.finish_reason == "error":
            body = {"error": {"message": reply.error, "type": "server_error"}}
            yield f"data: {json.dumps(body)}\n\n"
        else:
            yield reply.chunk({}, reply.finish_reason)
            if include_usage:
                yield reply.usage_chunk()
        yield "data: [DONE]\n\n"
    finally:
        # Reached on a normal end, and also when the client disconnects, because the web
        # server then closes this generator.
        if not finished:
            engine.cancel(reply.request_id)


def render_metrics(engine: EngineLoop) -> str:
    """Report in the Prometheus text format, written by hand to avoid a dependency."""
    scheduler = engine.scheduler
    stats = scheduler.stats
    lines = [
        f"llm_engine_slots_total {scheduler.cache.num_slots}",
        f"llm_engine_slots_free {len(scheduler.free_slots)}",
        f"llm_engine_requests_waiting {len(scheduler.waiting)}",
        f"llm_engine_requests_prefilling {len(scheduler.prefilling)}",
        f"llm_engine_requests_decoding {len(scheduler.decoding)}",
        f"llm_engine_cache_bytes {scheduler.cache.nbytes}",
        f"llm_engine_steps_total {stats.steps}",
        f"llm_engine_prefill_passes_total {stats.prefill_passes}",
        f"llm_engine_prefill_tokens_total {stats.prefill_tokens}",
        f"llm_engine_decode_passes_total {stats.decode_passes}",
        f"llm_engine_decode_rows_total {stats.decode_rows}",
        f"llm_engine_tokens_generated_total {stats.tokens_generated}",
        f"llm_engine_step_seconds_total {stats.step_seconds}",
        f"llm_engine_prefill_seconds_total {stats.prefill_seconds}",
        f"llm_engine_decode_seconds_total {stats.decode_seconds}",
        f"llm_engine_sampling_seconds_total {stats.sampling_seconds}",
        f"llm_engine_crashed_steps_total {engine.crashed_steps}",
    ]
    for reason, count in sorted(stats.finished.items()):
        lines.append(f'llm_engine_requests_finished_total{{reason="{reason}"}} {count}')
    for reason, count in sorted(stats.refused.items()):
        lines.append(f'llm_engine_requests_refused_total{{reason="{reason}"}} {count}')
    return "\n".join(lines) + "\n"


def main() -> None:
    import uvicorn
    from transformers import AutoTokenizer, GenerationConfig

    from engine.loader import load_pretrained

    parser = argparse.ArgumentParser(description="Serve a Qwen2 checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--prefill-tokens-per-step", type=int, default=256)
    parser.add_argument("--max-waiting", type=int, default=1024)
    parser.add_argument("--cache-budget-mb", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model = load_pretrained(
        args.model_path, dtype=getattr(torch, args.dtype), device=args.device
    )
    budget = args.cache_budget_mb * 2**20 if args.cache_budget_mb else None
    scheduler = Scheduler(
        model,
        args.slots,
        args.max_length,
        prefill_tokens_per_step=args.prefill_tokens_per_step,
        max_waiting=args.max_waiting,
        cache_budget_bytes=budget,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eos = GenerationConfig.from_pretrained(args.model_path).eos_token_id
    stop_tokens = tuple(eos) if isinstance(eos, list) else (eos,)
    app = create_app(scheduler, tokenizer, args.model_name, stop_tokens)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

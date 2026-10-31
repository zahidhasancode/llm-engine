"""Hugging Face generate behind the same API as the engine, one request at a time.

This is the baseline for the benchmark. Requests queue up and `model.generate` runs each one
to the end before the next starts, which is how a simple server around generate behaves.
It uses Hugging Face's default attention (SDPA), since eager gives NaN on this model in float16.

Run from the repository root:
PYTHONPATH=. python scripts/hf_server.py --model-path DIR --port 8001
"""

import argparse
import concurrent.futures
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Qwen2ForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer

from engine.sampling import SamplingParams
from engine.scheduler import Event, InvalidRequest, PromptTooLong
from engine.server import create_app


class CancelFlag(StoppingCriteria):
    def __init__(self) -> None:
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],), self.cancelled, dtype=torch.bool, device=input_ids.device
        )


class TokenStreamer(BaseStreamer):
    """Pass each new token on. Hugging Face sends the prompt through first, so skip it."""

    def __init__(self, on_token: Callable[[int], None]) -> None:
        self.on_token = on_token
        self.seen_prompt = False

    def put(self, value: torch.Tensor) -> None:
        if not self.seen_prompt:
            self.seen_prompt = True
            return
        self.on_token(value.flatten()[0].item())

    def end(self) -> None:
        pass


@dataclass
class Job:
    id: int
    prompt: torch.Tensor
    params: SamplingParams
    sink: Callable[[Event], None]
    flag: CancelFlag


class HfLoop:
    """One worker thread, one request at a time. Same interface as EngineLoop."""

    def __init__(self, model: Qwen2ForCausalLM, max_length: int) -> None:
        self.model = model
        self.max_length = max_length
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run, name="hf-loop", daemon=True)
        self.running = False
        self.flags: dict[int, CancelFlag] = {}
        self.next_id = 0
        self.max_running = 0
        self.waiting = 0
        self.busy = False

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.jobs.put(None)
        self.thread.join(timeout=10)

    @property
    def alive(self) -> bool:
        return self.running and self.thread.is_alive()

    def status(self) -> dict:
        return {"active": int(self.busy), "waiting": self.waiting}

    def metrics(self) -> str:
        return f"hf_requests_waiting {self.waiting}\n"

    def submit(self, prompt, params: SamplingParams, sink) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        if prompt.numel() + params.max_tokens > self.max_length:
            future.set_exception(
                PromptTooLong(
                    f"prompt of {prompt.numel()} tokens plus max_tokens {params.max_tokens} "
                    f"is over the limit of {self.max_length}"
                )
            )
            return future
        if prompt.numel() == 0:
            future.set_exception(InvalidRequest("prompt must have at least one token"))
            return future
        job = Job(self.next_id, prompt, params, sink, CancelFlag())
        self.next_id += 1
        self.flags[job.id] = job.flag
        self.waiting += 1
        self.jobs.put(job)
        future.set_result(job.id)
        return future

    def cancel(self, request_id: int) -> None:
        flag = self.flags.get(request_id)
        if flag is not None:
            flag.cancelled = True

    def _run(self) -> None:
        while self.running:
            job = self.jobs.get()
            if job is None:
                continue
            self.waiting -= 1
            self.busy = True
            try:
                self._generate(job)
            except Exception as error:
                job.sink(
                    Event(job.id, finish_reason="error", error=f"{type(error).__name__}: {error}")
                )
            finally:
                self.busy = False
                self.flags.pop(job.id, None)

    def _generate(self, job: Job) -> None:
        if job.flag.cancelled:
            job.sink(Event(job.id, finish_reason="cancelled"))
            return

        params = job.params
        sampled = params.temperature > 0
        config = GenerationConfig(
            do_sample=sampled,
            temperature=params.temperature if sampled else None,
            top_k=params.top_k if sampled else None,
            top_p=params.top_p if sampled else None,
            repetition_penalty=params.repetition_penalty,
            max_new_tokens=params.max_tokens,
            eos_token_id=list(params.stop_tokens) or None,
            pad_token_id=0,
        )
        last = []

        def on_token(token: int) -> None:
            last[:] = [token]
            job.sink(Event(job.id, token))

        ids = job.prompt[None].to(self.model.device)
        self.model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            generation_config=config,
            streamer=TokenStreamer(on_token),
            stopping_criteria=StoppingCriteriaList([job.flag]),
        )
        if job.flag.cancelled:
            reason = "cancelled"
        elif last and last[0] in params.stop_tokens:
            reason = "stop"
        else:
            reason = "length"
        job.sink(Event(job.id, finish_reason=reason))


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    model = Qwen2ForCausalLM.from_pretrained(args.model_path, dtype=getattr(torch, args.dtype))
    model = model.to(args.device).eval()
    # The checkpoint's defaults turn on sampling and a repetition penalty.
    eos = GenerationConfig.from_pretrained(args.model_path).eos_token_id
    model.generation_config = GenerationConfig()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    stop_tokens = tuple(eos) if isinstance(eos, list) else (eos,)
    app = create_app(HfLoop(model, args.max_length), tokenizer, args.model_name, stop_tokens)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

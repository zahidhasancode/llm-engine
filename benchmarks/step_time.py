"""Measure one decode step of my engine directly, with no HTTP, and split up its time.

Step time comes from a difference of two runs: one that stops after prefill, and one that also
decodes N more tokens. Everything except the decode steps cancels out, so the same method works
on vLLM, which does not expose its steps. See benchmarks/vllm_step_time.py.

Attention variants show what the attention function costs, and a timed attention shows where
the time goes inside a step. Every piece waits for the GPU, so the pieces are real but slower
than a normal run, and they are for comparing shares only.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from benchmarks.real_model import SdpaAttention, SdpaNoMaskAttention, with_attention
from engine.loader import load_pretrained
from engine.model import Attention, Qwen2, apply_rotary
from engine.sampling import SamplingParams
from engine.scheduler import Scheduler


def synchronise(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def prompt_ids(vocab_size: int, length: int, seed: int) -> Tensor:
    """Fixed token ids. vllm_step_time.py builds the identical list."""
    return torch.randint(0, vocab_size, (length,), generator=torch.Generator().manual_seed(seed))


def run_once(model: Qwen2, batch: int, prompt_length: int, new_tokens: int) -> tuple[float, Scheduler]:
    engine = Scheduler(
        model, batch, prompt_length + new_tokens + 1, prefill_tokens_per_step=prompt_length * batch
    )
    params = SamplingParams(temperature=0.0, max_tokens=new_tokens)
    device = str(model.tok_embed.weight.device)
    for seed in range(batch):
        engine.submit(prompt_ids(model.config.vocab_size, prompt_length, seed), params)
    synchronise(device)
    start = time.perf_counter()
    engine.run()
    synchronise(device)
    return time.perf_counter() - start, engine


def ms_per_step(
    model: Qwen2, batch: int, prompt_length: int, steps: int, repeats: int
) -> dict:
    """Time per decode pass, by the difference between a short run and a long run."""
    run_once(model, batch, prompt_length, 1 + steps)  # warm up
    short = [run_once(model, batch, prompt_length, 1)[0] for _ in range(repeats)]
    long_runs = [run_once(model, batch, prompt_length, 1 + steps) for _ in range(repeats)]
    long_seconds = [seconds for seconds, _ in long_runs]
    stats = long_runs[-1][1].stats
    step = (statistics.median(long_seconds) - statistics.median(short)) / steps
    return {
        "batch": batch,
        "ms_per_step": step * 1000,
        "tokens_per_second": batch / step,
        "sampling_share": stats.sampling_seconds / stats.step_seconds,
        "model_share": (stats.decode_seconds + stats.prefill_seconds) / stats.step_seconds,
        "short_runs_s": short,
        "long_runs_s": long_seconds,
    }


class TimedAttention(Attention):
    """The same attention with a stopwatch around each part."""

    clock: dict = defaultdict(float)
    device: str = "cuda"

    def _tick(self, name: str, started: float) -> float:
        synchronise(self.device)
        now = time.perf_counter()
        self.clock[name] += now - started
        return now

    def forward(self, x, cos, sin, future, cache, layer):
        batch, seq_len, _ = x.shape
        start = time.perf_counter()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        start = self._tick("q_k_v_projections", start)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        start = self._tick("rotary_rotation", start)
        k, v = cache.update(layer, k, v)
        start = self._tick("cache_write_and_read", start)
        k = k.repeat_interleave(self.queries_per_kv, dim=1)
        v = v.repeat_interleave(self.queries_per_kv, dim=1)
        start = self._tick("repeat_keys_and_values", start)
        out = self.attend(q, k, v, future)
        start = self._tick("attention_product", start)
        out = self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, -1))
        self._tick("output_projection", start)
        return out


@torch.no_grad()
def step_parts(model: Qwen2, batch: int, prompt_length: int, steps: int, device: str) -> dict:
    """Split decode steps into parts. Timing is only switched on after prefill."""
    timed = with_attention(model, TimedAttention)
    clock: dict = defaultdict(float)
    TimedAttention.clock = clock
    TimedAttention.device = device

    def timed_block(module, name):
        def before(*_):
            synchronise(device)
            module._started = time.perf_counter()

        def after(*_):
            synchronise(device)
            clock[name] += time.perf_counter() - module._started

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)

    for block in timed.blocks:
        timed_block(block.mlp, "mlp")
        timed_block(block.norm_attn, "rms_norm")
        timed_block(block.norm_mlp, "rms_norm")
    timed_block(timed.lm_head, "lm_head")
    timed_block(timed.tok_embed, "embedding")

    engine = Scheduler(
        timed, batch, prompt_length + steps + 2, prefill_tokens_per_step=prompt_length * batch
    )
    params = SamplingParams(temperature=0.0, max_tokens=steps + 1)
    for seed in range(batch):
        engine.submit(prompt_ids(model.config.vocab_size, prompt_length, seed), params)
    engine._admit()
    events: list = []
    engine._prefill(events)
    clock.clear()
    stats = engine.stats
    stats.sampling_seconds = 0.0
    started = time.perf_counter()
    while engine.has_work:
        engine.step()
    wall = time.perf_counter() - started

    clock["sampling"] = stats.sampling_seconds
    measured = sum(clock.values())
    parts = {name: seconds / steps * 1000 for name, seconds in sorted(clock.items())}
    parts["other_python_ms"] = (wall - measured) / steps * 1000
    return {"batch": batch, "steps": steps, "ms_per_step_with_stopwatches": wall / steps * 1000,
            "parts_ms": parts}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    model = load_pretrained(args.model_path, dtype=getattr(torch, args.dtype), device=args.device)
    variants = {
        "mine": model,
        "mine_with_sdpa": with_attention(model, SdpaAttention),
        "mine_with_sdpa_no_mask": with_attention(model, SdpaNoMaskAttention),
    }
    result = {
        "what": "Decode step time with no HTTP, by the difference between two runs",
        "dtype": args.dtype,
        "prompt_length": args.prompt_length,
        "steps": args.steps,
        "repeats": args.repeats,
        "search": "greedy, stop tokens off",
        "variants": {
            name: [
                ms_per_step(variant, batch, args.prompt_length, args.steps, args.repeats)
                for batch in args.batches
            ]
            for name, variant in variants.items()
        },
        "parts": [
            step_parts(model, batch, args.prompt_length, args.steps, args.device)
            for batch in args.batches
        ],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in result["variants"].items():
        print(name, [(row["batch"], round(row["ms_per_step"], 2)) for row in rows], flush=True)


if __name__ == "__main__":
    main()

"""Measure time per generated token with and without the KV cache, and save the result.

Uses a tiny random model on CPU. It shows how each mode scales with sequence length.
It says nothing about the speed of a real model.

Run from the repository root: PYTHONPATH=. .venv/bin/python benchmarks/cache_benchmark.py
"""

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from engine.cache import KVCache
from engine.generate import generate
from engine.model import ModelConfig, Qwen2
from engine.sampling import SamplingParams

OUTPUT = Path(__file__).parent / "cache_cpu_tiny.json"

SEED = 0
THREADS = 1
VOCAB_SIZE = 97
PROMPT_LENGTH = 16
NEW_TOKEN_COUNTS = [32, 128, 512, 1024]
REPEATS = 7
WARMUP_RUNS = 2

TINY = ModelConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=32,
    intermediate_size=64,
    num_layers=3,
    num_heads=4,
    num_kv_heads=4,
)

# Arithmetic only, nothing is loaded. These are the defaults of transformers' Qwen2Config.
QWEN2_DEFAULT = ModelConfig(
    vocab_size=151936,
    hidden_size=4096,
    intermediate_size=22016,
    num_layers=32,
    num_heads=32,
    num_kv_heads=32,
)


def hardware() -> dict:
    sysctl = lambda name: subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "memory_gb": int(sysctl("hw.memsize")) / 2**30,
        "os": platform.platform(),
    }


def measure(model: Qwen2, ids: torch.Tensor, new_tokens: int, use_cache: bool) -> dict:
    params = SamplingParams(temperature=0.0, max_tokens=new_tokens)
    for _ in range(WARMUP_RUNS):
        generate(model, ids, params, use_cache=use_cache)

    seconds = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        output = generate(model, ids, params, use_cache=use_cache)[0]
        seconds.append(time.perf_counter() - start)

    median = statistics.median(seconds)
    return {
        "new_tokens": new_tokens,
        "ms_per_token_median": median / new_tokens * 1000,
        "ms_per_token_min": min(seconds) / new_tokens * 1000,
        "ms_per_token_max": max(seconds) / new_tokens * 1000,
        "output": output,
    }


def bytes_per_token(config: ModelConfig, dtype: torch.dtype) -> int:
    return KVCache(config, batch_size=1, max_length=1, dtype=dtype).nbytes


def main() -> None:
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    model = Qwen2(TINY).eval()
    ids = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LENGTH))

    rows = []
    for new_tokens in NEW_TOKEN_COUNTS:
        cached = measure(model, ids, new_tokens, use_cache=True)
        uncached = measure(model, ids, new_tokens, use_cache=False)
        if not torch.equal(cached.pop("output"), uncached.pop("output")):
            raise RuntimeError("cached and uncached output differ")
        rows.append(
            {
                "new_tokens": new_tokens,
                "final_length": PROMPT_LENGTH + new_tokens,
                "with_cache": cached,
                "without_cache": uncached,
                "speedup": uncached["ms_per_token_median"] / cached["ms_per_token_median"],
                # Forward-pass work: how many tokens go through the model in total.
                "tokens_processed_with_cache": PROMPT_LENGTH + new_tokens - 1,
                "tokens_processed_without_cache": sum(
                    range(PROMPT_LENGTH, PROMPT_LENGTH + new_tokens)
                ),
            }
        )

    longest = PROMPT_LENGTH + NEW_TOKEN_COUNTS[-1]
    result = {
        "what": "Greedy generation, ms per generated token, with and without the KV cache",
        "caveat": "Tiny random model. Shows scaling only, not real model speed.",
        "hardware": hardware(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "model": {
            "name": "Qwen2, tiny, random weights",
            "layers": TINY.num_layers,
            "hidden_size": TINY.hidden_size,
            "intermediate_size": TINY.intermediate_size,
            "heads": TINY.num_heads,
            "kv_heads": TINY.num_kv_heads,
            "vocab_size": TINY.vocab_size,
            "dtype": "float32",
        },
        "settings": {
            "device": "cpu",
            "torch_threads": THREADS,
            "batch_size": 1,
            "prompt_length": PROMPT_LENGTH,
            "repeats": REPEATS,
            "warmup_runs_discarded": WARMUP_RUNS,
            "seed": SEED,
            "timing": "whole generate() call divided by new tokens, including the prompt pass",
        },
        "results": rows,
        "cache_memory": {
            "formula": "2 * layers * batch * kv_heads * max_length * head_dim * bytes_per_value",
            "tiny_float32": {
                "bytes_per_token": bytes_per_token(TINY, torch.float32),
                "max_length": longest,
                "total_bytes": bytes_per_token(TINY, torch.float32) * longest,
            },
            "qwen2_config_defaults_float16_computed_not_measured": {
                "layers": QWEN2_DEFAULT.num_layers,
                "kv_heads": QWEN2_DEFAULT.num_kv_heads,
                "head_dim": QWEN2_DEFAULT.head_dim,
                "bytes_per_token": bytes_per_token(QWEN2_DEFAULT, torch.float16),
                "max_length": 4096,
                "total_gib": bytes_per_token(QWEN2_DEFAULT, torch.float16) * 4096 / 2**30,
            },
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {OUTPUT}")
    for row in rows:
        print(
            f"{row['new_tokens']:>5} new tokens: "
            f"{row['with_cache']['ms_per_token_median']:8.3f} ms with cache, "
            f"{row['without_cache']['ms_per_token_median']:8.3f} ms without, "
            f"{row['speedup']:6.1f}x"
        )


if __name__ == "__main__":
    main()

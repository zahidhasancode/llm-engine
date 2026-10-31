"""Measure a decode step of vLLM by the same difference as benchmarks/step_time.py.

This runs in vLLM's own environment, so it imports nothing from the engine. Token ids must
match step_time.prompt_ids: both use torch.randint with the same seed, length and vocabulary.
Prefix caching is off, or the second run would skip its prefill and the difference would be wrong.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

VOCAB_SIZE = 151936


def prompt_ids(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (length,), generator=generator).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--enforce-eager", action="store_true", help="no compile or CUDA graphs")
    parser.add_argument("--attention-backend", default=None, help="vLLM's name for the backend")
    args = parser.parse_args()

    extra = {"attention_config": {"backend": args.attention_backend}} if args.attention_backend else {}
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        generation_config="vllm",
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        **extra,
    )

    def run(batch: int, tokens: int) -> float:
        prompts = [
            {"prompt_token_ids": prompt_ids(args.prompt_length, seed)} for seed in range(batch)
        ]
        params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        assert all(len(output.outputs[0].token_ids) == tokens for output in outputs)
        return seconds

    rows = []
    for batch in args.batches:
        run(batch, 1 + args.steps)  # warm up
        short = [run(batch, 1) for _ in range(args.repeats)]
        long_runs = [run(batch, 1 + args.steps) for _ in range(args.repeats)]
        step = (statistics.median(long_runs) - statistics.median(short)) / args.steps
        rows.append(
            {
                "batch": batch,
                "ms_per_step": step * 1000,
                "tokens_per_second": batch / step,
                "short_runs_s": short,
                "long_runs_s": long_runs,
            }
        )
        print(batch, round(step * 1000, 2), flush=True)

    result = {
        "what": "vLLM decode step time, by the same difference as step_time.py",
        "dtype": args.dtype,
        "prompt_length": args.prompt_length,
        "steps": args.steps,
        "repeats": args.repeats,
        "enforce_eager": args.enforce_eager,
        "attention_backend": args.attention_backend,
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

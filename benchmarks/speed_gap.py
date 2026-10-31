"""Split the throughput gap between my engine and vLLM into measured pieces.

The serving benchmark shows my engine serving far fewer tokens than its raw step time allows.
Each run here removes one thing, so the difference shows what that thing costs:

    http           the full server, as in the benchmark, as a control in this session
    direct         the same requests through the Scheduler, with no HTTP
    direct_short   the same again with short prompts only. A decode pass reads every slot out
                   to the longest one, so one long prompt slows all the others
    step_time      raw decode steps at larger batches, to see what more slots would give

The server runs first and is stopped before the model loads here, so nothing shares the GPU.
Run from the repository root, normally through kaggle/serving_run.py.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor

from benchmarks import serving_benchmark as bench
from benchmarks.step_time import ms_per_step
from engine.loader import load_pretrained
from engine.model import Qwen2
from engine.sampling import SamplingParams
from engine.scheduler import Scheduler
from scripts import loadtest


def direct_run(
    model: Qwen2,
    prompts: list[Tensor],
    slots: int,
    max_length: int,
    prefill_tokens_per_step: int,
    max_tokens: int,
) -> dict:
    """Run every prompt with all slots kept full, and say where the time went."""
    engine = Scheduler(
        model, slots, max_length, prefill_tokens_per_step=prefill_tokens_per_step,
        max_waiting=len(prompts),
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    for prompt in prompts:
        engine.submit(prompt, params)
    start = time.perf_counter()
    engine.run()
    seconds = time.perf_counter() - start
    stats = engine.stats
    return {
        "seconds": seconds,
        "tokens": stats.tokens_generated,
        "tokens_per_second": stats.tokens_generated / seconds,
        "prefill_seconds": stats.prefill_seconds,
        "decode_seconds": stats.decode_seconds,
        "sampling_seconds": stats.sampling_seconds,
        "decode_passes": stats.decode_passes,
        "mean_rows_per_decode_pass": stats.decode_rows / stats.decode_passes,
        "ms_per_decode_pass": stats.decode_seconds / stats.decode_passes * 1000,
    }


def sample_prompts(
    prompts: list[Tensor], count: int, seed: int, longest: int | None = None
) -> list[Tensor]:
    """The prompts a load test with this seed would send, from those short enough."""
    pool = [p for p in prompts if longest is None or p.numel() <= longest]
    return [pool[index] for index in loadtest.pick_prompts(len(pool), count, seed)]


def medians(runs: list[dict]) -> dict:
    return {key: statistics.median(run[key] for run in runs) for key in runs[0]} | {"runs": runs}


def http_runs(args: argparse.Namespace, out: Path, slots: int, requests: int) -> dict:
    settings = argparse.Namespace(**vars(args) | {"slots": slots})
    system = next(s for s in bench.systems(settings) if s.name == "my_engine")
    server = bench.Server(system, out / "logs" / f"gap_http_slots{slots}.log")
    runs = []
    try:
        server.wait_until_ready()
        bench.measure(settings, server, out, f"gap_slots{slots}_warmup", 999, "--mode", "closed",
                      "--users", "2", "--requests", "6")
        for repeat in range(args.repeats):
            point = bench.measure(settings, server, out, f"gap_slots{slots}", repeat, "--mode",
                                  "closed", "--users", str(system.closed_users),
                                  "--requests", str(requests))
            runs.append({"tokens_per_second": point["summary"]["tokens_per_second"]}
                        | point["time_split"])
            print("http", slots, round(runs[-1]["tokens_per_second"], 1), flush=True)
    finally:
        server.stop()
    return medians(runs)


def direct_runs(
    args: argparse.Namespace,
    model: Qwen2,
    prompts: list[Tensor],
    slots: int,
    requests: int,
    longest: int | None = None,
) -> dict:
    run = lambda seed: direct_run(
        model, sample_prompts(prompts, requests, seed, longest), slots, args.max_length,
        args.prefill_tokens_per_step, args.max_tokens,
    )
    run(999)  # warm up, with a different sample
    runs = [run(repeat) for repeat in range(args.repeats)]
    print("direct", slots, longest, round(medians(runs)["tokens_per_second"], 1), flush=True)
    return medians(runs)


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--prefill-tokens-per-step", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--short-prompt-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64, 128])
    args = parser.parse_args(argv)
    args.vllm_bin = "vllm"

    out = Path(args.output_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in ("output_dir", "vllm_bin")}
    result: dict = {"settings": settings, "search": bench.GREEDY}
    path = out / "speed_gap.json"

    def save() -> None:
        path.write_text(json.dumps(result, indent=2) + "\n")

    result["http_16_slots"] = http_runs(args, out, 16, 256)
    save()
    result["http_64_slots"] = http_runs(args, out, 64, 512)
    save()

    model = load_pretrained(args.model_path, dtype=getattr(torch, args.dtype), device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [
        torch.tensor(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=True, return_dict=False,
            ),
            dtype=torch.long,
        )
        for text in loadtest.load_prompts(args.prompts)
    ]
    result["prompt_tokens"] = {
        "median": statistics.median(p.numel() for p in prompts),
        "longest": max(p.numel() for p in prompts),
        "short_prompts": sum(p.numel() <= args.short_prompt_tokens for p in prompts),
        "prompts": len(prompts),
    }
    result["direct_16_slots"] = direct_runs(args, model, prompts, 16, 256)
    save()
    result["direct_16_slots_short_prompts"] = direct_runs(
        args, model, prompts, 16, 256, args.short_prompt_tokens
    )
    save()
    result["direct_64_slots"] = direct_runs(args, model, prompts, 64, 512)
    save()
    result["direct_64_slots_short_prompts"] = direct_runs(
        args, model, prompts, 64, 512, args.short_prompt_tokens
    )
    save()
    result["step_time"] = [
        ms_per_step(model, batch, 256, 64, args.repeats) for batch in args.batches
    ]
    save()


if __name__ == "__main__":
    main()

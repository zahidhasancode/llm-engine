"""Raise the request rate on each system until it saturates, and save everything.

Each system runs as its own server process, so nothing leaks between them, and the same load
tool, prompts and settings are used for all. Each point is repeated with a different seed,
and the same seeds are used for every system, so they all see identical traffic.

Run from the repository root, normally through kaggle/serving_run.py.
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from scripts import loadtest

GREEDY = '{"top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}'


@dataclass
class System:
    name: str
    command: list[str]
    port: int
    # A closed-loop test uses this many users, enough to keep every slot busy.
    closed_users: int


def systems(args: argparse.Namespace) -> list[System]:
    python = sys.executable
    return [
        System(
            "huggingface",
            [
                python, "scripts/hf_server.py",
                "--model-path", args.model_path,
                "--model-name", "qwen",
                "--dtype", args.dtype,
                "--max-length", str(args.max_length),
                "--port", "8001",
            ],
            8001,
            2,
        ),
        System(
            "my_engine",
            [
                python, "-m", "engine.server",
                "--model-path", args.model_path,
                "--model-name", "qwen",
                "--dtype", args.dtype,
                "--slots", str(args.slots),
                "--max-length", str(args.max_length),
                "--prefill-tokens-per-step", str(args.prefill_tokens_per_step),
                "--max-waiting", "100000",
                "--port", "8002",
            ],
            8002,
            2 * args.slots,
        ),
        System(
            "vllm",
            [
                args.vllm_bin, "serve", args.model_path,
                "--served-model-name", "qwen",
                "--dtype", args.dtype,
                "--max-model-len", str(args.max_length),
                "--gpu-memory-utilization", "0.85",
                "--generation-config", "vllm",
                "--no-enable-prefix-caching",
                "--port", "8003",
            ],
            8003,
            # vLLM has no fixed slot count, so it needs more users to show its maximum.
            8 * args.slots,
        ),
    ]


def is_saturated(summary: dict, offered_rate: float, latency_limit_ms: float) -> bool:
    """A point is saturated if the system falls behind arrivals, fails, or latency explodes."""
    if summary["errors"] > 0.01 * summary["requests"]:
        return True
    if summary["requests_per_second"] < 0.85 * offered_rate:
        return True
    p95 = summary["e2e_ms"]["p95"]
    return p95 is None or p95 > latency_limit_ms


def parse_metrics(text: str) -> dict[str, float]:
    """Read the Prometheus text format into a dictionary."""
    metrics = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            metrics[name] = float(value)
    return metrics


def time_split(before: dict[str, float], after: dict[str, float]) -> dict:
    """Where the engine's time went between two metric readings."""
    change = lambda name: after.get(name, 0.0) - before.get(name, 0.0)
    steps = change("llm_engine_step_seconds_total")
    prefill = change("llm_engine_prefill_seconds_total")
    decode = change("llm_engine_decode_seconds_total")
    sampling = change("llm_engine_sampling_seconds_total")
    passes = change("llm_engine_decode_passes_total")
    rows = change("llm_engine_decode_rows_total")
    return {
        "step_seconds": steps,
        "prefill_seconds": prefill,
        "decode_seconds": decode,
        "sampling_seconds": sampling,
        "other_step_seconds": steps - prefill - decode - sampling,
        "decode_passes": passes,
        "mean_rows_per_decode_pass": rows / passes if passes else None,
        "ms_per_decode_pass": decode / passes * 1000 if passes else None,
    }


def median_summary(points: list[dict]) -> dict:
    """The median across repeats, plus the spread, for the numbers the report uses."""
    def collect(pick: Callable[[dict], float | None]) -> dict | None:
        values = [pick(p["summary"]) for p in points]
        if any(value is None for value in values):
            return None
        return {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "tokens_per_second": collect(lambda s: s["tokens_per_second"]),
        "requests_per_second": collect(lambda s: s["requests_per_second"]),
        "ttft_p50_ms": collect(lambda s: s["ttft_ms"]["p50"]),
        "ttft_p95_ms": collect(lambda s: s["ttft_ms"]["p95"]),
        "ttft_p99_ms": collect(lambda s: s["ttft_ms"]["p99"]),
        "inter_token_p50_ms": collect(lambda s: s["inter_token_ms"]["p50"]),
        "inter_token_p95_ms": collect(lambda s: s["inter_token_ms"]["p95"]),
        "inter_token_p99_ms": collect(lambda s: s["inter_token_ms"]["p99"]),
        "e2e_p50_ms": collect(lambda s: s["e2e_ms"]["p50"]),
        "e2e_p95_ms": collect(lambda s: s["e2e_ms"]["p95"]),
        "e2e_p99_ms": collect(lambda s: s["e2e_ms"]["p99"]),
        "errors": collect(lambda s: s["errors"]),
        "requests": collect(lambda s: s["requests"]),
    }


def sweep(
    rates: list[float],
    repeats: int,
    run_point: Callable[[float, int], dict],
    latency_limit_ms: float,
    out_of_time: Callable[[], bool] = lambda: False,
) -> list[dict]:
    """Raise the rate until a point saturates, on the median of its repeats."""
    results = []
    for rate in rates:
        points = []
        for repeat in range(repeats):
            if out_of_time():
                break
            points.append(run_point(rate, repeat))
        if not points:
            break
        saturated = [is_saturated(p["summary"], rate, latency_limit_ms) for p in points]
        results.append(
            {
                "offered_rate": rate,
                "repeats": len(points),
                "saturated": sum(saturated) > len(saturated) / 2,
                "saturated_repeats": sum(saturated),
                "median": median_summary(points),
                "time_split": [p.get("time_split") for p in points],
            }
        )
        if results[-1]["saturated"]:
            break
    return results


class Server:
    """A server process, started and stopped cleanly so the GPU is free afterwards."""

    def __init__(self, system: System, log_path: Path, env: dict | None = None) -> None:
        self.system = system
        self.url = f"http://127.0.0.1:{system.port}"
        self.log = open(log_path, "w")
        self.process = subprocess.Popen(
            system.command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=(env or os.environ) | {"PYTHONPATH": os.getcwd()},
            start_new_session=True,
        )

    def wait_until_ready(self, timeout: float = 900) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.system.name} exited with {self.process.returncode}")
            try:
                if httpx.get(f"{self.url}/health", timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError(f"{self.system.name} did not start in {timeout} s")

    def metrics(self) -> dict[str, float]:
        try:
            return parse_metrics(httpx.get(f"{self.url}/metrics", timeout=30).text)
        except (httpx.HTTPError, ValueError):
            return {}

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.log.close()
        time.sleep(5)


def load_settings(args: argparse.Namespace, server: Server, seed: int, raw: Path) -> list[str]:
    return [
        "--base-url", server.url,
        "--model", "qwen",
        "--prompts", args.prompts,
        "--max-tokens", str(args.max_tokens),
        "--ignore-eos",
        "--temperature", "0",
        "--extra-body", GREEDY,
        "--seed", str(seed),
        "--output", str(raw),
        "--quiet",
    ]


def measure(args, server, out, label, seed, *load) -> dict:
    before = server.metrics()
    raw = out / "raw" / f"{server.system.name}_{label}_seed{seed}.json"
    result = loadtest.main(load_settings(args, server, seed, raw) + list(load))
    return {"summary": result["summary"], "time_split": time_split(before, server.metrics())}


def run_system(args: argparse.Namespace, system: System, out: Path, deadline: float) -> dict:
    print(f"=== {system.name}: {' '.join(system.command)}", flush=True)
    server = Server(system, out / "logs" / f"{system.name}.log")
    try:
        server.wait_until_ready()
        # A short warm-up, so the first point does not pay for compilation or caches.
        measure(args, server, out, "warmup", 999, "--mode", "closed", "--users", "2",
                "--requests", "6")

        def run_point(rate: float, repeat: int) -> dict:
            count = max(args.min_requests, round(rate * args.duration))
            point = measure(args, server, out, f"rate{rate}", repeat, "--mode", "poisson",
                            "--rate", str(rate), "--requests", str(count))
            summary = point["summary"]
            print(
                f"{system.name} rate {rate} repeat {repeat}: "
                f"{summary['tokens_per_second']:.1f} tok/s, "
                f"{summary['requests_per_second']:.2f} req/s, "
                f"e2e p95 {summary['e2e_ms']['p95']}, errors {summary['errors']}",
                flush=True,
            )
            return point

        points = sweep(
            args.rates, args.repeats, run_point, args.latency_limit_ms,
            lambda: time.monotonic() > deadline,
        )

        closed = []
        for repeat in range(args.repeats):
            if time.monotonic() > deadline:
                break
            closed.append(
                measure(args, server, out, "closed", repeat, "--mode", "closed", "--users",
                        str(system.closed_users), "--requests",
                        str(max(args.min_requests, 8 * system.closed_users)))
            )
        return {
            "name": system.name,
            "command": system.command,
            "points": points,
            "closed_users": system.closed_users,
            "closed_loop": median_summary(closed) if closed else None,
            "closed_time_split": [p.get("time_split") for p in closed],
        }
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--systems", nargs="+", default=["huggingface", "my_engine", "vllm"])
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--prefill-tokens-per-step", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--rates", type=float, nargs="+",
        default=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2],
    )
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--min-requests", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latency-limit-ms", type=float, default=120_000)
    parser.add_argument("--hours", type=float, default=8, help="stop starting new points after this")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600

    settings = {k: v for k, v in vars(args).items() if k not in ("output_dir", "vllm_bin")}
    results = {"settings": settings, "search": GREEDY, "systems": []}
    for system in systems(args):
        if system.name in args.systems:
            results["systems"].append(run_system(args, system, out, deadline))
            (out / "serving_benchmark.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

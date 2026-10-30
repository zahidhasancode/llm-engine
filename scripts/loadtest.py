"""Load test any OpenAI-compatible server, using streaming chat completions.

Two ways to create load:

    --mode poisson --rate R   requests arrive at random, R per second on average. This is an
                              open loop: arrivals do not wait for the server, like real users.
    --mode closed --users N   N users each send a request, wait for the answer, and repeat.

In Poisson mode latency is measured from when a request was scheduled to arrive, not from
when it was sent. If the client ever falls behind, that delay still counts. Measuring from
the send time would hide the slowness a load test exists to find.

Run it the same way against any server, so results can be compared directly:

    python scripts/loadtest.py --base-url http://127.0.0.1:8000 --model local \\
        --prompts data/prompts.jsonl --mode poisson --rate 2 --duration 60 --output out.json
"""

import argparse
import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx


@dataclass
class Record:
    """Everything measured about one request. Times are seconds from the test's start."""

    index: int
    prompt_index: int
    prompt_chars: int
    scheduled_at: float
    sent_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    completion_tokens: int = 0
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    gaps_ms: list[float] = field(default_factory=list)


def percentile(values: list[float], share: float) -> float | None:
    """The value below which `share` of the data falls, by linear interpolation."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * share / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "mean": sum(values) / len(values) if values else None,
        "count": len(values),
    }


def arrival_times(rate: float, count: int, seed: int) -> list[float]:
    """Poisson arrivals: the gaps between them follow an exponential distribution."""
    rng = random.Random(seed)
    now = 0.0
    times = []
    for _ in range(count):
        now += rng.expovariate(rate)
        times.append(now)
    return times


def pick_prompts(num_prompts: int, count: int, seed: int) -> list[int]:
    """Which prompt each request uses. Separate from arrivals, so the two never interact."""
    rng = random.Random(seed + 1)
    return [rng.randrange(num_prompts) for _ in range(count)]


def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line)["prompt"] for line in file if line.strip()]


def parse_event(line: str) -> dict | None:
    """Read one server-sent event line. Returns None for anything that is not data."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return None
    return json.loads(payload)


async def send(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    record: Record,
    started: float,
) -> None:
    """Send one streaming request and fill in its record. Never raises."""
    now = lambda: time.perf_counter() - started
    record.sent_at = now()
    previous = None
    chunks = 0
    usage_tokens = None
    try:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")[:200]
                record.error = f"HTTP {response.status_code}: {body}"
                return
            async for line in response.aiter_lines():
                event = parse_event(line)
                if event is None:
                    continue
                if "error" in event:
                    record.error = f"stream error: {event['error']}"
                    return
                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens")
                    record.prompt_tokens = event["usage"].get("prompt_tokens")
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        record.finish_reason = choice["finish_reason"]
                    if choice.get("delta", {}).get("content"):
                        arrived = now()
                        if previous is None:
                            record.first_token_at = arrived
                        else:
                            record.gaps_ms.append((arrived - previous) * 1000)
                        previous = arrived
                        chunks += 1
        if record.finish_reason is None:
            record.error = "stream ended without a finish reason"
    except Exception as error:
        record.error = f"{type(error).__name__}: {error}"
    finally:
        record.finished_at = now()
        record.completion_tokens = usage_tokens if usage_tokens is not None else chunks


def build_payload(prompt: str, args: argparse.Namespace) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    return payload


async def run_poisson(client, url, prompts, args) -> list[Record]:
    count = args.requests or max(1, round(args.rate * args.duration))
    times = arrival_times(args.rate, count, args.seed)
    choices = pick_prompts(len(prompts), count, args.seed)
    records = [
        Record(i, choice, len(prompts[choice]), scheduled)
        for i, (scheduled, choice) in enumerate(zip(times, choices))
    ]

    started = time.perf_counter()

    async def one(record: Record) -> None:
        delay = record.scheduled_at - (time.perf_counter() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        payload = build_payload(prompts[record.prompt_index], args)
        await send(client, url, payload, record, started)

    await asyncio.gather(*(one(record) for record in records))
    return records


async def run_closed(client, url, prompts, args) -> list[Record]:
    count = args.requests or args.users * 10
    choices = pick_prompts(len(prompts), count, args.seed)
    records = [Record(i, c, len(prompts[c]), 0.0) for i, c in enumerate(choices)]
    pending = iter(records)
    started = time.perf_counter()

    async def user() -> None:
        for record in pending:
            record.scheduled_at = time.perf_counter() - started
            payload = build_payload(prompts[record.prompt_index], args)
            await send(client, url, payload, record, started)

    await asyncio.gather(*(user() for _ in range(args.users)))
    return records


def summarise(records: list[Record], args: argparse.Namespace) -> dict:
    good = [r for r in records if r.error is None]
    errors: dict[str, int] = {}
    for record in records:
        if record.error is not None:
            kind = record.error.split(":")[0]
            errors[kind] = errors.get(kind, 0) + 1

    first = min((r.scheduled_at for r in records), default=0.0)
    last = max((r.finished_at for r in records if r.finished_at is not None), default=first)
    wall = max(last - first, 1e-9)
    tokens = sum(r.completion_tokens for r in good)

    return {
        "requests": len(records),
        "succeeded": len(good),
        "errors": len(records) - len(good),
        "errors_by_kind": errors,
        "wall_seconds": wall,
        "total_tokens": tokens,
        "tokens_per_second": tokens / wall,
        "requests_per_second": len(good) / wall,
        "ttft_ms": distribution(
            [
                (r.first_token_at - r.scheduled_at) * 1000
                for r in good
                if r.first_token_at is not None
            ]
        ),
        "inter_token_ms": distribution([gap for r in good for gap in r.gaps_ms]),
        "e2e_ms": distribution([(r.finished_at - r.scheduled_at) * 1000 for r in good]),
    }


async def run(args: argparse.Namespace) -> dict:
    prompts = load_prompts(args.prompts)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        runner = run_poisson if args.mode == "poisson" else run_closed
        records = await runner(client, url, prompts, args)
    return {
        "settings": {
            key: value for key, value in vars(args).items() if key not in ("output", "quiet")
        },
        "summary": summarise(records, args),
        "requests": [asdict(record) for record in records],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, help="JSONL file with a 'prompt' per line")
    parser.add_argument("--mode", choices=["poisson", "closed"], default="poisson")
    parser.add_argument("--rate", type=float, default=1.0, help="poisson: requests per second")
    parser.add_argument("--duration", type=float, default=60.0, help="poisson: seconds of arrivals")
    parser.add_argument("--users", type=int, default=1, help="closed: number of concurrent users")
    parser.add_argument("--requests", type=int, default=None, help="exact number of requests")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true", help="always run to max tokens")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", default=None, help="file for the raw per-request JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.rate <= 0 or args.users < 1:
        parser.error("--rate must be above 0 and --users at least 1")
    return args


def report(summary: dict) -> None:
    print(
        f"{summary['succeeded']}/{summary['requests']} succeeded, "
        f"{summary['errors']} errors {summary['errors_by_kind'] or ''}"
    )
    print(
        f"{summary['tokens_per_second']:.1f} tokens/s, "
        f"{summary['requests_per_second']:.2f} requests/s over {summary['wall_seconds']:.1f} s"
    )
    for name, key in (
        ("time to first token", "ttft_ms"),
        ("time between tokens", "inter_token_ms"),
        ("end-to-end latency", "e2e_ms"),
    ):
        values = summary[key]
        if values["count"]:
            print(
                f"{name:20} p50 {values['p50']:9.1f}  p95 {values['p95']:9.1f}  "
                f"p99 {values['p99']:9.1f} ms"
            )


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    result = asyncio.run(run(args))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    if not args.quiet:
        report(result["summary"])
    return result


if __name__ == "__main__":
    main()

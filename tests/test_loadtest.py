"""Tests for the load test tool, against a fake server with known delays.

If the tool reports the delays the fake server was told to use, its numbers can be trusted.
"""

import asyncio
import json
import statistics
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from engine.scheduler import Scheduler
from engine.server import create_app
from scripts import loadtest
from tiny import hf_config, hf_model, my_model

TTFT = 0.20
GAP = 0.05
CHUNKS = 5


class FakeServer:
    """Waits TTFT, then sends CHUNKS tokens with GAP between them."""

    def __init__(self, ttft=TTFT, gap=GAP, chunks=CHUNKS, usage=True, fail_every=0, truncate=False):
        self.in_flight = 0
        self.max_in_flight = 0
        self.received = []
        self.count = 0
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def completions(request: Request):
            body = await request.json()
            self.received.append(body)
            self.count += 1
            if fail_every and self.count % fail_every == 0:
                return JSONResponse(status_code=503, content={"error": {"message": "busy"}})

            async def events():
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                try:
                    await asyncio.sleep(ttft)
                    for i in range(chunks):
                        if i:
                            await asyncio.sleep(gap)
                        chunk = {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    if truncate:
                        return
                    final = {"choices": [{"delta": {}, "finish_reason": "length"}]}
                    yield f"data: {json.dumps(final)}\n\n"
                    if usage:
                        tail = {
                            "choices": [],
                            "usage": {"prompt_tokens": 3, "completion_tokens": chunks},
                        }
                        yield f"data: {json.dumps(tail)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    self.in_flight -= 1

            return StreamingResponse(events(), media_type="text/event-stream")

        self.app = app


class Running:
    def __init__(self, app):
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    def __exit__(self, *_):
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def prompts(tmp_path):
    path = tmp_path / "prompts.jsonl"
    lines = [json.dumps({"prompt": f"prompt number {i}"}) for i in range(5)]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def run_tool(url, prompts, *extra, **values):
    settings = ["--base-url", url, "--model", "fake", "--prompts", prompts, "--quiet"]
    for key, value in values.items():
        settings += [f"--{key.replace('_', '-')}", str(value)]
    return loadtest.main(settings + list(extra))


def assert_near(value, expected, tolerance=0.04):
    assert expected <= value <= expected + tolerance, f"{value} is not near {expected}"


# Known delays give back known numbers


def test_reports_the_known_delays(prompts):
    with Running(FakeServer().app) as url:
        result = run_tool(url, prompts, mode="closed", users=1, requests=4)

    summary = result["summary"]
    assert summary["errors"] == 0
    assert_near(summary["ttft_ms"]["p50"] / 1000, TTFT)
    assert_near(summary["inter_token_ms"]["p50"] / 1000, GAP, 0.02)
    assert_near(summary["e2e_ms"]["p50"] / 1000, TTFT + (CHUNKS - 1) * GAP)


def test_tokens_per_second_is_right(prompts):
    with Running(FakeServer().app) as url:
        result = run_tool(url, prompts, mode="closed", users=1, requests=4)
    one_request = TTFT + (CHUNKS - 1) * GAP
    assert result["summary"]["tokens_per_second"] == pytest.approx(CHUNKS / one_request, rel=0.1)


def test_concurrent_users_multiply_throughput(prompts):
    with Running(FakeServer().app) as url:
        result = run_tool(url, prompts, mode="closed", users=4, requests=16)
    one_request = TTFT + (CHUNKS - 1) * GAP
    assert result["summary"]["tokens_per_second"] == pytest.approx(
        4 * CHUNKS / one_request, rel=0.15
    )


def test_poisson_latency_is_measured_from_the_scheduled_time(prompts):
    """A server that handles one request at a time must show queueing in the numbers."""
    lock_app = FastAPI()
    lock = asyncio.Lock()

    @lock_app.post("/v1/chat/completions")
    async def serial(_: Request):
        async def events():
            async with lock:
                await asyncio.sleep(0.2)
                chunk = {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    with Running(lock_app) as url:
        result = run_tool(url, prompts, mode="poisson", rate=20, requests=10)

    # Ten requests, 0.2 s each, arriving almost at once: the last waits about 2 s.
    assert result["summary"]["e2e_ms"]["p99"] > 1500


# Arrival patterns


def test_poisson_arrivals_have_the_right_rate():
    times = loadtest.arrival_times(rate=5.0, count=20_000, seed=3)
    assert times[-1] / len(times) == pytest.approx(1 / 5.0, rel=0.02)


def test_poisson_gaps_are_exponential():
    times = loadtest.arrival_times(rate=5.0, count=20_000, seed=3)
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    # For an exponential distribution the standard deviation equals the mean.
    assert statistics.stdev(gaps) == pytest.approx(statistics.mean(gaps), rel=0.03)


def test_same_seed_gives_the_same_schedule_and_prompts(prompts):
    with Running(FakeServer(ttft=0.01, gap=0.0).app) as url:
        first = run_tool(url, prompts, mode="poisson", rate=50, requests=30, seed=7)
        second = run_tool(url, prompts, mode="poisson", rate=50, requests=30, seed=7)
    keys = ("scheduled_at", "prompt_index")
    pick = lambda result: [[r[k] for k in keys] for r in result["requests"]]
    assert pick(first) == pick(second)


def test_different_seeds_give_different_schedules():
    assert loadtest.arrival_times(1, 10, 1) != loadtest.arrival_times(1, 10, 2)
    assert loadtest.pick_prompts(100, 10, 1) != loadtest.pick_prompts(100, 10, 2)


def test_requests_really_are_sent_on_schedule(prompts):
    with Running(FakeServer(ttft=0.01, gap=0.0).app) as url:
        result = run_tool(url, prompts, mode="poisson", rate=20, requests=40)
    for record in result["requests"]:
        assert record["sent_at"] - record["scheduled_at"] < 0.05


def test_closed_mode_never_exceeds_the_user_count(prompts):
    fake = FakeServer(ttft=0.05, gap=0.01)
    with Running(fake.app) as url:
        result = run_tool(url, prompts, mode="closed", users=3, requests=15)
    assert result["summary"]["succeeded"] == 15
    assert fake.max_in_flight == 3


def test_duration_sets_the_request_count(prompts):
    with Running(FakeServer(ttft=0.01, gap=0.0).app) as url:
        result = run_tool(url, prompts, mode="poisson", rate=10, duration=2)
    assert result["summary"]["requests"] == 20


# Prompts and payload


def test_prompts_come_from_the_file(prompts):
    fake = FakeServer(ttft=0.0, gap=0.0)
    with Running(fake.app) as url:
        run_tool(url, prompts, mode="closed", users=1, requests=12)
    contents = {body["messages"][0]["content"] for body in fake.received}
    assert len(contents) > 1
    assert all(content.startswith("prompt number") for content in contents)


def test_payload_has_the_settings(prompts):
    fake = FakeServer(ttft=0.0, gap=0.0)
    with Running(fake.app) as url:
        run_tool(
            url, prompts, "--ignore-eos", mode="closed", requests=1, max_tokens=33, temperature=0.5
        )
    body = fake.received[0]
    assert body["model"] == "fake"
    assert body["max_tokens"] == 33
    assert body["temperature"] == 0.5
    assert body["stream"] is True
    assert body["ignore_eos"] is True
    assert body["stream_options"] == {"include_usage": True}


# Errors


def test_http_errors_are_counted(prompts):
    with Running(FakeServer(ttft=0.0, gap=0.0, fail_every=2).app) as url:
        result = run_tool(url, prompts, mode="closed", users=1, requests=10)
    summary = result["summary"]
    assert summary["errors"] == 5
    assert summary["succeeded"] == 5
    assert summary["errors_by_kind"] == {"HTTP 503": 5}


def test_a_stream_cut_short_is_an_error(prompts):
    with Running(FakeServer(ttft=0.0, gap=0.0, truncate=True).app) as url:
        result = run_tool(url, prompts, mode="closed", users=1, requests=3)
    assert result["summary"]["errors"] == 3


def test_failed_requests_do_not_count_as_throughput(prompts):
    with Running(FakeServer(ttft=0.0, gap=0.0, fail_every=2).app) as url:
        result = run_tool(url, prompts, mode="closed", users=1, requests=10)
    assert result["summary"]["total_tokens"] == 5 * CHUNKS


def test_a_server_that_is_down_counts_as_errors(prompts):
    result = run_tool(
        "http://127.0.0.1:9", prompts, mode="closed", users=2, requests=4, timeout=2
    )
    assert result["summary"]["errors"] == 4
    assert result["summary"]["succeeded"] == 0
    assert result["summary"]["ttft_ms"]["p50"] is None


def test_tokens_are_counted_from_chunks_when_a_server_gives_no_usage(prompts):
    with Running(FakeServer(ttft=0.0, gap=0.0, usage=False).app) as url:
        result = run_tool(url, prompts, mode="closed", users=1, requests=2)
    assert result["summary"]["total_tokens"] == 2 * CHUNKS


# Numbers and output


def test_percentile_on_known_values():
    values = list(range(1, 101))
    assert loadtest.percentile(values, 50) == 50.5
    assert loadtest.percentile(values, 95) == pytest.approx(95.05)
    assert loadtest.percentile(values, 99) == pytest.approx(99.01)
    assert loadtest.percentile(values, 0) == 1
    assert loadtest.percentile(values, 100) == 100
    assert loadtest.percentile([7.0], 99) == 7.0
    assert loadtest.percentile([], 50) is None


def test_percentile_does_not_depend_on_order():
    assert loadtest.percentile([5, 1, 3], 50) == 3


def record(**fields):
    settings = dict(index=0, prompt_index=0, prompt_chars=1, scheduled_at=0.0, finish_reason="stop")
    return loadtest.Record(**(settings | fields))


def test_latency_counts_time_lost_before_sending():
    """Scheduled at 1 s, but sent only at 3 s: those 2 s are the user's wait too."""
    late = record(
        scheduled_at=1.0, sent_at=3.0, first_token_at=3.5, finished_at=4.0, completion_tokens=2
    )
    summary = loadtest.summarise([late], None)
    assert summary["ttft_ms"]["p50"] == pytest.approx(2500)
    assert summary["e2e_ms"]["p50"] == pytest.approx(3000)


def test_tokens_from_failed_requests_are_not_throughput():
    good = record(sent_at=0, first_token_at=1, finished_at=2, completion_tokens=10)
    failed = record(
        sent_at=0, first_token_at=1, finished_at=2, completion_tokens=99, error="HTTP 500: no"
    )
    summary = loadtest.summarise([good, failed], None)
    assert summary["total_tokens"] == 10
    assert summary["tokens_per_second"] == pytest.approx(5)
    assert summary["errors_by_kind"] == {"HTTP 500": 1}
    assert summary["e2e_ms"]["count"] == 1


def test_percentiles_in_the_summary_are_the_right_ones():
    records = [
        record(sent_at=0, first_token_at=i / 1000, finished_at=i / 1000, completion_tokens=1)
        for i in range(1, 101)
    ]
    summary = loadtest.summarise(records, None)["e2e_ms"]
    assert summary["p50"] == pytest.approx(50.5)
    assert summary["p95"] == pytest.approx(95.05)
    assert summary["p99"] == pytest.approx(99.01)
    assert summary["mean"] == pytest.approx(50.5)


def test_wall_time_runs_from_the_first_arrival_to_the_last_finish():
    records = [
        record(scheduled_at=2.0, sent_at=2, first_token_at=2, finished_at=3, completion_tokens=1),
        record(scheduled_at=4.0, sent_at=4, first_token_at=4, finished_at=7, completion_tokens=1),
    ]
    summary = loadtest.summarise(records, None)
    assert summary["wall_seconds"] == pytest.approx(5)
    assert summary["requests_per_second"] == pytest.approx(0.4)


def test_gaps_between_tokens_are_pooled_across_requests():
    records = [
        record(sent_at=0, first_token_at=0, finished_at=1, gaps_ms=[10.0, 20.0]),
        record(sent_at=0, first_token_at=0, finished_at=1, gaps_ms=[30.0]),
    ]
    assert loadtest.summarise(records, None)["inter_token_ms"]["p50"] == 20.0


def test_raw_data_is_saved_per_request(prompts, tmp_path):
    output = tmp_path / "deep" / "result.json"
    with Running(FakeServer().app) as url:
        run_tool(url, prompts, mode="closed", users=2, requests=6, output=output)

    saved = json.loads(output.read_text())
    assert len(saved["requests"]) == 6
    record = saved["requests"][0]
    assert {"scheduled_at", "first_token_at", "finished_at", "completion_tokens"} <= set(record)
    assert len(record["gaps_ms"]) == CHUNKS - 1
    assert saved["settings"]["seed"] == 0
    assert saved["summary"]["requests"] == 6


def test_bad_arguments_are_refused(prompts):
    with pytest.raises(SystemExit):
        loadtest.parse_args(
            ["--base-url", "x", "--model", "m", "--prompts", prompts, "--rate", "0"]
        )


# The real engine


def test_works_against_the_real_server(prompts):
    model = my_model(hf_model(hf_config(num_key_value_heads=2)))

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens=True):
            return "".join(chr(97 + token % 26) for token in tokens)

        def apply_chat_template(self, messages, **_):
            return [byte % 97 for byte in messages[0]["content"].encode()]

    app = create_app(Scheduler(model, 4, 256), Tokenizer(), "tiny")
    with Running(app) as url:
        result = run_tool(
            url, prompts, "--ignore-eos", mode="poisson", rate=30, requests=30, max_tokens=8
        )
    summary = result["summary"]
    assert summary["errors"] == 0
    assert summary["total_tokens"] == 30 * 8
    assert all(record["prompt_tokens"] for record in result["requests"])
    assert all(record["finish_reason"] == "length" for record in result["requests"])

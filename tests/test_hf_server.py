"""Tests for the Hugging Face baseline server, on a tiny random model."""

import json
import threading
import time

import httpx
import pytest
import torch
import uvicorn

from engine.server import create_app
from scripts import loadtest
from scripts.hf_server import CancelFlag, HfLoop
from tiny import hf_config, hf_model


class Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(97 + token % 26) for token in tokens)

    def apply_chat_template(self, messages, **_):
        return [byte % 97 for byte in messages[0]["content"].encode()]


class Running:
    def __init__(self, stop_tokens=(), max_length=512):
        self.hf = hf_model(hf_config(num_key_value_heads=2, max_position_embeddings=2048))
        self.loop = HfLoop(self.hf, max_length)
        app = create_app(self.loop, Tokenizer(), "hf", stop_tokens)
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.client = httpx.Client(base_url=self.url, timeout=30)
        return self

    def __exit__(self, *_):
        self.client.close()
        self.server.should_exit = True
        self.thread.join(timeout=10)


def body(content="hello", **fields):
    settings = {"temperature": 0.0, "max_tokens": 8}
    return {"messages": [{"role": "user", "content": content}]} | settings | fields


def expected(server, content, max_tokens):
    ids = torch.tensor([Tokenizer().apply_chat_template([{"content": content}])])
    output = server.hf.generate(
        ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=max_tokens
    )
    return Tokenizer().decode(output[0, ids.shape[1] :].tolist())


def test_reply_matches_generate():
    with Running() as server:
        reply = server.client.post("/v1/chat/completions", json=body()).json()
        assert reply["choices"][0]["message"]["content"] == expected(server, "hello", 8)
        assert reply["usage"]["completion_tokens"] == 8
        assert reply["choices"][0]["finish_reason"] == "length"


def test_stream_delivers_each_token():
    with Running() as server:
        payload = body(stream=True)
        with server.client.stream("POST", "/v1/chat/completions", json=payload) as reply:
            events = [line[6:] for line in reply.iter_lines() if line.startswith("data: ")]
        text = "".join(
            json.loads(event)["choices"][0]["delta"].get("content", "") for event in events[:-1]
        )
        assert text == expected(server, "hello", 8)
        assert events[-1] == "[DONE]"
        assert len(events) == 8 + 3


def test_requests_run_one_at_a_time():
    with Running() as server:
        running = []
        peak = []
        original = server.loop._generate

        def tracked(job):
            running.append(1)
            peak.append(len(running))
            original(job)
            running.pop()

        server.loop._generate = tracked
        threads = [
            threading.Thread(
                target=server.client.post, args=("/v1/chat/completions",), kwargs={"json": body()}
            )
            for _ in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(peak) == 5 and max(peak) == 1


def test_stop_token_ends_a_request():
    with Running() as probe:
        ids = torch.tensor([Tokenizer().apply_chat_template([{"content": "hello"}])])
        output = probe.hf.generate(
            ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=8
        )
    stop = output[0, -3].item()
    with Running(stop_tokens=(stop,)) as server:
        reply = server.client.post("/v1/chat/completions", json=body()).json()
        ignored = server.client.post("/v1/chat/completions", json=body(ignore_eos=True)).json()
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert ignored["choices"][0]["finish_reason"] == "length"
    assert ignored["usage"]["completion_tokens"] == 8


def test_disconnect_stops_generation():
    with Running(max_length=2000) as server:
        tokens = []
        original = server.loop._generate

        def counted(job):
            sink = job.sink
            job.sink = lambda event: (tokens.append(event), sink(event))[1]
            original(job)

        server.loop._generate = counted
        payload = body(max_tokens=1900, stream=True)
        with server.client.stream("POST", "/v1/chat/completions", json=payload) as reply:
            lines = reply.iter_lines()
            for _ in range(4):
                next(lines)

        deadline = time.monotonic() + 10
        while not any(event.finish_reason for event in tokens):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert tokens[-1].finish_reason == "cancelled"
        assert len(tokens) < 1900


def test_too_long_prompt_is_refused():
    with Running(max_length=20) as server:
        reply = server.client.post("/v1/chat/completions", json=body("x" * 50))
    assert reply.status_code == 400
    assert reply.json()["error"]["code"] == "context_length_exceeded"


def test_cancel_flag_stays_on_the_device_of_its_input():
    """A flag on the wrong device breaks generate on a GPU, and a CPU run cannot show it."""
    flag = CancelFlag()
    result = flag(torch.zeros(2, 3, dtype=torch.long, device="meta"), None)
    assert result.device.type == "meta"
    assert result.shape == (2,)


def test_health():
    with Running() as server:
        assert server.client.get("/health").json() == {"status": "ok", "active": 0, "waiting": 0}


def test_load_tool_works_against_it(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "one"}\n{"prompt": "two words"}\n')
    with Running() as server:
        result = loadtest.main(
            [
                "--base-url", server.url, "--model", "hf", "--prompts", str(prompts),
                "--mode", "poisson", "--rate", "20", "--requests", "10",
                "--max-tokens", "6", "--ignore-eos", "--quiet",
            ]
        )
    assert result["summary"]["errors"] == 0
    assert result["summary"]["total_tokens"] == 60

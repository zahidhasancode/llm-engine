"""Tests for the HTTP server, using a real uvicorn server and the tiny model.

A real socket matters here: disconnects and slow clients only exist on a real connection.
"""

import json
import socket
import threading
import time

import httpx
import pytest
import torch
import uvicorn

from engine.generate import generate
from engine.sampling import SamplingParams
from engine.scheduler import Scheduler
from engine.server import EngineLoop, create_app
from engine.text import IncrementalDecoder
from tiny import hf_config, hf_model, my_model

EMOJI = "😀".encode()


class ByteTokenizer:
    """Each token is one or more bytes, so a character can span several tokens."""

    def __init__(self, table=None):
        self.table = table or {}

    def token_bytes(self, token):
        return self.table.get(token, bytes([32 + token % 90]))

    def decode(self, tokens, skip_special_tokens=True):
        return b"".join(self.token_bytes(t) for t in tokens).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return [byte % 97 for byte in text.encode()]


@pytest.fixture(scope="module")
def model():
    return my_model(hf_model(hf_config(num_key_value_heads=2)))


class LiveServer:
    def __init__(self, model, tokenizer=None, slots=3, max_length=2000, stop_tokens=(), **settings):
        self.tokenizer = tokenizer or ByteTokenizer()
        self.scheduler = Scheduler(model, slots, max_length, **settings)
        self.app = create_app(EngineLoop(self.scheduler), self.tokenizer, "tiny", stop_tokens)
        config = uvicorn.Config(self.app, host="127.0.0.1", port=0, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.client = httpx.Client(base_url=self.url, timeout=30)
        return self

    def __exit__(self, *_):
        self.client.close()
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def wait_until_idle(self, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            health = self.client.get("/health").json()
            if health["free_slots"] == health["total_slots"] and health["waiting"] == 0:
                return health
            time.sleep(0.02)
        raise AssertionError(f"still busy: {health}")


def body(content="hello", **fields):
    settings = {"temperature": 0.0, "max_tokens": 12}
    return {"messages": [{"role": "user", "content": content}]} | settings | fields


def sse_events(lines):
    for line in lines:
        if line.startswith("data: "):
            yield line[len("data: ") :]


def read_stream(server, payload):
    with server.client.stream("POST", "/v1/chat/completions", json=payload | {"stream": True}) as r:
        assert r.status_code == 200
        return list(sse_events(r.iter_lines()))


def stream_text(events):
    text = ""
    for event in events:
        if event != "[DONE]":
            for choice in json.loads(event)["choices"]:
                text += choice["delta"].get("content") or ""
    return text


def expected_tokens(server, model, content, max_tokens=12):
    prompt = server.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], True, True, False
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    ids = torch.tensor(prompt)
    return generate(model, [ids], params)[0][ids.numel() :].tolist()


def expected_text(server, model, content, max_tokens=12):
    return server.tokenizer.decode(expected_tokens(server, model, content, max_tokens))


# Non-streaming


def test_completion_has_the_openai_shape(model):
    with LiveServer(model) as server:
        reply = server.client.post("/v1/chat/completions", json=body(model="my-model")).json()

    assert reply["object"] == "chat.completion"
    assert reply["id"].startswith("chatcmpl-")
    assert isinstance(reply["created"], int)
    assert reply["model"] == "my-model"
    choice = reply["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "length"
    usage = reply["usage"]
    assert usage["completion_tokens"] == 12
    assert usage["total_tokens"] == usage["prompt_tokens"] + 12


def test_completion_text_matches_generate(model):
    with LiveServer(model) as server:
        reply = server.client.post("/v1/chat/completions", json=body("tell me")).json()
        assert reply["choices"][0]["message"]["content"] == expected_text(server, model, "tell me")


def test_content_can_be_a_list_of_parts(model):
    with LiveServer(model) as server:
        parts = [{"type": "text", "text": "tell "}, {"type": "text", "text": "me"}]
        reply = server.client.post("/v1/chat/completions", json=body(parts)).json()
        assert reply["choices"][0]["message"]["content"] == expected_text(server, model, "tell me")


def test_max_completion_tokens_is_accepted(model):
    with LiveServer(model) as server:
        payload = body(max_completion_tokens=3) | {"max_tokens": None}
        reply = server.client.post("/v1/chat/completions", json=payload).json()
    assert reply["usage"]["completion_tokens"] == 3


def test_same_seed_gives_the_same_text(model):
    with LiveServer(model) as server:
        payload = body(temperature=0.9, seed=5, max_tokens=30)
        first = server.client.post("/v1/chat/completions", json=payload).json()
        second = server.client.post("/v1/chat/completions", json=payload).json()
    assert first["choices"][0]["message"] == second["choices"][0]["message"]


def test_stop_token_ends_with_reason_stop_and_is_not_in_the_text(model):
    tokenizer = ByteTokenizer()
    with LiveServer(model, tokenizer) as probe:
        tokens = expected_tokens(probe, model, "hello", 12)
    stop = tokens[5]
    with LiveServer(model, tokenizer, stop_tokens=(stop,)) as server:
        reply = server.client.post("/v1/chat/completions", json=body()).json()
    choice = reply["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == tokenizer.decode(tokens[: tokens.index(stop)])


def test_ignore_eos_runs_to_max_tokens(model):
    tokenizer = ByteTokenizer()
    with LiveServer(model, tokenizer) as probe:
        stop = expected_tokens(probe, model, "hello", 12)[5]
    with LiveServer(model, tokenizer, stop_tokens=(stop,)) as server:
        reply = server.client.post("/v1/chat/completions", json=body(ignore_eos=True)).json()
    assert reply["choices"][0]["finish_reason"] == "length"
    assert reply["usage"]["completion_tokens"] == 12


# Streaming


def test_stream_has_the_openai_shape(model):
    with LiveServer(model) as server:
        events = read_stream(server, body(stream_options={"include_usage": True}))

    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 12


def test_no_usage_chunk_unless_asked(model):
    with LiveServer(model) as server:
        events = read_stream(server, body())
    assert all("usage" not in json.loads(event) for event in events[:-1])


def test_stream_text_equals_non_stream_text(model):
    with LiveServer(model) as server:
        streamed = stream_text(read_stream(server, body("same words", max_tokens=40)))
        plain = server.client.post(
            "/v1/chat/completions", json=body("same words", max_tokens=40)
        ).json()
    assert streamed == plain["choices"][0]["message"]["content"]


def test_tokens_arrive_one_at_a_time(model):
    with LiveServer(model) as server:
        events = read_stream(server, body(max_tokens=10))
    # Role chunk, ten content chunks, finish chunk, [DONE].
    assert len(events) == 13


def test_several_streams_at_once_each_match_running_alone(model):
    with LiveServer(model, slots=3) as server:
        contents = [f"request number {i} " * (i + 1) for i in range(9)]
        results = {}

        def run(content):
            results[content] = stream_text(read_stream(server, body(content, max_tokens=25)))

        threads = [threading.Thread(target=run, args=(content,)) for content in contents]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for content in contents:
            assert results[content] == expected_text(server, model, content, 25)
        server.wait_until_idle()


# Partial characters


def find_tokens_for_emoji(tokens):
    """Find four neighbouring tokens that appear nowhere else in the output."""
    for start in range(len(tokens) - 3):
        window = tokens[start : start + 4]
        if len(set(window)) == 4 and all(tokens.count(token) == 1 for token in window):
            return window
    return None


def test_a_character_split_across_tokens_arrives_whole(model):
    window = None
    with LiveServer(model) as server:
        for attempt in range(50):
            content = f"find a good stretch {attempt}"
            window = find_tokens_for_emoji(expected_tokens(server, model, content, 30))
            if window:
                break
    assert window, "no usable stretch of output found"
    table = {token: bytes([byte]) for token, byte in zip(window, EMOJI)}

    with LiveServer(model, ByteTokenizer(table)) as server:
        events = read_stream(server, body(content, max_tokens=30))

    pieces = [
        json.loads(event)["choices"][0]["delta"].get("content", "")
        for event in events[:-1]
    ]
    assert "😀" in pieces
    assert not any("\ufffd" in piece for piece in pieces)


@pytest.mark.parametrize("stream", [True, False])
def test_text_held_back_at_the_end_is_still_sent(model, stream):
    """An answer ending partway through a character must not lose those bytes."""
    with LiveServer(model) as server:
        tokens = expected_tokens(server, model, "hello", 12)
    assert tokens.count(tokens[-1]) == 1, "the last token must not appear earlier"
    tokenizer = ByteTokenizer({tokens[-1]: EMOJI[:2]})

    with LiveServer(model, tokenizer) as server:
        if stream:
            text = stream_text(read_stream(server, body()))
        else:
            reply = server.client.post("/v1/chat/completions", json=body()).json()
            text = reply["choices"][0]["message"]["content"]

    assert text == tokenizer.decode(tokens)
    assert text.endswith("\ufffd")


def test_decoder_holds_back_until_a_character_completes():
    tokenizer = ByteTokenizer({i: bytes([b]) for i, b in enumerate(EMOJI)})
    decoder = IncrementalDecoder(tokenizer)
    assert [decoder.push(i) for i in range(4)] == ["", "", "", "😀"]


def test_decoder_mixes_text_and_split_characters():
    tokenizer = ByteTokenizer({1: b"hi ", 2: EMOJI[:2], 3: EMOJI[2:], 4: b" ok"})
    decoder = IncrementalDecoder(tokenizer)
    assert [decoder.push(t) for t in (1, 2, 3, 4)] == ["hi ", "", "😀", " ok"]


def test_decoder_flushes_what_never_completed():
    tokenizer = ByteTokenizer({1: b"a", 2: EMOJI[:2]})
    decoder = IncrementalDecoder(tokenizer)
    assert decoder.push(1) == "a"
    assert decoder.push(2) == ""
    assert decoder.flush() == "�"
    assert decoder.flush() == ""


def test_decoder_keeps_spaces_that_depend_on_context():
    class SpacedTokenizer:
        """Like real tokenizers, the first token of a decode loses its leading space."""

        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(f"w{t}" for t in tokens)

    decoder = IncrementalDecoder(SpacedTokenizer())
    assert "".join(decoder.push(t) for t in (1, 2, 3)) == "w1 w2 w3"


def test_decoder_skips_tokens_with_no_text():
    tokenizer = ByteTokenizer({1: b"a", 2: b"", 3: b"b"})
    decoder = IncrementalDecoder(tokenizer)
    assert [decoder.push(t) for t in (1, 2, 3)] == ["a", "", "b"]


# Disconnects and slow clients


def metric(server, name):
    for line in server.client.get("/metrics").text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    return 0.0


def test_disconnect_during_streaming_cancels_and_frees_the_slot(model):
    with LiveServer(model, slots=2) as server:
        payload = body(max_tokens=1900, stream=True)
        with server.client.stream("POST", "/v1/chat/completions", json=payload) as reply:
            lines = reply.iter_lines()
            for _ in range(5):
                next(lines)
        # Leaving the block closes the connection in the middle of the stream.

        server.wait_until_idle()
        reasons = 'llm_engine_requests_finished_total{reason="cancelled"}'
        assert metric(server, reasons) == 1
        assert metric(server, "llm_engine_tokens_generated_total") < 1900


def test_disconnect_while_waiting_for_a_non_stream_reply_cancels(model):
    with LiveServer(model, slots=1) as server:
        payload = json.dumps(body(max_tokens=1900)).encode()
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        connection = socket.create_connection(("127.0.0.1", server.port))
        connection.sendall(request)
        deadline = time.monotonic() + 5
        while metric(server, "llm_engine_tokens_generated_total") == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        connection.close()

        server.wait_until_idle()
        reasons = 'llm_engine_requests_finished_total{reason="cancelled"}'
        assert metric(server, reasons) == 1


def test_a_client_that_never_reads_does_not_block_others(model):
    with LiveServer(model, slots=2) as server:
        payload = json.dumps(body(max_tokens=1000, stream=True)).encode()
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        slow = socket.socket()
        slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512)
        slow.connect(("127.0.0.1", server.port))
        slow.sendall(request)  # and never read anything

        reply = server.client.post("/v1/chat/completions", json=body(max_tokens=20))
        assert reply.json()["usage"]["completion_tokens"] == 20

        deadline = time.monotonic() + 20
        while metric(server, "llm_engine_tokens_generated_total") < 1020:
            assert time.monotonic() < deadline, "the unread stream held up the model"
            time.sleep(0.02)
        slow.close()


def test_the_model_keeps_running_while_the_web_side_is_frozen(model):
    with LiveServer(model, slots=2) as server:

        @server.app.get("/freeze")
        async def freeze():
            time.sleep(1.0)  # blocks the event loop on purpose
            return {}

        payload = body(max_tokens=1900, stream=True)
        with server.client.stream("POST", "/v1/chat/completions", json=payload) as reply:
            lines = reply.iter_lines()
            next(lines)
            freezer = threading.Thread(target=server.client.get, args=("/freeze",))
            freezer.start()
            time.sleep(0.2)
            before = server.scheduler.stats.tokens_generated
            time.sleep(0.6)
            during = server.scheduler.stats.tokens_generated - before
            freezer.join()

        assert during > 20, "the engine waited for the event loop"


# Health and metrics


def test_health_reports_slots(model):
    with LiveServer(model, slots=3) as server:
        health = server.client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok", "active": 0, "waiting": 0, "free_slots": 3, "total_slots": 3,
    }


def test_health_fails_when_the_engine_loop_is_stopped(model):
    with LiveServer(model) as server:
        server.app.state.engine.stop()
        assert server.client.get("/health").status_code == 503


def test_metrics_count_work(model):
    with LiveServer(model, slots=2) as server:
        server.client.post("/v1/chat/completions", json=body(max_tokens=7))
        server.client.post("/v1/chat/completions", json=body(max_tokens=-5))
        assert metric(server, "llm_engine_tokens_generated_total") == 7
        assert metric(server, 'llm_engine_requests_finished_total{reason="length"}') == 1
        assert metric(server, "llm_engine_slots_total") == 2
        assert metric(server, "llm_engine_decode_seconds_total") > 0
        assert metric(server, "llm_engine_prefill_seconds_total") > 0
        assert metric(server, "llm_engine_cache_bytes") == server.scheduler.cache.nbytes


# Errors


@pytest.mark.parametrize(
    "payload, message",
    [
        (body(n=2), "n=1"),
        (body(stop="END"), "stop strings"),
        (body(stop=["END"]), "stop strings"),
        (body(temperature=-1), "temperature"),
        (body(top_p=2), "top_p"),
        (body(max_tokens=-1), "max_tokens"),
        (body(frequency_penalty=0.5), "frequency_penalty"),
        ({"messages": []}, "at least 1"),
        ({"model": "x"}, "required"),
    ],
)
def test_bad_requests_get_a_400_in_the_openai_format(model, payload, message):
    with LiveServer(model) as server:
        reply = server.client.post("/v1/chat/completions", json=payload)
    assert reply.status_code == 400
    error = reply.json()["error"]
    assert message in error["message"]
    assert error["type"] == "invalid_request_error"


def test_a_prompt_that_is_too_long_says_so(model):
    with LiveServer(model, max_length=40) as server:
        reply = server.client.post("/v1/chat/completions", json=body("x" * 100))
    assert reply.status_code == 400
    assert reply.json()["error"]["code"] == "context_length_exceeded"
    assert "limit of 40" in reply.json()["error"]["message"]


def test_a_full_queue_gets_a_429(model):
    with LiveServer(model, slots=1, max_waiting=1) as server:
        payload = body(max_tokens=1900, stream=True)
        with server.client.stream("POST", "/v1/chat/completions", json=payload) as running:
            # Keep each line iterator alive: dropping it would close the connection.
            running_lines = running.iter_lines()
            next(running_lines)
            while metric(server, "llm_engine_tokens_generated_total") == 0:
                time.sleep(0.01)
            with server.client.stream("POST", "/v1/chat/completions", json=payload) as queued:
                queued_lines = queued.iter_lines()
                next(queued_lines)
                refused = server.client.post("/v1/chat/completions", json=body())
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "queue_full"


# One failure must not break the others


def test_a_failing_request_ends_its_stream_and_leaves_others(model):
    with LiveServer(model, slots=2) as server:
        good_tokens = expected_tokens(server, model, "good", 25)
        bad_tokens = expected_tokens(server, model, "bad!", 25)
        poison = next(t for t in bad_tokens[3:] if t not in good_tokens)

        def explode(_, inputs):
            if (inputs[0] == poison).any():
                raise RuntimeError("poisoned token")

        hook = model.tok_embed.register_forward_pre_hook(explode)
        try:
            results = {}

            def run(content):
                results[content] = read_stream(server, body(content, max_tokens=25))

            threads = [threading.Thread(target=run, args=(c,)) for c in ("good", "bad!")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            hook.remove()

        assert stream_text(results["good"]) == server.tokenizer.decode(good_tokens)
        assert results["bad!"][-1] == "[DONE]"
        assert "poisoned token" in results["bad!"][-2]
        server.wait_until_idle()


def test_a_failing_non_stream_request_gets_a_500(model):
    with LiveServer(model) as server:
        hook = model.tok_embed.register_forward_pre_hook(
            lambda *_: (_ for _ in ()).throw(RuntimeError("broken"))
        )
        try:
            reply = server.client.post("/v1/chat/completions", json=body())
        finally:
            hook.remove()
    assert reply.status_code == 500
    assert "broken" in reply.json()["error"]["message"]


def test_the_engine_loop_survives_a_broken_step(model, monkeypatch):
    with LiveServer(model) as server:
        original = server.scheduler.step
        calls = []

        def fail_once():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("step broke")
            return original()

        monkeypatch.setattr(server.scheduler, "step", fail_once)
        first = server.client.post("/v1/chat/completions", json=body())
        assert first.status_code == 500

        second = server.client.post("/v1/chat/completions", json=body())
        assert second.status_code == 200
        assert server.client.get("/health").status_code == 200
        assert metric(server, "llm_engine_crashed_steps_total") == 1

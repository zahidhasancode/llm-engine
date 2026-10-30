"""Tests for continuous batching, on tiny random models. Nothing here uses the network."""

import random
import socket
from collections import defaultdict

import pytest
import torch

from engine.generate import generate
from engine.sampling import SamplingParams
from engine.scheduler import Scheduler
from tiny import hf_config, hf_model, my_model

MAX_LENGTH = 300


@pytest.fixture(scope="module")
def model():
    return my_model(hf_model(hf_config(num_key_value_heads=2)))


def prompt_of(length, seed=0):
    return torch.randint(0, 97, (length,), generator=torch.Generator().manual_seed(seed))


def greedy(max_tokens=20, **settings):
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, **settings)


def alone(model, prompt, params):
    return generate(model, [prompt], params)[0][prompt.numel() :].tolist()


def collect(events):
    tokens = defaultdict(list)
    reasons = {}
    for event in events:
        if event.token is not None:
            tokens[event.request_id].append(event.token)
        if event.finish_reason is not None:
            reasons[event.request_id] = event.finish_reason
    return tokens, reasons


def scheduler(model, slots=4, **settings):
    return Scheduler(model, slots, MAX_LENGTH, **settings)


# Joining mid-run


@pytest.mark.parametrize("join_at_step", [1, 2, 5, 13, 29])
@pytest.mark.parametrize("prompt_length", [1, 40, 180])
def test_a_request_joining_mid_run_matches_running_alone(model, join_at_step, prompt_length):
    engine = scheduler(model)
    first_prompt, joiner_prompt = prompt_of(30, 1), prompt_of(prompt_length, 2)
    params = greedy(40)
    first = engine.submit(first_prompt, params)
    events = []
    for _ in range(join_at_step):
        events.extend(engine.step())

    joiner = engine.submit(joiner_prompt, params)
    events.extend(engine.run())

    tokens, _ = collect(events)
    assert tokens[first] == alone(model, first_prompt, params)
    assert tokens[joiner] == alone(model, joiner_prompt, params)


def test_a_joining_sampled_request_matches_running_alone(model):
    engine = scheduler(model)
    prompts = [prompt_of(10, 1), prompt_of(60, 2), prompt_of(3, 3)]
    params = [
        SamplingParams(seed=i, top_k=20, top_p=0.9, repetition_penalty=1.1, max_tokens=30)
        for i in range(3)
    ]
    ids = [engine.submit(prompts[0], params[0])]
    events = [event for _ in range(7) for event in engine.step()]
    ids.append(engine.submit(prompts[1], params[1]))
    events += [event for _ in range(4) for event in engine.step()]
    ids.append(engine.submit(prompts[2], params[2]))
    events += engine.run()

    tokens, _ = collect(events)
    for request, prompt, setting in zip(ids, prompts, params):
        assert tokens[request] == alone(model, prompt, setting)


@pytest.mark.parametrize("budget", [1, 7, 256])
def test_output_does_not_depend_on_the_prefill_budget(model, budget):
    engine = scheduler(model, prefill_tokens_per_step=budget)
    prompts = [prompt_of(n, n) for n in (5, 90, 33)]
    ids = [engine.submit(prompt, greedy()) for prompt in prompts]
    tokens, _ = collect(engine.run())
    for request, prompt in zip(ids, prompts):
        assert tokens[request] == alone(model, prompt, greedy())


# Requests leave and join between steps


def test_short_requests_finish_before_long_ones(model):
    engine = scheduler(model)
    long_id = engine.submit(prompt_of(10, 1), greedy(80))
    short_id = engine.submit(prompt_of(10, 2), greedy(4))
    order = [e.request_id for e in engine.run() if e.finish_reason is not None]
    assert order == [short_id, long_id]


def test_a_finished_request_frees_its_slot_in_the_same_step(model):
    engine = scheduler(model, slots=1)
    # The first step gives two tokens: one ending prefill and one from decode.
    engine.submit(prompt_of(5), greedy(3))
    engine.step()
    assert engine.free_slots == []
    engine.step()
    assert engine.free_slots == [0]
    assert not engine.has_work
    assert engine.cache.lengths.tolist() == [0]


def test_a_waiting_request_joins_without_stopping_the_others(model):
    engine = scheduler(model, slots=2)
    long_id = engine.submit(prompt_of(5, 1), greedy(30))
    short_id = engine.submit(prompt_of(5, 2), greedy(3))
    waiting_id = engine.submit(prompt_of(5, 3), greedy(3))

    per_step = [engine.step() for _ in range(12)]

    # The long request gets a token on every step, including the one where its
    # neighbour leaves and the one where the replacement joins.
    assert all(any(e.request_id == long_id and e.token is not None for e in s) for s in per_step)
    first_seen = next(i for i, s in enumerate(per_step) if any(e.request_id == waiting_id for e in s))
    last_short = max(i for i, s in enumerate(per_step) if any(e.request_id == short_id for e in s))
    assert first_seen == last_short + 1


def test_slots_are_reused(model):
    engine = scheduler(model, slots=2)
    ids = [engine.submit(prompt_of(4, i), greedy(3)) for i in range(9)]
    _, reasons = collect(engine.run())
    assert set(reasons) == set(ids)
    assert sorted(engine.free_slots) == [0, 1]


# Prefill mixing


def test_a_long_prompt_is_cut_into_chunks(model):
    engine = scheduler(model, prefill_tokens_per_step=30)
    shapes = []
    handle = model.tok_embed.register_forward_hook(
        lambda _, inputs, __: shapes.append(tuple(inputs[0].shape))
    )
    engine.submit(prompt_of(100), greedy(2))
    engine.run()
    handle.remove()
    assert shapes == [(1, 30), (1, 30), (1, 30), (1, 10), (1, 1)]


def test_decoding_continues_during_a_long_prefill(model):
    engine = scheduler(model, prefill_tokens_per_step=10)
    running = engine.submit(prompt_of(5, 1), greedy(50))
    engine.step()

    late = engine.submit(prompt_of(95, 2), greedy(5))
    per_step = [engine.step() for _ in range(9)]

    assert all(any(e.request_id == running for e in step) for step in per_step)
    assert not any(e.request_id == late for step in per_step[:8] for e in step)
    assert any(e.request_id == late for e in engine.step() + per_step[-1])


def test_prefill_budget_is_shared_between_prompts_in_arrival_order(model):
    engine = scheduler(model, prefill_tokens_per_step=10)
    shapes = []
    handle = model.tok_embed.register_forward_hook(
        lambda _, inputs, __: shapes.append(tuple(inputs[0].shape))
    )
    engine.submit(prompt_of(6, 1), greedy(1))
    engine.submit(prompt_of(6, 2), greedy(1))
    engine.step()
    handle.remove()
    assert shapes == [(1, 6), (1, 4)]


# Starvation


def test_requests_start_in_arrival_order(model):
    engine = scheduler(model, slots=2)
    ids = [engine.submit(prompt_of(4, i), greedy(i % 5 + 1)) for i in range(12)]
    first_tokens = []
    for event in engine.run():
        if event.request_id not in first_tokens:
            first_tokens.append(event.request_id)
    assert first_tokens == ids


def test_a_stream_of_short_requests_cannot_starve_a_long_one(model):
    engine = scheduler(model, slots=2)
    engine.submit(prompt_of(4, 0), greedy(3))
    engine.submit(prompt_of(4, 1), greedy(3))
    victim = engine.submit(prompt_of(200, 2), greedy(30))

    finished_at = None
    for step in range(200):
        engine.submit(prompt_of(4, step), greedy(2))
        for event in engine.step():
            if event.request_id == victim and event.finish_reason is not None:
                finished_at = step
        if finished_at is not None:
            break

    assert finished_at is not None


def test_a_large_prompt_cannot_be_starved_by_steady_decoding(model):
    engine = scheduler(model, slots=2, prefill_tokens_per_step=1)
    engine.submit(prompt_of(5, 1), greedy(250))
    while not engine.decoding:
        engine.step()
    late = engine.submit(prompt_of(100, 2), greedy(2))
    finished = None
    for step in range(300):
        if any(e.request_id == late and e.finish_reason for e in engine.step()):
            finished = step
            break
    assert finished is not None and finished <= 102


# Random arrival


@pytest.mark.parametrize("seed", range(6))
def test_all_requests_finish_under_random_arrival(model, seed):
    rng = random.Random(seed)
    engine = scheduler(model, slots=rng.randint(1, 5), prefill_tokens_per_step=rng.choice([8, 64]))
    requests = {}
    events = []
    to_send = 40
    while to_send or engine.has_work:
        while to_send and rng.random() < 0.3:
            prompt = prompt_of(rng.randint(1, 120), rng.randint(0, 10_000))
            params = greedy(rng.randint(1, 25))
            requests[engine.submit(prompt, params)] = (prompt, params)
            to_send -= 1
        events.extend(engine.step())

    tokens, reasons = collect(events)
    assert set(reasons) == set(requests)
    assert all(reason == "length" for reason in reasons.values())
    for request, (prompt, params) in requests.items():
        assert tokens[request] == alone(model, prompt, params)
    assert sorted(engine.free_slots) == list(range(engine.cache.num_slots))
    assert engine.cache.lengths.sum() == 0


# Finishing


def test_stop_token_ends_a_request_and_is_kept(model):
    engine = scheduler(model)
    prompt = prompt_of(6)
    expected = alone(model, prompt, greedy(30))
    stop = expected[7]
    request = engine.submit(prompt, greedy(30, stop_tokens=(stop,)))
    tokens, reasons = collect(engine.run())
    assert tokens[request] == expected[: expected.index(stop) + 1]
    assert reasons[request] == "stop"


def test_first_token_can_finish_a_request(model):
    engine = scheduler(model)
    request = engine.submit(prompt_of(6), greedy(1))
    tokens, reasons = collect(engine.run())
    assert len(tokens[request]) == 1
    assert reasons[request] == "length"
    assert engine.free_slots == [0, 1, 2, 3]


def test_zero_max_tokens_finishes_without_a_slot(model):
    engine = scheduler(model, slots=1)
    request = engine.submit(prompt_of(6), greedy(0))
    assert not engine.waiting
    events = engine.step()
    assert events[0].request_id == request and events[0].finish_reason == "length"
    assert not engine.has_work


# Isolation


def test_works_with_the_network_disabled(model, monkeypatch):
    def refuse(*_, **__):
        raise AssertionError("the scheduler tried to use the network")

    monkeypatch.setattr(socket, "socket", refuse)
    engine = scheduler(model)
    request = engine.submit(prompt_of(5), greedy(5))
    tokens, _ = collect(engine.run())
    assert len(tokens[request]) == 5


def test_request_over_the_limit_raises(model):
    engine = Scheduler(model, 2, 16)
    with pytest.raises(ValueError, match="over the limit"):
        engine.submit(prompt_of(10), greedy(7))


def test_budget_must_be_positive(model):
    with pytest.raises(ValueError, match="prefill_tokens_per_step"):
        scheduler(model, prefill_tokens_per_step=0)

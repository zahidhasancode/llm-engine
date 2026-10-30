"""Tests for memory limits, admission, cancellation and failure isolation."""

import gc
import random
from collections import defaultdict

import pytest
import torch

from engine.cache import KVCache
from engine.generate import generate
from engine.sampling import SamplingParams
from engine.scheduler import InvalidRequest, PromptTooLong, QueueFull, Scheduler
from tiny import hf_config, hf_model, my_model

MAX_LENGTH = 200


@pytest.fixture
def model():
    return my_model(hf_model(hf_config(num_key_value_heads=2)))


def prompt_of(length, seed=0):
    return torch.randint(0, 97, (length,), generator=torch.Generator().manual_seed(seed))


def greedy(max_tokens=20, **settings):
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, **settings)


def alone(model, prompt, params):
    return generate(model, [prompt], params)[0][prompt.numel() :].tolist()


def collect(events):
    tokens, finished = defaultdict(list), {}
    for event in events:
        if event.token is not None:
            tokens[event.request_id].append(event.token)
        if event.finish_reason is not None:
            finished[event.request_id] = event
    return tokens, finished


def assert_empty(engine):
    assert not engine.has_work
    assert sorted(engine.free_slots) == list(range(engine.cache.num_slots))
    assert engine.cache.lengths.sum() == 0
    assert engine.requests == {}


# Memory budget


def test_bytes_per_slot_matches_a_real_cache(model):
    cache = KVCache(model.config, 3, 50)
    assert 3 * KVCache.bytes_per_slot(model.config, 50, torch.float32) == cache.nbytes


def test_bytes_per_slot_follows_the_dtype(model):
    full = KVCache.bytes_per_slot(model.config, 50, torch.float32)
    assert KVCache.bytes_per_slot(model.config, 50, torch.float16) * 2 == full


def test_budget_limits_the_number_of_slots(model):
    per_slot = KVCache.bytes_per_slot(model.config, MAX_LENGTH, torch.float32)
    engine = Scheduler(model, 8, MAX_LENGTH, cache_budget_bytes=3 * per_slot + 1)
    assert engine.cache.num_slots == 3
    assert engine.cache.nbytes <= 3 * per_slot + 1


def test_a_budget_above_what_is_needed_changes_nothing(model):
    per_slot = KVCache.bytes_per_slot(model.config, MAX_LENGTH, torch.float32)
    engine = Scheduler(model, 2, MAX_LENGTH, cache_budget_bytes=100 * per_slot)
    assert engine.cache.num_slots == 2


def test_a_budget_below_one_slot_is_a_clear_error(model):
    with pytest.raises(ValueError, match="below one slot"):
        Scheduler(model, 2, MAX_LENGTH, cache_budget_bytes=100)


def test_memory_scales_with_slots_and_max_length(model):
    base = KVCache(model.config, 1, 10).nbytes
    assert KVCache(model.config, 4, 10).nbytes == 4 * base
    assert KVCache(model.config, 1, 30).nbytes == 3 * base
    assert KVCache(model.config, 4, 30).nbytes == 12 * base


# Admission control


def test_requests_wait_when_the_budget_is_full(model):
    per_slot = KVCache.bytes_per_slot(model.config, MAX_LENGTH, torch.float32)
    engine = Scheduler(model, 8, MAX_LENGTH, cache_budget_bytes=2 * per_slot)
    ids = [engine.submit(prompt_of(4, i), greedy(5)) for i in range(5)]

    engine.step()

    assert engine.free_slots == []
    assert [request.id for request in engine.waiting] == ids[2:]
    assert engine.cache.nbytes == 2 * per_slot


def test_waiting_requests_run_when_slots_free_up(model):
    engine = Scheduler(model, 1, MAX_LENGTH)
    ids = [engine.submit(prompt_of(4, i), greedy(3)) for i in range(4)]
    _, finished = collect(engine.run())
    assert set(finished) == set(ids)
    assert_empty(engine)


# Waiting queue limit


def test_queue_limit_refuses_with_a_clear_error(model):
    engine = Scheduler(model, 1, MAX_LENGTH, max_waiting=2)
    engine.submit(prompt_of(4, 1), greedy())
    engine.submit(prompt_of(4, 2), greedy())
    with pytest.raises(QueueFull, match="full at 2"):
        engine.submit(prompt_of(4, 3), greedy())
    assert len(engine.waiting) == 2


def test_queue_has_room_again_after_requests_start(model):
    engine = Scheduler(model, 1, MAX_LENGTH, max_waiting=1)
    engine.submit(prompt_of(4, 1), greedy(10))
    engine.step()
    engine.submit(prompt_of(4, 2), greedy(10))
    with pytest.raises(QueueFull):
        engine.submit(prompt_of(4, 3), greedy(10))


def test_a_refused_request_leaves_nothing_behind(model):
    engine = Scheduler(model, 1, MAX_LENGTH, max_waiting=0)
    with pytest.raises(QueueFull):
        engine.submit(prompt_of(4), greedy())
    assert_empty(engine)


# Bad input


@pytest.mark.parametrize(
    "prompt, message",
    [
        (torch.tensor([], dtype=torch.long), "at least one token"),
        (torch.tensor([[1, 2]]), "1-D"),
        (torch.tensor([1.5, 2.5]), "1-D"),
        ([1, 2, 3], "1-D"),
        (torch.tensor([1, 97]), "outside 0..96"),
        (torch.tensor([-1, 2]), "outside 0..96"),
    ],
)
def test_bad_prompts_are_clear_errors(model, prompt, message):
    engine = Scheduler(model, 1, MAX_LENGTH)
    with pytest.raises(InvalidRequest, match=message):
        engine.submit(prompt, greedy())
    assert_empty(engine)


def test_bad_params_are_a_clear_error(model):
    engine = Scheduler(model, 1, MAX_LENGTH)
    with pytest.raises(InvalidRequest, match="SamplingParams"):
        engine.submit(prompt_of(4), {"max_tokens": 5})


def test_prompt_too_long_is_a_clear_error(model):
    engine = Scheduler(model, 1, 16)
    with pytest.raises(PromptTooLong, match="prompt of 10 tokens plus max_tokens 7.*limit of 16"):
        engine.submit(prompt_of(10), greedy(7))
    assert_empty(engine)


def test_a_request_that_exactly_fits_runs(model):
    engine = Scheduler(model, 1, 16)
    request = engine.submit(prompt_of(10), greedy(6))
    tokens, finished = collect(engine.run())
    assert len(tokens[request]) == 6
    assert finished[request].finish_reason == "length"


def test_a_bad_request_does_not_disturb_a_running_one(model):
    engine = Scheduler(model, 2, MAX_LENGTH)
    prompt = prompt_of(6)
    good = engine.submit(prompt, greedy())
    events = engine.step()
    with pytest.raises(InvalidRequest):
        engine.submit(torch.tensor([999]), greedy())
    events += engine.run()
    tokens, _ = collect(events)
    assert tokens[good] == alone(model, prompt, greedy())
    assert_empty(engine)


# Cancellation


def test_cancelling_a_decoding_request_frees_its_slot_at_once(model):
    engine = Scheduler(model, 2, MAX_LENGTH)
    request = engine.submit(prompt_of(6), greedy(50))
    engine.step()
    assert engine.free_slots == [1]

    assert engine.cancel(request) is True

    assert sorted(engine.free_slots) == [0, 1]
    assert engine.decoding == {}
    assert engine.cache.lengths.sum() == 0
    _, finished = collect(engine.step())
    assert finished[request].finish_reason == "cancelled"


def test_cancelling_a_waiting_request_removes_it(model):
    engine = Scheduler(model, 1, MAX_LENGTH)
    engine.submit(prompt_of(6, 1), greedy(50))
    queued = engine.submit(prompt_of(6, 2), greedy(50))
    assert engine.cancel(queued) is True
    assert len(engine.waiting) == 1


def test_cancelling_during_prefill_frees_the_slot(model):
    engine = Scheduler(model, 1, MAX_LENGTH, prefill_tokens_per_step=4)
    request = engine.submit(prompt_of(50), greedy(5))
    engine.step()
    assert len(engine.prefilling) == 1 and engine.cache.lengths[0] == 4

    engine.cancel(request)

    assert engine.free_slots == [0]
    assert not engine.prefilling and engine.cache.lengths[0] == 0


def test_cancelling_twice_or_an_unknown_id_is_harmless(model):
    engine = Scheduler(model, 1, MAX_LENGTH)
    request = engine.submit(prompt_of(6), greedy(50))
    engine.step()
    assert engine.cancel(request) is True
    assert engine.cancel(request) is False
    assert engine.cancel(12345) is False


def test_cancelling_a_finished_request_is_harmless(model):
    engine = Scheduler(model, 1, MAX_LENGTH)
    request = engine.submit(prompt_of(6), greedy(2))
    engine.run()
    assert engine.cancel(request) is False


def test_cancelling_lets_a_waiting_request_start(model):
    engine = Scheduler(model, 1, MAX_LENGTH)
    prompt = prompt_of(6, 2)
    first = engine.submit(prompt_of(6, 1), greedy(100))
    second = engine.submit(prompt, greedy())
    engine.step()
    engine.cancel(first)
    tokens, finished = collect(engine.run())
    assert finished[first].finish_reason == "cancelled"
    assert tokens[second] == alone(model, prompt, greedy())


def test_cancelling_one_request_leaves_the_others_exact(model):
    engine = Scheduler(model, 3, MAX_LENGTH)
    prompts = [prompt_of(5, 1), prompt_of(40, 2), prompt_of(12, 3)]
    ids = [engine.submit(prompt, greedy(30)) for prompt in prompts]
    events = [event for _ in range(6) for event in engine.step()]

    engine.cancel(ids[1])
    events += engine.run()

    tokens, finished = collect(events)
    assert finished[ids[1]].finish_reason == "cancelled"
    assert tokens[ids[0]] == alone(model, prompts[0], greedy(30))
    assert tokens[ids[2]] == alone(model, prompts[2], greedy(30))
    assert_empty(engine)


# One failure must not break the others


class Poison:
    """Make the model raise whenever a chosen token passes through it."""

    def __init__(self, model, token):
        self.token = token
        self.hook = model.tok_embed.register_forward_pre_hook(self.check)

    def check(self, _, inputs):
        if (inputs[0] == self.token).any():
            raise RuntimeError("poisoned token")


def test_an_exception_in_decode_fails_only_that_request(model):
    good_prompt, bad_prompt = prompt_of(8, 1), prompt_of(8, 2)
    bad_tokens = alone(model, bad_prompt, greedy(30))
    good_tokens = alone(model, good_prompt, greedy(30))
    poison = next(token for token in bad_tokens[3:] if token not in good_tokens)
    hook = Poison(model, poison)

    engine = Scheduler(model, 2, MAX_LENGTH)
    good = engine.submit(good_prompt, greedy(30))
    bad = engine.submit(bad_prompt, greedy(30))
    tokens, finished = collect(engine.run())
    hook.hook.remove()

    assert finished[bad].finish_reason == "error"
    assert "poisoned token" in finished[bad].error
    assert tokens[good] == good_tokens
    assert finished[good].finish_reason == "length"
    assert_empty(engine)


def test_an_exception_in_prefill_fails_only_that_request(model):
    good_prompt = prompt_of(8, 1)
    good_tokens = alone(model, good_prompt, greedy(10))
    clean = set(range(97)) - set(good_prompt.tolist()) - set(good_tokens)
    poison = min(clean)
    bad_prompt = torch.tensor([poison] * 8)
    hook = Poison(model, poison)

    engine = Scheduler(model, 2, MAX_LENGTH)
    good = engine.submit(good_prompt, greedy(10))
    bad = engine.submit(bad_prompt, greedy(10))
    tokens, finished = collect(engine.run())
    hook.hook.remove()

    assert finished[bad].finish_reason == "error"
    assert tokens[good] == good_tokens
    assert_empty(engine)


def test_nan_logits_fail_only_that_request(model):
    good_prompt, bad_prompt = prompt_of(8, 1), prompt_of(30, 2)
    good_tokens = alone(model, good_prompt, greedy(12))
    engine = Scheduler(model, 2, MAX_LENGTH)
    good = engine.submit(good_prompt, greedy(12))
    bad = engine.submit(bad_prompt, greedy(12))
    events = engine.step()

    # Corrupt everything the bad request has stored.
    slot = engine.requests[bad].slot
    engine.cache.values[:, slot, :, :30] = torch.nan

    events += engine.run()
    tokens, finished = collect(events)
    assert finished[bad].finish_reason == "error"
    assert "NaN" in finished[bad].error
    assert tokens[good] == good_tokens
    assert_empty(engine)


def test_a_slot_can_be_reused_after_a_failure(model):
    engine = Scheduler(model, 1, MAX_LENGTH)
    prompt = prompt_of(30, 1)
    bad = engine.submit(prompt_of(30, 2), greedy(12))
    events = engine.step()
    engine.cache.values[:, 0, :, :30] = torch.nan
    events += engine.step()
    assert collect(events)[1][bad].finish_reason == "error"

    good = engine.submit(prompt, greedy(12))
    tokens, finished = collect(engine.run())
    assert finished[good].finish_reason == "length"
    assert tokens[good] == alone(model, prompt, greedy(12))


def test_a_broken_sampler_fails_only_that_request(model, monkeypatch):
    good_prompt = prompt_of(6, 1)
    engine = Scheduler(model, 2, MAX_LENGTH)
    good = engine.submit(good_prompt, greedy(8))
    bad = engine.submit(prompt_of(6, 2), greedy(8))
    engine._admit()

    def explode(_):
        raise ArithmeticError("sampler broke")

    monkeypatch.setattr(engine.requests[bad].sampler, "next_token", explode)
    tokens, finished = collect(engine.run())

    assert "sampler broke" in finished[bad].error
    assert tokens[good] == alone(model, good_prompt, greedy(8))
    assert_empty(engine)


# A long run


@pytest.mark.parametrize("seed", [0, 1])
def test_a_thousand_random_requests_leave_nothing_behind(model, seed):
    rng = random.Random(seed)
    engine = Scheduler(model, 6, MAX_LENGTH, prefill_tokens_per_step=32, max_waiting=30)
    cache_bytes = engine.cache.nbytes
    keys, values = engine.cache.keys, engine.cache.values

    def tensors():
        gc.collect()
        return sum(isinstance(item, torch.Tensor) for item in gc.get_objects())

    finished = {}
    live = set()
    refused = 0
    sent = 0
    before = None
    while sent < 1000 or engine.has_work:
        for _ in range(rng.randint(0, 3)):
            if sent >= 1000:
                break
            sent += 1
            try:
                prompt = prompt_of(rng.randint(1, 60), rng.randint(0, 99_999))
                live.add(engine.submit(prompt, greedy(rng.randint(0, 12))))
            except QueueFull:
                refused += 1
        if live and rng.random() < 0.3:
            engine.cancel(rng.choice(sorted(live)))
        for event in engine.step():
            if event.finish_reason is not None:
                finished[event.request_id] = event.finish_reason
                live.discard(event.request_id)
        if sent >= 200 and before is None:
            before = tensors()
    for event in engine.step():
        if event.finish_reason is not None:
            finished[event.request_id] = event.finish_reason
            live.discard(event.request_id)

    assert not live
    assert len(finished) + refused == 1000
    assert "cancelled" in finished.values() and "length" in finished.values()
    assert refused > 0, "the queue limit was never exercised"
    assert_empty(engine)
    assert not engine._ready
    assert engine.cache.nbytes == cache_bytes
    assert engine.cache.keys is keys and engine.cache.values is values
    assert (engine.cache.keys == 0).all() and (engine.cache.values == 0).all()
    assert tensors() <= before + 50

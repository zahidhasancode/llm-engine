"""Tests for each sampling option, on tiny random models and hand-made logits."""

import math

import pytest
import torch
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from engine.generate import generate
from engine.sampling import (
    Sampler,
    SamplingParams,
    apply_repetition_penalty,
    apply_temperature,
    draw,
    process_logits,
    top_k_filter,
    top_p_filter,
)
from tiny import greedy, hf_config, hf_model, my_model, random_ids

VOCAB = 97


def tiny():
    return my_model(hf_model(hf_config(num_key_value_heads=2)))


def sample(model, ids, use_cache=True, **settings):
    settings.setdefault("max_tokens", 40)
    return generate(model, ids, SamplingParams(**settings), use_cache=use_cache)


def same(first: list[torch.Tensor], second: list[torch.Tensor]) -> bool:
    return len(first) == len(second) and all(map(torch.equal, first, second))


def random_logits(seed=0, size=VOCAB):
    return torch.randn(size, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def logits_from(probs):
    return torch.tensor(probs, dtype=torch.float64).log()


def record_logits(model):
    """Collect the raw logits of every generation step for one request."""
    steps = []
    model.lm_head.register_forward_hook(lambda _, __, out: steps.append(out[0, -1].clone()))
    return steps


def new_sampler(seed=0, **settings):
    prompt = torch.tensor([0])
    return Sampler(SamplingParams(seed=seed, **settings), prompt, VOCAB)


# Temperature


def test_temperature_zero_equals_greedy_from_hugging_face():
    hf = hf_model(hf_config())
    ids = random_ids(2, 8)
    reference = hf.generate(
        ids, attention_mask=torch.ones_like(ids), max_new_tokens=30, do_sample=False
    )
    assert torch.equal(greedy(my_model(hf), ids, 30), reference)


def test_temperature_zero_ignores_the_seed():
    model, ids = tiny(), random_ids()
    assert same(
        sample(model, ids, temperature=0.0, seed=1), sample(model, ids, temperature=0.0, seed=2)
    )


def test_temperature_zero_uses_no_random_numbers():
    sampler = new_sampler(temperature=0.0)
    before = sampler.generator.get_state()
    sampler.next_token(random_logits())
    assert torch.equal(sampler.generator.get_state(), before)


def test_tiny_temperature_is_greedy_and_not_nan():
    model, ids = tiny(), random_ids()
    assert same(sample(model, ids, temperature=1e-300, seed=0), sample(model, ids, temperature=0.0))


def test_temperature_changes_the_distribution_as_expected():
    logits = logits_from([0.7, 0.2, 0.1])
    cold = apply_temperature(logits, 0.5).softmax(-1)
    hot = apply_temperature(logits, 2.0).softmax(-1)
    assert cold[0] > 0.7 > hot[0]


# Seed


def test_same_seed_gives_the_same_output():
    model, ids = tiny(), random_ids()
    assert same(sample(model, ids, seed=7), sample(model, ids, seed=7))


def test_different_seeds_give_different_output():
    model, ids = tiny(), random_ids()
    assert not same(sample(model, ids, seed=7), sample(model, ids, seed=8))


def test_no_seed_gives_different_output_each_time():
    model, ids = tiny(), random_ids()
    assert not same(sample(model, ids), sample(model, ids))


def test_cached_and_uncached_sampling_agree():
    model, ids = tiny(), random_ids()
    assert same(sample(model, ids, seed=7), sample(model, ids, seed=7, use_cache=False))


# Top-k


def test_top_k_keeps_exactly_the_k_highest():
    logits = random_logits()
    kept = top_k_filter(logits, 5).isfinite().nonzero().squeeze(-1)
    assert set(kept.tolist()) == set(logits.argsort(descending=True)[:5].tolist())


def test_top_k_keeps_exactly_k_on_ties():
    assert top_k_filter(torch.zeros(10), 3).isfinite().sum() == 3


def test_top_k_larger_than_vocabulary_keeps_everything():
    logits = random_logits()
    assert torch.equal(top_k_filter(logits, VOCAB + 10), logits)


def test_top_k_never_picks_outside_the_k_highest():
    logits = random_logits()
    allowed = set(logits.argsort(descending=True)[:5].tolist())
    picks = {new_sampler(seed, top_k=5).next_token(logits) for seed in range(2000)}
    assert picks == allowed


def test_top_k_holds_at_every_step_of_real_generation():
    model, ids = tiny(), random_ids(1, 8)
    steps = record_logits(model)

    result = sample(model, ids, top_k=3, seed=1, temperature=2.0)[0]

    for logits, token in zip(steps, result[8:].tolist()):
        assert token in logits.topk(3).indices.tolist()


def test_top_k_of_one_equals_greedy():
    model, ids = tiny(), random_ids()
    assert same(sample(model, ids, top_k=1, seed=3), sample(model, ids, temperature=0.0))


# Top-p


@pytest.mark.parametrize(
    "p, expected",
    [(0.3, {0}), (0.5, {0}), (0.51, {0, 1}), (0.8, {0, 1, 2}), (0.95, {0, 1, 2, 3})],
)
def test_top_p_keeps_the_smallest_set_reaching_p(p, expected):
    logits = logits_from([0.5, 0.25, 0.15, 0.1])
    assert set(top_p_filter(logits, p).isfinite().nonzero().flatten().tolist()) == expected


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.99])
def test_top_p_set_is_minimal_on_random_logits(seed, p):
    logits = random_logits(seed)
    probs = logits.softmax(-1)
    kept = top_p_filter(logits, p).isfinite()

    assert probs[kept].sum() >= p
    # Every kept token beats every hidden one, and dropping the weakest falls short of p.
    assert kept.all() or probs[kept].min() >= probs[~kept].max()
    assert probs[kept].sum() - probs[kept].min() < p


def test_top_p_always_keeps_one_token():
    logits = logits_from([0.9, 0.1])
    assert top_p_filter(logits, 1e-9).isfinite().sum() == 1


def test_top_p_never_picks_outside_the_set():
    logits = random_logits()
    allowed = set(top_p_filter(logits, 0.5).isfinite().nonzero().flatten().tolist())
    picks = {new_sampler(seed, top_p=0.5).next_token(logits) for seed in range(2000)}
    assert picks == allowed


def test_top_p_holds_at_every_step_of_real_generation():
    model, ids = tiny(), random_ids(1, 8)
    steps = record_logits(model)

    result = sample(model, ids, top_p=0.6, seed=1)[0]

    for logits, token in zip(steps, result[8:].tolist()):
        assert top_p_filter(logits.double(), 0.6)[token].isfinite()


# Repetition penalty


def test_repetition_penalty_lowers_seen_tokens_of_either_sign():
    logits = torch.tensor([2.0, -2.0, 2.0, -2.0])
    seen = torch.tensor([True, True, False, False])
    result = apply_repetition_penalty(logits, seen, 2.0)
    assert result.tolist() == [1.0, -4.0, 2.0, -2.0]


def test_repetition_penalty_counts_the_prompt_and_new_tokens():
    sampler = new_sampler(temperature=0.0, repetition_penalty=1.5)
    assert sampler.seen.nonzero().flatten().tolist() == [0]
    token = sampler.next_token(random_logits())
    assert sampler.seen.nonzero().flatten().tolist() == sorted({0, token})


def test_repetition_penalty_matches_hugging_face_generate():
    hf = hf_model(hf_config())
    ids = random_ids(2, 8)
    reference = hf.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=30,
        do_sample=False,
        repetition_penalty=1.3,
    )
    assert torch.equal(greedy(my_model(hf), ids, 30, repetition_penalty=1.3), reference)


def test_repetition_penalty_changes_the_output():
    model, ids = tiny(), random_ids()
    assert not same(
        sample(model, ids, temperature=0.0),
        sample(model, ids, temperature=0.0, repetition_penalty=1.5),
    )


# All filters together


def test_processed_distribution_matches_hugging_face():
    """Same penalty, temperature, top-k and top-p, in Hugging Face's order."""
    logits = random_logits()
    history = torch.tensor([[1, 5, 9]])
    seen = torch.zeros(VOCAB, dtype=torch.bool)
    seen[history] = True
    params = SamplingParams(temperature=0.8, top_k=20, top_p=0.9, repetition_penalty=1.2)

    reference = logits[None]
    reference = RepetitionPenaltyLogitsProcessor(1.2)(history, reference)
    reference = TemperatureLogitsWarper(0.8)(history, reference)
    reference = TopKLogitsWarper(20)(history, reference)
    reference = TopPLogitsWarper(0.9)(history, reference)

    mine = process_logits(logits, params, seen).softmax(-1)
    assert torch.allclose(mine, reference[0].softmax(-1), atol=1e-12)


# Drawing


def test_draw_frequencies_match_the_probabilities():
    probs = [0.5, 0.3, 0.15, 0.05]
    logits = logits_from(probs)
    generator = torch.Generator().manual_seed(0)
    picks = torch.tensor([draw(logits, generator) for _ in range(20000)])
    for token, expected in enumerate(probs):
        assert (picks == token).float().mean().item() == pytest.approx(expected, abs=0.01)


def test_draw_never_picks_a_hidden_token_at_the_ends_of_the_line():
    logits = torch.tensor([-math.inf, 0.0, 0.0, -math.inf], dtype=torch.float64)
    picks = {draw(logits, torch.Generator().manual_seed(seed)) for seed in range(500)}
    assert picks == {1, 2}


def test_each_sampled_token_uses_exactly_one_random_number():
    sampler = new_sampler(seed=3, top_k=5, top_p=0.9, temperature=0.7)
    reference = torch.Generator().manual_seed(3)
    for step in range(10):
        sampler.next_token(random_logits(step))
        torch.rand(1, generator=reference, dtype=torch.float64)
    assert torch.equal(sampler.generator.get_state(), reference.get_state())


# Stop tokens and max tokens


def test_stop_token_ends_generation_and_is_kept():
    model, ids = tiny(), random_ids(1, 8)
    full = sample(model, ids, seed=5)[0]
    stop = full[12].item()
    first = (full[8:] == stop).nonzero()[0].item() + 8

    result = sample(model, ids, seed=5, stop_tokens=(stop,))[0]

    assert torch.equal(result, full[: first + 1])


def test_any_of_several_stop_tokens_ends_generation():
    model, ids = tiny(), random_ids(1, 8)
    full = sample(model, ids, seed=5)[0]
    stops = (full[20].item(), full[12].item())
    first = min((full[8:] == stop).nonzero()[0].item() for stop in stops) + 8

    assert torch.equal(sample(model, ids, seed=5, stop_tokens=stops)[0], full[: first + 1])


def test_stop_token_in_the_prompt_does_not_stop_generation():
    model, ids = tiny(), random_ids(1, 8)
    result = sample(model, ids, seed=5, max_tokens=4, stop_tokens=(ids[0, 3].item(),))[0]
    assert len(result) > 8


def test_stop_token_works_with_greedy():
    model, ids = tiny(), random_ids(1, 8)
    full = greedy(model, ids, 20)[0]
    stop = full[12].item()
    first = (full[8:] == stop).nonzero()[0].item() + 8
    assert torch.equal(greedy(model, ids, 20, stop_tokens=(stop,))[0], full[: first + 1])


@pytest.mark.parametrize("max_tokens", [0, 1, 7])
def test_max_tokens_is_exact(max_tokens):
    model, ids = tiny(), random_ids(2, 8)
    results = sample(model, ids, seed=1, max_tokens=max_tokens)
    assert [len(result) for result in results] == [8 + max_tokens] * 2


@pytest.mark.parametrize("use_cache", [True, False])
def test_each_request_has_its_own_max_tokens(use_cache):
    model, ids = tiny(), random_ids(3, 8)
    params = [SamplingParams(seed=i, max_tokens=n) for i, n in enumerate([3, 0, 9])]
    assert [len(r) for r in generate(model, ids, params, use_cache=use_cache)] == [11, 8, 17]


# Requests stay independent


def test_batching_does_not_change_a_request():
    model = tiny()
    ids = random_ids(3, 8)
    params = [
        SamplingParams(seed=1, temperature=0.9, max_tokens=30),
        SamplingParams(temperature=0.0, max_tokens=30),
        SamplingParams(seed=3, top_k=10, top_p=0.9, repetition_penalty=1.2, max_tokens=30),
    ]

    together = generate(model, ids, params)
    alone = [generate(model, ids[[i]], params[i])[0] for i in range(3)]

    assert same(together, alone)


def test_order_in_the_batch_does_not_change_a_request():
    model = tiny()
    ids = random_ids(2, 8)
    params = [SamplingParams(seed=1, max_tokens=30), SamplingParams(seed=2, max_tokens=30)]

    forward = generate(model, ids, params)
    backward = generate(model, ids.flip(0), params[::-1])

    assert same(forward, backward[::-1])


def test_a_request_finishing_early_does_not_change_the_others():
    model = tiny()
    ids = random_ids(2, 8)
    long_request = SamplingParams(seed=2, max_tokens=30)

    with_short = generate(model, ids, [SamplingParams(seed=1, max_tokens=2), long_request])
    with_long = generate(model, ids, [SamplingParams(seed=1, max_tokens=30), long_request])

    assert torch.equal(with_short[1], with_long[1])


def test_samplers_share_no_random_state():
    """Using one sampler heavily between another's draws must not move the second."""
    logits = random_logits()
    undisturbed = new_sampler(seed=1)
    alone = [undisturbed.next_token(logits) for _ in range(20)]

    target, noisy = new_sampler(seed=1), new_sampler(seed=2)
    mixed = []
    for _ in range(20):
        for _ in range(5):
            noisy.next_token(logits)
        mixed.append(target.next_token(logits))

    assert mixed == alone


def test_same_seed_in_a_batch_is_two_separate_generators():
    model = tiny()
    ids = random_ids(1, 8).repeat(2, 1)
    first, second = generate(model, ids, SamplingParams(seed=4, max_tokens=20))
    assert torch.equal(first, second)
    assert torch.equal(first, generate(model, ids[:1], SamplingParams(seed=4, max_tokens=20))[0])


def test_global_random_state_is_neither_used_nor_changed():
    model, ids = tiny(), random_ids()

    torch.manual_seed(1)
    first = sample(model, ids, seed=7)
    state = torch.get_rng_state()
    torch.manual_seed(1)
    torch.rand(50)
    state_moved = torch.get_rng_state()
    second = sample(model, ids, seed=7)

    assert same(first, second)
    assert not torch.equal(state, state_moved)
    assert torch.equal(torch.get_rng_state(), state_moved)


# Validation


@pytest.mark.parametrize(
    "bad",
    [
        {"temperature": -0.1},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"repetition_penalty": 0.0},
        {"max_tokens": -1},
    ],
)
def test_invalid_settings_raise(bad):
    with pytest.raises(ValueError):
        SamplingParams(**bad)

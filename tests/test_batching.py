"""Tests for several requests sharing one forward pass, on tiny random models.

The key test: a request's tokens in a batch exactly match the same request run alone.
"""

import pytest
import torch

from engine.cache import KVCache
from engine.generate import generate
from engine.model import future_mask, rotary_embedding
from engine.sampling import SamplingParams
from tiny import hf_config, hf_model, my_model, random_ids

VERY_DIFFERENT_LENGTHS = [1, 3, 200, 7, 64, 2, 120, 3]

VARIANTS = [
    pytest.param({}, id="plain"),
    pytest.param({"num_key_value_heads": 2}, id="grouped-query"),
    pytest.param({"num_key_value_heads": 1}, id="single-kv-head"),
    pytest.param({"tie_word_embeddings": True}, id="tied"),
]


def prompts_of(lengths, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (length,), generator=generator) for length in lengths]


def greedy(max_tokens=40, **settings):
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, **settings)


def run_alone(model, prompts, params):
    params = params if isinstance(params, list) else [params] * len(prompts)
    return [generate(model, [p], r)[0] for p, r in zip(prompts, params)]


def assert_same(first, second):
    assert len(first) == len(second)
    for one, other in zip(first, second):
        assert torch.equal(one, other)


# The key test


@pytest.mark.parametrize("overrides", VARIANTS)
@pytest.mark.parametrize("seed", range(5))
def test_batch_tokens_exactly_match_running_alone(overrides, seed):
    model = my_model(hf_model(hf_config(**overrides)))
    prompts = prompts_of(VERY_DIFFERENT_LENGTHS, seed)
    assert_same(generate(model, prompts, greedy()), run_alone(model, prompts, greedy()))


def test_batch_matches_hugging_face_for_each_request():
    hf = hf_model(hf_config(max_position_embeddings=512))
    prompts = prompts_of([2, 90, 11])

    results = generate(my_model(hf), prompts, greedy(30))

    for prompt, result in zip(prompts, results):
        ids = prompt[None]
        reference = hf.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=30, do_sample=False
        )
        assert torch.equal(result, reference[0])


def test_output_has_variety():
    """Guards the key test: identical tokens would prove nothing if they were constant."""
    model = my_model(hf_model(hf_config()))
    results = generate(model, prompts_of(VERY_DIFFERENT_LENGTHS), greedy())
    assert len({tuple(result[-40:].tolist()) for result in results}) == len(results)


def test_order_in_the_batch_does_not_matter():
    model = my_model(hf_model(hf_config()))
    prompts = prompts_of(VERY_DIFFERENT_LENGTHS)
    forward = generate(model, prompts, greedy())
    backward = generate(model, prompts[::-1], greedy())
    assert_same(forward, backward[::-1])


def test_each_request_can_stop_at_its_own_point():
    model = my_model(hf_model(hf_config()))
    prompts = prompts_of([5, 150, 20])
    params = [greedy(3), greedy(50), greedy(0)]
    results = generate(model, prompts, params)
    assert_same(results, run_alone(model, prompts, params))
    assert [len(r) for r in results] == [8, 200, 20]


def test_a_stop_token_in_one_request_leaves_the_others_alone():
    model = my_model(hf_model(hf_config()))
    prompts = prompts_of([5, 150, 20])
    full = generate(model, prompts, greedy(30))
    stop = full[0][10].item()
    params = [greedy(30, stop_tokens=(stop,)), greedy(30), greedy(30)]

    results = generate(model, prompts, params)

    assert_same(results, run_alone(model, prompts, params))
    assert len(results[0]) < len(full[0])


def test_sampled_batch_matches_running_alone():
    model = my_model(hf_model(hf_config()))
    prompts = prompts_of(VERY_DIFFERENT_LENGTHS)
    params = [
        SamplingParams(seed=i, top_k=20, top_p=0.9, max_tokens=30)
        for i in range(len(prompts))
    ]
    assert_same(generate(model, prompts, params), run_alone(model, prompts, params))


def test_cached_batch_equals_uncached():
    model = my_model(hf_model(hf_config()))
    prompts = prompts_of([3, 70, 9])
    assert_same(generate(model, prompts, greedy(20)), generate(model, prompts, greedy(20), False))


# One forward pass


@torch.no_grad()
def test_decode_runs_all_open_requests_in_one_forward_pass():
    model = my_model(hf_model(hf_config()))
    shapes = []
    model.tok_embed.register_forward_hook(lambda _, inputs, __: shapes.append(tuple(inputs[0].shape)))

    generate(model, prompts_of([4, 9, 4]), [greedy(3), greedy(2), greedy(3)])

    # Equal prompts share a prefill, the other prefills alone, and then all decode.
    assert shapes == [(2, 4), (1, 9), (3, 1), (2, 1)]


@torch.no_grad()
def test_no_row_is_padded_in_decode():
    """Rows at very different lengths still pass exactly one token each."""
    model = my_model(hf_model(hf_config()))
    cache = KVCache(model.config, 2, 64)
    model(random_ids(1, 40), cache, slots=torch.tensor([0]))
    model(random_ids(1, 3), cache, slots=torch.tensor([1]))
    assert cache.lengths.tolist() == [40, 3]

    model(random_ids(2, 1), cache)

    assert cache.lengths.tolist() == [41, 4]


@torch.no_grad()
def test_logits_in_a_batch_match_logits_alone():
    model = my_model(hf_model(hf_config()))
    first, second = prompts_of([50, 5])

    cache = KVCache(model.config, 2, 64)
    model(first[None][:, :-1], cache, slots=torch.tensor([0]))
    model(second[None][:, :-1], cache, slots=torch.tensor([1]))
    together = model(torch.stack((first[-1:], second[-1:])), cache)

    assert (together[0, 0] - model(first[None])[0, -1]).abs().max() < 1e-5
    assert (together[1, 0] - model(second[None])[0, -1]).abs().max() < 1e-5


@torch.no_grad()
def test_a_subset_of_slots_leaves_the_others_untouched():
    model = my_model(hf_model(hf_config()))
    cache = KVCache(model.config, 3, 32)
    model(random_ids(3, 6), cache)
    keys = cache.keys.clone()

    model(random_ids(2, 2), cache, slots=torch.tensor([0, 2]))

    assert cache.lengths.tolist() == [8, 6, 8]
    assert torch.equal(cache.keys[:, 1], keys[:, 1])


@torch.no_grad()
def test_slots_need_not_be_side_by_side_or_in_order():
    model = my_model(hf_model(hf_config()))
    ids = random_ids(3, 6)

    ordered = KVCache(model.config, 3, 32)
    expected = model(ids, ordered)

    shuffled = KVCache(model.config, 3, 32)
    result = model(ids, shuffled, slots=torch.tensor([2, 0, 1]))

    assert (result - expected).abs().max() < 1e-5
    assert (shuffled.keys[:, 2] - ordered.keys[:, 0]).abs().max() < 1e-5


# Per-slot state


def test_positions_are_per_row():
    model = my_model(hf_model(hf_config()))
    cache = KVCache(model.config, 2, 16)
    cache.lengths[:] = torch.tensor([5, 2])
    positions = cache.begin(cache.all_slots, 2, 3)
    assert positions.tolist() == [[5, 6, 7], [2, 3, 4]]


def test_mask_is_causal_and_per_row():
    mask = future_mask(torch.tensor([[3], [1]]))
    assert mask.shape == (2, 1, 1, 4)
    assert mask[0, 0, 0].tolist() == [False, False, False, False]
    assert mask[1, 0, 0].tolist() == [False, False, True, True]


def test_mask_of_a_prefill_is_triangular():
    mask = future_mask(torch.arange(3)[None])[0, 0]
    assert torch.equal(mask, torch.triu(torch.ones(3, 3, dtype=torch.bool), 1))


def test_each_row_rotates_by_its_own_position():
    config = my_model(hf_model(hf_config())).config
    cos, sin = rotary_embedding(config, torch.tensor([[0], [9]]), torch.float32)
    alone, _ = rotary_embedding(config, torch.tensor([[9]]), torch.float32)
    assert torch.equal(cos[1], alone[0])
    assert (cos[0] == 1).all() and (sin[0] == 0).all()
    assert not torch.equal(cos[0], cos[1])


# Safety of shared memory


def test_new_cache_is_all_zeros():
    model = my_model(hf_model(hf_config()))
    cache = KVCache(model.config, 2, 8)
    assert (cache.keys == 0).all() and (cache.values == 0).all()


@torch.no_grad()
def test_masked_positions_cannot_change_the_output():
    """A short row reads past its end. Whatever sits there must have no effect."""
    model = my_model(hf_model(hf_config()))
    prompts = prompts_of([30, 4])

    def tokens_with_cache_filled_with(value):
        cache = KVCache(model.config, 2, 64)
        cache.keys.fill_(value)
        cache.values.fill_(value)
        model(prompts[0][None], cache, slots=torch.tensor([0]))
        model(prompts[1][None], cache, slots=torch.tensor([1]))
        return model(torch.tensor([[1], [2]]), cache)

    clean = tokens_with_cache_filled_with(0.0)
    dirty = tokens_with_cache_filled_with(1e4)
    assert (clean - dirty).abs().max() < 1e-4


@torch.no_grad()
def test_release_clears_a_slot_and_leaves_the_others():
    model = my_model(hf_model(hf_config()))
    cache = KVCache(model.config, 2, 16)
    model(random_ids(2, 5), cache)
    other = cache.keys[:, 1].clone()

    cache.release(0)

    assert cache.lengths.tolist() == [0, 5]
    assert (cache.keys[:, 0] == 0).all() and (cache.values[:, 0] == 0).all()
    assert torch.equal(cache.keys[:, 1], other)


@torch.no_grad()
def test_a_released_nan_cannot_reach_the_next_request():
    model = my_model(hf_model(hf_config()))
    cache = KVCache(model.config, 2, 32)
    model(random_ids(2, 10), cache)
    cache.keys[:, 0, :, :10] = torch.nan
    cache.values[:, 0, :, :10] = torch.nan

    cache.release(0)
    model(random_ids(1, 2), cache, slots=torch.tensor([0]))
    logits = model(random_ids(2, 1), cache)

    assert logits.isfinite().all()


@torch.no_grad()
def test_a_full_slot_does_not_block_a_shorter_one():
    model = my_model(hf_model(hf_config()))
    cache = KVCache(model.config, 2, 4)
    model(random_ids(1, 4), cache, slots=torch.tensor([0]))

    model(random_ids(1, 4), cache, slots=torch.tensor([1]))

    with pytest.raises(ValueError, match="cache is full"):
        model(random_ids(2, 1), cache)
    assert cache.lengths.tolist() == [4, 4]


@torch.no_grad()
def test_wrong_number_of_rows_for_the_slots_raises():
    model = my_model(hf_model(hf_config()))
    cache = KVCache(model.config, 3, 8)
    with pytest.raises(ValueError, match="batch size"):
        model(random_ids(2, 1), cache, slots=torch.tensor([0]))


def test_empty_prompt_raises():
    model = my_model(hf_model(hf_config()))
    with pytest.raises(ValueError, match="at least one token"):
        generate(model, [torch.tensor([], dtype=torch.long)], greedy())

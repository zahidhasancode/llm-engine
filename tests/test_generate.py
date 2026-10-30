"""Tests for the KV cache and greedy generation, on tiny random models."""

import pytest
import torch

from engine.cache import KVCache
from engine.generate import generate
from tiny import hf_config, hf_model, my_model, random_ids

VARIANTS = [
    pytest.param({}, id="plain"),
    pytest.param({"num_key_value_heads": 2}, id="grouped-query"),
    pytest.param({"num_key_value_heads": 1}, id="single-kv-head"),
    pytest.param({"tie_word_embeddings": True}, id="tied"),
]


def new_cache(model, batch=2, max_length=64):
    return KVCache(model.config, batch, max_length)


def hf_generate(hf, ids, max_new_tokens, **kwargs):
    return hf.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        **kwargs,
    )


@pytest.mark.parametrize("overrides", VARIANTS)
@pytest.mark.parametrize("batch, prompt_length", [(1, 1), (1, 16), (3, 5)])
def test_cached_generation_equals_uncached(overrides, batch, prompt_length):
    model = my_model(hf_model(hf_config(**overrides)))
    ids = random_ids(batch, prompt_length)

    cached = generate(model, ids, 40, use_cache=True)
    uncached = generate(model, ids, 40, use_cache=False)

    assert torch.equal(cached, uncached)
    assert cached.shape == (batch, prompt_length + 40)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("overrides", VARIANTS)
def test_generation_matches_hugging_face(overrides, attn_implementation):
    config = hf_config(**overrides)
    config._attn_implementation = attn_implementation
    hf = hf_model(config)
    ids = random_ids(2, 8)

    mine = generate(my_model(hf), ids, 40)

    assert torch.equal(mine, hf_generate(hf, ids, 40))
    assert mine[:, 8:].unique().numel() >= 5, "output too repetitive to prove anything"


def test_eos_stops_generation_and_matches_hugging_face():
    hf = hf_model(hf_config())
    mine = my_model(hf)
    ids = random_ids(2, 8)
    # A token that really appears in the output, partway through the first sequence.
    eos = generate(mine, ids, 20)[0, 13].item()

    result = generate(mine, ids, 20, eos_token_id=eos)
    uncached = generate(mine, ids, 20, use_cache=False, eos_token_id=eos)
    reference = hf_generate(hf, ids, 20, eos_token_id=eos, pad_token_id=eos)

    assert torch.equal(result, uncached)
    assert torch.equal(result, reference)


def test_single_sequence_stops_right_after_eos():
    mine = my_model(hf_model(hf_config()))
    ids = random_ids(1, 8)
    full = generate(mine, ids, 20)
    eos = full[0, 13].item()
    stop = (full[0, 8:] == eos).nonzero()[0].item() + 8

    assert torch.equal(generate(mine, ids, 20, eos_token_id=eos), full[:, : stop + 1])


def test_finished_sequence_is_filled_with_eos_until_the_batch_is_done():
    mine = my_model(hf_model(hf_config()))
    ids = random_ids(2, 8)
    full = generate(mine, ids, 20)
    eos = full[0, 13].item()
    stop = (full[0, 8:] == eos).nonzero()[0].item() + 8
    assert eos not in full[1, 8:], "needs a token the second sequence never produces"

    result = generate(mine, ids, 20, eos_token_id=eos)

    assert (result[0, stop:] == eos).all()
    assert torch.equal(result[1], full[1])


def test_eos_in_the_prompt_does_not_stop_generation():
    mine = my_model(hf_model(hf_config()))
    ids = random_ids(1, 8)
    assert generate(mine, ids, 5, eos_token_id=ids[0, 3].item()).shape[1] > 8


def test_zero_new_tokens_returns_the_prompt():
    mine = my_model(hf_model(hf_config()))
    ids = random_ids()
    assert torch.equal(generate(mine, ids, 0), ids)


def test_negative_new_tokens_raises():
    mine = my_model(hf_model(hf_config()))
    with pytest.raises(ValueError):
        generate(mine, random_ids(), -1)


@pytest.mark.parametrize("split", [[16], [1] * 16, [10, 1, 5], [3, 3, 10]])
@torch.no_grad()
def test_cached_logits_match_a_full_forward_pass(split):
    """Feeding tokens in any pieces must equal seeing them all at once."""
    mine = my_model(hf_model(hf_config(num_key_value_heads=2)))
    ids = random_ids()
    cache = new_cache(mine)

    pieces = [mine(piece, cache=cache) for piece in ids.split(split, dim=1)]

    assert cache.length == 16
    assert (torch.cat(pieces, dim=1) - mine(ids)).abs().max() < 1e-5


@torch.no_grad()
def test_last_token_only_matches_full_logits():
    mine = my_model(hf_model(hf_config()))
    ids = random_ids()
    # Not exact: a matrix product rounds differently on one row than on many.
    assert (mine(ids, last_token_only=True) - mine(ids)[:, -1:]).abs().max() < 1e-5


@torch.no_grad()
def test_empty_cache_slots_are_never_read():
    """Slots past the current length hold garbage. NaN there would poison any read."""
    mine = my_model(hf_model(hf_config()))
    ids = random_ids()
    cache = new_cache(mine)
    cache.keys.fill_(torch.nan)
    cache.values.fill_(torch.nan)

    logits = mine(ids, cache=cache)
    logits = mine(ids[:, :1], cache=cache)

    assert logits.isfinite().all()


@torch.no_grad()
def test_each_cached_step_processes_one_token():
    """The claim behind the cache: after the prompt, one new token per step."""
    mine = my_model(hf_model(hf_config()))
    seen = []
    mine.tok_embed.register_forward_hook(lambda _, inputs, __: seen.append(inputs[0].shape[1]))

    generate(mine, random_ids(1, 8), 5, use_cache=True)
    assert seen == [8, 1, 1, 1, 1]

    seen.clear()
    generate(mine, random_ids(1, 8), 5, use_cache=False)
    assert seen == [8, 9, 10, 11, 12]


@torch.no_grad()
def test_cache_writes_in_place():
    mine = my_model(hf_model(hf_config()))
    cache = new_cache(mine)
    keys, values = cache.keys, cache.values
    pointer = keys.data_ptr()

    for token in random_ids(length=10).split(1, dim=1):
        mine(token, cache=cache)

    assert cache.keys is keys and cache.values is values
    assert cache.keys.data_ptr() == pointer


@torch.no_grad()
def test_cache_can_be_filled_exactly_but_not_past_capacity():
    mine = my_model(hf_model(hf_config()))
    cache = new_cache(mine, max_length=16)
    mine(random_ids(length=16), cache=cache)
    assert cache.length == 16

    with pytest.raises(ValueError, match="cache is full"):
        mine(random_ids(length=1), cache=cache)
    assert cache.length == 16


@torch.no_grad()
def test_prompt_larger_than_cache_raises_before_writing():
    mine = my_model(hf_model(hf_config()))
    cache = new_cache(mine, max_length=4)
    with pytest.raises(ValueError, match="cache is full"):
        mine(random_ids(length=5), cache=cache)
    assert cache.length == 0


@torch.no_grad()
def test_wrong_batch_size_raises():
    mine = my_model(hf_model(hf_config()))
    with pytest.raises(ValueError, match="batch size"):
        mine(random_ids(batch=1), cache=new_cache(mine, batch=2))


def test_cache_shape_and_size_follow_the_formula():
    mine = my_model(hf_model(hf_config(num_key_value_heads=2)))
    cache = new_cache(mine, batch=3, max_length=50)
    layers, kv_heads, head_dim = 3, 2, 8

    assert cache.keys.shape == (layers, 3, kv_heads, 50, head_dim)
    assert cache.nbytes == 2 * layers * 3 * kv_heads * 50 * head_dim * 4

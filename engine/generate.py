"""Text generation: run the model, let each request's Sampler pick the next token, repeat."""

from collections.abc import Sequence

import torch
from torch import Tensor

from engine.cache import KVCache
from engine.model import Qwen2
from engine.sampling import Sampler, SamplingParams


@torch.no_grad()
def generate(
    model: Qwen2,
    prompts: Tensor | Sequence[Tensor],
    params: SamplingParams | Sequence[SamplingParams],
    use_cache: bool = True,
) -> list[Tensor]:
    """Return one tensor per request, holding its prompt plus its generated tokens.

    prompts is a sequence of 1-D tensors, which can differ in length, or a 2-D tensor.
    params is one setting for the whole batch, or one per request. Each request gets its
    own Sampler, so requests in a batch have separate random generators, even when they
    share a seed.
    """
    prompts = list(prompts)
    if isinstance(params, SamplingParams):
        params = [params] * len(prompts)
    if len(params) != len(prompts):
        raise ValueError(f"got {len(params)} SamplingParams for {len(prompts)} prompts")
    if any(prompt.numel() == 0 for prompt in prompts):
        raise ValueError("a prompt must have at least one token")

    samplers = [
        Sampler(request, prompt, model.config.vocab_size)
        for request, prompt in zip(params, prompts)
    ]
    if use_cache:
        generated = _generate_cached(model, prompts, samplers)
    else:
        generated = [_generate_uncached(model, *pair) for pair in zip(prompts, samplers)]
    return [
        torch.cat((prompt, prompt.new_tensor(tokens)))
        for prompt, tokens in zip(prompts, generated)
    ]


def _finished(sampler: Sampler, tokens: list[int]) -> bool:
    params = sampler.params
    return len(tokens) == params.max_tokens or tokens[-1] in params.stop_tokens


def _generate_uncached(model: Qwen2, prompt: Tensor, sampler: Sampler) -> list[int]:
    """Run one request with no cache, feeding the whole sequence through at every step."""
    tokens = []
    sequence = prompt
    while len(tokens) < sampler.params.max_tokens:
        logits = model(sequence[None], last_token_only=True)
        tokens.append(sampler.next_token(logits[0, -1]))
        if _finished(sampler, tokens):
            break
        sequence = torch.cat((sequence, sequence.new_tensor(tokens[-1:])))
    return tokens


def _generate_cached(
    model: Qwen2, prompts: list[Tensor], samplers: list[Sampler]
) -> list[list[int]]:
    """Fill the cache with every prompt, then decode all open requests together."""
    weights = model.tok_embed.weight
    capacity = max(
        prompt.numel() + sampler.params.max_tokens for prompt, sampler in zip(prompts, samplers)
    )
    cache = KVCache(model.config, len(prompts), capacity, weights.device, weights.dtype)
    generated = [[] for _ in prompts]
    active = []

    def choose(slot: int, logits: Tensor) -> None:
        generated[slot].append(samplers[slot].next_token(logits))
        if not _finished(samplers[slot], generated[slot]):
            active.append(slot)

    # Prefill: T must match inside one forward pass, so only prompts of equal length
    # share one. No prompt is ever padded.
    by_length: dict[int, list[int]] = {}
    for slot, prompt in enumerate(prompts):
        if samplers[slot].params.max_tokens > 0:
            by_length.setdefault(prompt.numel(), []).append(slot)
    for slots in by_length.values():
        input_ids = torch.stack([prompts[slot] for slot in slots])
        logits = model(input_ids, cache, True, torch.tensor(slots))
        for row, slot in enumerate(slots):
            choose(slot, logits[row, -1])

    # Decode: each open request adds one token, whatever its length so far. A finished
    # request leaves at once, so nothing is computed for it again.
    while active:
        slots, active = sorted(active), []
        input_ids = prompts[0].new_tensor([generated[slot][-1:] for slot in slots])
        logits = model(input_ids, cache, True, torch.tensor(slots))
        for row, slot in enumerate(slots):
            choose(slot, logits[row, -1])
    return generated

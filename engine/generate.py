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
    input_ids: Tensor,
    params: SamplingParams | Sequence[SamplingParams],
    use_cache: bool = True,
) -> list[Tensor]:
    """Return one tensor per request, holding its prompt plus its generated tokens.

    Without padding support, every prompt in the batch must have the same length.
    params is one setting for the whole batch, or one per request. Each request gets its
    own Sampler, so requests in a batch have separate random generators, even when they
    share a seed.
    """
    batch, prompt_length = input_ids.shape
    if isinstance(params, SamplingParams):
        params = [params] * batch
    if len(params) != batch:
        raise ValueError(f"got {len(params)} SamplingParams for {batch} prompts")

    samplers = [
        Sampler(request, prompt, model.config.vocab_size)
        for request, prompt in zip(params, input_ids)
    ]
    max_new_tokens = max(request.max_tokens for request in params)
    capacity = prompt_length + max_new_tokens

    # Zeros, because a finished request keeps feeding its row through the model until the
    # whole batch is done, and those ids must be valid.
    output = input_ids.new_zeros(batch, capacity)
    output[:, :prompt_length] = input_ids
    length = prompt_length

    cache = None
    if use_cache:
        weights = model.tok_embed.weight
        cache = KVCache(model.config, batch, capacity, weights.device, weights.dtype)

    produced = [0] * batch
    finished = [request.max_tokens == 0 for request in params]
    model_input = input_ids

    for _ in range(max_new_tokens):
        logits = model(model_input, cache=cache, last_token_only=True)[:, -1]

        # Requests are handled one at a time. Each draws from its own generator, so what
        # one request does here cannot change the numbers another one gets.
        for row, sampler in enumerate(samplers):
            if finished[row]:
                continue
            token = sampler.next_token(logits[row])
            output[row, length] = token
            produced[row] += 1
            finished[row] = (
                token in sampler.params.stop_tokens
                or produced[row] == sampler.params.max_tokens
            )
        length += 1

        if all(finished):
            break

        # The cache already holds everything but the newest token.
        model_input = output[:, length - 1 : length] if use_cache else output[:, :length]

    return [output[row, : prompt_length + produced[row]] for row in range(batch)]

"""Greedy text generation: at every step, take the single most likely next token."""

import torch
from torch import Tensor

from engine.cache import KVCache
from engine.model import Qwen2


@torch.no_grad()
def generate(
    model: Qwen2,
    input_ids: Tensor,
    max_new_tokens: int,
    use_cache: bool = True,
    eos_token_id: int | None = None,
) -> Tensor:
    """Return prompts plus generated tokens, of shape (B, prompt + new tokens).

    Without padding support, every prompt in the batch must have the same length.
    Generation stops once every sequence has produced eos_token_id. A sequence that
    finishes early is filled with eos_token_id until the others are done.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")

    batch, prompt_length = input_ids.shape
    capacity = prompt_length + max_new_tokens

    output = input_ids.new_empty(batch, capacity)
    output[:, :prompt_length] = input_ids
    length = prompt_length

    cache = None
    if use_cache:
        weights = model.tok_embed.weight
        cache = KVCache(model.config, batch, capacity, weights.device, weights.dtype)

    finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
    model_input = input_ids

    for _ in range(max_new_tokens):
        logits = model(model_input, cache=cache, last_token_only=True)
        next_token = logits[:, -1].argmax(dim=-1)

        if eos_token_id is not None:
            next_token = next_token.masked_fill(finished, eos_token_id)
            finished |= next_token == eos_token_id

        output[:, length] = next_token
        length += 1

        if finished.all():
            break

        # The cache already holds everything but the newest token.
        model_input = output[:, length - 1 : length] if use_cache else output[:, :length]

    return output[:, :length]

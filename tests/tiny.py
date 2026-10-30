"""Tiny random models shared by the tests. Nothing is downloaded."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from engine.generate import generate
from engine.loader import config_from_hf, load_hf_state_dict
from engine.model import Qwen2
from engine.sampling import SamplingParams


def hf_config(**overrides) -> Qwen2Config:
    settings = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        # Larger than the default 0.02. With tiny weights logits sit near zero, and
        # a 1e-4 tolerance would prove little.
        initializer_range=0.2,
    )
    return Qwen2Config(**(settings | overrides))


def hf_model(config: Qwen2Config) -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config).eval()


def my_model(hf: Qwen2ForCausalLM) -> Qwen2:
    # Clone, because the loader keeps the tensors it is given. Shared memory would make
    # a change to one model show up in the other and hide real differences.
    model = Qwen2(config_from_hf(hf.config.to_dict())).eval()
    load_hf_state_dict(model, {name: t.clone() for name, t in hf.state_dict().items()})
    return model


def random_ids(batch: int = 2, length: int = 16) -> torch.Tensor:
    return torch.randint(0, 97, (batch, length), generator=torch.Generator().manual_seed(1))


def greedy(model: Qwen2, ids: torch.Tensor, max_tokens: int, **kwargs) -> torch.Tensor:
    """Greedy generation for requests that all end up the same length."""
    use_cache = kwargs.pop("use_cache", True)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, **kwargs)
    return torch.stack(generate(model, ids, params, use_cache=use_cache))

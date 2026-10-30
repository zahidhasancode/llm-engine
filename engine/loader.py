"""Load Hugging Face Qwen2 weights into the engine's own model.

The checkpoint is only data: a config.json and some .safetensors files. Nothing here
imports transformers.

Anything the model cannot represent faithfully raises an error instead of loading and
then producing wrong output.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from engine.model import ModelConfig, Qwen2

_TOP_LEVEL_KEYS = {
    "model.embed_tokens.weight": "tok_embed.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}

_BLOCK_KEYS = {
    "input_layernorm.weight": "norm_attn.weight",
    "post_attention_layernorm.weight": "norm_mlp.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.q_proj.bias": "attn.q_proj.bias",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.k_proj.bias": "attn.k_proj.bias",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.v_proj.bias": "attn.v_proj.bias",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    "mlp.gate_proj.weight": "mlp.gate.weight",
    "mlp.up_proj.weight": "mlp.up.weight",
    "mlp.down_proj.weight": "mlp.down.weight",
}

_BLOCK_KEY_PATTERN = re.compile(r"model\.layers\.(\d+)\.(.+)")


def map_hf_key(name: str) -> str:
    """Translate one Hugging Face weight name into this model's name for it."""
    if name in _TOP_LEVEL_KEYS:
        return _TOP_LEVEL_KEYS[name]
    match = _BLOCK_KEY_PATTERN.fullmatch(name)
    if match and match[2] in _BLOCK_KEYS:
        return f"blocks.{match[1]}.{_BLOCK_KEYS[match[2]]}"
    raise KeyError(f"unknown Qwen2 weight: {name}")


def config_from_hf(hf: Mapping) -> ModelConfig:
    """Build a ModelConfig from the contents of a Hugging Face config.json.

    Older checkpoints keep rope_theta at the top level. Newer ones nest it in
    rope_parameters. Both are read here.
    """
    if hf.get("model_type") != "qwen2":
        raise ValueError(f"not a Qwen2 config: model_type={hf.get('model_type')!r}")
    if hf.get("use_sliding_window"):
        raise ValueError("sliding window attention is not supported")
    if hf.get("hidden_act", "silu") != "silu":
        raise ValueError(f"only silu is supported, got {hf['hidden_act']!r}")
    if hf.get("rope_scaling"):
        raise ValueError("rope_scaling is not supported")

    rope = hf.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"only default rope is supported, got {rope['rope_type']!r}")

    num_heads = hf["num_attention_heads"]
    if hf.get("head_dim") not in (None, hf["hidden_size"] // num_heads):
        raise ValueError("a custom head_dim is not supported")

    return ModelConfig(
        vocab_size=hf["vocab_size"],
        hidden_size=hf["hidden_size"],
        intermediate_size=hf["intermediate_size"],
        num_layers=hf["num_hidden_layers"],
        num_heads=num_heads,
        num_kv_heads=hf.get("num_key_value_heads") or num_heads,
        rope_theta=rope.get("rope_theta", hf.get("rope_theta", 10000.0)),
        rms_norm_eps=hf.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=hf.get("tie_word_embeddings", False),
    )


def load_hf_state_dict(
    model: Qwen2, state: Mapping[str, Tensor], dtype: torch.dtype | None = None
) -> None:
    """Copy Hugging Face weights into model. A missing, extra or misshapen weight raises."""
    mapped = {map_hf_key(name): tensor for name, tensor in state.items()}
    if dtype is not None:
        mapped = {name: tensor.to(dtype) for name, tensor in mapped.items()}

    # A tied checkpoint usually leaves lm_head out, so point it at the embedding.
    tied = model.config.tie_word_embeddings
    if tied:
        mapped["lm_head.weight"] = mapped["tok_embed.weight"]

    # assign=True keeps the checkpoint's tensors instead of copying them, so weights are
    # never held in memory twice and the model takes on the checkpoint's dtype. It also
    # means the model shares memory with state, and it wraps each key in a new Parameter,
    # so a tie has to be made again afterwards.
    model.load_state_dict(mapped, strict=True, assign=True)
    if tied:
        model.lm_head.weight = model.tok_embed.weight


def load_pretrained(
    path: str | Path, dtype: torch.dtype | None = None, device: str = "cpu"
) -> Qwen2:
    """Build a model from a local Hugging Face checkpoint directory."""
    path = Path(path)
    config = config_from_hf(json.loads((path / "config.json").read_text()))

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")

    state = {}
    for file in files:
        with safe_open(file, framework="pt", device=device) as checkpoint:
            for name in checkpoint.keys():
                state[name] = checkpoint.get_tensor(name)

    # The meta device creates the model's structure without allocating any weights.
    with torch.device("meta"):
        model = Qwen2(config)
    load_hf_state_dict(model, state, dtype)
    return model.eval()

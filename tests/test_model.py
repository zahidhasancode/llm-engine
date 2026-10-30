"""Parity tests: the engine's Qwen2 against Hugging Face's, on tiny random weights.

Nothing is downloaded. Each model is built from Qwen2Config on the spot.
"""

import subprocess
import sys

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from engine.loader import (
    config_from_hf,
    load_hf_state_dict,
    load_pretrained,
    map_hf_key,
)
from engine.model import ModelConfig, Qwen2, RMSNorm

TOLERANCE = 1e-4


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


@torch.no_grad()
def max_difference(mine: Qwen2, hf: Qwen2ForCausalLM, ids: torch.Tensor) -> float:
    return (mine(ids) - hf(ids).logits).abs().max().item()


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="plain"),
        pytest.param({"num_key_value_heads": 2}, id="grouped-query"),
        pytest.param({"num_key_value_heads": 1}, id="single-kv-head"),
        pytest.param({"tie_word_embeddings": True}, id="tied"),
        pytest.param(
            {"rope_parameters": {"rope_type": "default", "rope_theta": 500.0}}, id="custom-theta"
        ),
    ],
)
def test_logits_match_hugging_face(overrides, attn_implementation):
    config = hf_config(**overrides)
    config._attn_implementation = attn_implementation
    hf = hf_model(config)
    mine = my_model(hf)
    ids = random_ids()

    with torch.no_grad():
        reference = hf(ids).logits
    assert reference.std() > 0.1, "logits too small for the tolerance to mean anything"
    assert max_difference(mine, hf, ids) < TOLERANCE


@pytest.mark.parametrize("length", [1, 2, 64])
def test_logits_match_at_sequence_length(length):
    hf = hf_model(hf_config())
    assert max_difference(my_model(hf), hf, random_ids(length=length)) < TOLERANCE


def test_one_wrong_weight_is_detected():
    """Proves the parity check can fail: a small change must push it over tolerance."""
    hf = hf_model(hf_config())
    mine = my_model(hf)
    with torch.no_grad():
        mine.blocks[1].attn.q_proj.weight[0, 0] += 0.001
    assert max_difference(mine, hf, random_ids()) > TOLERANCE


def test_tied_model_has_one_shared_parameter():
    mine = my_model(hf_model(hf_config(tie_word_embeddings=True)))
    assert mine.lm_head.weight is mine.tok_embed.weight
    names = [name for name, _ in mine.named_parameters()]
    assert "tok_embed.weight" in names and "lm_head.weight" not in names


def test_future_tokens_do_not_change_earlier_logits():
    mine = my_model(hf_model(hf_config()))
    ids = random_ids(batch=1, length=8)
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 97
    with torch.no_grad():
        assert torch.equal(mine(ids)[:, :-1], mine(changed)[:, :-1])
        assert not torch.equal(mine(ids)[:, -1], mine(changed)[:, -1])


def test_rmsnorm_matches_hugging_face():
    x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(2))
    assert torch.equal(RMSNorm(8, 1e-6)(x), Qwen2RMSNorm(8, eps=1e-6)(x))


@pytest.mark.parametrize("tied", [False, True])
def test_load_pretrained_from_sharded_checkpoint(tmp_path, tied):
    hf = hf_model(hf_config(tie_word_embeddings=tied))
    hf.save_pretrained(tmp_path, max_shard_size="2KB")
    assert len(list(tmp_path.glob("*.safetensors"))) > 1

    mine = load_pretrained(tmp_path)

    assert max_difference(mine, hf, random_ids()) < TOLERANCE
    assert (mine.lm_head.weight is mine.tok_embed.weight) == tied


def test_load_pretrained_converts_dtype(tmp_path):
    hf = hf_model(hf_config())
    hf.save_pretrained(tmp_path)
    mine = load_pretrained(tmp_path, dtype=torch.float16)
    assert mine.tok_embed.weight.dtype == torch.float16
    assert mine(random_ids()).dtype == torch.float16


def test_missing_weight_raises():
    hf = hf_model(hf_config())
    state = dict(hf.state_dict())
    del state["model.layers.1.mlp.up_proj.weight"]
    model = Qwen2(config_from_hf(hf.config.to_dict()))
    with pytest.raises(RuntimeError, match="blocks.1.mlp.up.weight"):
        load_hf_state_dict(model, state)


def test_wrong_shape_raises():
    hf = hf_model(hf_config())
    state = dict(hf.state_dict())
    state["model.norm.weight"] = torch.ones(5)
    model = Qwen2(config_from_hf(hf.config.to_dict()))
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_hf_state_dict(model, state)


def test_unknown_weight_raises():
    with pytest.raises(KeyError):
        map_hf_key("model.layers.0.self_attn.mystery.weight")


def test_key_mapping():
    assert map_hf_key("model.layers.12.mlp.gate_proj.weight") == "blocks.12.mlp.gate.weight"
    assert map_hf_key("model.embed_tokens.weight") == "tok_embed.weight"


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "llama"},
        {"use_sliding_window": True},
        {"hidden_act": "gelu"},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
        {"rope_parameters": {"rope_type": "linear", "rope_theta": 10000.0}},
        {"head_dim": 16},
    ],
)
def test_unsupported_config_raises(change):
    with pytest.raises(ValueError):
        config_from_hf(hf_config().to_dict() | change)


def test_config_reads_older_top_level_rope_theta():
    settings = hf_config().to_dict()
    del settings["rope_parameters"]
    assert config_from_hf(settings | {"rope_theta": 500.0}).rope_theta == 500.0


def test_invalid_head_split_raises():
    with pytest.raises(ValueError):
        ModelConfig(97, 30, 64, 1, num_heads=4, num_kv_heads=4)


def test_engine_does_not_import_transformers():
    code = "import sys, engine.model, engine.loader; sys.exit('transformers' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

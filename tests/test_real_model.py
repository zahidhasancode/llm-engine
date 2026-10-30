"""Runs the real-model benchmark end to end on a tiny random checkpoint, on the CPU.

This proves the script works before spending GPU time. The numbers it produces mean nothing.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from benchmarks import real_model
from tiny import hf_config, hf_model, my_model, random_ids

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def tokenizer():
    words = " ".join(real_model.PARITY_TEXTS.values()) + real_model.PROMPT_PASSAGE
    vocabulary = {"<unk>": 0}
    for word in dict.fromkeys(words.split()):
        if len(vocabulary) < 97:
            vocabulary[word] = len(vocabulary)
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    hf_model(hf_config(num_key_value_heads=2, tie_word_embeddings=True)).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def result(checkpoint, tokenizer, tmp_path_factory):
    output = tmp_path_factory.mktemp("result") / "nested" / "result.json"
    with pytest.MonkeyPatch.context() as patch:
        # The tiny checkpoint has no tokenizer of its own.
        patch.setattr(real_model.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
        code = real_model.main(
            [
                "--model-path", str(checkpoint),
                "--output", str(output),
                "--device", "cpu",
                "--prompt-lengths", "4", "24",
                "--new-tokens", "6",
                "--repeats", "2",
                "--warmups", "1",
            ]
        )
    assert code == 0
    return json.loads(output.read_text())


def test_parity_passes_on_every_text(result):
    for attention in ("eager", "sdpa"):
        parity = result["parity"][attention]
        assert parity["within_tolerance"]
        assert parity["greedy_tokens_identical"]
        assert set(parity["texts"]) == set(real_model.PARITY_TEXTS)


def test_every_engine_is_measured_at_every_prompt_length(result):
    prompts = result["benchmark"]["prompts"]
    assert [prompt["prompt_tokens"] for prompt in prompts] == [4, 24]
    for prompt in prompts:
        assert len(prompt["engines"]) == 9
        for engine in prompt["engines"].values():
            assert engine["ttft_ms"]["median"] > 0
            assert engine["decode_tokens_per_s"]["median"] > 0


def test_all_engines_produce_the_same_tokens(result):
    # float16 on CPU, tiny model: rounding is small enough here for exact agreement.
    for prompt in result["benchmark"]["prompts"]:
        for key in ("sdpa", "eager"):
            differences = prompt[f"first_token_position_differing_from_hf_{key}"]
            assert set(differences.values()) == {None}


def test_my_model_is_recorded_as_finite(result):
    for prompt in result["benchmark"]["prompts"]:
        assert prompt["first_token_choice"]["mine"]["finite"] is True


def test_step_overheads_are_measured(result):
    overheads = result["benchmark"]["my_step_overheads"]
    assert overheads["ms_per_step_total"] > 0
    assert overheads["layers"] == 3


def test_broken_numbers_are_saved_as_null():
    assert real_model.json_number(float("nan")) is None
    assert real_model.json_number(float("inf")) is None
    assert real_model.json_number(1.5) == 1.5


def test_a_nan_model_fails_the_run(checkpoint, tokenizer, tmp_path, monkeypatch):
    monkeypatch.setattr(real_model.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    monkeypatch.setattr(
        real_model, "top_two", lambda *_: {"finite": False, "tokens": [], "logits": []}
    )
    code = real_model.main(
        [
            "--model-path", str(checkpoint),
            "--output", str(tmp_path / "x.json"),
            "--device", "cpu",
            "--prompt-lengths", "4",
            "--new-tokens", "2",
            "--repeats", "1",
            "--warmups", "0",
        ]
    )
    assert code == 1


def test_result_states_model_hardware_and_settings(result):
    assert result["hardware"]["device"] == "cpu"
    assert {"torch", "transformers"} <= set(result["software"])
    assert result["settings"]["new_tokens"] == 6
    assert result["settings"]["repeats"] == 2
    assert result["benchmark"]["dtype"] == "float16"


def test_refuses_to_run_without_a_gpu(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no GPU"):
        real_model.main(["--model-path", str(checkpoint), "--output", str(tmp_path / "x.json")])


@pytest.mark.parametrize("length", [1, 5, 40])
def test_prompt_has_the_exact_length(tokenizer, length):
    assert real_model.prompt_of_length(tokenizer, length, "cpu").shape == (1, length)


def test_timer_rejects_the_wrong_number_of_model_calls():
    model = my_model(hf_model(hf_config()))
    with pytest.raises(RuntimeError, match="model calls"):
        real_model.timed(model, lambda: model(random_ids()), 2, "cpu")


def test_forward_only_loops_equal_generate():
    hf = hf_model(hf_config())
    mine = my_model(hf)
    ids = random_ids(1, 8)
    params = real_model.SamplingParams(temperature=0.0, max_tokens=12)
    expected = real_model.generate(mine, ids, params)[0]

    assert torch.equal(real_model.forward_only_mine(mine, ids, 12), expected)
    assert torch.equal(real_model.forward_only_hf(hf, ids, 12), expected)


@pytest.mark.parametrize(
    "attention",
    [
        real_model.SdpaAttention,
        real_model.SdpaNoMaskAttention,
        real_model.Float16ScoresAttention,
    ],
)
def test_attention_variants_keep_the_same_output(attention):
    mine = my_model(hf_model(hf_config(num_key_value_heads=2)))
    variant = real_model.with_attention(mine, attention)
    ids = random_ids(1, 8)
    params = real_model.SamplingParams(temperature=0.0, max_tokens=20)

    with torch.no_grad():
        assert (variant(ids) - mine(ids)).abs().max() < 1e-4
    assert torch.equal(
        real_model.generate(variant, ids, params)[0], real_model.generate(mine, ids, params)[0]
    )


def test_attention_variant_shares_the_weights():
    mine = my_model(hf_model(hf_config()))
    variant = real_model.with_attention(mine, real_model.SdpaAttention)
    assert variant.tok_embed.weight.data_ptr() == mine.tok_embed.weight.data_ptr()
    assert type(variant.blocks[0].attn) is real_model.SdpaAttention
    assert type(mine.blocks[0].attn) is real_model.Attention


def test_first_difference():
    assert real_model.first_difference(torch.tensor([1, 2]), torch.tensor([1, 2])) is None
    assert real_model.first_difference(torch.tensor([1, 2]), torch.tensor([1, 3])) == 1


@pytest.fixture(scope="module")
def upload():
    """What push.sh would upload, built without touching the network."""
    environment = os.environ | {"DRY_RUN": "1"}
    folder = subprocess.check_output(
        ["bash", str(ROOT / "kaggle/push.sh")], env=environment, text=True
    ).strip()
    return Path(folder)


def test_upload_is_only_code(upload):
    files = {str(p.relative_to(upload)) for p in upload.rglob("*") if p.is_file()}
    assert "data/code/engine/model.py" in files
    assert "data/code/benchmarks/real_model.py" in files
    assert "kernel/run.py" in files
    assert not any(".venv" in f or ".git" in f or "CLAUDE" in f for f in files)
    assert not any(f.endswith((".json", ".safetensors")) for f in files if "/code/" in f)


def test_upload_contains_no_credentials(upload):
    token_file = Path.home() / ".kaggle/access_token"
    if not token_file.exists():
        pytest.skip("no Kaggle token on this machine")
    token = token_file.read_text().strip()
    # A plain list of file names, so a failure can never print the token itself.
    leaked = [
        path.name
        for path in upload.rglob("*")
        if path.is_file() and token in path.read_text(errors="ignore")
    ]
    assert leaked == []


def test_kernel_is_private_and_on_a_t4(upload):
    metadata = json.loads((upload / "kernel/kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["zahidhasancodeml/llm-engine-code"]


def test_kaggle_run_pins_every_library_exactly():
    source = (ROOT / "kaggle/run.py").read_text()
    for library in ("torch", "transformers", "safetensors", "pytest"):
        assert f'"{library}==' in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in source

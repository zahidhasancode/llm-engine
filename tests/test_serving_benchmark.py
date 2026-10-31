"""Tests for the benchmark's logic and step timing, on a tiny random model."""

import argparse
import json

import pytest
import torch

from benchmarks import serving_benchmark as bench
from benchmarks import step_time
from tiny import hf_config, hf_model, my_model


def summary(rps=1.0, errors=0, requests=100, p95=1000.0):
    timing = {"p50": 1.0, "p95": p95, "p99": p95, "mean": 1.0, "count": requests}
    return {
        "requests": requests,
        "errors": errors,
        "requests_per_second": rps,
        "tokens_per_second": rps * 128,
        "ttft_ms": timing,
        "inter_token_ms": timing,
        "e2e_ms": timing,
    }


# Saturation


def test_keeping_up_is_not_saturated():
    assert not bench.is_saturated(summary(rps=0.98), 1.0, 60_000)


def test_falling_behind_arrivals_is_saturated():
    assert bench.is_saturated(summary(rps=0.8), 1.0, 60_000)


def test_errors_are_saturated():
    assert bench.is_saturated(summary(errors=5), 1.0, 60_000)
    assert not bench.is_saturated(summary(errors=1), 1.0, 60_000)


def test_exploding_latency_is_saturated():
    assert bench.is_saturated(summary(p95=70_000), 1.0, 60_000)


def test_no_successes_is_saturated():
    assert bench.is_saturated(summary(p95=None), 1.0, 60_000)


# Sweeping


def test_sweep_stops_at_the_first_saturated_rate():
    calls = []

    def run_point(rate, repeat):
        calls.append((rate, repeat))
        return {"summary": summary(rps=min(rate, 2.0))}

    results = bench.sweep([1, 2, 4, 8], 3, run_point, 60_000)

    assert [r["offered_rate"] for r in results] == [1, 2, 4]
    assert [r["saturated"] for r in results] == [False, False, True]
    assert len(calls) == 9


def test_every_rate_is_repeated():
    results = bench.sweep([1], 3, lambda r, n: {"summary": summary(rps=r)}, 60_000)
    assert results[0]["repeats"] == 3


def test_seeds_differ_between_repeats_and_match_between_rates():
    seen = []
    bench.sweep([1, 2], 3, lambda r, n: seen.append(n) or {"summary": summary(rps=r)}, 60_000)
    assert seen == [0, 1, 2, 0, 1, 2]


def test_one_bad_repeat_does_not_decide_a_point():
    outcomes = iter([0.5, 1.0, 1.0])
    results = bench.sweep(
        [1], 3, lambda r, n: {"summary": summary(rps=next(outcomes))}, 60_000
    )
    assert not results[0]["saturated"]
    assert results[0]["saturated_repeats"] == 1


def test_sweep_stops_when_out_of_time():
    calls = iter([False, False, True])
    results = bench.sweep(
        [1, 2], 2, lambda r, n: {"summary": summary(rps=r)}, 60_000, lambda: next(calls)
    )
    assert [r["repeats"] for r in results] == [2]


def test_median_summary_gives_median_and_spread():
    points = [{"summary": summary(rps=value)} for value in (1.0, 3.0, 2.0)]
    median = bench.median_summary(points)
    assert median["requests_per_second"] == {"median": 2.0, "min": 1.0, "max": 3.0}


def test_median_summary_survives_a_point_with_no_successes():
    points = [{"summary": summary(p95=None)}]
    assert bench.median_summary(points)["e2e_p95_ms"] is None


# Metrics


def test_parse_metrics_reads_labels():
    text = 'a_total 3\nb{reason="x"} 2.5\n# comment\n\n'
    assert bench.parse_metrics(text) == {"a_total": 3.0, 'b{reason="x"}': 2.5}


def test_time_split_is_the_change_between_readings():
    before = {
        "llm_engine_step_seconds_total": 1.0,
        "llm_engine_decode_seconds_total": 0.5,
        "llm_engine_decode_passes_total": 10,
        "llm_engine_decode_rows_total": 10,
    }
    after = {
        "llm_engine_step_seconds_total": 11.0,
        "llm_engine_decode_seconds_total": 6.5,
        "llm_engine_prefill_seconds_total": 2.0,
        "llm_engine_sampling_seconds_total": 1.0,
        "llm_engine_decode_passes_total": 110,
        "llm_engine_decode_rows_total": 410,
    }
    split = bench.time_split(before, after)
    assert split["decode_seconds"] == 6.0
    assert split["other_step_seconds"] == pytest.approx(1.0)
    assert split["mean_rows_per_decode_pass"] == 4.0
    assert split["ms_per_decode_pass"] == 60.0


def test_time_split_with_no_metrics_is_safe():
    assert bench.time_split({}, {})["ms_per_decode_pass"] is None


# Commands


def arguments():
    return argparse.Namespace(
        model_path="/m", vllm_bin="/v/vllm", dtype="float16", max_length=2048, slots=16,
        prefill_tokens_per_step=256,
    )


def command(name):
    return next(s for s in bench.systems(arguments()) if s.name == name).command


def test_all_three_systems_exist_on_different_ports():
    found = bench.systems(arguments())
    assert [s.name for s in found] == ["huggingface", "my_engine", "vllm"]
    assert len({s.port for s in found}) == 3


def test_precision_matches_across_systems():
    for name in ("huggingface", "my_engine", "vllm"):
        settings = command(name)
        assert settings[settings.index("--dtype") + 1] == "float16"


def test_vllm_does_not_use_the_checkpoint_sampling_defaults():
    settings = command("vllm")
    assert settings[settings.index("--generation-config") + 1] == "vllm"


def test_vllm_prefix_cache_is_off_because_prompts_repeat():
    assert "--no-enable-prefix-caching" in command("vllm")


def test_requests_turn_off_every_sampling_extra():
    assert json.loads(bench.GREEDY) == {"top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}


# Step timing, on a tiny model


@pytest.fixture(scope="module")
def model():
    return my_model(hf_model(hf_config(num_key_value_heads=2)))


def test_step_time_gives_a_positive_time(model):
    row = step_time.ms_per_step(model, 2, 8, 6, 1)
    assert row["ms_per_step"] > 0
    assert row["tokens_per_second"] > 0
    assert 0 < row["model_share"] <= 1


def test_prompt_ids_are_fixed_by_the_seed():
    assert torch.equal(step_time.prompt_ids(97, 8, 3), step_time.prompt_ids(97, 8, 3))
    assert not torch.equal(step_time.prompt_ids(97, 8, 3), step_time.prompt_ids(97, 8, 4))


def test_timed_attention_gives_the_same_output(model):
    from benchmarks.real_model import with_attention
    from engine.generate import generate
    from engine.sampling import SamplingParams

    step_time.TimedAttention.device = "cpu"
    timed = with_attention(model, step_time.TimedAttention)
    prompts = [torch.arange(1, 12), torch.arange(3, 9)]
    params = SamplingParams(temperature=0.0, max_tokens=8)
    for mine, other in zip(generate(model, prompts, params), generate(timed, prompts, params)):
        assert torch.equal(mine, other)


def test_step_parts_cover_the_step(model):
    result = step_time.step_parts(model, 2, 8, 6, "cpu")
    parts = result["parts_ms"]
    for name in ("attention_product", "cache_write_and_read", "mlp", "sampling", "lm_head"):
        assert parts[name] > 0
    total = sum(parts.values())
    assert total == pytest.approx(result["ms_per_step_with_stopwatches"], rel=0.05)

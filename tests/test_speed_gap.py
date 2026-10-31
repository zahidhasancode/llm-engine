"""Tests for the speed gap measurements, on a tiny random model."""

import pytest
import torch

from benchmarks import speed_gap
from scripts import loadtest
from tiny import hf_config, hf_model, my_model


@pytest.fixture(scope="module")
def model():
    return my_model(hf_model(hf_config()))


def lengths(*sizes):
    return [torch.arange(1, size + 1) for size in sizes]


def test_direct_run_serves_every_request_in_full(model):
    result = speed_gap.direct_run(model, lengths(3, 9, 5, 2, 7), 2, 32, 8, 4)
    assert result["tokens"] == 5 * 4
    assert result["tokens_per_second"] == pytest.approx(result["tokens"] / result["seconds"])
    assert result["ms_per_decode_pass"] == pytest.approx(
        result["decode_seconds"] / result["decode_passes"] * 1000
    )


def test_all_slots_stay_full_while_work_is_waiting(model):
    result = speed_gap.direct_run(model, lengths(*[4] * 6), 3, 32, 12, 5)
    # Each group of three prefills together, then decodes the 4 remaining tokens as one row set.
    assert result["decode_passes"] == 2 * 4
    assert result["mean_rows_per_decode_pass"] == 3


def test_a_sample_matches_what_the_load_test_sends():
    prompts = lengths(*range(1, 21))
    chosen = speed_gap.sample_prompts(prompts, 50, 7)
    indices = loadtest.pick_prompts(20, 50, 7)
    assert [p.numel() for p in chosen] == [prompts[i].numel() for i in indices]


def test_a_sample_of_short_prompts_has_no_long_ones():
    prompts = lengths(2, 3, 50, 4, 60)
    chosen = speed_gap.sample_prompts(prompts, 40, 1, longest=4)
    assert len(chosen) == 40
    assert max(p.numel() for p in chosen) == 4


def test_medians_are_per_key_and_keep_the_runs():
    runs = [{"a": 1.0, "b": 9.0}, {"a": 3.0, "b": 8.0}, {"a": 2.0, "b": 7.0}]
    assert speed_gap.medians(runs) == {"a": 2.0, "b": 8.0, "runs": runs}

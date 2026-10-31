"""The charts must draw from result files of the saved shape."""

import json

from scripts import make_charts


def timing(value):
    return {"median": value, "min": value, "max": value}


def system(name):
    points = [
        {"offered_rate": rate, "median": {"e2e_p95_ms": timing(rate * 1000),
                                          "ttft_p95_ms": timing(rate * 100)}}
        for rate in (0.1, 0.2)
    ]
    return {"name": name, "points": points,
            "closed_loop": {"tokens_per_second": timing(100.0)}}


def step_rows(ms):
    return [{"batch": batch, "ms_per_step": ms * batch} for batch in (16, 32, 64, 128)]


def write_results(folder):
    source = folder / "serving_benchmark.json"
    source.write_text(json.dumps({"systems": [system("my_engine"), system("vllm")]}))
    (folder / "speed_gap.json").write_text(json.dumps({"step_time": step_rows(2.0)}))
    (folder / "vllm_step_time_default.json").write_text(json.dumps({"rows": step_rows(0.5)}))
    (folder / "vllm_step_time_flex.json").write_text(json.dumps({"rows": step_rows(1.5)}))
    return source


def test_all_charts_are_written(tmp_path):
    source = write_results(tmp_path)
    output = tmp_path / "charts"
    make_charts.main(source, output, tmp_path)
    names = sorted(path.name for path in output.iterdir())
    assert names == ["latency_e2e.png", "latency_ttft.png", "max_throughput.png", "step_time.png"]
    assert all(path.stat().st_size > 1000 for path in output.iterdir())


def test_step_time_lines_come_from_the_right_files(tmp_path):
    write_results(tmp_path)
    rows = {row[0]: row[4][0]["ms_per_step"] for row in make_charts.step_time_rows(tmp_path)}
    assert rows == {"My engine": 32.0, "vLLM, Triton attention": 8.0, "vLLM, FLEX attention": 24.0}

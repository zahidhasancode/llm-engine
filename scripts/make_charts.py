"""Draw the benchmark charts from the saved results in benchmarks/.

Run from the repository root: python scripts/make_charts.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator

SOURCE = Path("benchmarks/serving_benchmark.json")
RESULTS = Path("benchmarks")
OUTPUT = Path("benchmarks/charts")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"
SYSTEMS = {
    "huggingface": ("Hugging Face generate", "#1baf7a", "s"),
    "my_engine": ("My engine", "#2a78d6", "o"),
    "vllm": ("vLLM", "#eb6834", "^"),
}


def styled_axes(figsize):
    figure, axes = plt.subplots(figsize=figsize, dpi=160, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.spines[["top", "right", "left"]].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=MUTED, length=0, labelsize=9)
    axes.grid(True, axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    return figure, axes


def latency_chart(systems: list[dict], key: str, title: str, path: Path) -> None:
    """Latency against offered load. Bands show the spread across repeats."""
    figure, axes = styled_axes((7.2, 4.2))
    for system in systems:
        label, color, marker = SYSTEMS[system["name"]]
        points = [p for p in system["points"] if p["median"][key]]
        rates = [p["offered_rate"] for p in points]
        medians = [p["median"][key]["median"] / 1000 for p in points]
        low = [p["median"][key]["min"] / 1000 for p in points]
        high = [p["median"][key]["max"] / 1000 for p in points]
        axes.fill_between(rates, low, high, color=color, alpha=0.15, linewidth=0)
        axes.plot(rates, medians, color=color, marker=marker, markersize=6, linewidth=2,
                  label=label, markeredgecolor=SURFACE, markeredgewidth=1)
        axes.annotate(label, (rates[-1], medians[-1]), xytext=(7, 0),
                      textcoords="offset points", va="center", fontsize=9, color=INK)
    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    axes.set_xticks(sorted({p["offered_rate"] for s in systems for p in s["points"]}))
    axes.xaxis.set_major_formatter(plain)
    axes.xaxis.set_minor_locator(NullLocator())
    low_limit, high_limit = axes.get_ylim()
    axes.set_yticks([tick for tick in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
                     if low_limit <= tick <= high_limit])
    axes.set_ylim(low_limit, high_limit)
    axes.yaxis.set_major_formatter(plain)
    axes.yaxis.set_minor_locator(NullLocator())
    axes.set_xlabel("Offered load (requests per second)", color=MUTED, fontsize=9)
    axes.set_ylabel("Seconds (log scale)", color=MUTED, fontsize=9)
    axes.set_title(title, loc="left", color=INK, fontsize=11)
    axes.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")
    axes.margins(x=0.12)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def throughput_chart(systems: list[dict], path: Path) -> None:
    """Best sustained token rate, from the closed-loop runs."""
    rows = [s for s in systems if s.get("closed_loop")]
    rows.sort(key=lambda s: s["closed_loop"]["tokens_per_second"]["median"])
    names = [SYSTEMS[s["name"]][0] for s in rows]
    values = [s["closed_loop"]["tokens_per_second"]["median"] for s in rows]
    colors = [SYSTEMS[s["name"]][1] for s in rows]

    figure, axes = styled_axes((7.2, 2.6))
    axes.grid(False)
    bars = axes.barh(names, values, color=colors, height=0.5)
    for bar, value in zip(bars, values):
        axes.annotate(f"{value:,.0f}", (value, bar.get_y() + bar.get_height() / 2),
                      xytext=(6, 0), textcoords="offset points", va="center",
                      fontsize=9, color=INK)
    axes.set_xlabel("Tokens per second", color=MUTED, fontsize=9)
    axes.set_title("Maximum throughput, closed loop", loc="left", color=INK, fontsize=11)
    axes.tick_params(axis="y", labelcolor=INK)
    axes.margins(x=0.12)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def step_time_rows(results: Path) -> list[tuple[str, str, str, str, list[dict]]]:
    """Each line of the step time chart: label, color, marker, line style and its rows."""
    read = lambda name: json.loads((results / name).read_text())
    _, vllm_color, _ = SYSTEMS["vllm"]
    _, mine_color, _ = SYSTEMS["my_engine"]
    return [
        ("My engine", mine_color, "o", "-", read("speed_gap.json")["step_time"]),
        ("vLLM, Triton attention", vllm_color, "^", "-",
         read("vllm_step_time_default.json")["rows"]),
        ("vLLM, FLEX attention", vllm_color, "s", "--",
         read("vllm_step_time_flex.json")["rows"]),
    ]


def step_time_chart(results: Path, path: Path) -> None:
    """Decode step time against batch size, from runs with no HTTP."""
    figure, axes = styled_axes((7.2, 4.2))
    batches = [16, 32, 64, 128]
    for label, color, marker, style, rows in step_time_rows(results):
        times = [r["ms_per_step"] for r in rows if r["batch"] in batches]
        axes.plot(batches, times, color=color, marker=marker, linestyle=style, markersize=6,
                  linewidth=2, label=label, markeredgecolor=SURFACE, markeredgewidth=1)
    axes.set_xscale("log", base=2)
    axes.set_xticks(batches)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes.xaxis.set_minor_locator(NullLocator())
    axes.set_ylim(bottom=0)
    axes.set_xlabel("Requests in one decode step", color=MUTED, fontsize=9)
    axes.set_ylabel("Milliseconds per step", color=MUTED, fontsize=9)
    axes.set_title("Decode step time, 256-token prompts", loc="left", color=INK, fontsize=11)
    axes.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def main(source: Path = SOURCE, output: Path = OUTPUT, results: Path = RESULTS) -> None:
    systems = json.loads(source.read_text())["systems"]
    output.mkdir(parents=True, exist_ok=True)
    latency_chart(systems, "e2e_p95_ms", "End-to-end latency, p95", output / "latency_e2e.png")
    latency_chart(systems, "ttft_p95_ms", "Time to first token, p95", output / "latency_ttft.png")
    throughput_chart(systems, output / "max_throughput.png")
    step_time_chart(results, output / "step_time.png")


if __name__ == "__main__":
    main()

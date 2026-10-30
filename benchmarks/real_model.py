"""Check a real checkpoint against Hugging Face, then time both on one request.

Three things are measured and saved as JSON:

1. Parity: my logits against Hugging Face's on real text, in float32.
2. Speed: time to first token and tokens per second through each generate function.
3. Forward only: the same model loop with no generate function around it, which shows
   how much of any speed difference comes from the model and how much from the wrapper.

Every run is greedy, one request, with stop tokens off, so each engine makes the same
number of model calls. Runs are timed by a hook on lm_head that waits for the GPU, placed
the same way on both models. That wait makes each step finish before the next clock
reading, and it adds the same small cost to both.

The real run happens on Kaggle through kaggle/run.py. The tests run this file on a tiny
random checkpoint on the CPU.

Run from the repository root:
PYTHONPATH=. python benchmarks/real_model.py --model-path DIR --output FILE
"""

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch
import transformers
from torch import Tensor
from transformers import AutoTokenizer, DynamicCache, GenerationConfig, Qwen2ForCausalLM

from engine.cache import KVCache
from engine.generate import generate
from engine.loader import load_pretrained
from engine.model import Qwen2
from engine.sampling import SamplingParams

PARITY_TOLERANCE = 1e-4
GREEDY_CHECK_TOKENS = 64

PARITY_TEXTS = {
    "english": "The Eiffel Tower is in Paris. It was finished in 1889 and it is still one of "
    "the most visited monuments in the world.",
    "spanish": "Barcelona es una ciudad situada en la costa del Mediterráneo, conocida por "
    "su arquitectura y por sus barrios llenos de vida.",
    "chinese": "长城是中国古代的军事防御工程，也是世界上最著名的历史遗迹之一。",
    "code": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
    "    return a\n",
    "chat": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n",
    "single_word": "Hello",
}

GREEDY_CHECK_TEXT = PARITY_TEXTS["chat"]

PROMPT_PASSAGE = (
    "A serving engine turns a trained model into something people can use. It loads the "
    "weights, accepts requests, runs the model, and returns tokens as they are produced. "
    "Most of the work happens in two phases. The first phase reads the whole prompt and "
    "fills the cache. The second phase produces one token at a time. The first phase "
    "decides how long a person waits before seeing anything, and the second decides how "
    "fast the answer appears on the screen. "
)


def synchronise(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def free_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def load_hf(path: str, dtype: torch.dtype, attention: str, device: str) -> Qwen2ForCausalLM:
    model = Qwen2ForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=attention)
    # The checkpoint's own config turns on sampling and a repetition penalty. Replacing it
    # leaves plain greedy search, which is what the engine runs here.
    model.generation_config = GenerationConfig()
    return model.to(device).eval()


def encode(tokenizer, text: str, device: str) -> Tensor:
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def prompt_of_length(tokenizer, length: int, device: str) -> Tensor:
    """Real text cut to an exact token count, so prompt length is a controlled setting."""
    passage = PROMPT_PASSAGE
    ids = encode(tokenizer, passage, device)
    while ids.shape[1] < length:
        passage += PROMPT_PASSAGE
        ids = encode(tokenizer, passage, device)
    return ids[:, :length]


# Parity


@torch.no_grad()
def compare_logits(mine: Qwen2, hf: Qwen2ForCausalLM, ids: Tensor) -> dict:
    my_logits = mine(ids)
    hf_logits = hf(ids).logits
    return {
        "tokens": ids.shape[1],
        "max_abs_difference": (my_logits - hf_logits).abs().max().item(),
        "largest_logit": hf_logits.abs().max().item(),
        "top_token_agreement": (
            (my_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
        ),
    }


def check_parity(path: str, tokenizer, device: str) -> dict:
    mine = load_pretrained(path, dtype=torch.float32, device=device)
    encoded = {name: encode(tokenizer, text, device) for name, text in PARITY_TEXTS.items()}
    check_ids = encode(tokenizer, GREEDY_CHECK_TEXT, device)
    greedy = SamplingParams(temperature=0.0, max_tokens=GREEDY_CHECK_TOKENS)
    my_output = generate(mine, check_ids, greedy)[0]

    results = {"dtype": "float32", "tolerance": PARITY_TOLERANCE}
    for attention in ("eager", "sdpa"):
        hf = load_hf(path, torch.float32, attention, device)
        texts = {name: compare_logits(mine, hf, ids) for name, ids in encoded.items()}
        hf_output = hf.generate(
            check_ids,
            attention_mask=torch.ones_like(check_ids),
            do_sample=False,
            max_new_tokens=GREEDY_CHECK_TOKENS,
        )[0]
        worst = max(text["max_abs_difference"] for text in texts.values())
        results[attention] = {
            "texts": texts,
            "worst_max_abs_difference": worst,
            "within_tolerance": worst < PARITY_TOLERANCE,
            "greedy_tokens_identical": torch.equal(my_output, hf_output),
        }
        del hf
        free_memory(device)

    results["greedy_sample"] = {
        "prompt": GREEDY_CHECK_TEXT,
        "tokens": GREEDY_CHECK_TOKENS,
        "completion": tokenizer.decode(my_output[check_ids.shape[1] :]),
    }
    del mine
    free_memory(device)
    return results


# Timing


def timed(model, run: Callable[[], Tensor], new_tokens: int, device: str) -> dict:
    """Run once, stamping the clock each time the model finishes a step."""
    stamps = []

    def stamp(*_) -> None:
        synchronise(device)
        stamps.append(time.perf_counter())

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    handle = model.lm_head.register_forward_hook(stamp)
    synchronise(device)
    start = time.perf_counter()
    output = run()
    synchronise(device)
    end = time.perf_counter()
    handle.remove()

    if len(stamps) != new_tokens:
        raise RuntimeError(f"expected {new_tokens} model calls, saw {len(stamps)}")
    steps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    return {
        "output": output,
        "ttft_ms": (stamps[0] - start) * 1000,
        # Excludes the first token, which belongs to the prompt phase.
        "decode_tokens_per_s": len(steps) / sum(steps),
        "ms_per_decode_token": statistics.median(steps) * 1000,
        "end_to_end_tokens_per_s": new_tokens / (end - start),
        "peak_gpu_mb": (
            torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
        ),
    }


@torch.no_grad()
def forward_only_mine(model: Qwen2, ids: Tensor, new_tokens: int) -> Tensor:
    """The smallest possible generation loop around my model."""
    weights = model.tok_embed.weight
    cache = KVCache(
        model.config, 1, ids.shape[1] + new_tokens, weights.device, weights.dtype
    )
    tokens = [ids]
    model_input = ids
    for _ in range(new_tokens):
        logits = model(model_input, cache=cache, last_token_only=True)
        model_input = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(model_input)
    return torch.cat(tokens, dim=1)[0]


@torch.no_grad()
def forward_only_hf(model: Qwen2ForCausalLM, ids: Tensor, new_tokens: int) -> Tensor:
    """The same loop around Hugging Face's model, with no generate function."""
    cache = DynamicCache(config=model.config)
    tokens = [ids]
    model_input = ids
    for _ in range(new_tokens):
        logits = model(
            model_input, past_key_values=cache, use_cache=True, logits_to_keep=1
        ).logits
        model_input = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(model_input)
    return torch.cat(tokens, dim=1)[0]


def summarise(runs: list[dict]) -> dict:
    summary = {}
    for metric in (
        "ttft_ms",
        "decode_tokens_per_s",
        "ms_per_decode_token",
        "end_to_end_tokens_per_s",
    ):
        values = [run[metric] for run in runs]
        summary[metric] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    summary["peak_gpu_mb"] = runs[-1]["peak_gpu_mb"]
    return summary


def first_difference(first: Tensor, second: Tensor) -> int | None:
    different = (first != second).nonzero()
    return different[0].item() if len(different) else None


def benchmark_prompt(
    models: dict, ids: Tensor, new_tokens: int, repeats: int, warmups: int, device: str
) -> dict:
    """Time every engine in turn, round by round, so drift affects them all alike."""
    prompt_length = ids.shape[1]
    params = SamplingParams(temperature=0.0, max_tokens=new_tokens)

    def run_mine() -> Tensor:
        return generate(models["mine"], ids, params)[0]

    def run_hf(model) -> Callable[[], Tensor]:
        return lambda: model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=new_tokens,
        )[0]

    engines = {
        "mine_generate": (models["mine"], run_mine),
        "hf_generate_sdpa": (models["sdpa"], run_hf(models["sdpa"])),
        "hf_generate_eager": (models["eager"], run_hf(models["eager"])),
        "mine_forward_only": (
            models["mine"], lambda: forward_only_mine(models["mine"], ids, new_tokens)
        ),
        "hf_forward_only_sdpa": (
            models["sdpa"], lambda: forward_only_hf(models["sdpa"], ids, new_tokens)
        ),
        "hf_forward_only_eager": (
            models["eager"], lambda: forward_only_hf(models["eager"], ids, new_tokens)
        ),
    }

    runs = {name: [] for name in engines}
    outputs = {}
    for round_number in range(warmups + repeats):
        for name, (model, run) in engines.items():
            result = timed(model, run, new_tokens, device)
            outputs[name] = result.pop("output")
            if round_number >= warmups:
                runs[name].append(result)

    reference = outputs["hf_generate_sdpa"]
    return {
        "prompt_tokens": prompt_length,
        "new_tokens": new_tokens,
        "engines": {name: summarise(results) for name, results in runs.items()},
        "raw_runs": runs,
        "first_token_position_differing_from_hf_sdpa": {
            name: first_difference(output, reference) for name, output in outputs.items()
        },
    }


def run_benchmark(
    path: str, tokenizer, device: str, prompt_lengths: list[int], new_tokens: int,
    repeats: int, warmups: int,
) -> dict:
    # float16 is what this model serves in on a T4, which has no native bfloat16.
    dtype = torch.float16
    models = {
        "mine": load_pretrained(path, dtype=dtype, device=device),
        "sdpa": load_hf(path, dtype, "sdpa", device),
        "eager": load_hf(path, dtype, "eager", device),
    }
    results = [
        benchmark_prompt(
            models, prompt_of_length(tokenizer, length, device), new_tokens, repeats, warmups, device
        )
        for length in prompt_lengths
    ]
    config = models["mine"].config
    return {
        "dtype": "float16",
        "my_kv_cache_mb_per_prompt": {
            result["prompt_tokens"]: KVCache(
                config, 1, result["prompt_tokens"] + new_tokens, "meta", dtype
            ).nbytes / 2**20
            for result in results
        },
        "prompts": results,
    }


# Reporting


def hardware(device: str) -> dict:
    info = {"device": device, "os": platform.platform(), "python": platform.python_version()}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpus_visible"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        info["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
            text=True,
        ).strip()
        info["cuda_runtime"] = torch.version.cuda
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="local checkpoint")
    parser.add_argument("--revision", default="unknown")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 1024])
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("no GPU visible, refusing to report CPU numbers as GPU numbers")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one GPU, found {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    result = {
        "what": "My Qwen2 engine against Hugging Face on one request",
        "model": {"name": args.model_name, "revision": args.revision},
        "hardware": hardware(args.device),
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "settings": {
            "search": "greedy, stop tokens off, no repetition penalty",
            "batch_size": 1,
            "prompt_lengths": args.prompt_lengths,
            "new_tokens": args.new_tokens,
            "repeats": args.repeats,
            "warmup_rounds_discarded": args.warmups,
            "timing": "hook on lm_head with GPU synchronisation, identical for both engines",
            "ttft": "from calling generate to the first logits being ready on the GPU",
            "decode_tokens_per_s": "tokens after the first, between first and last stamp",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    result["parity"] = check_parity(args.model_path, tokenizer, args.device)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("parity:", {k: result["parity"][k]["worst_max_abs_difference"] for k in ("eager", "sdpa")})

    result["benchmark"] = run_benchmark(
        args.model_path, tokenizer, args.device, args.prompt_lengths, args.new_tokens,
        args.repeats, args.warmups,
    )
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    for prompt in result["benchmark"]["prompts"]:
        print(f"prompt {prompt['prompt_tokens']} tokens, {prompt['new_tokens']} new:")
        for name, engine in prompt["engines"].items():
            print(
                f"  {name:24} ttft {engine['ttft_ms']['median']:9.2f} ms   "
                f"decode {engine['decode_tokens_per_s']['median']:8.2f} tok/s"
            )

    return 0 if result["parity"]["eager"]["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

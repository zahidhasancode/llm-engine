"""Check a real checkpoint against Hugging Face, then time both on one request.

Three things are measured and saved as JSON:

1. Parity: my logits against Hugging Face's on real text, in float32.
2. Speed: time to first token and tokens per second through each generate function.
3. Forward only: the same model loop with no generate function around it, which shows
   how much of any speed difference comes from the model and how much from the wrapper.
4. Attention variants: my model with its attention step swapped, which shows whether
   the attention function explains the difference. They exist only in this file.

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
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor
from transformers import AutoTokenizer, DynamicCache, GenerationConfig, Qwen2ForCausalLM

from engine.cache import KVCache
from engine.generate import generate
from engine.loader import load_pretrained
from engine.model import Attention, Qwen2, rotary_embedding
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


class SdpaAttention(Attention):
    """PyTorch's built-in attention, the function Hugging Face uses by default."""

    def attend(self, q: Tensor, k: Tensor, v: Tensor, future: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=~future)


class Float16ScoresAttention(Attention):
    """My attention with no float32 step. Faster or not, it can overflow, so it is only a probe."""

    def attend(self, q: Tensor, k: Tensor, v: Tensor, future: Tensor) -> Tensor:
        scores = (q * self.scale) @ k.transpose(-2, -1)
        return scores.masked_fill(future, -torch.inf).softmax(dim=-1) @ v


class SdpaNoMaskAttention(SdpaAttention):
    """Built-in attention that skips the mask when nothing is hidden, as in every decode step."""

    def attend(self, q: Tensor, k: Tensor, v: Tensor, future: Tensor) -> Tensor:
        if future.any():
            return super().attend(q, k, v, future)
        return F.scaled_dot_product_attention(q, k, v)


def with_attention(model: Qwen2, attention: type[Attention]) -> Qwen2:
    """A second model that shares the weights, so it costs no extra memory."""
    with torch.device("meta"):
        variant = Qwen2(model.config)
    variant.load_state_dict(model.state_dict(), assign=True)
    for block in variant.blocks:
        block.attn.__class__ = attention
    return variant.eval()


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


def json_number(value: float) -> float | None:
    """JSON has no NaN or inf, so a broken value is saved as null."""
    return value if math.isfinite(value) else None


def logits_of(model, ids: Tensor) -> Tensor:
    output = model(ids)
    return output if isinstance(output, Tensor) else output.logits


@torch.no_grad()
def compare_logits(first, second, ids: Tensor) -> dict:
    first_logits = logits_of(first, ids)
    second_logits = logits_of(second, ids)
    return {
        "tokens": ids.shape[1],
        "max_abs_difference": json_number((first_logits - second_logits).abs().max().item()),
        "largest_logit": json_number(second_logits.abs().max().item()),
        "top_token_agreement": (
            (first_logits.argmax(-1) == second_logits.argmax(-1)).float().mean().item()
        ),
    }


@torch.no_grad()
def top_two(model, ids: Tensor) -> dict:
    """The two best first tokens. A small gap means rounding alone can swap them."""
    logits = logits_of(model, ids)
    values, tokens = logits[0, -1].topk(2)
    return {
        "finite": logits.isfinite().all().item(),
        "tokens": tokens.tolist(),
        "logits": [json_number(value) for value in values.tolist()],
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


def median_ms(work: Callable[[], object], device: str, repeats: int = 200) -> float:
    times = []
    for _ in range(repeats):
        synchronise(device)
        start = time.perf_counter()
        work()
        synchronise(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000


@torch.no_grad()
def step_overheads(model: Qwen2, cached_tokens: int, device: str) -> dict:
    """What one decode step spends on work other than the attention product itself."""
    config = model.config
    dtype = model.tok_embed.weight.dtype
    target = torch.device(device)
    key = torch.empty(1, config.num_kv_heads, cached_tokens, config.head_dim, device=device, dtype=dtype)
    repeats_needed = config.num_heads // config.num_kv_heads
    per_layer = {
        "mask": median_ms(
            lambda: torch.triu(
                torch.ones(1, cached_tokens, dtype=torch.bool, device=target), cached_tokens
            ),
            device,
        ),
        "repeat_key_and_value": median_ms(
            lambda: (
                key.repeat_interleave(repeats_needed, dim=1),
                key.repeat_interleave(repeats_needed, dim=1),
            ),
            device,
        ),
    }
    rotary = median_ms(
        lambda: rotary_embedding(config, cached_tokens, 1, target, dtype), device
    )
    return {
        "cached_tokens": cached_tokens,
        "layers": config.num_layers,
        "rotary_ms_once_per_step": rotary,
        "ms_per_layer": per_layer,
        "ms_per_step_total": rotary + config.num_layers * sum(per_layer.values()),
        "note": "Each piece timed alone with GPU synchronisation, so this is an upper estimate",
    }


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

    def run_mine(model) -> Callable[[], Tensor]:
        return lambda: generate(model, ids, params)[0]

    def run_hf(model) -> Callable[[], Tensor]:
        return lambda: model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=new_tokens,
        )[0]

    engines = {
        "mine_generate": (models["mine"], run_mine(models["mine"])),
        "mine_with_sdpa_generate": (
            models["mine_sdpa"], run_mine(models["mine_sdpa"])
        ),
        "mine_with_sdpa_no_mask_generate": (
            models["mine_sdpa_no_mask"], run_mine(models["mine_sdpa_no_mask"])
        ),
        "mine_with_float16_scores_generate": (
            models["mine_float16_scores"], run_mine(models["mine_float16_scores"])
        ),
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

    return {
        "prompt_tokens": prompt_length,
        "new_tokens": new_tokens,
        "engines": {name: summarise(results) for name, results in runs.items()},
        "raw_runs": runs,
        "first_token_position_differing_from_hf_sdpa": {
            name: first_difference(output, outputs["hf_generate_sdpa"])
            for name, output in outputs.items()
        },
        "first_token_position_differing_from_hf_eager": {
            name: first_difference(output, outputs["hf_generate_eager"])
            for name, output in outputs.items()
        },
        "first_token_choice": {
            name: top_two(model, ids) for name, model in models.items()
        },
        "logits_in_benchmark_dtype": {
            "mine_vs_hf_eager": compare_logits(models["mine"], models["eager"], ids),
            "mine_vs_hf_sdpa": compare_logits(models["mine"], models["sdpa"], ids),
            "hf_eager_vs_hf_sdpa": compare_logits(models["eager"], models["sdpa"], ids),
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
    models["mine_sdpa"] = with_attention(models["mine"], SdpaAttention)
    models["mine_sdpa_no_mask"] = with_attention(models["mine"], SdpaNoMaskAttention)
    models["mine_float16_scores"] = with_attention(models["mine"], Float16ScoresAttention)
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
        "my_step_overheads": step_overheads(
            models["mine"], results[-1]["prompt_tokens"] + new_tokens // 2, device
        ),
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
            "peak_gpu_mb": "includes all loaded models, so it does not compare engines",
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
                f"  {name:34} ttft {engine['ttft_ms']['median']:9.2f} ms   "
                f"decode {engine['decode_tokens_per_s']['median']:8.2f} tok/s"
            )

    # A NaN run still produces timings, so it has to be failed on purpose.
    finite = all(
        prompt["first_token_choice"]["mine"]["finite"] for prompt in result["benchmark"]["prompts"]
    )
    return 0 if finite and result["parity"]["eager"]["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

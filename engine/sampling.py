"""Choosing the next token from the model's logits.

The steps run in this order, each one working on the result of the last:

    repetition penalty -> temperature -> top-k -> top-p -> draw one token

Everything random lives in one Sampler per request. Its generator is private, sits on the
CPU, and is used exactly once per sampled token, so a request's random numbers depend
only on how many tokens that request has produced. No other request can shift them, and
neither can the global torch seed.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    """Settings for one request.

    temperature: 0 always takes the most likely token. Higher values flatten the choice.
    top_k: keep only the k most likely tokens. 0 turns it off.
    top_p: keep the smallest set of most likely tokens whose total probability reaches p.
        1 turns it off.
    repetition_penalty: above 1, tokens already seen (prompt included) get less likely.
    seed: same seed, same output. None picks a fresh random seed.
    stop_tokens: generation ends on any of these, and that token is kept in the output.
    max_tokens: the most tokens to generate.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop_tokens: tuple[int, ...] = ()
    max_tokens: int = 16

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.top_k < 0:
            raise ValueError("top_k must not be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be above 0 and at most 1")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be above 0")
        if self.max_tokens < 0:
            raise ValueError("max_tokens must not be negative")


def apply_repetition_penalty(logits: Tensor, seen: Tensor, penalty: float) -> Tensor:
    """Make seen tokens less likely. A negative logit must be multiplied to go down."""
    penalised = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen, penalised, logits)


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Divide logits by temperature.

    Subtracting the maximum changes nothing after softmax. It keeps a very small
    temperature from pushing the top logit to +inf, which softmax would turn into NaN.
    """
    return (logits - logits.max()) / temperature


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Hide everything but exactly k tokens. Ties are broken rather than all kept."""
    keep = logits.topk(min(k, logits.numel())).indices
    filtered = torch.full_like(logits, -torch.inf)
    filtered[keep] = logits[keep]
    return filtered


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Hide everything outside the smallest top set whose probability reaches p."""
    sorted_logits, order = logits.sort(descending=True, stable=True)
    probs = sorted_logits.softmax(dim=-1)

    # Keep a token if the ones before it have not reached p yet. That way the token which
    # crosses p stays, and the first token always stays.
    total_before = torch.cat((probs.new_zeros(1), probs.cumsum(dim=-1)[:-1]))
    keep = order[total_before < p]

    filtered = torch.full_like(logits, -torch.inf)
    filtered[keep] = logits[keep]
    return filtered


def process_logits(logits: Tensor, params: SamplingParams, seen: Tensor) -> Tensor:
    """Apply every step except the draw. Logits are one token's scores, of shape (V,)."""
    if params.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, seen, params.repetition_penalty)
    if params.temperature == 0:
        return logits
    logits = apply_temperature(logits, params.temperature)
    if params.top_k > 0:
        logits = top_k_filter(logits, params.top_k)
    if params.top_p < 1.0:
        logits = top_p_filter(logits, params.top_p)
    return logits


def draw(logits: Tensor, generator: torch.Generator) -> int:
    """Draw one token by inverse transform sampling, using exactly one random number.

    Lay the probabilities end to end on a line, pick a uniform point on it, and return
    the token whose stretch of the line contains that point.
    """
    probs = logits.softmax(dim=-1)
    kept = probs.nonzero().squeeze(-1)
    cumulative = probs[kept].cumsum(dim=-1)

    # The generator stays on the CPU, so a seed gives the same draws on any device.
    uniform = torch.rand(1, generator=generator, dtype=torch.float64).to(logits.device)
    point = uniform * cumulative[-1]

    # Only kept tokens are on the line, so a hidden token can never be chosen. The clamp
    # covers rounding that lands the point exactly on the end of the line.
    index = torch.searchsorted(cumulative, point, right=True).clamp(max=kept.numel() - 1)
    return kept[index].item()


class Sampler:
    """All the sampling state of one request: its settings, generator and seen tokens."""

    def __init__(self, params: SamplingParams, prompt: Tensor, vocab_size: int) -> None:
        self.params = params
        self.generator = torch.Generator(device="cpu")
        if params.seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(params.seed)

        self.seen = torch.zeros(vocab_size, dtype=torch.bool, device=prompt.device)
        self.seen[prompt] = True

    def next_token(self, logits: Tensor) -> int:
        """Choose a token from logits of shape (V,) and remember it."""
        # float64 keeps the probabilities accurate when they are summed up.
        logits = process_logits(logits.double(), self.params, self.seen)
        if self.params.temperature == 0:
            token = logits.argmax().item()
        else:
            token = draw(logits, self.generator)
        self.seen[token] = True
        return token

"""Qwen2 written from scratch in PyTorch.

The data flow for one forward pass:

    token ids -> embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits

Each decoder block is pre-norm: normalise, transform, add back to the input.

    x = x + attention(norm(x))
    x = x + mlp(norm(x))

Tensor shape letters used in comments:
    B batch size, T sequence length, H number of heads, D head_dim.

Attention dropout is left out on purpose: this model is only ever used for inference.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from engine.cache import KVCache


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must divide evenly into num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must divide evenly into num_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even: rotary embeddings rotate pairs of values")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class RMSNorm(nn.Module):
    """Divide each vector by its root mean square, then scale by a learned weight.

    Unlike LayerNorm it does not subtract the mean, and it has no bias.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Squaring can overflow float16, so the norm is always computed in float32.
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


def rotary_embedding(
    config: ModelConfig, positions: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Return cos and sin tables of shape (B, 1, T, D) for positions of shape (B, T).

    Every position gets a set of angles. Each pair of values in a head is rotated by
    its own frequency, from fast to slow, so positions are encoded at many scales.
    The 1 lines up with the heads dimension, so each row can sit at its own position.
    """
    dim = config.head_dim
    steps = torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
    inv_freq = 1.0 / (config.rope_theta ** (steps / dim))
    angles = positions[:, None, :, None].float() * inv_freq  # (B, 1, T, D/2)
    angles = torch.cat((angles, angles), dim=-1)  # (B, 1, T, D)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def future_mask(positions: Tensor) -> Tensor:
    """Mark what each new token must not see, with shape (B, 1, T, total).

    A token may only look at itself and earlier tokens. Comparing every stored position
    with each token's own position gives the causal mask, and it also hides the unused
    positions of any row shorter than the longest one.
    """
    stored = torch.arange(positions.max() + 1, device=positions.device)
    return stored > positions[:, None, :, None]


def rotate_half(x: Tensor) -> Tensor:
    """Turn each pair (a, b), held at matching spots in the two halves, into (-b, a)."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate (B, H, T, D) vectors by their position's angles: x' = x cos + rot90(x) sin."""
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    """Causal multi-head self-attention with rotary positions.

    With num_kv_heads < num_heads this is grouped-query attention: several query heads
    share one key/value head, which makes the K and V projections smaller.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.queries_per_kv = config.num_heads // config.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        future: Tensor,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        batch, seq_len, _ = x.shape

        # Split into heads: (B, T, H*D) -> (B, H, T, D).
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Rotate Q and K only. V carries content, and position enters through the scores.
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # From here K and V cover every token so far, while Q covers only the new ones.
        if cache is not None:
            k, v = cache.update(layer, k, v)

        # Copy each K and V head so that every query head has a partner.
        k = k.repeat_interleave(self.queries_per_kv, dim=1)
        v = v.repeat_interleave(self.queries_per_kv, dim=1)

        out = self.attend(q, k, v, future)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, -1))  # (B, T, H*D)

    def attend(self, q: Tensor, k: Tensor, v: Tensor, future: Tensor) -> Tensor:
        """Scaled dot-product attention. Returns one mixed value per query, (B, H, T, D)."""
        # In float32, because float16 stops at 65504. Dot products can pass that on real
        # weights, and they would become inf, which softmax turns into NaN.
        scores = (q.float() @ k.float().transpose(-2, -1)) * self.scale  # (B, H, T, total)
        scores = scores.masked_fill(future, -torch.inf)
        weights = F.softmax(scores, dim=-1).to(q.dtype)
        return weights @ v


class MLP(nn.Module):
    """Gated feed-forward network: silu(gate(x)) * up(x), projected back down."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Attention(config)
        self.norm_mlp = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        future: Tensor,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin, future, cache, layer)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Qwen2(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embed.weight

    def forward(
        self,
        input_ids: Tensor,
        cache: KVCache | None = None,
        last_token_only: bool = False,
        slots: Tensor | None = None,
    ) -> Tensor:
        """Map token ids of shape (B, T) to logits of shape (B, T, vocab_size).

        With a cache, input_ids holds only the tokens the cache has not seen yet, and
        slots says which slot each row belongs to, in order. Without slots, row i is
        slot i. Slots can be at different lengths, but each row adds the same T tokens.
        With last_token_only, logits have shape (B, 1, vocab_size), which is all that
        generation needs.
        """
        batch, seq_len = input_ids.shape
        if cache is None:
            positions = torch.arange(seq_len).expand(batch, seq_len)
        else:
            slots = cache.all_slots if slots is None else slots
            positions = cache.begin(slots, batch, seq_len)
        positions = positions.to(input_ids.device)

        x = self.tok_embed(input_ids)
        cos, sin = rotary_embedding(self.config, positions, x.dtype)
        future = future_mask(positions)
        for layer, block in enumerate(self.blocks):
            x = block(x, cos, sin, future, cache, layer)

        if cache is not None:
            cache.advance(slots, seq_len)
        if last_token_only:
            x = x[:, -1:]
        return self.lm_head(self.norm(x))

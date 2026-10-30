"""A preallocated key/value cache for fast generation.

Attention needs the keys and values of every earlier token. Those never change, because
a token can only see the past. So each one is computed once and kept here, and a
generation step only has to process the newest token.

All memory is taken up front for max_length tokens. Nothing is reallocated or copied as
the sequence grows: a write fills the next empty slots and `length` moves forward.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from engine.model import ModelConfig


class KVCache:
    """Keys and values for every layer, each of shape (layers, B, kv_heads, max_length, D).

    Keys are stored after rotation, since each one is rotated by its own absolute position
    and that never changes. They are stored before being repeated across query heads, so
    with grouped-query attention the cache shrinks along with the K and V projections.
    """

    def __init__(
        self,
        config: "ModelConfig",
        batch_size: int,
        max_length: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (config.num_layers, batch_size, config.num_kv_heads, max_length, config.head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty(shape, device=device, dtype=dtype)
        self.batch_size = batch_size
        self.max_length = max_length
        self.length = 0

    @property
    def nbytes(self) -> int:
        """Memory held by the cache. It is fixed from creation, however full the cache is."""
        return self.keys.nbytes + self.values.nbytes

    def check_fits(self, batch_size: int, new_tokens: int) -> None:
        """Raise before anything is written if this input cannot be stored."""
        if batch_size != self.batch_size:
            raise ValueError(f"cache holds batch size {self.batch_size}, got {batch_size}")
        if self.length + new_tokens > self.max_length:
            raise ValueError(
                f"cache is full: {self.length} + {new_tokens} tokens > {self.max_length}"
            )

    def update(self, layer: int, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """Store one layer's new keys and values, and return everything held so far.

        The returned tensors are views into the cache, not copies. Slots past the end are
        never exposed, so uninitialised memory can never reach attention.
        """
        end = self.length + keys.shape[2]
        self.keys[layer, :, :, self.length : end] = keys
        self.values[layer, :, :, self.length : end] = values
        return self.keys[layer, :, :, :end], self.values[layer, :, :, :end]

    def advance(self, new_tokens: int) -> None:
        """Move the length forward. Called once per forward pass, after all layers wrote."""
        self.length += new_tokens

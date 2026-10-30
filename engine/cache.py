"""A preallocated key/value cache with a fixed number of slots.

Attention needs the keys and values of every earlier token. Those never change, because
a token can only see the past. So each one is computed once and kept here, and a
generation step only has to process the newest token.

Each slot holds one request and has its own length, so requests of different lengths can
share a forward pass without padding. All memory is taken up front for num_slots requests
of up to max_length tokens each. Nothing is reallocated or copied as sequences grow: a
write fills the slot's next empty positions and its length moves forward.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from engine.model import ModelConfig


class KVCache:
    """Keys and values for every layer, each of shape (layers, slots, kv_heads, max_length, D).

    Keys are stored after rotation, since each one is rotated by its own absolute position
    and that never changes. They are stored before being repeated across query heads, so
    with grouped-query attention the cache shrinks along with the K and V projections.

    Lengths stay on the CPU, so reading them never has to wait for the GPU.
    """

    def __init__(
        self,
        config: "ModelConfig",
        num_slots: int,
        max_length: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (config.num_layers, num_slots, config.num_kv_heads, max_length, config.head_dim)
        # Zeros, not empty memory. A shorter slot reads positions past its own end, and
        # although attention gives them weight 0, 0 * NaN is still NaN.
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)
        self.num_slots = num_slots
        self.max_length = max_length
        self.lengths = torch.zeros(num_slots, dtype=torch.long)
        self.all_slots = torch.arange(num_slots)
        self._rows = self._columns = self._reads = None
        self._total = 0

    @property
    def nbytes(self) -> int:
        """Memory held by the cache. It is fixed from creation, however full the cache is."""
        return self.keys.nbytes + self.values.nbytes

    def begin(self, slots: Tensor, batch_size: int, new_tokens: int) -> Tensor:
        """Check that a forward pass fits, and return its positions, of shape (R, T).

        Raises before anything is written if this input cannot be stored.
        """
        if batch_size != slots.numel():
            raise ValueError(f"got batch size {batch_size} for {slots.numel()} slots")
        positions = self.lengths[slots][:, None] + torch.arange(new_tokens)
        if positions.max() >= self.max_length:
            raise ValueError(
                f"cache is full: slot needs {positions.max() + 1} tokens, "
                f"holds {self.max_length}"
            )
        device = self.keys.device
        self._rows = slots[:, None].to(device)
        self._columns = positions.to(device)
        self._total = positions.max().item() + 1

        # Slots side by side can be read as a view. Any other selection needs a copy.
        first = slots[0].item()
        contiguous = torch.equal(slots, torch.arange(first, first + slots.numel()))
        self._reads = (
            slice(first, first + slots.numel()) if contiguous else slots.to(device)
        )
        return positions

    def update(self, layer: int, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """Store one layer's new keys and values, and return everything the slots hold.

        Incoming tensors have shape (R, kv_heads, T, D). The result covers as many
        positions as the longest slot, and shorter slots rely on their mask.
        """
        self.keys[layer][self._rows, :, self._columns] = keys.transpose(1, 2)
        self.values[layer][self._rows, :, self._columns] = values.transpose(1, 2)
        return (
            self._read(self.keys[layer]),
            self._read(self.values[layer]),
        )

    def _read(self, tensor: Tensor) -> Tensor:
        tensor = tensor[:, :, : self._total]
        if isinstance(self._reads, slice):
            return tensor[self._reads]
        return tensor.index_select(0, self._reads)

    def advance(self, slots: Tensor, new_tokens: int) -> None:
        """Move lengths forward. Called once per forward pass, after all layers wrote."""
        self.lengths[slots] += new_tokens

    def release(self, slot: int) -> None:
        """Empty a slot for its next request.

        The old values are cleared because a NaN left behind would reach the next
        occupant's output through the 0 * NaN case described above.
        """
        length = self.lengths[slot].item()
        self.keys[:, slot, :, :length] = 0
        self.values[:, slot, :, :length] = 0
        self.lengths[slot] = 0

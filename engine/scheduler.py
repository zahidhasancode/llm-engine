"""Continuous batching: requests join and leave between model steps.

Static batching waits for a whole batch to finish, so one long request holds every slot.
Here each step does this:

    1. Move waiting requests into free slots, oldest first.
    2. Run up to `prefill_tokens_per_step` prompt tokens, oldest request first.
    3. Run one decode pass, with one token for every request past its prefill.
    4. A finished request leaves at once and its slot is free for the next step.

Mixing prefill and decode. A long prompt is cut into chunks, so a decode pass runs after
each chunk. Decoding requests are held up by at most one chunk per token, and a joining
request cannot be starved, since prefill gets its budget every step. The budget sets the
tradeoff: larger means a faster first token but a longer stall for everyone decoding.

Everything is synchronous and in-process. There is no network in this file.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor

from engine.cache import KVCache
from engine.model import Qwen2
from engine.sampling import Sampler, SamplingParams


@dataclass(frozen=True)
class Event:
    """Something that happened to a request. The last event has a finish_reason."""

    request_id: int
    token: int | None = None
    finish_reason: str | None = None


@dataclass
class _Request:
    id: int
    prompt: Tensor
    params: SamplingParams
    sampler: Sampler | None = None
    slot: int | None = None
    prefilled: int = 0
    generated: list[int] = field(default_factory=list)


class Scheduler:
    def __init__(
        self,
        model: Qwen2,
        num_slots: int,
        max_length: int,
        prefill_tokens_per_step: int = 256,
    ) -> None:
        if prefill_tokens_per_step < 1:
            raise ValueError("prefill_tokens_per_step must be at least 1")
        weights = model.tok_embed.weight
        self.model = model
        self.device = weights.device
        self.max_length = max_length
        self.prefill_tokens_per_step = prefill_tokens_per_step
        self.cache = KVCache(model.config, num_slots, max_length, weights.device, weights.dtype)
        self.waiting: deque[_Request] = deque()
        self.prefilling: deque[_Request] = deque()
        self.decoding: dict[int, _Request] = {}  # by slot
        self.free_slots = list(range(num_slots))  # a heap, so the lowest slot is used first
        self._next_id = 0
        self._ready: list[Event] = []

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.prefilling or self.decoding)

    def submit(self, prompt: Tensor, params: SamplingParams) -> int:
        """Queue a request and return its id. It starts on a later step."""
        if prompt.numel() + params.max_tokens > self.max_length:
            raise ValueError(
                f"prompt of {prompt.numel()} tokens plus max_tokens {params.max_tokens} "
                f"is over the limit of {self.max_length}"
            )
        request = _Request(self._next_id, prompt.to(self.device), params)
        self._next_id += 1
        if params.max_tokens == 0:
            self._ready.append(Event(request.id, finish_reason="length"))
        else:
            self.waiting.append(request)
        return request.id

    @torch.no_grad()
    def step(self) -> list[Event]:
        """Advance every request by one scheduling step, and return what happened."""
        events, self._ready = self._ready, []
        self._admit()
        self._prefill(events)
        self._decode(events)
        return events

    def run(self) -> list[Event]:
        """Step until nothing is left, and return all events."""
        events = []
        while self.has_work or self._ready:
            events.extend(self.step())
        return events

    def _admit(self) -> None:
        while self.waiting and self.free_slots:
            request = self.waiting.popleft()
            request.slot = heapq.heappop(self.free_slots)
            request.sampler = Sampler(request.params, request.prompt, self.model.config.vocab_size)
            self.prefilling.append(request)

    def _prefill(self, events: list[Event]) -> None:
        budget = self.prefill_tokens_per_step
        while budget > 0 and self.prefilling:
            request = self.prefilling[0]
            chunk = request.prompt[request.prefilled : request.prefilled + budget]
            logits = self.model(
                chunk[None], self.cache, True, torch.tensor([request.slot])
            )
            budget -= chunk.numel()
            request.prefilled += chunk.numel()
            if request.prefilled == request.prompt.numel():
                self.prefilling.popleft()
                if not self._choose(request, logits[0, -1], events):
                    self.decoding[request.slot] = request

    def _decode(self, events: list[Event]) -> None:
        if not self.decoding:
            return
        slots = sorted(self.decoding)
        requests = [self.decoding[slot] for slot in slots]
        input_ids = torch.tensor(
            [[request.generated[-1]] for request in requests], device=self.device
        )
        logits = self.model(input_ids, self.cache, True, torch.tensor(slots))
        for row, (slot, request) in enumerate(zip(slots, requests)):
            if self._choose(request, logits[row, -1], events):
                del self.decoding[slot]

    def _choose(self, request: _Request, logits: Tensor, events: list[Event]) -> bool:
        """Pick the next token. Returns True if that finished the request."""
        token = request.sampler.next_token(logits)
        request.generated.append(token)
        reason = None
        if token in request.params.stop_tokens:
            reason = "stop"
        elif len(request.generated) == request.params.max_tokens:
            reason = "length"
        events.append(Event(request.id, token, reason))
        if reason is not None:
            self._free(request)
        return reason is not None

    def _free(self, request: _Request) -> None:
        self.cache.release(request.slot)
        heapq.heappush(self.free_slots, request.slot)
        request.slot = None

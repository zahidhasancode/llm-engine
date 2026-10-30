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

Limits. The cache is one preallocated block of num_slots * max_length tokens, so a memory
budget decides how many slots exist and nothing grows afterwards. When every slot is busy,
new requests wait, and past max_waiting they are refused. One bad request ends with an
error event, and the others carry on.

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


class InvalidRequest(ValueError):
    """The request itself is wrong. Sending it again will not help."""


class PromptTooLong(InvalidRequest):
    """The prompt plus max_tokens cannot fit in a slot."""


class QueueFull(RuntimeError):
    """The server is busy. The same request can be sent again later."""


@dataclass(frozen=True)
class Event:
    """Something that happened to a request. The last event has a finish_reason.

    finish_reason is "stop", "length", "cancelled" or "error", and error says why.
    """

    request_id: int
    token: int | None = None
    finish_reason: str | None = None
    error: str | None = None


@dataclass(eq=False)
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
        max_waiting: int = 1024,
        cache_budget_bytes: int | None = None,
    ) -> None:
        if prefill_tokens_per_step < 1:
            raise ValueError("prefill_tokens_per_step must be at least 1")
        weights = model.tok_embed.weight
        if cache_budget_bytes is not None:
            per_slot = KVCache.bytes_per_slot(model.config, max_length, weights.dtype)
            num_slots = min(num_slots, cache_budget_bytes // per_slot)
            if num_slots < 1:
                raise ValueError(
                    f"cache budget of {cache_budget_bytes} bytes is below one slot of "
                    f"{per_slot} bytes at max_length {max_length}"
                )
        self.model = model
        self.device = weights.device
        self.max_length = max_length
        self.prefill_tokens_per_step = prefill_tokens_per_step
        self.max_waiting = max_waiting
        self.cache = KVCache(model.config, num_slots, max_length, weights.device, weights.dtype)
        self.waiting: deque[_Request] = deque()
        self.prefilling: deque[_Request] = deque()
        self.decoding: dict[int, _Request] = {}  # by slot
        self.free_slots = list(range(num_slots))  # a heap, so the lowest slot is used first
        self.requests: dict[int, _Request] = {}  # every unfinished request, by id
        self._next_id = 0
        self._ready: list[Event] = []

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.prefilling or self.decoding)

    def submit(self, prompt: Tensor, params: SamplingParams) -> int:
        """Queue a request and return its id. It starts on a later step.

        Raises InvalidRequest or PromptTooLong for a bad request, and QueueFull when the
        waiting queue is at its limit. Neither leaves anything behind.
        """
        self._validate(prompt, params)
        if len(self.waiting) >= self.max_waiting:
            raise QueueFull(f"waiting queue is full at {self.max_waiting} requests")

        request = _Request(self._next_id, prompt.to(self.device), params)
        self._next_id += 1
        if params.max_tokens == 0:
            self._ready.append(Event(request.id, finish_reason="length"))
        else:
            self.requests[request.id] = request
            self.waiting.append(request)
        return request.id

    def _validate(self, prompt: Tensor, params: SamplingParams) -> None:
        if not isinstance(params, SamplingParams):
            raise InvalidRequest("params must be SamplingParams")
        if not isinstance(prompt, Tensor) or prompt.dim() != 1 or prompt.is_floating_point():
            raise InvalidRequest("prompt must be a 1-D tensor of token ids")
        if prompt.numel() == 0:
            raise InvalidRequest("prompt must have at least one token")
        vocab_size = self.model.config.vocab_size
        if prompt.min() < 0 or prompt.max() >= vocab_size:
            raise InvalidRequest(f"prompt has a token id outside 0..{vocab_size - 1}")
        if prompt.numel() + params.max_tokens > self.max_length:
            raise PromptTooLong(
                f"prompt of {prompt.numel()} tokens plus max_tokens {params.max_tokens} "
                f"is over the limit of {self.max_length}"
            )

    def cancel(self, request_id: int) -> bool:
        """Stop a request and free its slot at once. Returns False if it is not running."""
        request = self.requests.get(request_id)
        if request is None:
            return False
        self._finish(request, Event(request_id, finish_reason="cancelled"))
        return True

    @torch.no_grad()
    def step(self) -> list[Event]:
        """Advance every request by one scheduling step, and return what happened."""
        events = []
        self._admit()
        self._prefill(events)
        self._decode(events)
        # Cancellations and failures go out with this step, not one step late.
        ready, self._ready = self._ready, []
        return ready + events

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
            self.prefilling.append(request)
            try:
                request.sampler = Sampler(
                    request.params, request.prompt, self.model.config.vocab_size
                )
            except Exception as error:
                self._fail(request, error)

    def _forward(self, input_ids: Tensor, slots: list[int]) -> tuple[Tensor, list[bool]]:
        """Run the model, and say which rows came back with usable logits."""
        logits = self.model(input_ids, self.cache, True, torch.tensor(slots))[:, -1]
        return logits, logits.isfinite().all(dim=-1).tolist()

    def _prefill(self, events: list[Event]) -> None:
        budget = self.prefill_tokens_per_step
        while budget > 0 and self.prefilling:
            request = self.prefilling[0]
            chunk = request.prompt[request.prefilled : request.prefilled + budget]
            try:
                logits, finite = self._forward(chunk[None], [request.slot])
            except Exception as error:
                self._fail(request, error)
                continue
            budget -= chunk.numel()
            request.prefilled += chunk.numel()
            if request.prefilled == request.prompt.numel():
                self.prefilling.popleft()
                if self._choose(request, logits[0], finite[0], events):
                    continue
                self.decoding[request.slot] = request

    def _decode(self, events: list[Event]) -> None:
        if not self.decoding:
            return
        slots = sorted(self.decoding)
        requests = [self.decoding[slot] for slot in slots]
        input_ids = torch.tensor(
            [[request.generated[-1]] for request in requests], device=self.device
        )
        try:
            logits, finite = self._forward(input_ids, slots)
        except Exception:
            self._decode_one_by_one(requests, input_ids, slots, events)
            return
        for row, (slot, request) in enumerate(zip(slots, requests)):
            if self._choose(request, logits[row], finite[row], events):
                self.decoding.pop(slot, None)

    def _decode_one_by_one(
        self, requests: list[_Request], input_ids: Tensor, slots: list[int], events: list[Event]
    ) -> None:
        """A shared pass failed, so find out which request caused it.

        Lengths only move forward after a pass succeeds, so running a row again simply
        overwrites whatever the failed pass wrote.
        """
        for row, (slot, request) in enumerate(zip(slots, requests)):
            try:
                logits, finite = self._forward(input_ids[row : row + 1], [slot])
            except Exception as error:
                self._fail(request, error)
                continue
            if self._choose(request, logits[0], finite[0], events):
                self.decoding.pop(slot, None)

    def _choose(
        self, request: _Request, logits: Tensor, finite: bool, events: list[Event]
    ) -> bool:
        """Pick the next token. Returns True if that finished the request."""
        if not finite:
            self._fail(request, ValueError("the model produced NaN or inf logits"))
            return True
        try:
            token = request.sampler.next_token(logits)
        except Exception as error:
            self._fail(request, error)
            return True

        request.generated.append(token)
        reason = None
        if token in request.params.stop_tokens:
            reason = "stop"
        elif len(request.generated) == request.params.max_tokens:
            reason = "length"
        if reason is None:
            events.append(Event(request.id, token))
            return False
        self._finish(request, Event(request.id, token, reason), events)
        return True

    def _fail(self, request: _Request, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        self._finish(request, Event(request.id, finish_reason="error", error=message))

    def _finish(
        self, request: _Request, event: Event, events: list[Event] | None = None
    ) -> None:
        """Take a request out of wherever it is, free its slot, and report the event."""
        (self._ready if events is None else events).append(event)
        del self.requests[request.id]
        if request.slot is None:
            self.waiting.remove(request)
            return
        if request in self.prefilling:
            self.prefilling.remove(request)
        self.decoding.pop(request.slot, None)
        self.cache.release(request.slot)
        heapq.heappush(self.free_slots, request.slot)
        request.slot = None

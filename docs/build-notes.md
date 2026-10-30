# Build notes

One section per step, from step 5 onward: what was built, why, the main tradeoff, and three
interview questions.

## Step 5: batching

**What was built.** The cache now has a fixed number of slots, each with its own length. A
forward pass takes any subset of slots, and every row gets its own positions, rotary angles and
mask. `generate` accepts prompts of different lengths. Decode passes one token per open request.
Prefill groups prompts of equal length, and nothing is padded.

**Why.** One forward pass over several requests uses the GPU far better than one pass each,
because the matrix work is shared. Per-slot state lets a 3-token prompt and a 200-token prompt
decode side by side, so short requests pay no padding cost for their own tokens.

**How the mask works.** Compare every stored position with each new token's own position, and
hide anything larger. One rule covers both the causal mask and the unused end of shorter rows.

**A trap.** A short row reads past its own end, and those positions get weight 0. But
`0 * NaN = NaN`, so garbage there would poison real output. The cache therefore starts as zeros
and a freed slot is cleared.

**Main tradeoff.** No padding for tokens, but every row in a pass still reads as many positions
as the longest row. A short request sharing a batch with a long one pays for attention over the
long request's width, so batches mixing very different lengths are less efficient than sorted
ones. In exchange there is no need to wait for similar requests, which matters for step 6.

**Exactness.** Greedy tokens matched running alone exactly in every test, and eight planted bugs
were all caught. This is not a mathematical guarantee: batch shape changes float rounding slightly,
so a near tie could in principle flip a token. It did not happen here.

### Interview questions

**1. Why not pad every prompt to the longest one?**
Padding runs real work on fake tokens, and it wastes cache space. With per-slot lengths, each row
in decode passes exactly one real token, and a mask hides whatever a shorter row should not see.

**2. A masked position has weight 0. Why does its content still matter?**
Attention multiplies weights by values, and `0 * NaN` is NaN, which then spreads through the row.
So memory past a slot's end must stay finite. That is why the cache starts as zeros instead of
uninitialised memory, and why a released slot is cleared.

**3. If rows pass one token each, why does a long neighbour slow a short request?**
All rows in a pass are read to the same width, so attention compares every query against as many
stored positions as the longest row. The mask removes those positions from the result but not from
the arithmetic. Decode time depends on the longest active slot.

## Step 6: continuous batching

**What was built.** `engine/scheduler.py` has a waiting queue and active slots. Each `step()`
admits waiting requests into free slots, runs prompt tokens up to a budget, then runs one decode
pass for every request past its prefill. A finished request frees its slot in that same step.

**Why.** Static batching holds every slot until the longest request ends. With continuous
batching a slot goes back to work the moment it is free, so short requests leave early and the
GPU is not left running half-empty batches.

**Mixing prefill and decode.** A prompt is cut into chunks of at most `prefill_tokens_per_step`
tokens, oldest request first, and a decode pass runs after the chunks of every step. A request
that just finished prefill also decodes in that step, so its first two tokens can come together.

**Main tradeoff.** The budget trades time to first token against smooth streaming. A large budget
gets a long prompt through quickly, but everyone decoding stalls for that long between tokens. A
small budget keeps decoding smooth but a long prompt waits many steps. Two extreme choices are
worse: prefill-first freezes running streams, and decode-first can starve new requests under steady
load.

**Fairness.** Admission and prefill are strictly first in, first out, and prefill gets its budget
on every step, so no request can be starved. The cost is that one long prompt at the front delays
the prompts behind it.

### Interview questions

**1. What does continuous batching change compared with static batching?**
Batch membership changes between model steps instead of once per batch. Finished requests leave
immediately and waiting ones take their slots, so a short request never waits for a long
neighbour and slots spend less time idle.

**2. Why cut prompts into chunks?**
A forward pass cannot be interrupted, so one huge prompt would block every decoding request for
its whole duration. Chunks put a bound on that stall: at most one budget's worth of prefill runs
between decode passes.

**3. How do you know no request is starved?**
Admission is first in, first out, and every step gives prefill a budget even when decoding is busy.
So a request's queue position only moves forward. The tests push a stream of short requests past a
long one and check that the long one still finishes, and that requests start in arrival order.

## Step 7: memory limits and failures

**What was built.** A cache memory budget, a limit on the waiting queue, cancellation that frees a
slot at once, clear errors (`InvalidRequest`, `PromptTooLong`, `QueueFull`), and failure isolation.
A request that raises, or whose logits turn NaN, ends with an error event while the rest carry on.
If a shared decode pass raises, each row is run alone to find the one at fault. Lengths only move
after a pass succeeds, so a retry simply overwrites what the failed pass wrote.

**How memory scales.** The cache holds keys and values, so:

    bytes = 2 * layers * slots * kv_heads * max_length * head_dim * bytes_per_value

It grows linearly with slots and linearly with max_length, so together they multiply. Qwen2.5-1.5B
in float16 costs 2 * 28 * 2 * 128 * 2 = 28,672 bytes per token position. So 8 slots of 4,096 tokens
take about 940 MB, and 32 slots of the same length about 3.7 GB, on top of the weights.

**Main tradeoff.** The cache is a fixed rectangle, so the budget is settled at startup: it decides
how many slots exist, and "budget full" means every slot is busy and new requests wait. That makes
memory completely predictable, and it can never grow. The cost is waste: a 20-token request holds a
full `max_length` row. Sharing memory by real token use would need a block allocator, as vLLM has.
It is not built here, which is an honest limit, not a hidden one.

**Proof of no leaks.** After 1,000 random requests with cancellations and refusals, every slot is
free, the cache tensors are the same objects at the same size and all zeros, and the count of live
tensors has not grown.

### Interview questions

**1. Why reject at admission rather than letting requests fail later?**
A request refused at the door costs nothing and tells the client straight away. One that fails
after a long prefill has wasted GPU time and held a slot. It also gives the client a clear signal:
a 400 means fix the request, and a 429 means try again later.

**2. What happens to a batch if one request causes an exception?**
The forward pass fails for everyone, so the scheduler retries each row alone and fails only the
one that raises. That is safe because lengths advance only after success, so nothing half-written
is ever treated as valid.

**3. Why does a preallocated cache make memory safety simple, and what does it give up?**
Nothing allocates while serving, so running out of memory mid-request cannot happen, and memory
cannot grow. The price is internal waste, because every slot reserves the maximum length. A block
allocator shares memory better but adds fragmentation and eviction to handle.

## Step 8: API and streaming

**What was built.** `engine/server.py` is a FastAPI server with `POST /v1/chat/completions`
(OpenAI format, streaming and not), `/health` and `/metrics`. `engine/text.py` turns tokens into
text without breaking characters. `engine/scheduler.py` now keeps running totals of where time goes.

**Why this shape.** A model step blocks for tens of milliseconds, which would freeze an asyncio
server. So the scheduler lives on its own thread and nothing else touches it. Handlers send it
commands through a queue, and tokens come back through one asyncio queue per request. The engine
only ever drops an event into a queue, so a slow client cannot slow the model. It can only grow its
own queue, which holds at most `max_tokens` events.

**Partial characters.** One character can span several tokens. Each new token is decoded with the
tokens just before it, and text is held back while it ends in a replacement character. Anything
still held at the end is flushed, so no bytes are lost.

**Disconnects.** Streaming requests cancel when the web server closes their generator. A request
waiting for a full reply has no such signal, so it asks whether the client is still there on a
timer. The first version only checked when no event had arrived for 0.25 s. A busy request always
has an event ready, so it never checked, and a real-socket test caught that.

**Main tradeoff.** One engine thread keeps the design simple and free of locks, but in Python the
web side and the engine share the interpreter lock, so each can briefly delay the other. Stop
strings are refused with a 400 rather than ignored, because silently ignoring one would be worse.

### Interview questions

**1. Why does the scheduler run on a thread rather than as an asyncio task?**
A forward pass blocks until the GPU finishes, and an asyncio task that blocks freezes every other
request on that event loop, including health checks. A dedicated thread keeps the web side free,
and PyTorch releases the interpreter lock during the heavy work, so both make progress.

**2. How can a slow client not slow everyone down?**
The model never writes to a socket. It puts events on an unbounded per-request queue, which cannot
block, and a separate coroutine drains it at the client's pace. The memory that can pile up is
bounded by `max_tokens`. A test freezes the event loop and checks tokens keep being generated.

**3. Why can't you decode each token by itself when streaming?**
Tokens are not characters. An emoji can span several tokens, and decoding them apart gives broken
output. Spacing can also depend on neighbouring tokens. So you decode a window of recent tokens,
emit only the new text, and wait while the text ends in an incomplete character.

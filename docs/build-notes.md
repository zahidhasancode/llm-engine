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

## Step 9: load test tool

**What was built.** `scripts/loadtest.py` drives any OpenAI-compatible server with streaming chat
completions. It supports Poisson arrivals at a set rate and a fixed number of concurrent users,
both with a fixed seed. It reports time to first token, time between tokens, end-to-end latency
(p50, p95, p99), total tokens per second and errors, and saves raw per-request data as JSON.
`data/prompts.jsonl` holds 200 real prompts from Dolly 15k, with 120 short, 60 medium and 20 long.

**Why.** The same tool must test my engine and vLLM, so nothing in it knows which server it talks
to. Its numbers are proven against a fake server told to use known delays.

**Open loop and closed loop.** Poisson is an open loop: arrivals ignore the server, like real users,
so queues can build without limit and saturation is visible. Fixed users are a closed loop: a slow
server slows the arrivals down, which hides overload but gives a steady, repeatable load.

**Coordinated omission.** In Poisson mode latency starts at the scheduled arrival time, not the
send time. If the client falls behind, the delay still counts. Measuring from the send time would
hide exactly the slowness a load test exists to find.

**Main tradeoff.** A Python client is simple, but it shares CPU with the server on a small machine
and adds its own delay at high rates. The same client is used for every system, so comparisons stay
fair, but absolute time between tokens at high load includes client overhead. Also, a server that
sends several tokens in one chunk would look like one long gap, so that metric is per chunk.

### Interview questions

**1. What is coordinated omission?**
A load generator that waits for each reply before sending the next sends less just when the server
is slow, so it under-samples the worst moments. The fix is to schedule arrivals in advance and
measure latency from the scheduled time, so waiting caused by the server is always counted.

**2. Why report p99 rather than the mean?**
Latency has a long tail, and the mean hides it. If one request in a hundred takes ten seconds,
the mean barely moves, yet a user making many requests will hit that often. The tail also shows
saturation first, because queueing delay hits the unluckiest requests hardest.

**3. How do you know a load test tool is telling the truth?**
Point it at a fake server with known behaviour and check that it reports those delays. Then test
the arithmetic on hand-made records with answers worked out on paper. Planted bugs showed that the
fake server alone was not enough: it could not tell latency measured from the send time from
latency measured from the schedule, so exact tests were added.

## Step 10: full benchmark

**What was built.** `benchmarks/serving_benchmark.py` starts each system as its own server process
and raises the rate until it saturates, with the same prompts, seeds and load tool for all three.
`scripts/hf_server.py` wraps Hugging Face `generate`, one request at a time, behind the same API.
`benchmarks/step_time.py` and `benchmarks/vllm_step_time.py` measure a decode step, and
`benchmarks/speed_gap.py` removes one thing at a time to split up the gap. `kaggle/serving_run.py`
runs it all on a private T4 with pinned versions. vLLM has its own virtual environment, because it
pins an older torch.

**Why.** A benchmark only means something if everything except the system is the same, and a claim
about why one system is slower needs a measurement. My first guess was wrong: I expected HTTP in a
shared Python process to be a large cost, and it was 3–5%.

**Results.** On one T4, in float16, serving 128 tokens per request: Hugging Face 32 tokens/s, my
engine 229, vLLM 1,680. The main causes are that long prompts set the attention cost for every
request in a pass, and that vLLM's Triton attention kernel makes each extra request about four times
cheaper than mine. With its FLEX backend, vLLM is no faster than my engine at batches of 16 or more.

**Measuring a step on a closed system.** vLLM does not expose its steps, so I time a run with 1 token
and a run with 1 + N tokens and divide the difference by N. Prefill cancels out, and the same method
works on both engines. Prefix caching must be off, or the second run skips its prefill.

**Main tradeoff.** vLLM's prefix cache is off, because the prompts repeat and it would measure
repetition instead of serving. That makes the test fair between systems but unlike a normal vLLM
deployment. It is stated in the README rather than hidden.

### Interview questions

**1. Your engine is 7 times slower than vLLM. Why?**
I measured it by removing one thing at a time. HTTP costs only 3–5%. Decode passes take 85% of the
time. My cache is a rectangle, so one long prompt makes every row attend over that length: at 64
slots, real prompts give 238 tokens/s and short prompts 720. vLLM's Triton kernel also grows far more
slowly with batch size: going from 16 to 128 requests, each one adds 0.3 ms per step, against 1.2 ms
in mine. When I switched vLLM to its FLEX backend, it was no faster than my engine.

**2. How do you tell a saturated system from a slow one?**
Under an open loop, a saturated system falls behind arrivals, so its queue and latency keep growing
for as long as the test runs. I call a point saturated if it serves under 85% of the offered rate,
has over 1% errors or has p95 above 120 seconds, on the majority of repeats. A slow system that is
keeping up has high but steady latency.

**3. Why did more slots not help your engine?**
At 64 slots, nearly every pass holds a long prompt, so a pass took 216 ms against 58 ms at 16 slots.
Four times the rows at about four times the time gives the same throughput, 238 against 234 tokens/s,
with much worse time between tokens. Throughput only tripled when I removed the long prompts. So the
next fix is how passes are grouped, not the slot count.

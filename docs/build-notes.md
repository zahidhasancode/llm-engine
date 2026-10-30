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

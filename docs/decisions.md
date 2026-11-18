# Design decisions and alternatives

These decisions explain the maintained implementation. They distinguish an implemented behavior from a possible future extension.

## 1. Use an untrained CPU decoder to isolate serving behavior

**Decision:** retain a small attention computation with seeded random weights and explicit positions.

**Reason:** a pretrained checkpoint introduces download size, accelerator requirements and language-quality variables before the cache/scheduler is understandable. A small model lets the regression suite compare cached and uncached outputs frequently.

**Cost:** compute/memory balance differs from a large multilayer Transformer. Byte outputs are not meaningful chat. The benchmark cannot predict GPU tokens/s, user satisfaction, model quality or commercial cost savings.

## 2. Reserve worst-case KV capacity before admission

**Decision:** reserve `prompt_length + max_new_tokens - 1`, rounded to blocks for the paged layout; fixed slabs reserve the model context limit.

**Reason:** prompt-only admission can start requests whose later cache growth exhausts the pool. Failing halfway through a response is harder to reason about than rejecting or queueing it before generation.

**Cost:** early EOS makes reservations conservative. Real systems can overcommit, preempt and recompute, or offload. Those alternatives require another experiment and are not claimed here.

## 3. Keep FIFO as the measured admission policy

**Decision:** sort by priority and stable arrival order; stop when the next request cannot fit. Offer `admission="fit"` as a separately tested alternative.

**Reason:** a fit-first scan can bypass a large waiting request to use otherwise idle capacity. That can improve occupancy but may starve large requests under sustained arrivals. A throughput gain without queue fairness evidence would be incomplete.

**Cost:** FIFO can exhibit head-of-line blocking. The reference grid keeps admission fixed so it isolates scheduling and cache layout. Fit-first is not mixed into the headline numbers.

## 4. Read physical blocks directly

**Decision:** implement online softmax across pages without a full-sequence KV gather.

**Reason:** gathering would duplicate the sequence history before every decode and obscure the difference between block ownership and attention access. Direct page reads make the intended algorithm and its numerical invariants explicit.

**Cost:** this Python loop is slower than a contiguous NumPy matmul in the measured CPU workload. It is a readable algorithmic reference, not a CUDA-kernel optimization. The negative throughput result is reported.

## 5. Make randomness independent of scheduling order

**Decision:** each request owns its seeded generator.

**Reason:** a shared RNG couples generated outputs to interleaving. Scheduler comparisons should change resource use and timing, not consume a different random stream for an unrelated request.

**Cost:** request RNG state must travel with the request. Resuming a preempted sequence would need to retain it along with generated tokens and cache state.

## 6. Measure events at their origin

**Decision:** stamp real submission, admission, token and finish events with a monotonic high-resolution clock.

**Reason:** output chunk index is not a timestamp. Reconstructing evenly spaced chunk times after completion can produce plausible-looking TTFT and ITL without observing either.

**Cost:** instrumentation is part of the in-process timing scope. It adds Python overhead. The benchmark does not claim that these timestamps equal a network client's receive times.

## 7. Compare a full 2×2 grid

**Decision:** run static/continuous scheduling with fixed/paged layout under the same pool budget, model, workloads and output budgets.

**Reason:** changing scheduling and memory layout together cannot identify which change caused the result. The grid exposes that continuous refill helps the fixed layout while Python paging overhead can dominate throughput despite higher active-request counts.

**Cost:** four configurations and multiple workload seeds require more evidence than one before/after screenshot. The repository keeps raw runs, representative traces and canonical outputs to make that evidence reviewable.

## 8. Isolate legacy interfaces from the maintained runtime

**Decision:** retain the earlier functional interfaces in `archive/`, expose the maintained runtime through `kvserve/`, and document the migration.

**Reason:** the legacy functions use different state conventions and reconstructed timings. The maintained package makes cache ownership, request progression and observed event timing authoritative while retaining a technical reference for earlier interfaces.

**Cost:** this is a deliberate API change. Callers of the legacy interfaces must use the archived snapshot or migrate to `Engine`, `EngineConfig`, cache classes and the CLI. Published measurements use the maintained runtime.

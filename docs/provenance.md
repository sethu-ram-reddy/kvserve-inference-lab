# Implementation history

KVServe consolidates earlier functional cache and scheduling interfaces into a maintained runtime with explicit ownership, request lifecycles and reproducible measurement. This document records the technical changes and their rationale.

## Baseline implementation

Commit `392f3b995edb27002a846ad43ed230f718fb36c9` records the earlier implementation. Its functional interfaces and integration driver are retained in `archive/` for comparison. Neither the HTTP server nor the published benchmark imports those files.

## Changes to the maintained runtime

| Baseline area | Current implementation | Engineering rationale |
| --- | --- | --- |
| Sampling helpers | `kvserve/sampling.py` | Validate parameters and retain request-local RNG semantics |
| Character vocabulary | `ByteTokenizer` | Stable, corpus-independent input IDs; explicit UTF-8 handling |
| Prefill and decode helpers | `AttentionDecoder` | One authoritative cache/input lifecycle, explicit positions and uncached numerical reference |
| Paged cache and gather | `PagedCache` | Atomic append, ownership checks, and direct online softmax over physical pages |
| Independent sequence/batch helpers | `Engine` | Consistent request state across scheduling policies |
| Prompt-only capacity checks | Lifetime reservations | Prevent admission without enough KV capacity for the declared output budget |
| Dictionary serving interface | FastAPI shared worker and SSE | Provide HTTP delivery with bounded buffering and cancellation cleanup |
| Chunk timing reconstruction | Observed event timestamps | Measure TTFT and ITL at their source |
| Single integration run | 180-run paired grid and separate ablations | Measure controls, repeated outcomes and counterexamples |
| Function checklist | Architecture, theory, decisions and evidence | Make implementation choices and numerical claims reviewable |

The current implementation separates cache ownership, scheduler policy, model computation and client delivery. Its evaluation reports both improvements and regressions: continuous refill improves the measured fixed-cache workload, while Python page traversal adds overhead despite reducing assigned capacity.

## Compatibility

The legacy integration driver imports a module named `model`. To execute the archived pair in isolation, copy `step_solutions.py` and `original_scaffold.py` into a separate temporary directory as `model.py` and `scaffold.py`, then run the latter there. Current callers should use the public classes under `kvserve/` or the installed `kvserve` command.

Source attribution and research references are recorded in [NOTICE.md](../NOTICE.md) and [theory.md](theory.md). No pretrained model or external text dataset is redistributed.

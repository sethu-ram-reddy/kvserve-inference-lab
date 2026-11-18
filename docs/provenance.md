# Provenance and the engineering extension

The repository began with the Deep-ML *Build a Mini LLM Inference Server* 51-step exercise. The uploaded exercise commit is `392f3b995edb27002a846ad43ed230f718fb36c9`. The original `model.py` and `scaffold.py` are preserved as `archive/step_solutions.py` and `archive/original_scaffold.py`.

The maintained package is a deliberate integration and evaluation extension. The archive preserves the original solution state; it is not imported by the benchmark or server, and its reconstructed latency figures are not reused.

| Original area | Maintained implementation | Why it changed |
| --- | --- | --- |
| Sampling helpers | `kvserve/sampling.py` | Validate parameters and retain request-local RNG semantics |
| Character vocabulary | `ByteTokenizer` | Stable, corpus-independent input IDs; explicit UTF-8 handling |
| Prefill and decode helpers | `AttentionDecoder` | One authoritative cache/input lifecycle, explicit positions and uncached numerical reference |
| Paged cache and gather | `PagedCache` | Atomic append, ownership checks, and direct online softmax over physical pages |
| Independent sequence/batch helpers | `Engine` | Consistent request state across scheduling policies |
| Prompt-only capacity checks | Lifetime reservations | Prevent a request from being admitted without enough room to finish its declared budget |
| Dictionary serving interface | FastAPI shared worker and SSE | Expose an actual HTTP delivery path with cleanup |
| Chunk timing reconstruction | Observed event timestamps | Establish valid TTFT/ITL accounting |
| One small scaffold run | 180-run paired grid and separate ablations | Measure controls, repeated outcomes and counterexamples |
| Checklist README | Theory, architecture, decisions and evidence | Explain the engineering question and make claims reviewable |

The exercise's primitives are established techniques. Continuous scheduling and paging are not new inventions of this repository. Its distinct value is the combination of transparent implementation, failure handling, equivalence checks and an experiment that reports both the benefits and the costs.

## Run the preserved snapshot

The archived scaffold imports a module named `model`. To run the exact archived pair, copy `step_solutions.py` and `original_scaffold.py` into a separate temporary directory as `model.py` and `scaffold.py`, then run the latter there. The archive is primarily provided for inspection; the maintained `python scaffold.py` at the repository root is the supported command.

No pretrained model or external dataset is redistributed. Source-paper credit belongs to the Orca and PagedAttention authors; the original learning exercise is credited to Deep-ML.

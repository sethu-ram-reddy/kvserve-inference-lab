# Model and system card

## Intended use

KVServe is an inspectable reference for inference mechanics: incremental state, KV ownership, request scheduling, sampling and streamed delivery. It supports engineering discussion and small controlled experiments. It is not a trained assistant.

| Property | Reference configuration |
| --- | --- |
| Architecture | One causal attention block, one head, sinusoidal positions |
| Width | 64 |
| Vocabulary | 256 UTF-8 byte values + BOS + EOS |
| Weight count | 49,408 |
| Weights / KV rows | float32 |
| Training data | None |
| Training process | None; seeded random initialization |
| Benchmark context limit | 256 |
| Default CLI/API context limit | 512 |
| Accelerator | CPU / NumPy |

The block omits MLPs, residual pathways, normalization and multiple layers. Random outputs can contain arbitrary bytes, special IDs and Unicode replacement characters after decoding. `output_ids` is the canonical representation for correctness checks.

## Implemented system behaviors

- Stable temperature/top-k/top-p sampling and request-local RNGs.
- Separate prefill and incremental decode, with vectorized projections across active requests.
- Fixed slab and paged cache layouts; direct blockwise online-softmax attention.
- Static wave and continuous iteration scheduling with stable priority order.
- Conservative lifetime KV reservations; optional fit-first admission.
- A shared local HTTP worker, completion JSON, SSE events, bounded client channels and disconnect cancellation.
- Real in-process event timing, reproducible workload generation and evidence-derived summaries.

## Explicit boundaries

There is no trained-model quality evaluation, factuality result, accuracy improvement, quantization, CUDA kernel, GPU benchmark, distributed inference, tensor parallelism, prefix sharing, speculative decoding, preemptive scheduler, swap/offload, or chunked prefill. The optional fit-first policy has no sustained-load fairness benchmark.

The main grid uses fixed output budgets with EOS disabled and all-at-once bursts. It is not a simulation of a full production traffic distribution. There is no throughput comparison against another serving framework. Capacity savings are sequence-assigned KV slots, not measured process memory or VRAM savings.

The HTTP implementation is a local demonstration. It lacks user authentication, tenant isolation, durable request state and deployment orchestration. It binds to loopback by default. The implementation is suitable for understanding mechanics before replacing the backend and load-testing the service for a real deployment.

## Provenance

The original learning path is the Deep-ML *Build a Mini LLM Inference Server* exercise. `archive/` preserves the uploaded implementation. Maintained code and evidence are described in [provenance.md](provenance.md), with source hashes in the reference manifest. Project context is ChIEAC applied AI engineering, as described by the repository owner.

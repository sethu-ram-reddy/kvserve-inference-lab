"""Paired 2x2 scheduling/cache ablation with real CPU timestamps and output checks."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np

from kvserve.cache import ContiguousCache, PagedCache
from kvserve.engine import Engine, EngineConfig
from kvserve.metrics import summarize
from kvserve.model import AttentionDecoder

MODES = [(schedule, layout) for schedule in ("static", "continuous") for layout in ("fixed", "paged")]
SEEDS = [7, 19, 42]
WORKLOADS = ["uniform", "mixed", "long_tail"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def workload(name, seed, count):
    rng = np.random.default_rng(seed)
    requests = []
    for i in range(count):
        if name == "uniform":
            prompt_len, budget = 32, 24
        elif name == "mixed":
            prompt_len, budget = int(rng.choice([12, 24, 48, 80])), int(rng.choice([4, 8, 16, 64]))
        elif name == "long_tail":
            prompt_len, budget = int(rng.choice([16, 32, 64])), 96 if i % 8 == 0 else 8
        else:
            raise ValueError(name)
        tokens = [256, *rng.integers(0, 256, size=prompt_len - 1).tolist()]
        requests.append({"prompt_ids": tokens, "max_new_tokens": budget, "seed": i})
    return requests


def run_once(model, requests, scheduling, layout, block_size, total_slots, max_running):
    config = EngineConfig(layout, scheduling, "fifo", total_slots, block_size, max_running)
    engine = Engine(model, config)
    # Start before submission: Python submission overhead is included in elapsed time.
    start = perf_counter()
    for request in requests:
        engine.submit(request["prompt_ids"], request["max_new_tokens"], seed=request["seed"])
    outputs = engine.run()
    elapsed = perf_counter() - start
    result = summarize(engine.events, elapsed, engine.snapshots)
    result["kv_released"] = engine.cache.assigned_slots == 0 and engine.reserved == 0
    if not result["kv_released"]:
        raise RuntimeError("Benchmark leaked KV ownership")
    return result, outputs, engine


def memory_study(model, requests, block_sizes):
    result = []
    pool_slots = model.max_context * len(requests)
    fixed = ContiguousCache(pool_slots, model.width, model.max_context)
    for i, request in enumerate(requests):
        fixed.add(i)
        length = len(request["prompt_ids"]) + request["max_new_tokens"] - 1
        rows = np.zeros((length, model.width), np.float32)
        fixed.append(i, rows, rows)
    for size in block_sizes:
        paged = PagedCache(pool_slots, model.width, size)
        for i, request in enumerate(requests):
            paged.add(i)
            length = len(request["prompt_ids"]) + request["max_new_tokens"] - 1
            rows = np.zeros((length, model.width), np.float32)
            paged.append(i, rows, rows)
        paged.check()
        result.append({"block_size": size, "requests": len(requests),
                       "useful_kv_slots": fixed.used_slots,
                       "fixed_assigned_slots": fixed.assigned_slots,
                       "paged_assigned_slots": paged.assigned_slots,
                       "paged_tail_slots": paged.assigned_slots - paged.used_slots,
                       "assigned_capacity_reduction_pct": 100 * (1 - paged.assigned_slots / fixed.assigned_slots),
                       "bytes_per_slot": 2 * model.width * np.dtype(np.float32).itemsize,
                       "paged_preallocated_pool_bytes": paged.keys.nbytes + paged.values.nbytes})
    return result


def cache_study(model, repeats):
    rng = np.random.default_rng(91)
    prompt, budget = [256, *rng.integers(0, 256, 95).tolist()], 32
    rows = []
    for repeat in range(repeats + 1):
        start = perf_counter()
        ids = list(prompt)
        uncached = []
        for _ in range(budget):
            token = int(model.uncached_logits(ids)[-1].argmax())
            uncached.append(token)
            ids.append(token)
        uncached_time = perf_counter() - start
        cache = ContiguousCache(model.max_context, model.width, model.max_context)
        cache.add("s")
        start = perf_counter()
        logits = model.prefill("s", prompt, cache)
        cached = []
        for i in range(budget):
            if i:
                logits = model.decode_batch(["s"], [cached[-1]], cache)[0]
            cached.append(int(logits.argmax()))
        cached_time = perf_counter() - start
        if cached != uncached:
            raise RuntimeError("KV-cached and full-recompute outputs differ")
        if repeat:
            rows.append({"repeat": repeat - 1, "uncached_s": uncached_time, "cached_s": cached_time})
    return {"prompt_tokens": len(prompt), "output_tokens": budget,
            "output_equivalent": True, "runs": rows,
            "uncached_median_ms": 1000 * median(r["uncached_s"] for r in rows),
            "cached_median_ms": 1000 * median(r["cached_s"] for r in rows),
            "median_speedup": median(r["uncached_s"] for r in rows) / median(r["cached_s"] for r in rows),
            "uncached_projected_token_rows": sum(len(prompt) + i for i in range(budget)),
            "cached_projected_token_rows": len(prompt) + budget - 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/reference"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--requests", type=int, default=32)
    args = parser.parse_args()
    if args.repeats < 1 or args.requests < 8:
        parser.error("Positive repeats and at least eight requests required")
    model = AttentionDecoder(width=64, max_context=256, seed=7)
    model.ignore_eos = True
    output, records = args.output, []
    output.mkdir(parents=True, exist_ok=True)
    config = {"width": 64, "max_context": 256, "model_seed": 7, "parameters": model.parameter_count,
              "requests_per_workload": args.requests, "repeats": args.repeats, "workload_seeds": SEEDS,
              "block_size": 16, "total_slots": 1024, "max_running": 16,
              "sampling": "greedy", "eos": "disabled for fixed output budgets", "arrivals": "all-at-once burst",
              "timing_scope": "in-process CPU; submit through last completion; excludes model/pool construction"}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "fastapi", "uvicorn", "matplotlib")}
    cpu_line = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.processor())
    source_files = sorted(Path("kvserve").glob("*.py")) + [Path("benchmarks/run.py")]
    manifest = {"config": config, "environment": {"python": platform.python_version(),
                "platform": platform.platform(), "cpu": cpu_line, "packages": versions,
                "thread_environment": {name: os.environ.get(name) for name in
                                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    write_json(output / "manifest.json", manifest)
    reference_outputs = {}
    for name in WORKLOADS:
        for seed in SEEDS:
            requests = workload(name, seed, args.requests)
            write_json(output / "workloads" / f"{name}-seed{seed}.json", requests)
            for scheduling, layout in MODES:
                mode = f"{scheduling}-{layout}"
                run_once(model, requests[:4], scheduling, layout, 16, 1024, 16)  # warm-up, discarded
            # Rotate execution order across workload seeds/repeats to limit order bias.
            for repeat in range(args.repeats):
                offset = (repeat + SEEDS.index(seed)) % len(MODES)
                modes = MODES[offset:] + MODES[:offset]
                for scheduling, layout in modes:
                    mode = f"{scheduling}-{layout}"
                    metrics, outputs, engine = run_once(model, requests, scheduling, layout, 16, 1024, 16)
                    key = (name, seed)
                    if key not in reference_outputs:
                        reference_outputs[key] = outputs
                        write_json(output / "outputs" / f"{name}-seed{seed}.json", outputs)
                    if outputs != reference_outputs[key]:
                        raise RuntimeError(f"Output mismatch in {key}, {mode}, repeat {repeat}")
                    record = {"workload": name, "workload_seed": seed, "mode": mode,
                              "repeat": repeat, "output_sha256": checksum(outputs), "output_equivalent": True, **metrics}
                    records.append(record)
                    if name == "mixed" and seed == SEEDS[0] and repeat == 0:
                        write_json(output / "traces" / f"{mode}.json", {"events": engine.events, "snapshots": engine.snapshots})
            print(f"Completed {name}, seed={seed}", flush=True)
    with (output / "runs.jsonl").open("w") as file:
        for row in records:
            file.write(json.dumps(row, allow_nan=False) + "\n")
    summary = []
    for name in WORKLOADS:
        for scheduling, layout in MODES:
            mode = f"{scheduling}-{layout}"
            group = [r for r in records if r["workload"] == name and r["mode"] == mode]
            numeric = [k for k, v in group[0].items() if type(v) in (int, float) and k not in ("workload_seed", "repeat")]
            summary.append({"workload": name, "mode": mode, "runs": len(group),
                            **{k: median(r[k] for r in group) for k in numeric},
                            "tokens_per_s_p10": float(np.percentile([r["tokens_per_s"] for r in group], 10)),
                            "tokens_per_s_p90": float(np.percentile([r["tokens_per_s"] for r in group], 90))})
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(output / "memory.json", memory_study(model, workload("mixed", 7, 8), [4, 8, 16, 32, 64]))
    write_json(output / "cache.json", cache_study(model, max(args.repeats, 7)))
    print(f"Saved {len(records)} measured runs to {output}", flush=True)


if __name__ == "__main__":
    main()

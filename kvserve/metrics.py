"""Metrics derived only from real scheduler events stamped by perf_counter."""
import numpy as np


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def summarize(events, elapsed_s, snapshots):
    if elapsed_s <= 0:
        raise ValueError("Elapsed time must be positive")
    submits, admitted, tokens, finishes = {}, {}, {}, {}
    for event in events:
        rid, timestamp = event["request_id"], event["time_s"]
        kind = event["event"]
        if kind == "submit":
            submits[rid] = timestamp
        elif kind == "admit":
            admitted[rid] = timestamp
        elif kind == "token":
            tokens.setdefault(rid, []).append(timestamp)
        elif kind == "finish":
            finishes[rid] = timestamp
    if set(submits) != set(finishes):
        raise ValueError("Incomplete requests must not enter a complete-run benchmark")
    ttft = [1000 * (times[0] - submits[rid]) for rid, times in tokens.items()]
    gaps = [1000 * (b - a) for times in tokens.values() for a, b in zip(times, times[1:])]
    e2e = [1000 * (end - submits[rid]) for rid, end in finishes.items()]
    waits = [1000 * (time - submits[rid]) for rid, time in admitted.items()]
    return {
        "completed_requests": len(finishes), "output_tokens": sum(map(len, tokens.values())),
        "elapsed_s": elapsed_s, "tokens_per_s": sum(map(len, tokens.values())) / elapsed_s,
        "requests_per_s": len(finishes) / elapsed_s,
        "ttft_p50_ms": percentile(ttft, 50), "ttft_p95_ms": percentile(ttft, 95),
        "itl_p50_ms": percentile(gaps, 50), "itl_p95_ms": percentile(gaps, 95),
        "e2e_p95_ms": percentile(e2e, 95), "queue_p95_ms": percentile(waits, 95),
        "decode_rounds": len(snapshots),
        "mean_active_requests": float(np.mean([s["active"] for s in snapshots])) if snapshots else 0,
        "peak_assigned_kv_slots": max((s["assigned_slots"] for s in snapshots), default=0),
        "peak_reserved_kv_slots": max((s["reserved_slots"] for s in snapshots), default=0),
    }

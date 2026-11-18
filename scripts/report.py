"""Generate the publication figures and evidence-derived claim ledger."""
import argparse
import json
from pathlib import Path
from statistics import median
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/reference"))
    parser.add_argument("--assets", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    root = args.results
    summary = json.loads((root / "summary.json").read_text())
    records = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
    memory = json.loads((root / "memory.json").read_text())
    cache = json.loads((root / "cache.json").read_text())
    mixed = {r["mode"]: r for r in summary if r["workload"] == "mixed"}
    fixed_static, fixed_cont = mixed["static-fixed"], mixed["continuous-fixed"]
    mem = next(row for row in memory if row["block_size"] == 16)
    claims = {
        "experiment_runs": len(records), "workloads": 3, "workload_seeds": 3, "repeats_per_seed": 5,
        "requests_completed_across_runs": sum(r["completed_requests"] for r in records),
        "all_outputs_equivalent": all(r["output_equivalent"] for r in records),
        "all_kv_released": all(r["kv_released"] for r in records),
        "mixed_fixed_throughput_gain_pct": 100 * (fixed_cont["tokens_per_s"] / fixed_static["tokens_per_s"] - 1),
        "mixed_fixed_ttft_p95_reduction_pct": 100 * (1 - fixed_cont["ttft_p95_ms"] / fixed_static["ttft_p95_ms"]),
        "mixed_fixed_static_tokens_per_s": fixed_static["tokens_per_s"],
        "mixed_fixed_continuous_tokens_per_s": fixed_cont["tokens_per_s"],
        "assigned_kv_capacity_reduction_pct": mem["assigned_capacity_reduction_pct"],
        "fixed_assigned_slots": mem["fixed_assigned_slots"], "paged_assigned_slots": mem["paged_assigned_slots"],
        "cache_microbenchmark_speedup": cache["median_speedup"],
        "cache_microbenchmark_uncached_ms": cache["uncached_median_ms"],
        "cache_microbenchmark_cached_ms": cache["cached_median_ms"],
        "scope": "CPU NumPy untrained attention decoder; in-process timings; assigned capacity is not process RSS",
    }
    (root / "claims.json").write_text(json.dumps(claims, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7), layout="constrained")
    modes = ["static-fixed", "continuous-fixed", "static-paged", "continuous-paged"]
    labels = ["Static / fixed", "Continuous / fixed", "Static / paged", "Continuous / paged"]
    for axis, metric, label, title in (
        (axes[0, 0], "tokens_per_s", "Output tokens / second", "(a) Throughput · mixed workload"),
        (axes[0, 1], "ttft_p95_ms", "Request p95 TTFT (ms)", "(b) First-token latency · mixed workload"),
    ):
        for i, mode in enumerate(modes):
            values = [r[metric] for r in records if r["workload"] == "mixed" and r["mode"] == mode]
            value, lo, hi = median(values), np.percentile(values, 10), np.percentile(values, 90)
            axis.errorbar(value, i, xerr=np.array([[value - lo], [hi - value]]), fmt="o",
                          color="black", markerfacecolor="white" if "static" in mode else "black",
                          markersize=5, capsize=3, elinewidth=.8)
        axis.set_yticks(range(4), labels)
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        axis.set_xlabel(label)
        axis.set_title(title, loc="left", fontsize=10, pad=12)
    axis = axes[1, 0]
    sizes = [r["block_size"] for r in memory]
    assigned = [r["paged_assigned_slots"] for r in memory]
    axis.plot(sizes, assigned, "o-", color="black", linewidth=.9, markersize=4, label="Paged: assigned capacity")
    axis.axhline(mem["fixed_assigned_slots"], color=".35", linestyle="--", linewidth=.9, label="Fixed: 2,048 slots")
    axis.axhline(mem["useful_kv_slots"], color=".55", linestyle=":", linewidth=.9, label="Useful KV: 584 slots")
    axis.set_xticks(sizes)
    axis.set_ylim(0, 2400)
    axis.set_xlabel("Tokens per block")
    axis.set_ylabel("KV token slots")
    axis.set_title("(c) Assigned capacity · same eight requests", loc="left", fontsize=10, pad=12)
    axis.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    axis = axes[1, 1]
    durations = [cache["uncached_median_ms"], cache["cached_median_ms"]]
    axis.barh([0, 1], durations, color=[".8", ".3"], height=.45, edgecolor="black", linewidth=.6)
    axis.set_yticks([0, 1], ["Full-prefix recompute", "KV-cached decode"])
    axis.invert_yaxis()
    axis.set_xlim(0, max(durations) * 1.23)
    for i, value in enumerate(durations):
        axis.text(value + max(durations) * .03, i, f"{value:.2f}", va="center", fontsize=9)
    axis.set_xlabel("Median generation time (ms)")
    axis.set_title("(d) Cache ablation · 96 input + 32 output tokens", loc="left", fontsize=10, pad=12)
    for axis in axes.flat:
        axis.tick_params(direction="out", length=3)
    args.assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "benchmark.png", dpi=180, facecolor="white")
    fig.savefig(args.assets / "benchmark.svg", facecolor="white")
    plt.close(fig)

    # Representative real scheduler trace, not an illustrative schedule.
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 4.8), sharex=True, layout="constrained")
    for axis, mode in zip(axes, ("static-fixed", "continuous-fixed")):
        trace = json.loads((root / "traces" / f"{mode}.json").read_text())
        admitted, finished = {}, {}
        for event in trace["events"]:
            if event["event"] == "admit":
                admitted[event["request_id"]] = event["tick"]
            if event["event"] == "finish":
                finished[event["request_id"]] = event["tick"] + 1
        for rid, start in admitted.items():
            index = int(rid.split("-")[1])
            axis.broken_barh([(start, finished[rid] - start)], (index - .35, .7), facecolors=".3")
        axis.set_ylabel("Request index")
        axis.set_title(mode.replace("-", " / "), loc="left", fontsize=10)
        axis.set_ylim(-1, len(admitted))
        axis.set_yticks([0, 8, 16, 24, 31])
    axes[-1].set_xlabel("Decode round (logical work; not wall time)")
    fig.savefig(args.assets / "schedule.svg", facecolor="white")
    plt.close(fig)
    print(json.dumps(claims, indent=2))


if __name__ == "__main__":
    main()

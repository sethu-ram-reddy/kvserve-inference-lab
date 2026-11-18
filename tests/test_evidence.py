"""Audit committed claims against raw run records and observed request traces."""
import hashlib
import json
from pathlib import Path
from statistics import median
import unittest
from kvserve.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/reference"


def load(path):
    return json.loads(path.read_text())


class EvidenceTests(unittest.TestCase):
    def test_complete_paired_grid_and_canonical_output_digests(self):
        records = [json.loads(line) for line in (RESULTS / "runs.jsonl").read_text().splitlines()]
        keys = {(r["workload"], r["workload_seed"], r["mode"], r["repeat"]) for r in records}
        self.assertEqual(len(keys), 180)
        self.assertEqual(len(records), 180)
        for record in records:
            output = load(RESULTS / "outputs" / f"{record['workload']}-seed{record['workload_seed']}.json")
            digest = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(digest, record["output_sha256"])
            self.assertTrue(record["output_equivalent"])
            self.assertTrue(record["kv_released"])
            self.assertEqual(record["completed_requests"], 32)
        self.assertEqual(sum(r["completed_requests"] for r in records), 5760)
        for summary in load(RESULTS / "summary.json"):
            group = [r for r in records if r["mode"] == summary["mode"] and r["workload"] == summary["workload"]]
            self.assertEqual(len(group), 15)
            self.assertAlmostEqual(median(r["tokens_per_s"] for r in group), summary["tokens_per_s"])

    def test_trace_timing_metrics_are_observed_and_recompute_exactly(self):
        records = [json.loads(line) for line in (RESULTS / "runs.jsonl").read_text().splitlines()]
        for path in (RESULTS / "traces").glob("*.json"):
            trace = load(path)
            record = next(r for r in records if r["mode"] == path.stem and r["workload"] == "mixed"
                          and r["workload_seed"] == 7 and r["repeat"] == 0)
            metrics = summarize(trace["events"], record["elapsed_s"], trace["snapshots"])
            for key, value in metrics.items():
                self.assertEqual(value, record[key])
            times = [e["time_s"] for e in trace["events"]]
            self.assertEqual(times, sorted(times))
            for snapshot in trace["snapshots"]:
                self.assertLessEqual(snapshot["used_slots"], snapshot["assigned_slots"])
                self.assertLessEqual(snapshot["assigned_slots"], snapshot["reserved_slots"])
                self.assertLessEqual(snapshot["reserved_slots"], 1024)

    def test_claim_ledger_and_measured_source_identity(self):
        claims = load(RESULTS / "claims.json")
        memory = next(r for r in load(RESULTS / "memory.json") if r["block_size"] == 16)
        self.assertAlmostEqual(claims["assigned_kv_capacity_reduction_pct"],
                               100 * (1 - memory["paged_assigned_slots"] / memory["fixed_assigned_slots"]))
        self.assertEqual(claims["cache_microbenchmark_speedup"], load(RESULTS / "cache.json")["median_speedup"])
        for path, digest in load(RESULTS / "manifest.json")["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)

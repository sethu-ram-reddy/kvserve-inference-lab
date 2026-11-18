import unittest
import numpy as np
from kvserve.cache import CacheFull, ContiguousCache, PagedCache
from kvserve.engine import Engine, EngineConfig
from kvserve.model import AttentionDecoder, ByteTokenizer
from kvserve.metrics import summarize
from kvserve.sampling import Sampling, softmax


class CacheTests(unittest.TestCase):
    def test_paged_attention_matches_contiguous_across_block_boundaries(self):
        rng = np.random.default_rng(12)
        for length in (1, 3, 4, 5, 16, 31):
            page = PagedCache(64, 8, 4)
            dense = ContiguousCache(64, 8, 32)
            for cache in (page, dense):
                cache.add("a")
            keys, values, query = rng.normal(size=(length, 8)), rng.normal(size=(length, 8)), rng.normal(size=8)
            for start in range(0, length, 3):
                for cache in (page, dense):
                    cache.append("a", keys[start:start + 3], values[start:start + 3])
            np.testing.assert_allclose(page.attend("a", query), dense.attend("a", query), atol=1e-6, rtol=1e-5)
            page.check()
            page.free("a")
            page.free("a")
            page.check()
            self.assertEqual(page.assigned_slots, 0)

    def test_failed_append_is_atomic_and_blocks_reuse_without_stale_tail(self):
        cache = PagedCache(8, 4, 4)
        cache.add("a")
        cache.append("a", np.ones((3, 4)), np.ones((3, 4)))
        before = (list(cache.free_blocks), list(cache.tables["a"]), dict(cache.lengths))
        with self.assertRaises(CacheFull):
            cache.append("a", np.ones((6, 4)), np.ones((6, 4)))
        self.assertEqual(before, (cache.free_blocks, cache.tables["a"], cache.lengths))
        cache.free("a")
        cache.add("b")
        cache.append("b", np.zeros((1, 4)), np.full((1, 4), 7))
        np.testing.assert_allclose(cache.attend("b", np.ones(4)), np.full(4, 7))
        cache.check()


class ModelTests(unittest.TestCase):
    def test_prefill_and_incremental_decode_match_full_causal_recomputation(self):
        model = AttentionDecoder(width=16, max_context=64)
        tokens = [256, 7, 13, 200, 0, 22]
        for cache in (PagedCache(64, 16, 4), ContiguousCache(64, 16, 64)):
            cache.add("s")
            logits = model.prefill("s", tokens[:3], cache)
            np.testing.assert_allclose(logits, model.uncached_logits(tokens[:3])[-1], atol=2e-6, rtol=1e-4)
            for end in range(4, len(tokens) + 1):
                logits = model.decode_batch(["s"], [tokens[end - 1]], cache)[0]
                np.testing.assert_allclose(logits, model.uncached_logits(tokens[:end])[-1], atol=2e-6, rtol=1e-4)

    def test_byte_tokenizer_roundtrips_unicode(self):
        tokenizer = ByteTokenizer()
        for text in ("hello", "తెలుగు", "café", ""):
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_sampling_validation_and_stability(self):
        np.testing.assert_allclose(softmax(np.array([10000., 10001.])), softmax(np.array([0., 1.])))
        rng = np.random.default_rng(1)
        self.assertEqual(Sampling(1, 1).select(np.array([1., 2., 3.]), rng), 2)
        self.assertEqual(Sampling(1, None, 0.01).select(np.array([1., 2., 3.]), rng), 2)
        for sampling in (Sampling(-1), Sampling(1, 0), Sampling(1, None, 0), Sampling(float("nan"))):
            with self.assertRaises(ValueError):
                sampling.select(np.ones(3), rng)
        with self.assertRaises(ValueError):
            softmax(np.array([-np.inf, -np.inf]))


class EngineTests(unittest.TestCase):
    def model(self):
        model = AttentionDecoder(width=16, max_context=64)
        model.ignore_eos = True
        return model

    def test_all_policies_preserve_greedy_and_seeded_stochastic_outputs(self):
        for sampling in (Sampling(), Sampling(0.8, 20, 0.9)):
            reference = None
            for layout in ("fixed", "paged"):
                for scheduling in ("static", "continuous"):
                    model = self.model()
                    engine = Engine(model, EngineConfig(layout, scheduling, total_slots=128, block_size=8, max_running=4))
                    for i in range(8):
                        engine.submit([256, i + 10, i + 40], [1, 13, 2, 9][i % 4], sampling, seed=i)
                    result = engine.run()
                    if reference is None:
                        reference = result
                    self.assertEqual(result, reference)
                    self.assertEqual(engine.cache.assigned_slots, 0)
                    self.assertEqual(engine.reserved, 0)

    def test_continuous_refills_slots_before_static_wave_ends(self):
        rounds = {}
        for policy in ("static", "continuous"):
            engine = Engine(self.model(), EngineConfig("fixed", policy, total_slots=128, block_size=8, max_running=2))
            for budget in (1, 9, 1, 9):
                engine.submit([256, 7], budget)
            engine.run()
            rounds[policy] = engine.tick
        self.assertLess(rounds["continuous"], rounds["static"])

    def test_zero_budget_eos_context_rejection_and_cancel(self):
        engine = Engine(self.model(), EngineConfig(total_slots=64, block_size=8))
        zero = engine.submit([256], 0)
        self.assertEqual(engine.completed[zero]["output_ids"], [])
        one = engine.submit([256] * 64, 1)
        engine.run()
        self.assertEqual(len(engine.completed[one]["output_ids"]), 1)
        with self.assertRaises(ValueError):
            engine.submit([256] * 64, 2)
        rid = engine.submit([256, 9], 10)
        engine.step()
        self.assertTrue(engine.cancel(rid))
        self.assertFalse(engine.cancel(rid))
        self.assertEqual(engine.cache.assigned_slots, 0)
        self.assertEqual(engine.reserved, 0)
        class AlwaysEOS(Sampling):
            def select(self, logits, rng):
                return ByteTokenizer.eos
        engine.model.ignore_eos = False
        rid = engine.submit([256], 10, AlwaysEOS())
        engine.run()
        self.assertEqual(engine.completed[rid]["finish_reason"], "eos")
        self.assertEqual(engine.completed[rid]["output_ids"], [257])

    def test_memory_pressure_backpressure_and_timeout_cleanup(self):
        engine = Engine(self.model(), EngineConfig(total_slots=16, block_size=8, max_pending=2))
        engine.submit([256] * 8, 9)
        engine.submit([256] * 8, 9)
        with self.assertRaises(OverflowError):
            engine.submit([256], 1)
        with self.assertRaises(TimeoutError):
            engine.run(max_ticks=1)
        self.assertEqual(engine.cache.assigned_slots, 0)
        self.assertEqual(engine.reserved, 0)
        engine.cache.check()

    def test_metrics_use_observed_events_and_reject_incomplete_runs(self):
        events = [
            {"event": "submit", "request_id": "a", "time_s": 0.},
            {"event": "admit", "request_id": "a", "time_s": .01},
            {"event": "token", "request_id": "a", "time_s": .02},
            {"event": "token", "request_id": "a", "time_s": .05},
            {"event": "finish", "request_id": "a", "time_s": .051},
        ]
        result = summarize(events, .1, [])
        self.assertEqual(result["ttft_p50_ms"], 20)
        self.assertAlmostEqual(result["itl_p50_ms"], 30)
        self.assertEqual(result["output_tokens"], 2)
        with self.assertRaises(ValueError):
            summarize(events[:-1], .1, [])

    def test_fifo_vs_fit_admission_and_stable_priority(self):
        for admission in ("fifo", "fit"):
            engine = Engine(self.model(), EngineConfig("paged", "continuous", admission,
                                                        total_slots=64, block_size=8, max_running=4))
            first = engine.submit([256] * 24, 9)   # reserves 32
            blocked = engine.submit([256] * 32, 9) # reserves 40
            small = engine.submit([256], 1)        # reserves 8
            engine.step()
            admitted = [e["request_id"] for e in engine.events if e["event"] == "admit"]
            self.assertEqual(admitted, [first] if admission == "fifo" else [first, small])
            engine.run()
            engine.cache.check()
        engine = Engine(self.model(), EngineConfig(total_slots=64, block_size=8))
        low = engine.submit([256], 1, priority=5)
        high1 = engine.submit([256], 1, priority=0)
        high2 = engine.submit([256], 1, priority=0)
        engine.run()
        self.assertEqual([e["request_id"] for e in engine.events if e["event"] == "admit"], [high1, high2, low])

    def test_late_request_joins_without_changing_existing_output(self):
        engine = Engine(self.model(), EngineConfig(total_slots=128, block_size=8))
        first = engine.submit([256, 7], 10)
        engine.step()
        engine.step()
        late = engine.submit([256, 9], 2)
        engine.run()
        first_finish = next(e["tick"] for e in engine.events if e["event"] == "finish" and e["request_id"] == first)
        late_admit = next(e["tick"] for e in engine.events if e["event"] == "admit" and e["request_id"] == late)
        self.assertLess(late_admit, first_finish)
        baseline = Engine(self.model(), EngineConfig(total_slots=128, block_size=8))
        baseline.submit([256, 7], 10)
        baseline.run()
        self.assertEqual(engine.completed[first]["output_ids"], baseline.completed[first]["output_ids"])

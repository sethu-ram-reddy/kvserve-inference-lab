import json
import unittest
from fastapi.testclient import TestClient
from kvserve.api import create_app
from kvserve.engine import EngineConfig
from kvserve.model import AttentionDecoder


class APITests(unittest.TestCase):
    def test_completion_stream_and_validation_release_all_cache(self):
        model = AttentionDecoder(width=16, max_context=64)
        model.ignore_eos = True
        app = create_app(model, EngineConfig(total_slots=128, block_size=8, max_pending=8))
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json()["trained"], False)
            result = client.post("/v1/completions", json={"prompt": "cache", "max_tokens": 6}).json()
            self.assertEqual(len(result["output_ids"]), 6)
            response = client.post("/v1/completions", json={"prompt": "cache", "max_tokens": 6, "stream": True})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            messages = [json.loads(line[6:]) for line in response.text.splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"]
            self.assertEqual([e["token_id"] for e in messages if e["event"] == "token"], result["output_ids"])
            self.assertEqual(messages[-1]["event"], "finish")
            zero = client.post("/v1/completions", json={"prompt": "a", "max_tokens": 0}).json()
            self.assertEqual(zero["output_ids"], [])
            self.assertEqual(client.post("/v1/completions", json={"prompt": "a" * 64, "max_tokens": 2}).status_code, 422)
            self.assertEqual(client.post("/v1/completions", json={"prompt": "a", "top_p": 0}).status_code, 422)
            metrics = client.get("/metrics").json()
            self.assertEqual(metrics["finished"], 3)
            self.assertEqual(metrics["assigned_kv_slots"], 0)
            self.assertEqual(metrics["reserved_kv_slots"], 0)
            self.assertEqual(len(app.state.channels), 0)
            self.assertEqual(len(app.state.engine.events), 0)

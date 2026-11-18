"""Runnable demonstration and optional HTTP server entry point."""
import argparse
import json
from .engine import Engine, EngineConfig
from .model import AttentionDecoder, ByteTokenizer
from .sampling import Sampling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "serve"), nargs="?", default="demo")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("kvserve.api:app", host="127.0.0.1", port=args.port)
        return
    tokenizer = ByteTokenizer()
    engine = Engine(AttentionDecoder(), EngineConfig())
    for prompt in ("What is a KV cache?", "How does scheduling work?", "Explain prefill."):
        engine.submit(tokenizer.encode(prompt), 12, Sampling(temperature=0.7, top_k=16), seed=3)
    outputs = engine.run()
    print(json.dumps({"note": "Untrained decoder: tokens demonstrate execution, not language quality",
                      "requests": outputs, "decode_rounds": engine.tick,
                      "assigned_slots_after_completion": engine.cache.assigned_slots}, indent=2))


if __name__ == "__main__":
    main()

"""A seeded, untrained single-head attention decoder for systems experiments.

This deliberately omits a full Transformer's MLP, layer stack and normalization.
Weights and KV rows are float32. The sinusoidal positions make decode order explicit.
"""
import numpy as np
from .sampling import softmax


class ByteTokenizer:
    bos, eos, vocab_size = 256, 257, 258

    def encode(self, text):
        return [self.bos, *text.encode("utf-8")]

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


class AttentionDecoder:
    def __init__(self, width=64, max_context=512, seed=7):
        if width < 2 or width % 2 or max_context < 1:
            raise ValueError("width must be positive and even; context must be positive")
        self.width, self.max_context, self.seed = width, max_context, seed
        rng = np.random.default_rng(seed)
        def weight(shape):
            return (rng.standard_normal(shape) / np.sqrt(width)).astype(np.float32)
        self.embedding = weight((ByteTokenizer.vocab_size, width))
        self.wq, self.wk, self.wv, self.wo = [weight((width, width)) for _ in range(4)]
        self.head = weight((width, ByteTokenizer.vocab_size))
        positions = np.arange(max_context)[:, None]
        frequencies = np.exp(np.arange(0, width, 2) * (-np.log(10000.0) / width))
        self.positions = np.empty((max_context, width), np.float32)
        self.positions[:, 0::2] = np.sin(positions * frequencies)
        self.positions[:, 1::2] = np.cos(positions * frequencies)

    def project(self, ids, positions):
        if len(ids) == 0 or min(positions) < 0 or max(positions) >= self.max_context:
            raise ValueError("Invalid token positions")
        if min(ids) < 0 or max(ids) >= ByteTokenizer.vocab_size:
            raise ValueError("Token outside vocabulary")
        inputs = self.embedding[np.asarray(ids)] + self.positions[np.asarray(positions)]
        return inputs @ self.wq, inputs @ self.wk, inputs @ self.wv

    def prefill(self, seq, ids, cache):
        queries, keys, values = self.project(ids, list(range(len(ids))))
        cache.append(seq, keys, values)
        # A single attention layer only needs its final query to predict the next token.
        return (cache.attend(seq, queries[-1]) @ self.wo) @ self.head

    def decode_batch(self, seqs, ids, cache):
        # Q/K/V and output projections are genuinely vectorized across active requests.
        # Ragged attention reads each request's own cache; there is no GPU batch kernel.
        positions = [cache.lengths[s] for s in seqs]
        queries, keys, values = self.project(ids, positions)
        contexts = []
        for i, seq in enumerate(seqs):
            cache.append(seq, keys[i:i + 1], values[i:i + 1])
            contexts.append(cache.attend(seq, queries[i]))
        return (np.stack(contexts) @ self.wo) @ self.head

    def uncached_logits(self, ids):
        queries, keys, values = self.project(ids, list(range(len(ids))))
        scores = (queries @ keys.T) / np.sqrt(self.width)
        scores[np.triu_indices(len(ids), 1)] = -np.inf
        return (softmax(scores) @ values @ self.wo) @ self.head

    @property
    def parameter_count(self):
        return sum(w.size for w in (self.embedding, self.wq, self.wk, self.wv, self.wo, self.head))

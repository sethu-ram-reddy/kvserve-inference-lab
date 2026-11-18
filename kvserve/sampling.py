"""Stable sampling, with request-local random generators owned by the scheduler."""
from dataclasses import dataclass
import numpy as np


def softmax(logits):
    logits = np.asarray(logits)
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("Logits must not contain NaN or positive infinity")
    if not np.isfinite(logits).any(axis=-1).all():
        raise ValueError("Each row needs at least one finite logit")
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0

    def validate(self, vocab_size):
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k is not None and (type(self.top_k) is not int or not 1 <= self.top_k <= vocab_size):
            raise ValueError("top_k must be between 1 and vocabulary size")
        if not np.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")

    def select(self, logits, rng):
        self.validate(len(logits))
        if self.temperature == 0:
            return int(np.argmax(logits))
        scores = np.asarray(logits, dtype=np.float64) / self.temperature
        order = np.argsort(-scores, kind="stable")
        if self.top_k is not None:
            scores[order[self.top_k:]] = -np.inf
        probabilities = softmax(scores)
        order = np.argsort(-probabilities, kind="stable")
        cumulative = np.cumsum(probabilities[order])
        # Keep the token that crosses the nucleus threshold.
        remove = np.r_[False, cumulative[:-1] >= self.top_p]
        scores[order[remove]] = -np.inf
        return int(rng.choice(len(scores), p=softmax(scores)))

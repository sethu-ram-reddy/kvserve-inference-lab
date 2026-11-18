"""Iteration scheduler with conservative KV reservation and timestamped events."""
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from .cache import ContiguousCache, PagedCache
from .model import ByteTokenizer
from .sampling import Sampling


@dataclass(frozen=True)
class EngineConfig:
    layout: str = "paged"
    scheduling: str = "continuous"
    admission: str = "fifo"
    total_slots: int = 1024
    block_size: int = 16
    max_running: int = 16
    max_pending: int = 128

    def validate(self, model):
        if self.layout not in ("fixed", "paged") or self.scheduling not in ("static", "continuous"):
            raise ValueError("Unknown layout or scheduling policy")
        if self.admission not in ("fifo", "fit"):
            raise ValueError("Unknown admission policy")
        for name in ("total_slots", "block_size", "max_running", "max_pending"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.total_slots % self.block_size:
            raise ValueError("total_slots must be divisible by block_size")
        if self.layout == "fixed" and self.total_slots < model.max_context:
            raise ValueError("Fixed cache cannot hold one full context")


@dataclass
class Request:
    id: str
    prompt: tuple
    max_new_tokens: int
    sampling: Sampling
    seed: int
    priority: int
    ordinal: int
    submitted_at: float
    reservation: int
    generated: list = field(default_factory=list)
    logits: object = None
    rng: object = None


class Engine:
    def __init__(self, model, config=EngineConfig(), clock=perf_counter, retain_events=True):
        config.validate(model)
        self.model, self.config, self.clock = model, config, clock
        self.retain_events = retain_events
        self.started_at = clock()
        self.cache = (PagedCache(config.total_slots, model.width, config.block_size)
                      if config.layout == "paged" else
                      ContiguousCache(config.total_slots, model.width, model.max_context))
        self.waiting, self.running, self.completed = [], [], {}
        self.events, self.snapshots = [], []
        self.tick, self.ordinal, self.reserved = 0, 0, 0
        self.last_step_events = []

    def _event(self, kind, request, **fields):
        event = {"event": kind, "request_id": request.id, "tick": self.tick,
                 "time_s": self.clock() - self.started_at, **fields}
        self.last_step_events.append(event)
        if self.retain_events:
            self.events.append(event)
        return event

    def submit(self, prompt, max_new_tokens=32, sampling=Sampling(), seed=0, priority=0):
        prompt = tuple(prompt)
        if (not prompt or any(type(t) is not int or not 0 <= t < ByteTokenizer.vocab_size for t in prompt)):
            raise ValueError("Prompt must contain valid integer token IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if type(seed) is not int or seed < 0 or type(priority) is not int:
            raise ValueError("seed must be nonnegative and priority must be an integer")
        length = len(prompt) + max(0, max_new_tokens - 1)
        if length > self.model.max_context:
            raise ValueError("Prompt plus decode inputs exceed model context")
        sampling.validate(ByteTokenizer.vocab_size)
        if len(self.waiting) + len(self.running) >= self.config.max_pending:
            raise OverflowError("Request queue is full")
        cost = (self.model.max_context if self.config.layout == "fixed" else
                ((length + self.config.block_size - 1) // self.config.block_size) * self.config.block_size)
        if max_new_tokens and cost > self.config.total_slots:
            raise ValueError("Request cannot fit even in an otherwise empty KV pool")
        rid = f"req-{self.ordinal}"
        request = Request(rid, prompt, max_new_tokens, sampling, seed, priority,
                          self.ordinal, self.clock() - self.started_at, cost)
        request.rng = np.random.default_rng(seed)
        self.ordinal += 1
        self._event("submit", request)
        if max_new_tokens == 0:
            self.completed[rid] = {"request_id": rid, "output_ids": [], "finish_reason": "length"}
            self._event("finish", request, finish_reason="length")
        else:
            self.waiting.append(request)
        return rid

    def _admit(self):
        if self.config.scheduling == "static" and self.running:
            return
        self.waiting.sort(key=lambda r: (r.priority, r.ordinal))
        for request in list(self.waiting):
            if len(self.running) >= self.config.max_running:
                break
            if self.reserved + request.reservation > self.config.total_slots:
                if self.config.admission == "fifo":
                    break
                continue
            self.cache.add(request.id)
            self._event("admit", request, reserved_slots=request.reservation)
            try:
                request.logits = self.model.prefill(request.id, request.prompt, self.cache)
            except Exception:
                self.cache.free(request.id)
                raise
            self.reserved += request.reservation
            self.waiting.remove(request)
            self.running.append(request)
            self._event("prefill_end", request)

    def step(self):
        """Admit, emit one token per active request, then release finished caches."""
        self.last_step_events = []
        if not self.pending:
            return []
        self._admit()
        if not self.running:
            raise RuntimeError("Pending requests made no progress")
        needs_decode = [r for r in self.running if r.generated]
        if needs_decode:
            logits = self.model.decode_batch([r.id for r in needs_decode],
                                             [r.generated[-1] for r in needs_decode], self.cache)
            for request, row in zip(needs_decode, logits):
                request.logits = row
        snapshot = {"tick": self.tick, "active": len(self.running), "waiting": len(self.waiting),
                    "assigned_slots": self.cache.assigned_slots, "used_slots": self.cache.used_slots,
                    "reserved_slots": self.reserved}
        if self.retain_events:
            self.snapshots.append(snapshot)
        for request in list(self.running):
            token = request.sampling.select(request.logits, request.rng)
            request.generated.append(token)
            # Benchmarks can disable EOS by setting this explicit flag on the model.
            eos = token == ByteTokenizer.eos and not getattr(self.model, "ignore_eos", False)
            done = eos or len(request.generated) >= request.max_new_tokens
            self._event("token", request, token_id=token)
            if done:
                reason = "eos" if eos else "length"
                self.completed[request.id] = {"request_id": request.id,
                                              "output_ids": list(request.generated),
                                              "finish_reason": reason}
                self.cache.free(request.id)
                self.reserved -= request.reservation
                self.running.remove(request)
                self._event("finish", request, finish_reason=reason)
        self.tick += 1
        return list(self.last_step_events)

    def cancel(self, rid):
        for group in (self.waiting, self.running):
            for request in list(group):
                if request.id == rid:
                    group.remove(request)
                    if group is self.running:
                        self.reserved -= request.reservation
                    self.cache.free(rid)
                    self._event("cancel", request)
                    return True
        return False

    @property
    def pending(self):
        return bool(self.waiting or self.running)

    def run(self, max_ticks=100000):
        try:
            while self.pending and self.tick < max_ticks:
                self.step()
            if self.pending:
                raise TimeoutError("Scheduler tick limit exceeded")
            return self.completed
        finally:
            if self.pending:
                for request in list(self.waiting) + list(self.running):
                    self.cancel(request.id)
            self.cache.check()

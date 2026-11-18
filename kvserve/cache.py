"""Two cache layouts under the same token-slot budget; no simulated memory timings."""
import numpy as np
from .sampling import softmax


class CacheFull(RuntimeError):
    """An append cannot commit without exceeding the physical KV capacity."""


class ContiguousCache:
    def __init__(self, total_slots, width, max_context):
        self.total_slots, self.width, self.max_context = total_slots, width, max_context
        self.records = {}
        self.lengths = {}

    def add(self, seq):
        if seq in self.records:
            raise ValueError("Sequence already owns a cache")
        if self.assigned_slots + self.max_context > self.total_slots:
            raise CacheFull("No fixed-size sequence slab available")
        self.records[seq] = (np.empty((self.max_context, self.width), np.float32),
                             np.empty((self.max_context, self.width), np.float32))
        self.lengths[seq] = 0

    def append(self, seq, keys, values):
        start = self.lengths[seq]
        end = start + len(keys)
        if keys.shape != values.shape or keys.shape[1:] != (self.width,):
            raise ValueError("K/V shape mismatch")
        if end > self.max_context:
            raise CacheFull("Sequence exceeds its contiguous slab")
        self.records[seq][0][start:end] = keys
        self.records[seq][1][start:end] = values
        self.lengths[seq] = end

    def attend(self, seq, query):
        length = self.lengths[seq]
        keys, values = self.records[seq]
        return softmax((query @ keys[:length].T) / np.sqrt(self.width)) @ values[:length]

    def free(self, seq):
        self.records.pop(seq, None)
        self.lengths.pop(seq, None)

    @property
    def assigned_slots(self):
        return len(self.records) * self.max_context

    @property
    def used_slots(self):
        return sum(self.lengths.values())

    def check(self):
        assert set(self.records) == set(self.lengths)
        assert self.assigned_slots <= self.total_slots
        assert all(0 <= n <= self.max_context for n in self.lengths.values())


class PagedCache:
    def __init__(self, total_slots, width, block_size):
        if total_slots < 1 or width < 1 or block_size < 1 or total_slots % block_size:
            raise ValueError("Positive pool dimensions and whole blocks required")
        self.total_slots, self.width, self.block_size = total_slots, width, block_size
        self.keys = np.empty((total_slots // block_size, block_size, width), np.float32)
        self.values = np.empty_like(self.keys)
        self.free_blocks = list(range(len(self.keys) - 1, -1, -1))
        self.tables, self.lengths = {}, {}

    def add(self, seq):
        if seq in self.tables:
            raise ValueError("Sequence already owns a cache")
        self.tables[seq], self.lengths[seq] = [], 0

    def append(self, seq, keys, values):
        if keys.shape != values.shape or keys.shape[1:] != (self.width,):
            raise ValueError("K/V shape mismatch")
        start, size = self.lengths[seq], self.block_size
        end = start + len(keys)
        needed = (end + size - 1) // size - len(self.tables[seq])
        # Check the entire operation before mutating ownership or lengths.
        if needed > len(self.free_blocks):
            raise CacheFull("Insufficient free KV blocks; append left unchanged")
        self.tables[seq].extend(self.free_blocks.pop() for _ in range(needed))
        cursor = 0
        while cursor < len(keys):
            position = start + cursor
            block, offset = self.tables[seq][position // size], position % size
            count = min(size - offset, len(keys) - cursor)
            self.keys[block, offset:offset + count] = keys[cursor:cursor + count]
            self.values[block, offset:offset + count] = values[cursor:cursor + count]
            cursor += count
        self.lengths[seq] = end

    def attend(self, seq, query):
        # Online softmax across physical blocks. Never gather the full KV sequence.
        maximum, denominator = -np.inf, 0.0
        numerator = np.zeros(self.width, dtype=np.float64)
        remaining = self.lengths[seq]
        for block in self.tables[seq]:
            count = min(self.block_size, remaining)
            scores = (query @ self.keys[block, :count].T) / np.sqrt(self.width)
            new_max = max(maximum, float(np.max(scores)))
            correction = np.exp(maximum - new_max)
            weights = np.exp(scores.astype(np.float64) - new_max)
            numerator = numerator * correction + weights @ self.values[block, :count]
            denominator = denominator * correction + weights.sum()
            maximum = new_max
            remaining -= count
        if denominator == 0:
            raise ValueError("Cannot attend to an empty cache")
        return (numerator / denominator).astype(np.float32)

    def free(self, seq):
        self.free_blocks.extend(self.tables.pop(seq, []))
        self.lengths.pop(seq, None)

    @property
    def assigned_slots(self):
        return sum(map(len, self.tables.values())) * self.block_size

    @property
    def used_slots(self):
        return sum(self.lengths.values())

    def check(self):
        owned = [b for table in self.tables.values() for b in table]
        all_blocks = owned + self.free_blocks
        assert sorted(all_blocks) == list(range(len(self.keys))), "Block ownership lost or duplicated"
        assert set(self.tables) == set(self.lengths)
        assert all(len(self.tables[s]) == (n + self.block_size - 1) // self.block_size
                   for s, n in self.lengths.items())

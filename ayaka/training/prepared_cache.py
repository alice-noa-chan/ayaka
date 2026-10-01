"""Bounded reuse of immutable CPU training rows from a verified dataset snapshot."""

import sys
from collections import OrderedDict
from dataclasses import fields, is_dataclass

import torch


def cpu_bytes(value):
    seen, storage = set(), set()

    def size(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        total = sys.getsizeof(obj)
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "cpu":
                raise ValueError("prepared cache must never retain GPU tensors or KV caches")
            data = obj.untyped_storage()
            key = (data.data_ptr(), data.nbytes())
            if key not in storage:
                storage.add(key)
                total += data.nbytes()
        elif is_dataclass(obj):
            if hasattr(obj, "__dict__"):
                total += sys.getsizeof(vars(obj))
            total += sum(size(getattr(obj, field.name)) for field in fields(obj))
        elif isinstance(obj, dict):
            total += sum(size(k) + size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            total += sum(map(size, obj))
        return total

    return size(value)


class PreparedSampleCache:
    def __init__(self, builder, max_bytes=256 * 1024 * 1024):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("prepared cache bytes must be a nonnegative integer")
        self.builder, self.max_bytes = builder, max_bytes
        self.entries, self.bytes = OrderedDict(), 0
        self.hits = self.misses = 0

    def get(self, sample):
        # Samples belong to an immutable, checksum-verified snapshot. Keep the
        # source object in each entry to prevent Python identity reuse collisions.
        key = id(sample)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][1]
        self.misses += 1
        items = self.builder(sample)
        if not self.max_bytes:
            return items
        size = cpu_bytes(items)
        if size <= self.max_bytes:
            while self.entries and self.bytes + size > self.max_bytes:
                _, (_, _, evicted) = self.entries.popitem(last=False)
                self.bytes -= evicted
            self.entries[key] = sample, items, size
            self.bytes += size
        return items

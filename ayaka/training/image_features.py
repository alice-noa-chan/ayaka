"""Bounded frozen vision reuse; language activations and KV caches are never cached."""

import hashlib
from collections import OrderedDict
from contextlib import contextmanager

import torch


class FrozenImageFeatures:
    def __init__(self, backend, max_bytes):
        self.backend, self.max_bytes = backend, max_bytes
        self.entries, self.bytes = OrderedDict(), 0
        self.hits = self.misses = 0
        self.signature = None

    def _check(self):
        native, model = self.backend.native, self.backend.model
        text_ids = {id(p) for p in model.backbone.parameters()}
        frozen = [p for p in native.parameters() if id(p) not in text_ids]
        if any(p.requires_grad for p in frozen):
            raise ValueError("image feature reuse requires frozen vision/projector weights")
        if native.training:
            raise ValueError("image feature reuse requires deterministic native evaluation mode")
        signature = [(id(p), p._version, p.device, p.dtype) for p in frozen]
        if self.signature != signature:
            self.entries.clear()
            self.bytes = 0
            self.signature = signature

    def get(self, payload):
        digest = hashlib.sha256()
        for key in ("pixel_values", "image_position_ids"):
            value = payload[key]
            if value.device.type != "cpu":
                raise ValueError("feature cache keys require prepared CPU media tensors")
            digest.update(str((key, value.shape, value.dtype)).encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        key = digest.hexdigest()
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        weight = self.backend.model.embed_weight()
        with torch.no_grad():
            features = self.backend.native.get_image_features(
                payload["pixel_values"].to(weight.device, weight.dtype),
                payload["image_position_ids"].to(weight.device),
                return_dict=True,
            ).pooler_output
            # Split views may retain padded parent storage. Own exact-sized
            # tensors so the LRU byte ceiling also bounds actual allocation.
            features = tuple(t.detach().clone() for t in features)
        size = sum(t.numel() * t.element_size() for t in features)
        if size <= self.max_bytes:
            while self.entries and self.bytes + size > self.max_bytes:
                _, evicted = self.entries.popitem(last=False)
                self.bytes -= sum(t.numel() * t.element_size() for t in evicted)
            self.entries[key], self.bytes = features, self.bytes + size
        return features

    @contextmanager
    def use(self, payloads):
        # Training is single-threaded. Replace only for this native forward and
        # restore in finally; checkpoint recomputation runs language layers only.
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        self._check()
        features = tuple(t for payload in payloads for t in self.get(payload))
        native = self.backend.native
        original = native.get_image_features
        native.get_image_features = lambda *args, **kwargs: BaseModelOutputWithPooling(
            pooler_output=features
        )
        try:
            yield
        finally:
            native.get_image_features = original

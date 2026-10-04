"""Isolated branches of native prefix caches, without sliding-window rollback."""

import copy

import torch


def branch_evidence_cache(cache, strategy):
    """Copy metadata while sharing immutable KV tensors in known dynamic layers.

    Stock DynamicLayer/DynamicSlidingWindowLayer.update assigns cat-produced
    tensors instead of writing the retained prefix in place. Independent layer
    objects isolate both full and sliding counters. Recurrent/static/offloaded
    or custom cache classes require the existing deep-copy strategy. No crop or
    fallback is used: a sliding window may already have discarded old states.
    """
    if strategy == "deepcopy":
        return copy.deepcopy(cache)
    if strategy != "copy_on_write":
        raise ValueError("unsupported evidence cache strategy")
    from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

    if (
        type(cache) is not DynamicCache
        or cache.offloading
        or not cache.layers
        or any(
            type(layer) not in (DynamicLayer, DynamicSlidingWindowLayer) for layer in cache.layers
        )
    ):
        raise ValueError("copy_on_write requires stock non-offloaded attention DynamicCache layers")
    branch = copy.copy(cache)
    branch.layers = [copy.copy(layer) for layer in cache.layers]
    return branch


def dynamic_kv_bytes(cache):
    """Logical KV bytes of known layers; not allocated/peak CUDA memory."""
    from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

    if type(cache) is not DynamicCache or any(
        type(layer) not in (DynamicLayer, DynamicSlidingWindowLayer) for layer in cache.layers
    ):
        return None
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.layers
        for tensor in (layer.keys, layer.values)
        if torch.is_tensor(tensor)
    )

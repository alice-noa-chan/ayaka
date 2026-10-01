import pytest
import torch

from ayaka.data.schema import Question, Sample
from ayaka.training.prepared_cache import PreparedSampleCache, cpu_bytes


def test_cpu_preparation_is_reused_without_aliasing_other_source_objects():
    a = Sample("first", [Question.noul("q", "Visible?", 1)])
    b = Sample("different", [Question.noul("q", "Visible?", 1)])
    calls = []

    def prepare(sample):
        calls.append(sample.state)
        return [sample.state]

    cache = PreparedSampleCache(prepare, 4096)
    first = cache.get(a)
    assert cache.get(a) is first and cache.get(b) != first
    assert calls == ["first", "different"] and cache.hits == 1 and cache.misses == 2
    assert cache.bytes <= cache.max_bytes


def test_prepared_memory_counts_shared_storage_once_and_evicts_cold_rows():
    tensor = torch.zeros(2048)
    assert cpu_bytes([tensor, tensor[:1]]) < cpu_bytes([tensor, tensor.clone()])
    cache = PreparedSampleCache(lambda sample: [torch.zeros(512)], 3200)
    a, b = object(), object()
    cache.get(a)
    cache.get(b)
    assert len(cache.entries) == 1 and cache.bytes <= 3200
    cache.get(a)
    assert cache.misses == 3


def test_disabled_and_oversized_preparation_never_accumulates_memory():
    sample = object()
    for limit in (0, 1):
        cache = PreparedSampleCache(lambda sample: [torch.zeros(512)], limit)
        cache.get(sample)
        cache.get(sample)
        assert cache.bytes == 0 and not cache.entries and cache.misses == 2
    with pytest.raises(ValueError, match="never retain GPU"):
        cpu_bytes(torch.empty(1, device="meta"))


def test_cached_stream_preserves_sampling_and_reuses_read_only_encoded_rows():
    from dataclasses import replace

    from test_multimodal import build

    from ayaka.data.candidate_v2 import candidate_curriculum
    from ayaka.training.run_v2 import sample_stream

    _, model, tok, backend = build("gemma4")
    model.cfg = replace(model.cfg, max_seq_len=4096)
    samples = candidate_curriculum("train", 2)
    cached = sample_stream(samples, tok, model.cfg, backend, 3, 7)
    uncached = sample_stream(samples, tok, model.cfg, backend, 3, 7, prepared_cache_bytes=0)
    seen = {}
    for _ in range(3):
        a, b = next(cached), next(uncached)
        assert [x.sample_id for x in a] == [x.sample_id for x in b]
        assert [x.enc.prefix_ids for x in a] == [x.enc.prefix_ids for x in b]
        for item in a:
            if item.sample_id in seen:
                assert item is seen[item.sample_id]
            seen[item.sample_id] = item

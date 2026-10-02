import hashlib
import json

import pytest

from ayaka.training.reference_profile import validate_reference


def test_reference_profile_rejects_stale_runtime_schedule_and_modified_bytes(tmp_path):
    binding = {"parent": "v1", "schedule": "fixed-200"}
    runtime = {"gpu": "same-gpu", "torch": "pinned"}
    profile = {
        "weights_unchanged": True,
        "optimizer_steps": 0,
        "schedule": {"schedule_sha256": "fixed-200"},
    }
    data = json.dumps(profile).encode()
    (tmp_path / "throughput.json").write_bytes(data)
    (tmp_path / "binding.json").write_text(
        json.dumps(
            {
                "binding": binding,
                "runtime": runtime,
                "throughput_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    )
    assert validate_reference(tmp_path, binding, runtime) == profile
    for expected, device in (
        ({**binding, "parent": "new-parent"}, runtime),
        ({**binding, "schedule": "different-plan"}, runtime),
        (binding, {**runtime, "torch": "changed"}),
    ):
        with pytest.raises(ValueError, match="mismatch"):
            validate_reference(tmp_path, expected, device)
    (tmp_path / "throughput.json").write_bytes(data + b" ")
    with pytest.raises(ValueError, match="checksum"):
        validate_reference(tmp_path, binding, runtime)


def test_reference_probe_refuses_changed_batch_shapes_and_never_scales_down(monkeypatch):
    from types import SimpleNamespace

    import torch

    from ayaka.training import reference_profile as module

    policy = {"micro_tokens": 4096, "micro_ckpt_tokens": 4096, "checkpoint_threshold": 0}
    reference = {
        "effective_policy": policy,
        "batches": [{"rows": 64}] * 3,
        "seconds": [6, 6, 6],
        "max_seconds": 20,
        "schedule": {"max_seconds": 9},
    }
    trainer = SimpleNamespace(
        device=torch.device("cuda"), micro_tokens=4096, micro_ckpt_tokens=4096, ckpt_threshold=0
    )
    monkeypatch.setattr(module, "profile_binding", lambda *a: {})
    monkeypatch.setattr(module, "runtime_signature", lambda *a: {})
    monkeypatch.setattr(module, "validate_reference", lambda *a: reference)
    probe = {"batches": [{"rows": 64}] * 3, "max_seconds": 4}
    monkeypatch.setattr(module, "profile_backward", lambda *a, **kw: probe)
    result = module.reuse_profile(None, trainer, None, None, None, None, 200)
    assert result["schedule"]["max_seconds"] == 9
    probe["max_seconds"] = 12
    assert (
        module.reuse_profile(None, trainer, None, None, None, None, 200)["schedule"]["max_seconds"]
        == 18
    )
    assert reference["schedule"]["max_seconds"] == 9
    probe["batches"] = [{"rows": 32}] * 3
    with pytest.raises(ValueError, match="identical"):
        module.reuse_profile(None, trainer, None, None, None, None, 200)

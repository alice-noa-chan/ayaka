"""The remote runtime's older decorator lacks the client-only marketplace flag."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.beam_v2 import worker


def test_remote_job_imports_without_new_client_only_constructor_keywords(monkeypatch):
    calls = []
    allowed = {
        "name",
        "gpu",
        "cpu",
        "memory",
        "timeout",
        "retries",
        "headless",
        "image",
        "volumes",
        "pool",
    }

    def old_function(**kwargs):
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"older remote SDK does not support {unknown}")
        calls.append(kwargs)
        return lambda fn: fn

    old_beam = SimpleNamespace(
        function=old_function,
        Image=lambda **_: SimpleNamespace(add_python_packages=lambda _: None),
        Volume=lambda **_: None,
    )
    monkeypatch.setitem(sys.modules, "beam", old_beam)
    monkeypatch.setitem(sys.modules, "worker", worker)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/beam_v2/job.py"))
    gpu = next(c for c in calls if c.get("gpu") == "RTX5090")
    assert gpu["timeout"] == 9700 and gpu["retries"] == 0 and gpu["headless"]
    assert gpu["cpu"] == 2 and gpu["memory"] == "32Gi" and "pool" not in gpu

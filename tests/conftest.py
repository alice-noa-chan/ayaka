"""Shared test configuration.

Some Swift tests read the prepared v2 corpus under ``runs/``. That directory is
local and git-ignored, so a clean checkout (for example CI) does not have it.
Mark such tests with ``@pytest.mark.prepared_corpus``; they run when the corpus
is present and are skipped with an explicit reason otherwise.
"""

from pathlib import Path

import pytest

PREPARED_CORPUS = Path(__file__).resolve().parents[1] / "runs/v2-pretraining-20261002-ready"


def pytest_collection_modifyitems(config, items):
    if (PREPARED_CORPUS / "calibration.jsonl").is_file() and (
        PREPARED_CORPUS / "dev.jsonl"
    ).is_file():
        return
    skip = pytest.mark.skip(reason=f"prepared corpus not found at {PREPARED_CORPUS}")
    for item in items:
        if "prepared_corpus" in item.keywords:
            item.add_marker(skip)

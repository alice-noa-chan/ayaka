"""Dataset manifest records (docs.md section 37).

Every ingested dataset records provenance: source url, exact
revision/config/split, license, language, task family, primitive
mapping, and the transformation/dedup/teacher/soft-target versions
applied — GitHub-sourced sets pin exact commits, not just names.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TRANSFORMATION_VERSION = "t1"
DEDUP_VERSION = "d1"


@dataclass
class DatasetManifest:
    dataset_id: str
    source_url: str
    revision: str
    config: str
    split: str
    license: str
    language: str
    task_family: str
    primitive_mapping: str
    original_label_schema: str
    transformation_version: str = TRANSFORMATION_VERSION
    dedup_version: str = DEDUP_VERSION
    teacher_version: str = ""
    soft_target_version: str = ""
    augmentation_version: str = ""
    notes: str = ""

    def to_jsonl_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_manifest(records: list[DatasetManifest], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_jsonl_line() + "\n")

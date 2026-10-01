import base64
import hashlib

import pytest

from ayaka.data.multimodal_v2 import image_curriculum
from ayaka.data.reasoning_v2 import SPLITS
from ayaka.data.schema import Candidate, Question, Sample
from ayaka.multimodal import decode_media


def test_native_data_has_disjoint_templates_rules_lineage_and_multilingual_pairs():
    previous = {
        key: set() for key in ("generator_template_id", "rule_combination", "source_lineage")
    }
    for split in SPLITS:
        samples = image_curriculum(split, 8)
        assert len(samples) == 192
        assert {s.metadata["language"] for s in samples} == {"en", "ko", "ja"}
        for key in previous:
            values = {s.metadata[key] for s in samples}
            assert not previous[key] & values
            previous[key] |= values
        for image, text in zip(samples[::2], samples[1::2], strict=True):
            assert image.questions == text.questions
            assert image.metadata["source_lineage"] == text.metadata["source_lineage"]
            raw = base64.b64decode(image.metadata["media"][0]["data"])
            assert hashlib.sha256(raw).hexdigest() == image.metadata["image_sha256"]
            decode_media(image.state, image.metadata["media"])
            assert Sample.from_json(image.to_json()).questions == image.questions
            assert sum(image.questions[0].target_distribution.values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "target", [{"a": float("nan")}, {"a": float("inf")}, {"a": -0.1, "b": 1.1}]
)
def test_invalid_training_targets_fail_closed(target):
    with pytest.raises(ValueError, match="finite"):
        Question("q", "choice", "Task", [Candidate("a", "a"), Candidate("b", "b")], target)


def test_duplicate_candidates_and_nonfinite_score_levels_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        Question("q", "choice", "Task", [Candidate("a", "a"), Candidate("a", "b")], {"a": 1})
    with pytest.raises(ValueError, match="finite"):
        Question(
            "q",
            "score",
            "Task",
            [Candidate("a", "a", float("nan")), Candidate("b", "b", 1)],
            {"b": 1},
        )

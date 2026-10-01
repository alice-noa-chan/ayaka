from ayaka.data.language_v2 import language_curriculum
from ayaka.data.reasoning_v2 import SPLITS


def test_native_language_rehearsal_is_balanced_and_split_specific():
    previous = set()
    for split in SPLITS:
        samples = language_curriculum(split, 8)
        assert len(samples) == 72
        templates = {s.metadata["generator_template_id"] for s in samples}
        assert not previous & templates
        previous |= templates
        assert any("회원" in s.state for s in samples if s.metadata["language"] == "ko")
        assert any("会員" in s.state for s in samples if s.metadata["language"] == "ja")
        for language in ("en", "ko", "ja"):
            group = [s for s in samples if s.metadata["language"] == language]
            assert {s.questions[0].type for s in group} == {"choice", "noul", "score"}
            assert {
                s.questions[0].target_distribution["true"]
                for s in group
                if s.questions[0].type == "noul"
            } == {0, 0.5, 1}
            for kind in ("choice", "score"):
                assert {
                    max(
                        s.questions[0].target_distribution,
                        key=s.questions[0].target_distribution.get,
                    )
                    for s in group
                    if s.questions[0].type == kind
                } == {"0", "1", "2", "3"}

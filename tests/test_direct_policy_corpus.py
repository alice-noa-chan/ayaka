import copy
import json
from collections import Counter

import pytest
import torch
from test_contract_nli import contract_registry
from test_direct_corpus import inputs
from test_direct_natural import raw_registry

from ayaka.data.contract_nli import SOURCE, ContractGoldRegistry
from ayaka.data.direct_natural import NaturalGoldRegistry, verify_raw_binding
from ayaka.data.natural_training_v2 import partition_sources
from ayaka.data.reasoning_v2 import SPLITS
from ayaka.data.reserved_evidence import ReservedEvidenceBlocker
from ayaka.data.schema import Sample
from ayaka.eval.read_artifact import fingerprint
from ayaka.input_errors import ContextLimitError
from ayaka.training import direct_audit, direct_bundle, direct_corpus
from ayaka.training.direct_corpus_plan import POLICY_VERSION, VERSION, plan_sha256, validate_plan
from ayaka.training.run_direct import run_pipeline


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def policy_inputs():
    original, cfg, tok, _, encoding, reserved = inputs()
    registry = NaturalGoldRegistry(raw_registry().raw, policy_registry=contract_registry())
    settings = copy.deepcopy(original["settings"])
    settings["natural_sample_quotas"][SOURCE] = dict.fromkeys(SPLITS, 1)
    settings["authored_per_type"] = dict.fromkeys(SPLITS, 2)
    settings["rows_per_step"] = 3
    settings["epochs"] = 2
    settings["minimum_english_question_fraction"] = 0.85
    plan = direct_corpus.create_plan(settings, cfg, tok, registry, encoding, reserved)
    return plan, cfg, tok, registry, encoding, reserved


def test_plan2_preserves_three_way_document_quotas_languages_and_whole_epochs(
    tmp_path, monkeypatch
):
    plan, cfg, tok, registry, encoding, reserved = policy_inputs()
    assert plan["version"] == POLICY_VERSION
    report = direct_corpus.prepare_corpus(
        tmp_path / "prepared", plan, cfg, tok, registry, encoding, reserved, allow_tiny=True
    )
    contract = report["corpus_contract"]
    assert contract["whole_epochs"] == {
        **contract["whole_epochs"],
        "rows": 33,
        "epochs": 2,
        "min_visits": 2,
        "max_visits": 2,
    }
    for summary in contract["splits"].values():
        assert summary["sources"][SOURCE] == {
            "samples": 1,
            "questions": 17,
            "types": {"choice": 17},
        }
        assert summary["language_questions"] == {"en": 29, "ko": 2, "ja": 2}
    assert sum(report["selected_policy_labels"]["train"].values()) == 17
    bundle = tmp_path / "prepared/bundle"
    regenerated = run_pipeline(
        bundle,
        None,
        action="audit",
        mechanics_only=True,
        natural_registry=registry,
        expected_bundle_sha256=report["bundle_manifest_sha256"],
    )
    assert (
        regenerated["audit_mode"] == "full_regeneration"
        and regenerated["model_weights_loaded"] is False
    )
    assert (
        SOURCE
        in json.loads((bundle / "recipe.json").read_bytes())["gold_sources"]["policy_sources"]
    )
    monkeypatch.setattr(
        "ayaka.data.direct_natural.local_raw_sources",
        lambda: pytest.fail("portable load opened raw annotations"),
    )
    monkeypatch.setattr(
        ContractGoldRegistry,
        "verify_files",
        lambda self: pytest.fail("portable load opened original contracts"),
    )
    loaded = direct_audit.load_audited_bundle(
        bundle,
        tmp_path / "prepared/cpu-audit.json",
        tok,
        allow_tiny=True,
        expected_manifest_sha256=report["bundle_manifest_sha256"],
        expected_receipt_sha256=report["audit_receipt_sha256"],
    )
    batches = list(direct_bundle.training_batches(loaded.recipe, loaded.inventory, loaded.groups))
    visits = Counter(id(item) for batch in batches for item in batch)
    assert len(batches) == 22 and len(visits) == 33 and set(visits.values()) == {2}
    assert (
        len(
            list(
                direct_bundle.training_batches(
                    loaded.recipe, loaded.inventory, loaded.groups, start_step=10
                )
            )
        )
        == 12
    )
    assert report["model_weights_loaded"] is False
    portable = run_pipeline(
        bundle,
        None,
        action="audit",
        mechanics_only=True,
        audit_receipt=tmp_path / "prepared/cpu-audit.json",
        expected_bundle_sha256=report["bundle_manifest_sha256"],
        expected_audit_receipt_sha256=report["audit_receipt_sha256"],
    )
    assert portable["audit_mode"] == "frozen_cpu_receipt"
    with pytest.raises(ValueError, match="must not receive"):
        run_pipeline(
            bundle,
            None,
            action="audit",
            natural_registry=registry,
            audit_receipt=tmp_path / "prepared/cpu-audit.json",
            expected_audit_receipt_sha256=report["audit_receipt_sha256"],
        )


def test_plan1_cannot_silently_gain_policy_source_and_plan2_cannot_drop_it():
    plan, _, _, _, _, _ = policy_inputs()
    damaged = copy.deepcopy(plan)
    damaged["version"] = VERSION
    with pytest.raises(ValueError):
        validate_plan(damaged)
    damaged = copy.deepcopy(plan)
    damaged["settings"]["natural_sample_quotas"].pop(SOURCE)
    with pytest.raises(ValueError):
        validate_plan(damaged)
    original, *_ = inputs()
    assert original["version"] == VERSION
    assert "reserved_evidence" not in original["selection_policy"]
    assert SOURCE not in original["settings"]["natural_sample_quotas"]


def test_whole_document_overflow_excludes_every_original_hypothesis_before_quotas(monkeypatch):
    plan, cfg, tok, registry, encoding, reserved = policy_inputs()
    actual = direct_corpus.encode_direct_sample
    failed = []

    def overflow_one_whole_document(sample, *args, **kwargs):
        if sample.metadata["source"] == SOURCE and not failed:
            failed.append(sample.metadata["source_example_id"])
            assert len(sample.questions) == 17
            raise ContextLimitError("full contract does not fit; annotated evidence span could fit")
        return actual(sample, *args, **kwargs)

    monkeypatch.setattr(direct_corpus, "encode_direct_sample", overflow_one_whole_document)
    selected, report = direct_corpus.select_corpus(plan, cfg, tok, registry, encoding, reserved)
    assert report["selection"]["removed_by_source"][SOURCE]["context_overflow_whole_sample"] == 1
    assert all(
        s.metadata["source_example_id"] not in failed for rows in selected.values() for s in rows
    )
    assert all(
        len(s.questions) == 17
        for rows in selected.values()
        for s in rows
        if s.metadata["source"] == SOURCE
    )


@pytest.mark.parametrize(
    "state",
    [
        "must be returned except archival legal copies",
        {"excerpt": "MUST be RETURNED except archival legal copies"},
        "기밀정보는반드시안전하게반환해야합니다",
        "앞부분 기밀정보는반드시안전하게반환해야합니다 뒷부분",
    ],
)
def test_private_state_excerpts_match_full_documents_without_schema_overblocking(state):
    private = Sample(state, [], {})
    blocker = ReservedEvidenceBlocker([private])
    candidate = Sample(
        "Entire document: All material must be returned except archival legal copies. 기밀정보는반드시안전하게반환해야합니다",
        [],
        {},
    )
    assert blocker.state_hit(candidate)
    template = contract_registry(1).sample(0)
    unrelated = Sample(
        "Different uniquely sourced private evidence has no common clauses.",
        copy.deepcopy(template.questions),
        {},
    )
    assert not ReservedEvidenceBlocker([unrelated]).state_hit(template)


def test_private_excerpt_blocks_the_entire_transitive_raw_document_group():
    policy = contract_registry(180)
    documents = policy.sources()[SOURCE]
    first = documents[0]
    first.state = "This unique contract clause requires alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu."
    # Source grouping applies to the whole raw inventory, before native fit.
    documents[1].metadata["derived_from"] = first.metadata["source_example_id"]
    private = Sample("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu", [], {})
    selected, report = partition_sources(
        {SOURCE: documents},
        [private],
        fits=lambda _: True,
        quotas={SOURCE: dict.fromkeys(SPLITS, 1)},
        reserved_evidence=ReservedEvidenceBlocker([private]),
    )
    assert report["removed_by_source"][SOURCE]["reserved_or_public_overlap"] == 2
    kept = [s for rows in selected.values() for s in rows]
    assert all(s.metadata["raw_row_index"] not in {0, 1} for s in kept)
    assert report["source_groups"][SOURCE]["largest_group_samples"] == 2


@pytest.mark.parametrize(
    "damage",
    ["fixture_native", "unknown_registry", "no_policy_registry", "changed_raw", "changed_binding"],
)
def test_composite_registry_cannot_bypass_raw_gold_policy(damage):
    _, _, _, registry, _, _ = policy_inputs()
    sample = registry.policy_registry.sample(0)
    if damage == "fixture_native":
        with pytest.raises(ValueError, match="tiny CPU mechanics"):
            direct_bundle._gold_verifier({"train": [sample]}, registry, allow_tiny=False)
    elif damage == "unknown_registry":
        with pytest.raises(ValueError, match="pinned raw-source"):
            direct_bundle._gold_verifier({"train": [sample]}, lambda *args: {}, allow_tiny=True)
    elif damage == "no_policy_registry":
        with pytest.raises(ValueError, match="explicit pinned"):
            direct_bundle._gold_verifier({"train": [sample]}, None, allow_tiny=True)
        with pytest.raises(ValueError, match="explicit pinned"):
            verify_raw_binding(registry.binding)
    else:
        if damage == "changed_raw":
            registry.policy_registry.raw["documents"][0]["text"] += " Changed contract."
        else:
            registry.binding["policy_sources"][SOURCE]["sha256"] = fingerprint("changed")
        with pytest.raises(ValueError):
            verify_raw_binding(registry.binding, registry=registry)


def test_plan_sources_and_composite_membership_are_bound_before_selection_or_publication(tmp_path):
    plan, cfg, tok, registry, encoding, reserved = policy_inputs()
    assert plan_sha256(plan)
    registry.policy_registry = None
    with pytest.raises(ValueError):
        direct_corpus.prepare_corpus(
            tmp_path / "must-not-exist",
            plan,
            cfg,
            tok,
            registry,
            encoding,
            reserved,
            allow_tiny=True,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_actual_random_native_teacher_transport_preserves_all_hypotheses_and_omits_annotation_metadata(
    tmp_path,
):
    from test_teacher_artifacts import exported, observations

    policy = contract_registry(1)
    sample = policy.sample(0)
    sample.metadata["split"] = "train"
    sample.metadata["source_group_id"] = "fixture-group"
    original = copy.deepcopy(sample)
    values = observations(tmp_path, samples=[sample])
    teachers, report = exported(values, gold_verifier=policy)
    _, _, _, direct, paired, reader = values
    assert len(direct) == len(paired) == len(teachers) == len(reader.trace_calls) == 17
    assert report["usage"]["backend_calls"] == 51
    for messages, _ in reader.trace_calls:
        text = json.dumps(messages)
        assert sample.state in text
        assert "annotation_sets" not in text and "spans" not in text
        assert policy.raw["documents"][0]["file_name"] not in text
        assert policy.raw["documents"][0]["url"] not in text
    assert sample == original
    altered = (copy.deepcopy(values[0]), *values[1:])
    altered[0][0].state = "Only the annotated evidence span."
    with pytest.raises(ValueError):
        exported(altered, gold_verifier=policy)

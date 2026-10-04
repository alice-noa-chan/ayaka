"""R12/R13 guards run before fitting; fixtures are entirely CPU reads."""

import copy

import pytest

from ayaka.eval.read_artifact import fingerprint
from ayaka.swift.collect import adapt_jevbench, collect, load_reads
from ayaka.swift.fit import fit_policy
from ayaka.swift.readers import FakeReader
from scripts.swift import select_variant


def bound_roles(tmp_path, *, dev_case=None, dev_state=None, dev_lineage=None):
    result = []
    for split in ("calibration", "dev"):
        rows = []
        for variant in ("min", "rules"):
            items = []
            for kind, labels in (
                ("choice", ["a", "b"]),
                ("noul", ["no", "yes"]),
                ("score", ["0", "1"]),
            ):
                items.append(
                    adapt_jevbench(
                        {
                            "id": f"{split}/{kind}",
                            "source": "corpus",
                            "case_id": dev_case if split == "dev" and dev_case else split,
                            "lineage_ids": [
                                dev_lineage if split == "dev" and dev_lineage else f"global:{split}"
                            ],
                            "state": dev_state if split == "dev" and dev_state else split,
                            "split": split,
                            "public": False,
                            "labels": labels,
                            "expected": labels[0],
                            "question": {"type": kind, "criteria": dict.fromkeys(labels, "option")},
                        }
                    )
                )
            path = tmp_path / f"{split}-{variant}.jsonl"
            reader = FakeReader()
            reader.backend = "hf"
            reader.logprobs_mode = "raw_logits"
            collect(
                iter(items),
                reader,
                path,
                model="fixture",
                revision="a" * 40,
                prompt_variant=variant,
            )
            rows.extend(load_reads([path]))
        result.append(rows)
    return result


@pytest.mark.parametrize("role", ["calibration", "dev"])
@pytest.mark.parametrize("split", ["test", "train", "unknown", None])
def test_selector_rejects_role_mismatch_before_fit(tmp_path, monkeypatch, role, split):
    cal, dev = bound_roles(tmp_path)
    rows = cal if role == "calibration" else dev
    rows[0]["split"] = split
    monkeypatch.setattr(
        select_variant, "fit_policy", lambda *a, **kw: pytest.fail("fitting reached")
    )
    with pytest.raises(ValueError, match="recorded split"):
        select_variant.select_variants(cal, dev, B=1)


def test_selector_rejects_conflicting_metadata(tmp_path):
    cal, _ = bound_roles(tmp_path)
    cal[0]["metadata"] = {"split": "test"}
    with pytest.raises(ValueError, match="split conflict"):
        select_variant.group_reads(cal, "calibration")


@pytest.mark.parametrize("change", ["case", "lineage", "input", "cross_variant_input"])
def test_selector_checks_all_global_overlap_before_fit(tmp_path, monkeypatch, change):
    options = (
        {"dev_case": "calibration"}
        if change == "case"
        else {"dev_lineage": "global:calibration"}
        if change == "lineage"
        else {"dev_state": "calibration"}
    )
    cal, dev = bound_roles(tmp_path, **options)
    if change == "cross_variant_input":
        # Overlap present only in a non-min variant must still be rejected.
        cal = [r for r in cal if r["prompt_variant"] != "rules"] + [
            copy.deepcopy(r) for r in cal if r["prompt_variant"] == "rules"
        ]
        for row in cal[:3]:
            row["binding"]["messages"][1]["content"] += "calibration-only prefix"
            row["binding"]["rendered_input_sha256"] = fingerprint(row["binding"]["messages"])
            row["rendered_input_sha256"] = row["binding"]["rendered_input_sha256"]
            # Synthetic token input must differ too.
            row["binding"]["token_inputs"][0]["input_token_ids"] = [123]
            row["binding"]["token_inputs"][0]["input_token_ids_sha256"] = fingerprint([123])
            row["pass_bindings"][0]["input_token_ids"] = [123]
            row["pass_bindings"][0]["messages"] = row["binding"]["messages"]
            rehash(row)
    monkeypatch.setattr(
        select_variant, "fit_policy", lambda *a, **kw: pytest.fail("fitting reached")
    )
    with pytest.raises(ValueError, match="overlap"):
        select_variant.select_variants(cal, dev, B=1)


def rehash(row):
    binding = row["binding"]
    binding["binding_sha256"] = fingerprint(
        {k: v for k, v in binding.items() if k != "binding_sha256"}
    )
    row["binding_sha256"] = binding["binding_sha256"]
    row["record_sha256"] = fingerprint({k: v for k, v in row.items() if k != "record_sha256"})


def test_fitter_rejects_unbound_before_temperature_fit(tmp_path, monkeypatch):
    from ayaka.swift import fit

    cal, _ = bound_roles(tmp_path)
    cal = [r for r in cal if r["prompt_variant"] == "min"]
    for row in cal:
        row.pop("binding")
    monkeypatch.setattr(fit, "fit_temperature", lambda *a: pytest.fail("temperature fit reached"))
    with pytest.raises(ValueError, match="binding"):
        fit_policy(cal)


def test_unbound_exploration_never_promotes_policy_or_selection(tmp_path):
    cal, dev = bound_roles(tmp_path)
    for row in cal + dev:
        row.pop("binding")
    with pytest.raises(ValueError, match="binding"):
        select_variant.select_variants(cal, dev, B=1)
    result = select_variant.select_variants(cal, dev, B=1, exploratory=True)
    assert result["promotable"] is False
    assert all(p["promotable"] is False for p in result["policies"].values())


@pytest.mark.parametrize(
    "field", ["model", "revision", "tokenizer_revision", "runtime", "readout", "prompt_variant"]
)
def test_fitter_requires_all_nonempty_binding_fields(tmp_path, field):
    cal, _ = bound_roles(tmp_path)
    rows = [r for r in cal if r["prompt_variant"] == "min"]
    rows[0]["binding"][field] = ""
    rehash(rows[0])
    with pytest.raises(ValueError, match="non-empty"):
        fit_policy(rows)


def test_fitter_checks_record_hash_and_diagnostic_mark(tmp_path):
    cal, _ = bound_roles(tmp_path)
    rows = [r for r in cal if r["prompt_variant"] == "min"]
    rows[0]["record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        fit_policy(rows)
    rows[0]["comparison_valid"] = False
    rehash(rows[0])
    with pytest.raises(ValueError, match="diagnostic"):
        fit_policy(rows)
    assert fit_policy(rows, exploratory=True).promotable is False


def test_role_overlap_checks_adaptive_pass_inputs_too():
    def row(role):
        return {
            "id": role,
            "source": "corpus",
            "case_id": role,
            "rendered_input_sha256": fingerprint(role),
            "pass_bindings": [
                {"messages": [{"role": "user", "content": "same adaptive winner prompt"}]}
            ],
        }

    with pytest.raises(ValueError, match="overlap"):
        select_variant.assert_roles_isolated([row("calibration")], [row("dev")])

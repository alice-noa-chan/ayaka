"""Opt-in raw paragraph closure for a future document-independent protocol.

Close the complete supplied inventory before filtering/quota selection. This
module checks context linkage, not human labels, raw-file provenance, public
decontamination, or independence from an unavailable historical inventory.
Existing corpus split keys and concluded Swift experiments are unchanged.
"""

from __future__ import annotations

import copy
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..eval.read_artifact import fingerprint
from .schema import Sample
from .source_groups import connected_groups, identities

VERSION = "ayaka-paragraph-closure-1"
ROLES = frozenset({"train", "router_train", "dev", "calibration", "test"})
KINDS = frozenset({"choice", "noul", "score"})


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"paragraph {name} must be a nonempty string")
    return value


def _normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _context(context, namespace):
    if (
        not isinstance(context, dict)
        or set(context) != {"title", "sentences"}
        or not isinstance(context["title"], (list, tuple))
        or not isinstance(context["sentences"], (list, tuple))
        or not context["title"]
        or len(context["title"]) != len(context["sentences"])
    ):
        raise ValueError("paragraph context requires paired titles and sentence lists")
    rendered, aliases = [], set()
    for title, sentences in zip(context["title"], context["sentences"], strict=True):
        _text(title, "title")
        if not isinstance(sentences, (list, tuple)) or not sentences:
            raise ValueError("paragraph sentences must be a nonempty sequence")
        if any(not isinstance(sentence, str) for sentence in sentences):
            raise ValueError("paragraph sentences must contain strings")
        text = _text("".join(sentences), "text")
        rendered.append(f"{title}: {text}")
        # A stable article namespace spans train/validation and all converters.
        # Text aliases also catch a copied paragraph under a different title.
        aliases.add("paragraph-title/" + fingerprint([namespace, _normalized(title)]))
        aliases.add("paragraph-text/" + fingerprint(_normalized(text)))
    return "\n\n".join(rendered), tuple(sorted(aliases))


def _records(records, namespace):
    _text(namespace, "namespace")
    records = copy.deepcopy(list(records))
    if not records:
        raise ValueError("paragraph inventory must be nonempty")
    seen = set()
    for row in records:
        if not isinstance(row, dict) or set(row) != {"id", "source", "type", "context"}:
            raise ValueError("paragraph records require id, source, type and raw context")
        row_id = _text(row["id"], "row id")
        _text(row["source"], "source")
        if row_id in seen:
            raise ValueError("paragraph inventory repeats a row id")
        seen.add(row_id)
        if row["type"] is not None and (
            not isinstance(row["type"], str) or row["type"] not in KINDS
        ):
            raise ValueError("paragraph type must be typed decision or an unselected bridge")
        _context(row["context"], namespace)
    return sorted(records, key=lambda row: row["id"])


def paragraph_inventory_digest(records, *, namespace):
    """Logical anchor of a complete label-free inventory, independent of order."""
    return fingerprint(
        {"version": VERSION, "namespace": namespace, "records": _records(records, namespace)}
    )


@dataclass(frozen=True, slots=True)
class ParagraphRow:
    id: str
    source: str
    type: str | None
    state: str
    aliases: tuple[str, ...]
    component_id: str


@dataclass(frozen=True, init=False, slots=True)
class ParagraphIndex:
    namespace: str = field(init=False)
    inventory_sha256: str = field(init=False)
    rows: tuple[ParagraphRow, ...] = field(init=False)
    _rows_by_id: Mapping[str, ParagraphRow] = field(init=False, repr=False, compare=False)

    def __init__(self, records, *, namespace, expected_inventory_sha256):
        # Derived groups are never constructor/replace inputs. Even direct
        # construction must rebuild closure from the anchored original data.
        rows, digest = _closed_rows(
            records, namespace=namespace, expected_inventory_sha256=expected_inventory_sha256
        )
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "inventory_sha256", digest)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "_rows_by_id", MappingProxyType({row.id: row for row in rows}))


def build_paragraph_index(records, *, namespace, expected_inventory_sha256):
    """Verify the declared inventory and close it before any row is omitted.

    A type=None record remains a component bridge but cannot be selected as a
    decision. Include unsupported raw questions before conversion/filtering.
    """
    return ParagraphIndex(
        records, namespace=namespace, expected_inventory_sha256=expected_inventory_sha256
    )


def _closed_rows(records, *, namespace, expected_inventory_sha256):
    records = _records(records, namespace)
    digest = fingerprint({"version": VERSION, "namespace": namespace, "records": records})
    if expected_inventory_sha256 != digest:
        raise ValueError("paragraph inventory differs from its declared logical anchor")
    parsed = [_context(row["context"], namespace) for row in records]
    samples = [
        Sample(
            fingerprint(["paragraph-row", namespace, row["id"]]),
            [],
            {"lineage_ids": list(aliases)},
        )
        for row, (_, aliases) in zip(records, parsed, strict=True)
    ]
    indices, groups = connected_groups(samples)
    rows = tuple(
        ParagraphRow(
            row["id"],
            row["source"],
            row["type"],
            state,
            aliases,
            "paragraph-component/" + fingerprint(groups[group]["identities"]),
        )
        for row, (state, aliases), group in zip(records, parsed, indices, strict=True)
    )
    return rows, digest


def bind_paragraph_sample(index, row_id, sample):
    """Copy verified context aliases onto a one-question converted decision.

    Question wording/labels need their separate raw-source verifier. The
    source example id must be the explicit inventory id, not a loop position.
    """
    if not isinstance(index, ParagraphIndex) or not isinstance(sample, Sample):
        raise ValueError("paragraph binding requires its index and converted Sample")
    row = index._rows_by_id.get(row_id) if isinstance(row_id, str) else None
    if row is None or row.type is None:
        raise ValueError("paragraph binding requires a selectable inventory row")
    if (
        sample.state != row.state
        or sample.metadata.get("source") != row.source
        or sample.metadata.get("source_example_id") != row.id
        or len(sample.questions) != 1
        or sample.questions[0].type != row.type
    ):
        raise ValueError("converted paragraph state/id/source/type differs from the raw inventory")
    result = copy.deepcopy(sample)
    result.metadata["lineage_ids"] = sorted(
        identities(sample) | set(row.aliases) | {row.component_id}
    )
    result.metadata["paragraph_binding"] = {
        "version": VERSION,
        "inventory_sha256": index.inventory_sha256,
        "component_id": row.component_id,
        "namespace": index.namespace,
    }
    return result


def audit_paragraph_roles(index, selections, *, minimum_counts):
    """Reject cross-role components and missing predeclared decision minima.

    Selections map roles to row IDs; omitted rows still bridge the full index.
    Every observed source/type/role cell needs an explicit positive minimum.
    No files, raw corpus, model or scoring policy are opened or changed here.
    """
    if (
        not isinstance(index, ParagraphIndex)
        or not isinstance(selections, dict)
        or not selections
        or set(selections) - ROLES
        or not isinstance(minimum_counts, dict)
        or not minimum_counts
    ):
        raise ValueError("paragraph audit requires explicit roles and nonempty declared minima")
    rows = {row.id: row for row in index.rows}
    for cell, minimum in minimum_counts.items():
        if (
            not isinstance(cell, tuple)
            or len(cell) != 3
            or not isinstance(cell[0], str)
            or cell[0] not in selections
            or not isinstance(cell[1], str)
            or not cell[1].strip()
            or not isinstance(cell[2], str)
            or cell[2] not in KINDS
            or type(minimum) is not int
            or minimum < 1
        ):
            raise ValueError("paragraph minima require (role, source, type): positive int")
    if {cell[0] for cell in minimum_counts} != set(selections):
        raise ValueError("paragraph every selected role requires a declared minimum")
    counts, assigned, component_roles = Counter(), set(), defaultdict(set)
    for role, selected in selections.items():
        if not isinstance(selected, (list, tuple)):
            raise ValueError("paragraph selected IDs must be an explicit sequence")
        for row_id in selected:
            if not isinstance(row_id, str) or row_id not in rows or row_id in assigned:
                raise ValueError("paragraph selected IDs must be unique existing inventory IDs")
            row = rows[row_id]
            if row.type is None:
                raise ValueError("paragraph bridge cannot be selected as a decision")
            assigned.add(row_id)
            counts[role, row.source, row.type] += 1
            component_roles[row.component_id].add(role)
    if any(len(roles) > 1 for roles in component_roles.values()):
        raise ValueError("selected roles share a transitive paragraph component")
    if set(counts) - set(minimum_counts):
        raise ValueError("paragraph observed source/type/role lacks a declared minimum")
    if any(counts[cell] < minimum for cell, minimum in minimum_counts.items()):
        raise ValueError("paragraph selection falls below a declared minimum")
    return {
        "version": VERSION,
        "inventory_sha256": index.inventory_sha256,
        "namespace": index.namespace,
        "inventory_rows": len(rows),
        "inventory_components": len({row.component_id for row in index.rows}),
        "selected_decisions": len(assigned),
        "selected_components": len(component_roles),
        "counts": {"/".join(cell): count for cell, count in sorted(counts.items())},
        "minimum_counts": {"/".join(cell): n for cell, n in sorted(minimum_counts.items())},
        "closure_before_selection": True,
        "scope": "supplied paragraph contexts and selected-ID mechanics only",
        "label_provenance_attested": False,
        "historical_or_public_decontamination_attested": False,
        "promotable": False,
    }

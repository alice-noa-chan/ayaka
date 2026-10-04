"""Transitive evidence/lineage groups, independent of annotations and row order."""

import hashlib
import json
import unicodedata
from collections import defaultdict


def evidence(sample):
    state = sample.state
    if isinstance(state, dict) and "prompt" in state:
        state = state["prompt"]
    return unicodedata.normalize(
        "NFKC", json.dumps(state, ensure_ascii=False, sort_keys=True)
    ).casefold()


def group_evidence(sample):
    """Whitespace-normalized evidence; legacy raw primary IDs remain unchanged."""

    def normalize(value):
        if isinstance(value, str):
            return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    state = sample.state
    if isinstance(state, dict) and "prompt" in state:
        state = state["prompt"]
    state = normalize(state)
    if "media" in sample.metadata:
        state = {"state": state, "media": sample.metadata["media"]}
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def identities(sample):
    """References can name either a parent example or a primary lineage."""
    result = set()
    for key in (
        "source_lineage",
        "source_example_id",
        "lineage_ids",
        "derived_from",
        "translation_of",
    ):
        value = sample.metadata.get(key)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        if key in {"source_lineage", "source_example_id"} and not isinstance(value, str):
            raise ValueError(f"source group {key} must be a nonempty string")
        if key == "lineage_ids" and not isinstance(value, (list, tuple)):
            raise ValueError("source group lineage_ids must be a sequence")
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError(f"source group {key} contains an invalid identity")
        result.update(values)
    return result


def connected_groups(samples):
    """Return row group indices and deterministic keys after closing all aliases.

    A group with one primary lineage retains that lineage as its split key.
    Human prompt variants and translated siblings share the complete closure.
    """
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    owners = {}
    row_ids = []
    for index, sample in enumerate(samples):
        refs = identities(sample)
        row_ids.append(refs)
        for node in [
            ("evidence", group_evidence(sample)),
            *(("identity", value) for value in refs),
        ]:
            previous = owners.setdefault(node, index)
            union(index, previous)
    rows = [find(index) for index in range(len(samples))]
    members = defaultdict(list)
    for index, group in enumerate(rows):
        members[group].append(index)
    groups = {}
    for group, indices in members.items():
        lineages = {samples[i].metadata.get("source_lineage") for i in indices} - {None}
        key = min(lineages) if lineages else min(group_evidence(samples[i]) for i in indices)
        groups[group] = {
            "key": key,
            "id": "natural-group/" + hashlib.sha256(key.encode()).hexdigest(),
            "identities": sorted(set().union(*(row_ids[i] for i in indices))),
        }
    return rows, groups

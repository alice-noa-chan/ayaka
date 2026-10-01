"""Supervised proposal CE in a separate context from the typed decision loss."""

import json
from dataclasses import replace

import torch

from ..candidates import policy_for, proposal_messages, validate_proposals
from ..evidence_generation import chat_ids
from .batching import sample_to_items
from .reasoning import supervised_forward, trace_ce


def proposal_items(sample, tok, cfg):
    annotation = sample.metadata.get("proposal_supervision")
    if not isinstance(annotation, dict) or annotation.get("validator") != "finite-domain-v1":
        raise ValueError("proposal SFT requires finite-domain validator provenance")
    from ..data.candidate_v2 import finite_partition_audit

    audit = finite_partition_audit(
        annotation["universe"], annotation["memberships"], annotation.get("parent")
    )
    if audit["overlap_pairs"] or audit["outside_parent"] or audit["equivalent_pairs"]:
        raise ValueError("proposal supervision contains an invalid partition")
    question = annotation["question"]
    policy = policy_for(question)
    clean = {k: v for k, v in question.items() if k != "candidate_generation"}
    text = json.dumps(annotation["rows"], ensure_ascii=False, separators=(",", ":"))
    rows = validate_proposals(text, policy, clean.get("criteria", {}))
    if {r["id"] for r in rows} != set(annotation["memberships"]):
        raise ValueError("proposal ids do not match validated membership annotations")
    prompt = chat_ids(tok, proposal_messages(sample.state, clean, policy))
    labels = tok.encode(text)
    eos = getattr(getattr(tok, "hf", None), "eos_token_id", getattr(tok, "eos_id", None))
    if eos is not None:
        labels.append(eos)
    if len(labels) > policy["max_tokens"] or len(prompt) + len(labels) > cfg.max_seq_len:
        raise ValueError("complete proposal supervision does not fit; do not truncate it")
    items = sample_to_items(sample, tok, cfg)
    if len(items) != 1:
        raise ValueError("proposal supervision needs exactly one independent readout question")
    return [
        replace(
            items[0],
            proposal_input_ids=prompt + labels,
            proposal_labels=labels,
            proposal_positions=list(range(len(prompt) - 1, len(prompt) + len(labels) - 1)),
        )
    ]


def proposal_ce(model, items, pad_id, chunk_tokens=128, *, prune=True):
    """One masked proposal batch; context remains separate from teacher readouts."""
    selected = [item for item in items if item.proposal_labels]
    if not selected:
        return model.gate.sum() * 0
    device = model.embed_weight().device
    width = max(len(item.proposal_input_ids) for item in selected)
    ids = torch.full((len(selected), width), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    proxies = []
    for row, item in enumerate(selected):
        length = len(item.proposal_input_ids)
        ids[row, :length] = torch.tensor(item.proposal_input_ids, device=device)
        mask[row, :length] = 1
        proxies.append(
            replace(
                item,
                reasoning_positions=item.proposal_positions,
                reasoning_labels=item.proposal_labels,
            )
        )
    hidden, _, proxies = supervised_forward(model, ids, mask, proxies, prune=prune)
    return trace_ce(model, hidden, proxies, chunk_tokens) * (len(selected) / len(items))

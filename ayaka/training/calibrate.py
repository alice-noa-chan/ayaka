"""Per-primitive scalar temperature fitting (sec 40.2/Stage 6, A8).

After decision training, one scalar T per primitive is fit on a
held-out split by minimizing NLL — vector scaling is rejected
(dynamic ontology + permutation equivariance, sec 40.2).
"""

from __future__ import annotations

import torch

from ..model.model import CHOICE, NOUL, SCORE, ElectraDecisionModel

PRIMITIVE_NAMES = {NOUL: "noul", CHOICE: "choice", SCORE: "score"}


def fit_temperatures(
    logits_by_primitive: dict[int, torch.Tensor],
    targets_by_primitive: dict[int, torch.Tensor],
    cand_cu_by_primitive: dict[int, torch.Tensor],
    iters: int = 200,
) -> dict[int, float]:
    """Fit scalar T per primitive via LBFGS on held-out NLL.

    Each dict maps primitive index -> (flat logits, flat targets,
    cand_cu) restricted to questions of that primitive.
    """
    temps: dict[int, float] = {}
    for prim, logits in logits_by_primitive.items():
        temps[prim] = _fit_one(
            logits, targets_by_primitive[prim], cand_cu_by_primitive[prim], iters
        )
    return temps


def _fit_one(logits: torch.Tensor, targets: torch.Tensor, cu: torch.Tensor, iters: int) -> float:
    log_t = torch.zeros(1, requires_grad=True, device=logits.device)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = _scaled_nll(logits / log_t.exp(), targets, cu)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def _scaled_nll(logits: torch.Tensor, targets: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    from ..model.pointer import ragged_log_softmax

    logp = ragged_log_softmax(logits, cu)
    seg = torch.repeat_interleave(
        torch.arange(cu.numel() - 1, device=logits.device), (cu[1:] - cu[:-1]).long()
    )
    per_q = torch.zeros(cu.numel() - 1, device=logits.device)
    per_q.index_add_(0, seg, -(targets * logp))
    return per_q.mean()


def apply_temperatures(model: ElectraDecisionModel, temps: dict[int, float]) -> None:
    """Write fitted temperatures into the model buffer (checkpoint
    metadata carries them, Stage 6)."""
    with torch.no_grad():
        for prim, t in temps.items():
            model.temperature[prim] = t


@torch.no_grad()
def collect_logits_by_primitive(
    model: ElectraDecisionModel, batches: list
) -> dict[int, dict[str, torch.Tensor]]:
    """Run held-out batches and bucket logits/targets by primitive."""
    model.eval()
    out: dict[int, dict[str, list]] = {}
    for batch in batches:
        res = model(**batch.inputs)
        prim = batch.inputs["primitive_index"]
        for qi in range(res.cand_cu.numel() - 1):
            s, e = int(res.cand_cu[qi]), int(res.cand_cu[qi + 1])
            p = int(prim[qi])
            slot = out.setdefault(p, {"logits": [], "targets": [], "cu": [0]})
            slot["logits"].append(res.logits[s:e])
            slot["targets"].append(batch.targets[s:e])
            slot["cu"].append(slot["cu"][-1] + (e - s))
    flat = {}
    for p, slot in out.items():
        flat[p] = {
            "logits": torch.cat(slot["logits"]),
            "targets": torch.cat(slot["targets"]),
            "cu": torch.tensor(slot["cu"]),
        }
    return flat

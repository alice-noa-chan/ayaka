"""TypeSafe-compatible decision server (``POST /v1/systemone``).

Request::

    {"state": ..., "model": "...", "questions": {
        "<name>": {"type": "noul",   "instructions": "...", "criteria": {"false": "...", "true": "..."}},
        "<name>": {"type": "choice", "instructions": "...", "criteria": {"<label>": "<description>", ...}},
        "<name>": {"type": "score",  "instructions": "...", "criteria": ["<level 0>", "<level 1>", ...]}}}

Response::

    {"model": "...", "answers": {
        "<name>": {"type": "noul",   "noul": P(true)},
        "<name>": {"type": "choice", "choice": "<label>", "probabilities": {"<label>": p},
                   "confidence": c},
        "<name>": {"type": "score",  "score": E[level], "probabilities": {"0": p, ...},
                   "confidence": c, "legend": {"0": "<level 0>", ...}}},
     "usage": {"input_tokens": n, "output_tokens": 0}}

All questions of a request share one state encoding (prefix KV cache)
and are evaluated in one batched forward, isolated from each other.

    python -m ayaka.serve --model exports/electra-small --device cuda --port 8000
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from threading import Lock

from .http_transport import ServiceConfig, add_service_arguments, config_from_args, make_api_handler
from .jev_api import ModelCatalog, ValidationError, build_answer, render_content
from .jev_api import parse_question as parse_jev_question
from .primitives import Decision, QuestionSpec

BadRequest = ValidationError


def parse_question(q: dict, *, field="questions") -> tuple[QuestionSpec, list[str]]:
    """TypeSafe question -> (QuestionSpec, exact answer labels)."""
    parsed = parse_jev_question(q, field=field)
    ordinals = None
    if parsed.type == "score":
        # Preserve v1's insertion order and historical Score-map ordinals.
        ordinals = [int(k) if k.lstrip("-").isdigit() else i for i, k in enumerate(parsed.labels)]
    return QuestionSpec(
        parsed.type, parsed.instruction, parsed.descriptions, ordinals
    ), parsed.labels


def answer(spec: QuestionSpec, labels: list[str], probs: list[float], *, legend=None) -> dict:
    dist = {lb: float(p) for lb, p in zip(labels, probs, strict=True)}
    if legend is None and spec.type == "score":
        legend = dict(zip(labels, spec.candidates, strict=True))
    return build_answer(spec.type, dist, legend=legend, ordinals=spec.ordinals)


class DecisionService:
    def __init__(
        self,
        decision: Decision,
        model_name: str,
        *,
        model_id: str | None = None,
        model_description: str = ModelCatalog.description,
        model_release_date: str = ModelCatalog.release_date,
    ):
        self.decision = decision
        self.model_name = model_name
        self.catalog = ModelCatalog(
            model_name, model_id or f"{model_name}-1.0.0", model_description, model_release_date
        )
        self.lock = Lock()  # one forward at a time: predictable latency, no VRAM spikes

    def handle(self, body: dict) -> dict:
        if not isinstance(body, dict) or "questions" not in body:
            raise BadRequest("body needs 'state' and 'questions'", "body")
        qs = body["questions"]
        if not isinstance(qs, dict) or not qs:
            raise BadRequest("'questions' must be a non-empty object")
        if not isinstance(body.get("ayaka", {}), dict):
            raise BadRequest("ayaka must be an object", "ayaka")
        # Reserve the namespace without enabling any v2 request controls.
        model_id = self.catalog.resolve(body.get("model"))
        names = list(qs)
        if any(not isinstance(n, str) for n in names):
            raise BadRequest("question names must be strings")
        parsed = [parse_question(qs[n], field=f"questions.{n}") for n in names]
        state = body.get("state", "")
        render_content(state, "state")
        with self.lock:
            results = self.decision.decide(state, [p[0] for p in parsed])
        answers = {
            n: answer(
                spec,
                labels,
                r.probs,
                legend=parse_jev_question(qs[n]).legend if spec.type == "score" else None,
            )
            for n, (spec, labels), r in zip(names, parsed, results, strict=True)
        }
        generated = [(r.extras.get("evidence") or {}).get("worked_steps", "") for r in results]
        return {
            "model": model_id,
            "answers": answers,
            "usage": {
                "input_tokens": self._count_tokens(state, parsed),
                "output_tokens": sum(len(self.decision.tok.encode(g)) for g in generated if g),
            },
        }

    def _count_tokens(self, state, parsed) -> int:
        from .collate import encode_decision

        prefix, items = encode_decision(
            state, [p[0].view() for p in parsed], self.decision.tok, self.decision.max_seq_len
        )
        return len(prefix) + sum(len(it.rendered.suffix_ids) for it in items)


def make_handler(service: DecisionService, *, config: ServiceConfig | None = None):
    return make_api_handler(service, config=config)


def serve(
    decision: Decision,
    model_name: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    config: ServiceConfig | None = None,
    model_id: str | None = None,
    model_description: str = ModelCatalog.description,
    model_release_date: str = ModelCatalog.release_date,
) -> ThreadingHTTPServer:
    service = DecisionService(
        decision,
        model_name,
        model_id=model_id,
        model_description=model_description,
        model_release_date=model_release_date,
    )
    return ThreadingHTTPServer((host, port), make_handler(service, config=config))


def main(argv: list[str] | None = None) -> None:
    import argparse
    import os

    import torch

    from .export import load_exported

    ap = argparse.ArgumentParser(description="TypeSafe-compatible Electra server")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="export folder (ayaka.export)")
    src.add_argument(
        "--ckpt", help="checkpoint dir or Hub repo <user>/<name> (adapter kept unmerged)"
    )
    ap.add_argument("--revision", default=None, help="Hub revision for --ckpt")
    ap.add_argument(
        "--reasoning",
        action="store_true",
        help="gated worked-steps route for low-confidence calculation questions",
    )
    ap.add_argument(
        "--reasoner-adapter",
        default="off",
        choices=["off", "on"],
        help="off: worked steps from the base model (needs --ckpt); on: with the decision "
        "LoRA, which also works on merged exports",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="float32 tracks the reference probabilities more closely (bf16 moved some "
        "near-tie noul answers by up to 0.2 on E2B) at ~1.3-1.5x CPU latency",
    )
    ap.add_argument(
        "--linear-mode",
        default="auto",
        choices=["auto", "dequant", "mixed", "dynamic"],
        help="int8 exports on CPU: auto keeps bf16 accuracy; mixed is ~30%% faster, less accurate",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="prompt token budget per question (default: the config's serve_max_seq_len)",
    )
    add_service_arguments(ap)
    args = ap.parse_args(argv)
    config = config_from_args(args)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.reasoning and args.reasoner_adapter == "off" and not args.ckpt:
        ap.error("--reasoning with --reasoner-adapter off needs --ckpt (exports are merged)")
    if args.ckpt:
        from .checkpoint import load_checkpoint, resolve_checkpoint
        from .tokenization import HFTokenizer

        args.ckpt = resolve_checkpoint(args.ckpt, args.revision)
        model = load_checkpoint(
            args.ckpt, device=args.device, dtype=getattr(torch, args.dtype), merge=False
        ).eval()
        tok = HFTokenizer.for_config(model.cfg)
        name = os.path.basename(os.path.normpath(os.path.dirname(os.path.abspath(args.ckpt))))
    else:
        model, tok = load_exported(
            args.model,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            linear_mode=args.linear_mode,
        )
        name = os.path.basename(os.path.normpath(args.model))
    if args.reasoning:
        from .evidence_pipeline import reasoning_decision

        decision = reasoning_decision(
            model,
            tok,
            max_seq_len=args.max_seq_len or None,
            reasoner_adapter=args.reasoner_adapter,
        )
    else:
        decision = Decision(model, tok, max_seq_len=args.max_seq_len or None)
    decision.decide("warm-up", [QuestionSpec("noul", "Is this a warm-up?", ["no", "yes"])])
    httpd = serve(
        decision,
        name,
        args.host,
        args.port,
        config=config,
        model_id=args.model_id,
        model_description=args.model_description,
        model_release_date=args.model_release_date,
    )
    print(
        f"[serve] {name} on {args.device} at http://{args.host}:{args.port}/v1/systemone",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()

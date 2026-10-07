"""TypeSafe-compatible decision server (``POST /v1/systemone``).

Request::

    {"state": ..., "model": "...", "questions": {
        "<name>": {"type": "noul",   "instructions": "...", "criteria": {"false": "...", "true": "..."}},
        "<name>": {"type": "choice", "instructions": "...", "criteria": {"<label>": "<description>", ...}},
        "<name>": {"type": "score",  "instructions": "...", "criteria": ["<level 0>", "<level 1>", ...]}}}

Response::

    {"model": "...", "answers": {
        "<name>": {"type": "noul",   "noul": P(true)},
        "<name>": {"type": "choice", "choice": "<label>", "probabilities": {"<label>": p}},
        "<name>": {"type": "score",  "score": E[level], "probabilities": {"0": p, ...}}},
     "usage": {"input_tokens": n, "output_tokens": 0}}

All questions of a request share one state encoding (prefix KV cache)
and are evaluated in one batched forward, isolated from each other.

    python -m ayaka.serve --model exports/electra-small --device cuda --port 8000
"""

from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from threading import Lock

from .http_transport import (
    RequestCache,
    ServiceConfig,
    add_service_arguments,
    config_from_args,
    make_api_handler,
    server_for,
)
from .jev_api import (
    ModelCatalog,
    ValidationError,
    build_answer,
    complete_answers,
    normalize_request,
)
from .jev_api import parse_question as parse_jev_question
from .primitives import Decision, QuestionSpec
from .reasoning import resolve_settings


class BadRequest(ValidationError):
    pass


def parse_question(q: dict, *, native_score=False) -> tuple[QuestionSpec, list[str]]:
    try:
        parsed = parse_jev_question(q)
    except ValidationError as exc:
        raise BadRequest(str(exc), exc.field) from exc
    ordinals = None
    if parsed.type == "score":
        if native_score:
            try:
                ordinals = [int(k) for k in parsed.labels]
            except ValueError as exc:
                raise BadRequest(
                    "Swift Score criteria require integer level keys", "criteria"
                ) from exc
            if len(set(ordinals)) != len(ordinals):
                raise BadRequest("Swift Score levels must be numerically unique", "criteria")
        else:
            ordinals = [
                int(k) if k.lstrip("-").isdigit() else i for i, k in enumerate(parsed.labels)
            ]
    return QuestionSpec(
        parsed.type,
        parsed.instruction,
        parsed.descriptions,
        ordinals=ordinals,
        candidate_ids=parsed.labels,
    ), parsed.labels


def answer(spec: QuestionSpec, labels: list[str], probs: list[float]) -> dict:
    return build_answer(
        spec.type,
        dict(zip(labels, probs, strict=True)),
        legend=dict(zip(labels, spec.candidates, strict=True)),
        ordinals=spec.ordinals,
    )


class DecisionService:
    def __init__(
        self,
        decision: Decision,
        model_name: str,
        reasoning_defaults=None,
        *,
        model_id=None,
        model_description=ModelCatalog.description,
        model_release_date=ModelCatalog.release_date,
    ):
        self.catalog = ModelCatalog(model_name, model_id, model_description, model_release_date)
        self.decision = decision
        self.model_name = model_name
        self.reasoning_defaults = reasoning_defaults
        self.lock = Lock()  # one forward at a time: predictable latency, no VRAM spikes

    def handle(self, body: dict, *, candidate_partition=False) -> dict:
        try:
            body = normalize_request(body, max_questions=None)
            self.catalog.resolve(body.get("model"))
        except ValidationError as exc:
            raise BadRequest(str(exc), exc.field) from exc
        response = self._handle(body, candidate_partition=candidate_partition)
        response["model"] = self.catalog.actual_id
        calibration = "unfitted"
        if (
            getattr(self.decision, "calibration", None) is not None
            or getattr(getattr(self.decision, "model", None), "calibration", None) is not None
        ):
            calibration = "fitted"
        if candidate_partition:
            calibration = "unvalidated_generated_partition"
        return complete_answers(response, body, calibration=calibration, include_diagnostics=False)

    def _handle(self, body: dict, *, candidate_partition=False) -> dict:
        if not isinstance(body, dict) or "questions" not in body:
            raise BadRequest("body needs 'state' and 'questions'")
        qs = body["questions"]
        if not isinstance(qs, dict) or not qs:
            raise BadRequest("'questions' must be a non-empty object")
        names = list(qs)
        if any(not isinstance(qs[n], dict) for n in names):
            raise BadRequest("question must be an object")
        if any("candidate_generation" in qs[n] for n in names):
            from .candidates import handle_candidates

            return handle_candidates(self, body)
        contract = getattr(getattr(self.decision, "model", None), "input_contract", None)
        native_score = (
            contract is not None and contract["input_encoding"]["encoder"] == "swift_canonical"
        )
        parsed = [parse_question(qs[n], native_score=native_score) for n in names]
        options = body.get("options", {})
        if not isinstance(options, dict):
            raise BadRequest("options must be an object")
        explicit = "reasoning" in options or any("reasoning" in qs[n] for n in names)
        cfg = getattr(getattr(self.decision, "model", None), "cfg", None)
        checkpoint = getattr(cfg, "reasoning_defaults", {})
        if getattr(cfg, "version", 1) < 2:
            checkpoint = {"mode": "off", **checkpoint}
        try:
            settings = [
                resolve_settings(
                    checkpoint,
                    self.reasoning_defaults,
                    options.get("reasoning"),
                    qs[n].get("reasoning"),
                )
                for n in names
            ]
            if explicit:
                if ("reasoning" in options and options["reasoning"] is None) or any(
                    "reasoning" in qs[n] and qs[n]["reasoning"] is None for n in names
                ):
                    raise ValueError("reasoning settings must be an object")
                if not getattr(self.decision, "supports_reasoning", False) and any(
                    s.budget for s in settings
                ):
                    raise ValueError(
                        "this server does not support reasoning; enable a reasoning backend"
                    )
        except ValueError as exc:
            raise BadRequest(str(exc)) from exc
        state = body.get("state", "")
        decision = self.decision
        if candidate_partition:
            decision = getattr(decision, "text", decision).for_unvalidated_partition()
        if "media" in body:
            if not getattr(self.decision, "supports_images", False):
                raise BadRequest(
                    "this server has no image backend; start with --images and --ckpt",
                    "ayaka.media",
                )
            try:
                state = self.decision.prepare_media(state, body["media"])
            except ValueError as exc:
                raise BadRequest(str(exc), "ayaka.media") from exc
        with self.lock:
            if getattr(self.decision, "supports_reasoning", False):
                try:
                    results = decision.decide(state, [p[0] for p in parsed], reasoning=settings)
                except ValueError as exc:
                    if "media" in body:
                        raise BadRequest(str(exc)) from exc
                    raise
            else:
                direct = (
                    getattr(self.decision, "original", self.decision) if explicit else self.decision
                )
                results = direct.decide(state, [p[0] for p in parsed])
        answers = {
            n: answer(spec, labels, r.probs)
            for n, (spec, labels), r in zip(names, parsed, results, strict=True)
        }
        generated = [(r.extras.get("evidence") or {}).get("worked_steps", "") for r in results]
        diagnostics = {
            n: r.extras["reasoning"]
            for n, r in zip(names, results, strict=True)
            if "reasoning" in r.extras
        }
        reasoning_tokens = sum(d["generated_tokens"] for d in diagnostics.values())
        return {
            "model": body.get("model") or self.model_name,
            "answers": answers,
            "usage": {
                "input_tokens": sum(d["input_tokens"] for d in diagnostics.values())
                if diagnostics
                else self._count_tokens(state, parsed),
                "output_tokens": reasoning_tokens
                if diagnostics
                else sum(len(self.decision.tok.encode(g)) for g in generated if g),
                "reasoning_tokens": reasoning_tokens,
            },
            **({"reasoning": diagnostics} if diagnostics else {}),
        }

    def _count_tokens(self, state, parsed) -> int:
        if hasattr(self.decision, "input_counts"):
            return sum(self.decision.input_counts(state, [p[0] for p in parsed]))
        from .collate import encode_decision

        prefix, items = encode_decision(
            state, [p[0].view() for p in parsed], self.decision.tok, self.decision.max_seq_len
        )
        return len(prefix) + sum(len(it.rendered.suffix_ids) for it in items)


def make_handler(service: DecisionService, *, config: ServiceConfig | None = None):
    return make_api_handler(
        service, config=config, cache=RequestCache(), validation_errors=(BadRequest,)
    )


def serve(
    decision: Decision,
    model_name: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    config: ServiceConfig | None = None,
    **service_options,
) -> ThreadingHTTPServer:
    config = config or ServiceConfig()
    config.validate_bind(host)
    return server_for(
        (host, port),
        make_handler(DecisionService(decision, model_name, **service_options), config=config),
    )


def main(argv: list[str] | None = None) -> None:
    import argparse
    import os

    import torch

    from .export import load_exported

    ap = argparse.ArgumentParser(description="TypeSafe-compatible Ayaka server")
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
    ap.add_argument("--reasoning-mode", choices=["off", "auto", "on"])
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    ap.add_argument("--max-reasoning-tokens", type=int)
    ap.add_argument("--reasoning-router", help="promoted v2 router JSON")
    ap.add_argument("--reasoning-calibration", help="v2 path-temperature JSON")
    ap.add_argument("--image-calibration", help="model-bound native image temperature artifact")
    ap.add_argument("--image-router", help="model-bound, dev-promoted native image router")
    ap.add_argument(
        "--images", action="store_true", help="native Gemma 4 image inputs (--ckpt only)"
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
    try:
        config = config_from_args(args)
        config.validate_bind(args.host)
    except ValueError as exc:
        ap.error(str(exc))
    overrides = {
        k: v
        for k, v in {
            "mode": args.reasoning_mode,
            "effort": args.reasoning_effort,
            "max_tokens": args.max_reasoning_tokens,
        }.items()
        if v is not None
    }
    try:
        resolve_settings(server=overrides)
    except ValueError as exc:
        ap.error(str(exc))
    if args.reasoning and (overrides or args.reasoning_router or args.reasoning_calibration):
        ap.error("use either legacy --reasoning or v2 reasoning flags")
    if args.images and (not args.ckpt or args.reasoning):
        ap.error("--images requires --ckpt and the v2 reasoning interface")
    if (args.image_calibration or args.image_router) and not args.images:
        ap.error("--image-calibration/--image-router require --images")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.reasoning and args.reasoner_adapter == "off" and not args.ckpt:
        ap.error("--reasoning with --reasoner-adapter off needs --ckpt (exports are merged)")
    image_decision = None
    if args.images:
        from .checkpoint import resolve_checkpoint
        from .multimodal import load_image_decision
        from .routing import BenefitRouter
        from .training.path_calibration import PathCalibration
        from .training.scoped_calibration import ScopedCalibration
        from .training.scoped_router import ScopedRouter

        args.ckpt = resolve_checkpoint(args.ckpt, args.revision)
        router = BenefitRouter.load(args.reasoning_router) if args.reasoning_router else None
        calibration = None
        if args.reasoning_calibration:
            with open(args.reasoning_calibration, encoding="utf-8") as f:
                calibration = PathCalibration(json.load(f))
        image_decision = load_image_decision(
            args.ckpt,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            max_seq_len=args.max_seq_len or None,
            router=router,
            calibration=calibration,
            image_calibration=ScopedCalibration.load(args.image_calibration)
            if args.image_calibration
            else None,
            image_router=ScopedRouter.load(args.image_router) if args.image_router else None,
        )
        model, tok = image_decision.model, image_decision.tok
        name = os.path.basename(os.path.normpath(args.ckpt))
    elif args.ckpt:
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
    if image_decision is not None:
        decision = image_decision
    elif args.reasoning:
        from .evidence_pipeline import reasoning_decision

        decision = reasoning_decision(
            model,
            tok,
            max_seq_len=args.max_seq_len or None,
            reasoner_adapter=args.reasoner_adapter,
        )
    elif overrides or model.cfg.version >= 2 or args.reasoning_router or args.reasoning_calibration:
        from .reasoning_pipeline import controlled_decision
        from .routing import BenefitRouter
        from .training.path_calibration import PathCalibration

        router = BenefitRouter.load(args.reasoning_router) if args.reasoning_router else None
        calibration = None
        if args.reasoning_calibration:
            with open(args.reasoning_calibration, encoding="utf-8") as f:
                calibration = PathCalibration(json.load(f))
        decision = controlled_decision(
            model, tok, max_seq_len=args.max_seq_len or None, router=router, calibration=calibration
        )
        if model.cfg.version < 2:
            overrides = {"mode": "auto", **overrides}
    else:
        decision = Decision(model, tok, max_seq_len=args.max_seq_len or None)
    decision.decide("warm-up", [QuestionSpec("noul", "Is this a warm-up?", ["no", "yes"])])
    service = DecisionService(
        decision,
        name,
        overrides or None,
        model_id=args.model_id,
        model_description=args.model_description,
        model_release_date=args.model_release_date,
    )
    httpd = server_for((args.host, args.port), make_handler(service, config=config))
    display_host = f"[{args.host}]" if ":" in args.host else args.host
    print(
        f"[serve] {name} on {args.device} at http://{display_host}:{args.port}/v1/systemone",
        flush=True,
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

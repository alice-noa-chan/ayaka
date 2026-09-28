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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from .primitives import Decision, QuestionSpec


class BadRequest(ValueError):
    pass


def parse_question(q: dict) -> tuple[QuestionSpec, list[str]]:
    """TypeSafe question -> (QuestionSpec, exact answer labels)."""
    if not isinstance(q, dict):
        raise BadRequest("question must be an object")
    qtype = q.get("type")
    instruction = q.get("instructions") or q.get("instruction") or ""
    crit = q.get("criteria")
    if qtype == "noul":
        crit = crit if isinstance(crit, dict) else {}
        return QuestionSpec(
            "noul", instruction, [str(crit.get("false", "false")), str(crit.get("true", "true"))]
        ), ["false", "true"]
    if qtype == "choice":
        if isinstance(crit, dict) and len(crit) >= 2:
            labels = [str(k) for k in crit]
            descs = [str(v) if v not in (None, "") else str(k) for k, v in crit.items()]
        elif isinstance(crit, list) and len(crit) >= 2:
            labels = descs = [str(x) for x in crit]
        else:
            raise BadRequest("choice needs criteria with at least two options")
        if len(set(labels)) != len(labels):
            raise BadRequest("choice option labels must be unique")
        return QuestionSpec("choice", instruction, descs), labels
    if qtype == "score":
        if isinstance(crit, list) and len(crit) >= 2:
            labels = [str(i) for i in range(len(crit))]
            return QuestionSpec(
                "score", instruction, [str(c) for c in crit], ordinals=list(range(len(crit)))
            ), labels
        if isinstance(crit, dict) and len(crit) >= 2:
            keys = list(crit)
            ords = [int(k) if str(k).lstrip("-").isdigit() else i for i, k in enumerate(keys)]
            return QuestionSpec(
                "score", instruction, [str(crit[k]) for k in keys], ordinals=ords
            ), [str(k) for k in keys]
        raise BadRequest("score needs criteria with at least two levels")
    raise BadRequest(f"unknown question type: {qtype!r}")


def answer(spec: QuestionSpec, labels: list[str], probs: list[float]) -> dict:
    if spec.type == "noul":
        return {"type": "noul", "noul": float(probs[1])}
    dist = {lb: float(p) for lb, p in zip(labels, probs, strict=True)}
    if spec.type == "choice":
        return {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist}
    expected = sum(o * p for o, p in zip(spec.ordinals, probs, strict=True))
    return {"type": "score", "score": float(expected), "probabilities": dist}


class DecisionService:
    def __init__(self, decision: Decision, model_name: str):
        self.decision = decision
        self.model_name = model_name
        self.lock = Lock()  # one forward at a time: predictable latency, no VRAM spikes

    def handle(self, body: dict) -> dict:
        if not isinstance(body, dict) or "questions" not in body:
            raise BadRequest("body needs 'state' and 'questions'")
        qs = body["questions"]
        if not isinstance(qs, dict) or not qs:
            raise BadRequest("'questions' must be a non-empty object")
        names = list(qs)
        parsed = [parse_question(qs[n]) for n in names]
        state = body.get("state", "")
        with self.lock:
            results = self.decision.decide(state, [p[0] for p in parsed])
        answers = {
            n: answer(spec, labels, r.probs)
            for n, (spec, labels), r in zip(names, parsed, results, strict=True)
        }
        generated = [(r.extras.get("evidence") or {}).get("worked_steps", "") for r in results]
        return {
            "model": body.get("model") or self.model_name,
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


def make_handler(service: DecisionService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("/health", "/v1/health"):
                self._send(200, {"status": "ok", "model": service.model_name})
            elif self.path.rstrip("/") == "/v1/models":
                self._send(200, {"data": [{"id": service.model_name}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                t0 = time.perf_counter()
                out = service.handle(body)
                out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                self._send(200, out)
            except (BadRequest, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # never leak a traceback to the caller
                self._send(500, {"error": f"internal error: {type(e).__name__}"})

        def log_message(self, *args):  # keep stdout for our own logs
            pass

    return Handler


def serve(
    decision: Decision, model_name: str, host: str = "0.0.0.0", port: int = 8000
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(DecisionService(decision, model_name)))


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
    args = ap.parse_args(argv)
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
    httpd = serve(decision, name, args.host, args.port)
    print(
        f"[serve] {name} on {args.device} at http://{args.host}:{args.port}/v1/systemone",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()

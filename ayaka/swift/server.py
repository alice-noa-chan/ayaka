"""Stdlib TypeSafe server for the Swift letter readout."""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ayaka.reasoning import resolve_settings

from .candidates import (
    CandidateEvaluation,
    CandidateGenerationError,
    CandidateGenerator,
    VLLMCandidateGenerator,
    evaluate_candidates,
    request_policies,
)
from .grouping import read_question, validate_option_count
from .policy import Policy
from .prompt import InvalidQuestion, parse_question, validate_prompt_variant
from .readers import LetterReader, VLLMChatReader, add_reader_arguments, reader_from_args
from .reasoning import system_read

MAX_HTTP_BYTES = 8 * 1024 * 1024
CLOSE_DRAIN_SECONDS = 0.10


class DecisionService:
    def __init__(
        self,
        reader: LetterReader,
        model_name: str,
        policy: Policy | None = None,
        *,
        max_parallel: int = 8,
        max_questions: int = 128,
        group_size: int = 20,
        state_format: str = "pretty",
        prompt_variant: str = "min",
        force_variant: bool = False,
        diagnostic: bool = False,
        candidate_generator: CandidateGenerator | None = None,
    ):
        if max_parallel < 1 or max_questions < 1:
            raise ValueError("parallelism and question limits must be positive")
        validate_option_count(2, group_size)
        if state_format not in ("pretty", "compact"):
            raise ValueError("state_format must be pretty or compact")
        validate_prompt_variant(prompt_variant)
        policy = policy or Policy(prompt_variant=prompt_variant)
        if policy.prompt_variant != prompt_variant and not force_variant:
            raise ValueError(
                f"policy prompt_variant={policy.prompt_variant!r} differs from "
                f"--prompt-variant {prompt_variant!r}; use --force-variant to override"
            )
        if diagnostic or force_variant:
            policy = replace(policy, promotable=False)
        else:
            if policy.promotable is not True:
                raise ValueError("unpromotable Swift policies require --diagnostic serving")
            from .adopt import validate_adopted_policy

            validate_adopted_policy(policy)
            if policy.adoption and policy.adoption["model"] != model_name:
                raise ValueError("adopted policy model differs from serving model")
            if (
                policy.adoption
                and getattr(reader, "backend", None) in ("hf", "vllm")
                and getattr(reader, "revision", None) != policy.adoption["revision"]
            ):
                raise ValueError("adopted policy revision differs from serving reader")
        self.reader = reader
        self.model_name = model_name
        self.policy = policy
        self.prompt_variant = prompt_variant
        self.max_questions = max_questions
        self.group_size = group_size
        self.state_format = state_format
        # Offline tests may inject a CPU fake; serving always uses the reader's
        # own frozen vLLM model, never a client-selected proposing backend.
        self.candidate_generator = candidate_generator
        self._pool = ThreadPoolExecutor(max_workers=max_parallel)

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def handle(self, body: object) -> dict:
        if not isinstance(body, dict) or "questions" not in body:
            raise InvalidQuestion("body needs questions")
        questions = body["questions"]
        if not isinstance(questions, dict) or not questions:
            raise InvalidQuestion("questions must be a non-empty object")
        if len(questions) > self.max_questions:
            raise InvalidQuestion(f"maximum {self.max_questions} questions per request")
        policies = request_policies(body)
        parsed = [
            parse_question(question) if policies[name]["mode"] != "open" else None
            for name, question in questions.items()
        ]
        options = body.get("options", {})
        if not isinstance(options, dict):
            raise InvalidQuestion("options must be an object")
        try:
            for question in questions.values():
                if ("reasoning" in options and options["reasoning"] is None) or (
                    "reasoning" in question and question["reasoning"] is None
                ):
                    raise ValueError("reasoning settings must be an object")
                settings = resolve_settings(
                    checkpoint={"mode": "off"},
                    request=options.get("reasoning"),
                    question=question.get("reasoning"),
                )
                if settings.budget > 0:
                    raise ValueError("reasoning not supported by this readout")
        except ValueError as exc:
            raise InvalidQuestion(str(exc)) from exc
        for question in parsed:
            if question is None:
                continue
            validate_option_count(len(question.labels), self.group_size)
            if question.type == "score":
                ordinals = [int(label) for label in question.labels]
                if any(b - a != 1 for a, b in zip(ordinals, ordinals[1:], strict=False)):
                    raise InvalidQuestion("score ordinals must be uniform and contiguous")
        if any(policy["mode"] != "fixed" for policy in policies.values()):
            generator = self.candidate_generator
            if generator is None:
                if not isinstance(self.reader, VLLMChatReader):
                    raise InvalidQuestion("candidate generation needs the frozen vLLM chat backend")
                generator = VLLMCandidateGenerator(self.reader)
            return self._handle_candidates(body, policies, parsed, generator)
        futures = [
            self._pool.submit(
                system_read,
                self.reader,
                body.get("state", ""),
                question,
                policy=self.policy,
                group_size=self.group_size,
                state_format=self.state_format,
                prompt_variant=self.prompt_variant,
            )
            for question in parsed
        ]
        try:
            reads = [future.result() for future in futures]
        except Exception:
            for future in futures:
                future.cancel()
            raise
        return {
            "model": body.get("model") or self.model_name,
            "answers": {
                name: self.policy.decide(
                    question.type,
                    result.raw_probs,
                    candidate_log_masses=result.candidate_log_masses,
                )
                for name, question, result in zip(questions, parsed, reads, strict=True)
            },
            "usage": {
                "input_tokens": sum(result.input_tokens for result in reads),
                "output_tokens": sum(result.output_tokens for result in reads),
            },
        }

    def _handle_candidates(self, body, policies, parsed, generator):
        """Experimental generated partitions; fixed JevBench never enters here."""
        generated_policy = replace(self.policy, letter_bias=None, reasoning_route=None)

        def score(question, *, generated=False):
            policy = generated_policy if generated else self.policy
            read = (read_question if generated else system_read)(
                self.reader,
                body.get("state", ""),
                question,
                **({} if generated else {"policy": policy}),
                group_size=self.group_size,
                state_format=self.state_format,
                prompt_variant=self.prompt_variant,
            )
            return policy.decide(
                question.type, read.raw_probs, candidate_log_masses=read.candidate_log_masses
            ), read

        def evaluate(name, question):
            if policies[name]["mode"] != "fixed":
                return evaluate_candidates(
                    body.get("state", ""), body["questions"][name], policies[name], score, generator
                )
            answer, read = score(question)
            return CandidateEvaluation(
                answer,
                {"input_tokens": read.input_tokens, "output_tokens": read.output_tokens},
                {"input_tokens": 0, "output_tokens": 0},
            )

        futures = [
            self._pool.submit(evaluate, name, question)
            for name, question in zip(body["questions"], parsed, strict=True)
        ]
        try:
            results = [future.result() for future in futures]
        except Exception:
            for future in futures:
                future.cancel()
            raise
        usage = {
            key: sum(result.usage[key] for result in results)
            for key in ("input_tokens", "output_tokens")
        }
        proposal_usage = {
            key: sum(result.proposal_usage[key] for result in results) for key in usage
        }
        extension_usage = {
            "proposal_input_tokens": proposal_usage["input_tokens"],
            "proposal_output_tokens": proposal_usage["output_tokens"],
            "scoring_input_tokens": usage["input_tokens"] - proposal_usage["input_tokens"],
            "scoring_output_tokens": usage["output_tokens"] - proposal_usage["output_tokens"],
        }
        failures = {
            name: result.extension
            for name, result in zip(body["questions"], results, strict=True)
            if result.answer is None
        }
        if failures:
            failed = next(iter(failures))
            raise CandidateGenerationError(
                f"open candidate generation failed for {failed}: {failures[failed]['diagnostics']['error']}",
                {"usage": usage, "ayaka": {"usage": extension_usage, "questions": failures}},
            )
        return {
            "model": body.get("model") or self.model_name,
            "answers": {
                name: result.answer for name, result in zip(body["questions"], results, strict=True)
            },
            "usage": usage,
            "ayaka": {"usage": extension_usage},
        }


def make_handler(service: DecisionService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def finish(self) -> None:
            # TCPServer.shutdown_request() half-closes then immediately closes.
            # On Windows, unread/in-flight input can turn that close into a RST
            # which discards the response, even when wfile.write() succeeded.
            # Drain *before* StreamRequestHandler closes rfile: it can already
            # hold body bytes read ahead while parsing the HTTP headers.
            # Drain before SHUT_WR too: Windows loopback filtering can report
            # receive EOF immediately after that half-close, hiding late input.
            try:
                self.wfile.flush()
                deadline = time.monotonic() + CLOSE_DRAIN_SECONDS
                while (remaining := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(remaining)
                    if not self.rfile.read1(64 * 1024):
                        break
            except OSError:
                pass  # Client disconnected, or the bounded drain timed out.
            finally:
                with contextlib.suppress(OSError):
                    self.connection.shutdown(socket.SHUT_WR)
                super().finish()

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            if path == "/health":
                self._send(
                    200,
                    {
                        "status": "ok",
                        "model": service.model_name,
                        "prompt_variant": service.prompt_variant,
                    },
                )
            elif path == "/v1/models":
                self._send(200, {"data": [{"id": service.model_name}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send(400, {"error": "invalid Content-Length"})
                return
            if not 0 <= length <= MAX_HTTP_BYTES:
                self._send(422, {"error": "request exceeds the 8 MiB limit"})
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request body")
                body = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                self._send(400, {"error": str(exc)})
                return
            try:
                response = service.handle(body)
            except InvalidQuestion as exc:
                self._send(422, getattr(exc, "response", {"error": str(exc)}))
            except Exception as exc:
                self._send(502, {"error": f"backend failure: {type(exc).__name__}"})
            else:
                self._send(200, response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def serve(
    service: DecisionService, host: str = "127.0.0.1", port: int = 8009
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(service))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_reader_arguments(parser)
    parser.add_argument("--policy", help="policy.json; default uses T=1 without commitment")
    parser.add_argument(
        "--force-variant",
        action="store_true",
        help="diagnostic override of policy/variant mismatch",
    )
    parser.add_argument(
        "--diagnostic", action="store_true", help="serve unadopted candidates; never promotable"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        service = DecisionService(
            reader_from_args(args),
            args.hf_model if args.backend == "hf" else args.model,
            Policy.load(args.policy) if args.policy else None,
            max_parallel=args.max_parallel,
            max_questions=args.max_questions,
            group_size=args.group_size,
            state_format=args.state_format,
            prompt_variant=args.prompt_variant,
            force_variant=args.force_variant,
            diagnostic=args.diagnostic,
        )
    except ValueError as exc:
        parser.error(str(exc))
    server = serve(service, args.host, args.port)
    print(f"Swift listening on http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()

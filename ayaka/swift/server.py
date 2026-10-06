"""Stdlib TypeSafe server for the Swift letter readout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ayaka.http_transport import (
    ServiceConfig,
    add_service_arguments,
    config_from_args,
    make_api_handler,
    server_for,
)
from ayaka.jev_api import ModelCatalog, ValidationError, complete_answers, normalize_request
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
from .reasoning import system_read, validate_reasoning

MAX_HTTP_BYTES = 24 * 1024 * 1024


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
        model_id: str | None = None,
        model_description: str = ModelCatalog.description,
        model_release_date: str = ModelCatalog.release_date,
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
        self.catalog = ModelCatalog(model_name, model_id, model_description, model_release_date)
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

    def _answer(self, question, read, policy):
        answer = policy.decide(
            question.type, read.raw_probs, candidate_log_masses=read.candidate_log_masses
        )
        answer["ayaka"] = {
            "route": "grouped"
            if read.readout == "grouped_approx"
            else "reasoned"
            if read.passes > 1
            else "direct"
        }
        return answer

    def handle(self, body: object) -> dict:
        try:
            body = normalize_request(body, max_questions=self.max_questions)
            self.catalog.resolve(body.get("model"))
        except ValidationError as exc:
            raise InvalidQuestion(str(exc), exc.field) from exc
        response = self._handle(body)
        response["model"] = self.catalog.actual_id
        calibration = "unfitted" if self.policy.fitted_on == "unfitted" else "fitted"
        response = complete_answers(response, body, calibration=calibration)
        for answer in response["answers"].values():
            if "media" in body:
                answer["ayaka"]["route"] = "image"
            elif "candidates" in answer["ayaka"]:
                answer["ayaka"]["route"] = "generated"
            elif len(answer.get("probabilities", {})) > 26:
                answer["ayaka"]["route"] = "grouped"
        return response

    def _handle(self, body: object) -> dict:
        reader, policy = self.reader, self.policy
        if not isinstance(body, dict) or "questions" not in body:
            raise InvalidQuestion("body needs questions")
        questions = body["questions"]
        if not isinstance(questions, dict) or not questions:
            raise InvalidQuestion("questions must be a non-empty object")
        if len(questions) > self.max_questions:
            raise InvalidQuestion(f"maximum {self.max_questions} questions per request")
        policies = request_policies(body)
        if "media" in body:
            from .media import ImageReader, image_parts

            reader = ImageReader(reader, image_parts(body["media"]))
            policy = replace(
                policy,
                t_choice=1,
                t_noul=1,
                t_score=1,
                letter_bias=None,
                reasoning_route=None,
                commit_margin=None,
            )
        parsed = [
            parse_question(question) if policies[name]["mode"] != "open" else None
            for name, question in questions.items()
        ]
        options = body.get("options", {})
        if not isinstance(options, dict):
            raise InvalidQuestion("options must be an object")
        reasoning = {}
        try:
            for (name, question), parsed_question in zip(questions.items(), parsed, strict=True):
                if ("reasoning" in options and options["reasoning"] is None) or (
                    "reasoning" in question and question["reasoning"] is None
                ):
                    raise ValueError("reasoning settings must be an object")
                settings = (
                    resolve_settings(
                        checkpoint={"mode": "auto" if policy.reasoning_route else "off"},
                        request=options.get("reasoning"),
                        question=question.get("reasoning"),
                    )
                    if "reasoning" in options or "reasoning" in question
                    else None
                )
                if settings is not None:
                    if settings.budget > 0 and policies[name]["mode"] != "fixed":
                        raise ValueError("reasoning not supported with candidate generation")
                    validate_reasoning(reader, parsed_question, settings)
                reasoning[name] = settings
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
            return self._handle_candidates(body, policies, parsed, generator, reasoning)
        futures = [
            self._pool.submit(
                system_read,
                reader,
                body.get("state", ""),
                question,
                policy=policy,
                reasoning=reasoning[name],
                group_size=self.group_size,
                state_format=self.state_format,
                prompt_variant=self.prompt_variant,
            )
            for name, question in zip(questions, parsed, strict=True)
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
                name: self._answer(question, result, policy)
                for name, question, result in zip(questions, parsed, reads, strict=True)
            },
            "usage": {
                "input_tokens": sum(result.input_tokens for result in reads),
                "output_tokens": sum(result.output_tokens for result in reads),
            },
        }

    def _handle_candidates(self, body, policies, parsed, generator, reasoning):
        """Experimental generated partitions; fixed JevBench never enters here."""
        generated_policy = replace(self.policy, letter_bias=None, reasoning_route=None)

        def score(name, question, *, generated=False):
            policy = generated_policy if generated else self.policy
            read = (read_question if generated else system_read)(
                self.reader,
                body.get("state", ""),
                question,
                **({} if generated else {"policy": policy, "reasoning": reasoning[name]}),
                group_size=self.group_size,
                state_format=self.state_format,
                prompt_variant=self.prompt_variant,
            )
            return self._answer(question, read, policy), read

        def evaluate(name, question):
            if policies[name]["mode"] != "fixed":
                return evaluate_candidates(
                    body.get("state", ""),
                    body["questions"][name],
                    policies[name],
                    lambda question, **kwargs: score(name, question, **kwargs),
                    generator,
                )
            answer, read = score(name, question)
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


def make_handler(
    service: DecisionService, *, config: ServiceConfig | None = None
) -> type[BaseHTTPRequestHandler]:
    return make_api_handler(
        service, config=config, max_http_bytes=MAX_HTTP_BYTES, validation_errors=(InvalidQuestion,)
    )


def serve(
    service: DecisionService,
    host: str = "127.0.0.1",
    port: int = 8009,
    *,
    config: ServiceConfig | None = None,
) -> ThreadingHTTPServer:
    config = config or ServiceConfig()
    config.validate_bind(host)
    return server_for((host, port), make_handler(service, config=config))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_reader_arguments(parser)
    add_service_arguments(parser)
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
        config = config_from_args(args)
        config.validate_bind(args.host)
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
            model_id=args.model_id,
            model_description=args.model_description,
            model_release_date=args.model_release_date,
        )
    except ValueError as exc:
        parser.error(str(exc))
    server = serve(service, args.host, args.port, config=config)
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

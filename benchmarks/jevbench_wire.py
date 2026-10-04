"""CPU wire check: JevBench's real typesafe adapter/scorer against ayaka.serve.

    python -m benchmarks.jevbench_wire --jevbench PATH_TO_JEVBENCH_CHECKOUT

A uniform stub replaces inference, so this measures wire validity only. Calling
the adapter in-process also avoids JevBench's Unix-only CLI dependency on fcntl.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from ayaka.http_transport import ServiceConfig
from ayaka.primitives import DecisionResult
from ayaka.serve import serve
from ayaka.tokenization import ToyTokenizer


class UniformDecision:
    tok = ToyTokenizer()
    max_seq_len = 8192

    def decide(self, state, questions):
        return [
            DecisionResult(q.type, [1 / len(q.candidates)] * len(q.candidates), {})
            for q in questions
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jevbench", required=True, type=Path, help="JevBench checkout")
    args = parser.parse_args(argv)
    # The external checkout is read-only, including Python's bytecode cache.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.jevbench.resolve()))
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.scoring import score_task
    from jevbench.tasks import load_jsonl

    public = args.jevbench / "datasets" / "public"
    tasks = [
        task
        for split in ("easy", "original", "hard")
        for task in load_jsonl(str(public / f"{split}.jsonl"))
    ]
    server = serve(
        UniformDecision(),
        "ayaka-wire",
        host="127.0.0.1",
        port=0,
        config=ServiceConfig(api_key=""),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Exercise the SDK/Jev default alias through JevBench's unchanged request builder.
    adapter = TypeSafeAdapter(endpoint=f"http://127.0.0.1:{server.server_port}", key_env="")
    failures = []
    try:
        for task in tasks:
            result = adapter.run(task)
            scored = score_task(result.probs or {}, task) if result.ok else {"valid": False}
            if not (result.ok and scored.get("valid")):
                failures.append((task.id, result.error or scored.get("error")))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    print(f"items: {len(tasks)}, valid: {len(tasks) - len(failures)}, invalid: {len(failures)}")
    for task_id, error in failures[:5]:
        print(f"  {task_id}: {error}")
    return 0 if tasks and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

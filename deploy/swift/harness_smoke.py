"""CPU wire-format smoke test: JevBench's own typesafe adapter and scorer against the Swift server.

No model, no GPU: the server uses a fake reader that returns uniform letters, so scores are
meaningless. It proves that every public item gets an answer that JevBench parses and scores
as valid. The adapter and ``score_task`` are called in-process because JevBench's CLI needs the
Unix-only ``fcntl`` module; on Linux, the CLI run in ``deploy/swift/README.md`` is the real check.

    python deploy/swift/harness_smoke.py --jevbench PATH_TO_JEVBENCH_CHECKOUT
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from ayaka.swift.readers import FakeReader
from ayaka.swift.server import DecisionService, serve

SPLITS = ("easy", "original", "hard")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jevbench", required=True, type=Path, help="JevBench checkout")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.jevbench.resolve()))
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.scoring import score_task
    from jevbench.tasks import load_jsonl

    public = args.jevbench / "datasets" / "public"
    tasks = [task for split in SPLITS for task in load_jsonl(str(public / f"{split}.jsonl"))]

    service = DecisionService(FakeReader(), "swift-smoke")
    server = serve(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    adapter = TypeSafeAdapter(
        endpoint=f"http://127.0.0.1:{server.server_port}", model="swift-smoke", key_env=""
    )
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
        service.close()
    print(f"items: {len(tasks)}, valid: {len(tasks) - len(failures)}, invalid: {len(failures)}")
    for task_id, error in failures[:5]:
        print(f"  {task_id}: {error}")
    return 0 if tasks and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

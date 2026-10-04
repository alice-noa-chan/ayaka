"""Supervised GPU-job helpers for P5; importing this module never loads a model."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.swift.inputs import sha256  # noqa: E402
from scripts.swift.latency_probe import write_json  # noqa: E402
from scripts.swift.matched_2x2 import NOTICE, public_items  # noqa: E402

BASE = "google/gemma-4-12B-it"
BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
CHECKPOINT = "alice-noa-chan/ayaka-large"
CHECKPOINT_REVISION = "267605ee22f2b5f934d81e5fbee691952d2e6f55"


def validate_config(config: dict) -> None:
    if (
        config.get("backbone") != BASE
        or config.get("backbone_revision") != BASE_REVISION
        or config.get("readout", "hybrid") != "hybrid"
        or config.get("version", 1) != 1
        or config.get("lora_r", 64) != 64
    ):
        raise ValueError("matched native cell requires the pinned v1 large hybrid config")


def native_command(cell: str, checkpoint: Path, output: Path) -> list[str]:
    if cell not in ("frozen_native", "lora_native"):
        raise ValueError("unknown native cell")
    return [
        sys.executable,
        "-m",
        "ayaka.eval.jevbench",
        *(
            ["--zero-shot", "electra-large"]
            if cell == "frozen_native"
            else ["--ckpt", str(checkpoint)]
        ),
        "--tiers",
        "easy,original,hard",
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--max-seq-len",
        "8192",
        "--out",
        str(output),
    ]


def swift_adapter_key(key: str, model_type: str) -> str:
    """Detached native text-model adapters need the full HF LM's module prefix."""
    if model_type not in ("gemma4", "gemma4_text"):
        raise ValueError("unsupported matched base model type")
    if not re.fullmatch(
        r"base_model\.model\.layers\.\d+\.(?:self_attn|mlp)\."
        r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_[AB]\.weight",
        key,
    ):
        raise ValueError(f"unexpected published adapter key: {key}")
    prefix = "base_model.model.model."
    if model_type == "gemma4":
        prefix += "language_model."
    return prefix + key.removeprefix("base_model.model.")


def convert_adapter(source: Path, target: Path, model_type: str) -> dict:
    """Rename safetensors header keys; stream unchanged payload without tensor allocation."""
    with source.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("adapter header is unexpectedly large")
        header = json.loads(stream.read(length))
        keys = [key for key in header if key != "__metadata__"]
        if not keys:
            raise ValueError("empty adapter")
        mapping = {key: swift_adapter_key(key, model_type) for key in keys}
        renamed = {mapping.get(key, key): value for key, value in header.items()}
        encoded = json.dumps(renamed, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as destination:
            destination.write(struct.pack("<Q", len(encoded)))
            destination.write(encoded)
            shutil.copyfileobj(stream, destination, length=1024 * 1024)
    return {"tensor_count": len(keys), "key_mapping": mapping, "payload": "unchanged bytes"}


def prepare_checkpoint(output: Path) -> dict:
    """Only called in an admitted P4 GPU child; checkpoint download is explicitly authorized."""
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    resolved = HfApi().model_info(CHECKPOINT, revision=CHECKPOINT_REVISION).sha
    if resolved != CHECKPOINT_REVISION:
        raise ValueError("published checkpoint revision did not resolve to the required SHA")
    checkpoint = Path(
        snapshot_download(
            repo_id=CHECKPOINT,
            revision=CHECKPOINT_REVISION,
            allow_patterns=["*config.json", "head.safetensors", "meta.json", "adapter/*"],
        )
    )
    config_path = checkpoint / "ayaka_config.json"
    if not config_path.is_file():
        config_path = checkpoint / "electra_config.json"
    validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    adapter = checkpoint / "adapter"
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        adapter_config.get("base_model_name_or_path") != BASE
        or adapter_config.get("revision") not in (None, BASE_REVISION)
        or adapter_config.get("r") != 64
    ):
        raise ValueError("published adapter base/revision mismatch")
    # Only the checkpoint may be fetched. The pinned base and tokenizer must be cached.
    model_type = AutoConfig.from_pretrained(
        BASE, revision=BASE_REVISION, local_files_only=True
    ).model_type
    swift_adapter = output / "swift_adapter"
    conversion = convert_adapter(
        adapter / "adapter_model.safetensors",
        swift_adapter / "adapter_model.safetensors",
        model_type,
    )
    write_json(swift_adapter / "adapter_config.json", {**adapter_config, "revision": BASE_REVISION})
    provenance_files = [config_path, checkpoint / "head.safetensors", *sorted(adapter.glob("*"))]
    receipt = {
        "role": "diagnostic",
        "used_for_fit_or_selection": False,
        "used_for_gates": False,
        "notice": NOTICE,
        "checkpoint_repo": CHECKPOINT,
        "checkpoint_revision": resolved,
        "checkpoint_path": str(checkpoint),
        "base_model": BASE,
        "base_revision": BASE_REVISION,
        "swift_adapter_path": str(swift_adapter),
        "preparation_s": time.perf_counter() - started,
        "source_sha256": {
            p.relative_to(checkpoint).as_posix(): sha256(p) for p in provenance_files if p.is_file()
        },
        "swift_adapter_sha256": {p.name: sha256(p) for p in sorted(swift_adapter.iterdir())},
        "conversion": conversion,
    }
    write_json(output / "checkpoint.json", receipt)
    return receipt


def run_native(cell: str, checkpoint: Path, output: Path) -> dict:
    """Execute the existing evaluator CLI main, instrumenting loads and flushed partial rows."""
    import dataclasses

    from ayaka.config import model_config
    from ayaka.eval import jevbench

    output.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint / "ayaka_config.json"
    if not config_path.is_file():
        config_path = checkpoint / "electra_config.json"
    validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    validate_config(dataclasses.asdict(model_config("electra-large")))
    items = public_items(Path(jevbench.data_dir()))
    if len(items) != 231:
        raise ValueError("native matched cell needs exactly 231 public items")
    command = native_command(cell, checkpoint, output / "report.json")
    receipt = {
        "cell": cell,
        "role": "diagnostic",
        "notice": NOTICE,
        "used_for_fit_or_selection": False,
        "used_for_gates": False,
        "complete": False,
        "base_model": BASE,
        "base_revision": BASE_REVISION,
        "tokenizer_revision": BASE_REVISION,
        "checkpoint_repo": CHECKPOINT if cell == "lora_native" else None,
        "checkpoint_revision": CHECKPOINT_REVISION if cell == "lora_native" else None,
        "equivalent_native_command": command,
        "execution": "ayaka.eval.jevbench.main with the recorded CLI argv; recording wrappers only",
        "launcher_argv": [sys.executable, *sys.argv],
        "dtype": "bfloat16",
        "device": "cuda",
        "max_seq_len": 8192,
        "config_readout": "hybrid",
        "effective_frozen_readout": "LM for labeled items (zero gates)",
        "hf_load_s": None,
        "load_timing_scope": "CLI model/checkpoint and tokenizer loading",
        "completed_items": 0,
    }
    write_json(output / "load_times.json", receipt)
    original_run = jevbench.run_jevbench
    original_evaluate = jevbench.evaluate
    completed = set()
    started = time.perf_counter()
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:

        def recording_run(*args, **kwargs):
            receipt["hf_load_s"] = time.perf_counter() - started
            write_json(output / "load_times.json", receipt)
            return original_run(*args, **kwargs)

        def recording_evaluate(decision, rows, device=None):
            context = iter(rows)

            class RecordingDecision:
                def decide(self, state, questions, device=None):
                    row = next(context)
                    tick = time.perf_counter()
                    results = decision.decide(state, questions, device=device)
                    item = jevbench.record_to_item(row)
                    probs = dict(zip(item.labels, results[0].probs, strict=True))
                    pred = max(probs, key=probs.get)
                    result = {
                        "id": item.id,
                        **items[item.id],
                        "pred": pred,
                        "ok": pred == item.expected,
                        "probs": probs,
                        "latency_s": time.perf_counter() - tick,
                        "diagnostic": True,
                        "used_for_fit_or_selection": False,
                        "used_for_gates": False,
                        "cell": cell,
                    }
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed.add(item.id)
                    receipt["completed_items"] = len(completed)
                    write_json(output / "load_times.json", receipt)
                    return results

            return original_evaluate(RecordingDecision(), rows, device=device)

        jevbench.run_jevbench = recording_run
        jevbench.evaluate = recording_evaluate
        try:
            report = jevbench.main(command[3:])
            if completed != set(items):
                raise ValueError("native evaluation did not complete the matched id set")
            report["matched_diagnostic"] = receipt
            receipt["complete"] = True
            write_json(output / "report.json", report)
        except BaseException as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            jevbench.run_jevbench = original_run
            jevbench.evaluate = original_evaluate
            receipt["elapsed_s"] = time.perf_counter() - started
            if receipt["hf_load_s"] is None:
                receipt["hf_load_attempt_s"] = receipt["elapsed_s"]
            write_json(output / "load_times.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-checkpoint", action="store_true")
    parser.add_argument("--cell", choices=("frozen_native", "lora_native"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prepare_checkpoint:
        prepare_checkpoint(args.output)
    elif args.cell and args.checkpoint:
        # Separate native processes need only cached base/tokenizer + the P4 snapshot.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(NOTICE, flush=True)
        run_native(args.cell, args.checkpoint, args.output)
    else:
        parser.error("use --prepare-checkpoint or both --cell and --checkpoint")


if __name__ == "__main__":
    main()

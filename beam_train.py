"""beam.cloud serverless entry point: any ``ayaka.pipeline`` command.

beam serverless offers T4 / A10G / RTX 4090 / RTX 5090 only (A100 and
H100 are reserved on-demand machines), so beam is used for small-model
work and evals; Large training runs on RunPod / vast.ai / Modal
(scripts/run_plan.sh, scripts/modal/train_v1.py). Run from the repo root under the
beam SDK python::

    python -c "import beam_train as b; b.pipeline_4090.remote(['eval', '--model', 'electra-small', '--zero-shot'])"
    python -c "import beam_train as b; b.pipeline_4090.remote(['train', '--model', 'electra-small', '--run', 'small-v1'])"

The file must live at the repo root: the SDK derives the handler name
from the module path, and a nested path breaks the handler on Windows.
GPU-attached CPU cores are billed separately and are expensive, so
containers get 2 cores.
"""

from __future__ import annotations

import os
import sys

from beam import GpuType, Image, Volume, function

IMAGE = Image(
    python_version="python3.11",
    python_packages=[
        "torch>=2.6",
        "transformers>=5.6",
        "peft>=0.19",
        "accelerate>=1.0",
        "datasets>=3.0",
        "huggingface_hub>=0.30",
        "safetensors",
        "numpy>=1.26",
    ],
)

VOLUME = Volume(name="ayaka-artifacts", mount_path="/artifacts")

ENV = {
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": "/artifacts/hf-cache",  # weights + datasets cached across runs
    "AYAKA_ARTIFACTS": "/artifacts",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

COMMON = {
    "image": IMAGE,
    "volumes": [VOLUME],
    "env": ENV,
    "gpu_count": 1,
    "retries": 0,
    "cpu": 2,
    "timeout": -1,
    "headless": True,  # keep running after the client disconnects
}


def _run(argv: list[str]) -> dict:
    sys.path.insert(0, os.getcwd())
    from ayaka.pipeline import main

    return main(list(argv))


@function(name="ayaka-4090", gpu=GpuType.RTX4090, memory="32Gi", **COMMON)
def pipeline_4090(argv: list[str]) -> dict:
    return _run(argv)


@function(name="ayaka-5090", gpu=GpuType.RTX5090, memory="40Gi", **COMMON)
def pipeline_5090(argv: list[str]) -> dict:
    return _run(argv)

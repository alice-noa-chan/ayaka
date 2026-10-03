"""Launch exactly one admitted inference diagnostic on serverless RTX5090."""

import argparse
import json

import mechanism_worker
from beam import Image, Volume, function

image = Image(
    base_image="docker.io/library/ubuntu:24.04", python_version="python3.11"
).add_python_packages(["zstandard==0.25.0"])
volume = Volume(name="ayaka-mechanism-20261003", mount_path="/ayaka-volume")


@function(
    name=mechanism_worker.RUN_NAME,
    gpu="RTX5090",
    cpu=2,
    memory="32Gi",
    timeout=mechanism_worker.SECONDS,
    retries=0,
    headless=True,
    image=image,
    volumes=[volume],
)
def diagnose(overlay_sha, plan_sha, admission):
    return mechanism_worker.execute("/ayaka-volume", overlay_sha, plan_sha, admission)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("build", "run"))
    ap.add_argument("--overlay-sha")
    ap.add_argument("--plan-sha")
    ap.add_argument("--admission")
    args = ap.parse_args()
    if args.action == "build":
        print(json.dumps({"image_built": image.build().success}))
    else:
        if not all((args.overlay_sha, args.plan_sha, args.admission)):
            ap.error("run requires a frozen overlay, plan and fresh credit admission")
        from pathlib import Path

        admission = json.loads(Path(args.admission).read_text())
        mechanism_worker.validate_admission(admission)
        print(json.dumps(diagnose.remote(args.overlay_sha, args.plan_sha, admission)))

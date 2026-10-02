"""Only this small directory is synced; private offline bytes live in a volume."""

import argparse
import json

import worker
from beam import Image, Volume, function

VOLUME = "ayaka-clean-20261002"
POOL = "ayaka-clean-20261002"
MOUNT = "/ayaka-volume"
image = Image(
    base_image="docker.io/library/ubuntu:24.04", python_version="python3.11"
).add_python_packages(["zstandard==0.25.0"])
volume = Volume(name=VOLUME, mount_path=MOUNT)


@function(
    name="ayaka-clean-cpu-20261002",
    cpu=2,
    memory="8Gi",
    timeout=1800,
    retries=0,
    image=image,
    volumes=[volume],
)
def prepare():
    return worker.prepare(MOUNT)


@function(
    name="ayaka-clean-pilot-20261002",
    gpu="A100-80",
    cpu=4,
    memory="64Gi",
    timeout=10000,
    retries=0,
    image=image,
    volumes=[volume],
    pool=POOL,
)
def pilot():
    return worker.execute(MOUNT, "clean-pilot-20261002")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pilot", "build"))
    args = parser.parse_args()
    if args.action == "build":
        print(json.dumps({"image_built": image.build().success}))
    else:
        print(json.dumps((prepare if args.action == "prepare" else pilot).remote()))

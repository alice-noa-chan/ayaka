import io
import json
import tarfile

import pytest
import zstandard

from scripts.runpod_v2.clean_package import trusted_base_member, verify_clean_archive
from scripts.runpod_v2.package import digest


def test_dependency_whitelist_excludes_old_training_and_private_cache():
    assert trusted_base_member(tarfile.TarInfo("ayaka-v2/runtime/bin/python3.11"))
    assert not trusted_base_member(tarfile.TarInfo("ayaka-v2/bundle/train.jsonl"))
    assert not trusted_base_member(tarfile.TarInfo("ayaka-v2/hf-cache/token"))
    link = tarfile.TarInfo("ayaka-v2/runtime/bin/python3")
    link.type, link.linkname = tarfile.SYMTYPE, "python3.11"
    assert trusted_base_member(link)
    for target in ("/usr/bin/python3", "../../../outside"):
        link.linkname = target
        with pytest.raises(ValueError, match="symlink"):
            trusted_base_member(link)
    with pytest.raises(ValueError, match="escaped"):
        trusted_base_member(tarfile.TarInfo("ayaka-v2/../outside"))


def test_zstd_delivery_verifies_actual_member_bytes_and_sidecar(tmp_path):
    import hashlib

    path = tmp_path / "kit.tar.zst"
    data = b"verified data"
    record = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def write(record):
        with (
            path.open("wb") as raw,
            zstandard.ZstdCompressor(write_checksum=True).stream_writer(raw) as compressed,
            tarfile.open(fileobj=compressed, mode="w|") as t,
        ):
            for name, body in (
                ("bundle/train.jsonl", data),
                (
                    "clean-kit-manifest.json",
                    json.dumps(
                        {"version": "ayaka-clean-kit-1", "files": {"bundle/train.jsonl": record}}
                    ).encode(),
                ),
            ):
                member = tarfile.TarInfo("ayaka-v2/" + name)
                member.size = len(body)
                t.addfile(member, io.BytesIO(body))
        path.with_name(path.name + ".sha256").write_text(digest(path))

    write(record)
    assert verify_clean_archive(path)["verified"]
    write({**record, "sha256": "a" * 64})
    with pytest.raises(ValueError, match="content manifest"):
        verify_clean_archive(path)

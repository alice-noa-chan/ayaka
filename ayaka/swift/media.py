"""Experimental bounded images forwarded as vLLM data-URI content parts."""

from __future__ import annotations

import base64
import binascii
import io
import warnings

from .prompt import InvalidQuestion


def image_parts(media):
    if not isinstance(media, list) or not 1 <= len(media) <= 4:
        raise InvalidQuestion("media needs 1..4 images", "ayaka.media")
    # Imported only on the experimental image path; text serving is stdlib-only.
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise InvalidQuestion(
            "image validation requires Pillow (install ayaka[vision])", "ayaka.media"
        ) from exc
    parts, total = [], 0
    formats = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
    for i, item in enumerate(media):
        field = f"ayaka.media.{i}"
        if not isinstance(item, dict) or set(item) != {"type", "mime_type", "data"}:
            raise InvalidQuestion("each image needs type, mime_type and base64 data", field)
        mime = item["mime_type"]
        if item["type"] != "image" or mime not in formats or not isinstance(item["data"], str):
            raise InvalidQuestion("media supports base64 PNG, JPEG and WebP images", field)
        if len(item["data"]) > 4 * ((8 * 1024 * 1024 + 2) // 3):
            raise InvalidQuestion("image exceeds 8 MiB", field)
        try:
            raw = base64.b64decode(item["data"], validate=True)
            total += len(raw)
            if not raw or len(raw) > 8 * 1024 * 1024 or total > 16 * 1024 * 1024:
                raise ValueError("images exceed 8 MiB per image or 16 MiB total")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as picture:
                    if picture.format != formats[mime]:
                        raise ValueError("mime_type does not match image contents")
                    if (
                        picture.width * picture.height > 16_000_000
                        or getattr(picture, "n_frames", 1) != 1
                    ):
                        raise ValueError("images need a single frame and at most 16 million pixels")
                    picture.load()
                    if picture.getexif().get(274, 1) != 1:
                        oriented = ImageOps.exif_transpose(picture)
                        encoded = io.BytesIO()
                        oriented.save(encoded, format="PNG")
                        raw = encoded.getvalue()
                        mime = "image/png"
                        if len(raw) > 8 * 1024 * 1024:
                            raise ValueError("oriented image exceeds 8 MiB")
            uri = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
        except (
            ValueError,
            OSError,
            binascii.Error,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise InvalidQuestion(str(exc), field) from exc
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


class ImageReader:
    """Per-request wrapper: no mutation of the shared reader or text policy."""

    def __init__(self, reader, parts):
        if getattr(reader, "backend", None) == "hf":
            raise InvalidQuestion("Swift image inputs require the vLLM chat backend", "ayaka.media")
        self.reader = reader
        self.parts = parts
        self.readout = getattr(reader, "readout", "undeclared")

    def read(self, messages, letters):
        messages = [dict(m) for m in messages]
        user = next(m for m in messages if m["role"] == "user")
        user["content"] = [{"type": "text", "text": user["content"]}, *self.parts]
        return self.reader.read(messages, letters)

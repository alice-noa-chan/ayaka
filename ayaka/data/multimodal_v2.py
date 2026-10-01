"""Deterministic MIT image/text pairs with exact, offline-only target oracles."""

import base64
import hashlib
import io
from datetime import date

from PIL import Image, ImageDraw, ImageFont

from ..evidence import add_months
from .reasoning_v2 import SPLITS
from .schema import Candidate, Question, Sample

INSTRUCTIONS = {
    "en": {
        "receipt": "Report the total in cents.",
        "chart": "Which series has the largest value?",
        "calendar": "Report the day after adding one calendar month, clamping to month end.",
        "rule": "Is approval allowed under the stated rule and exception?",
    },
    "ko": {
        "receipt": "총액을 센트 단위로 판정하세요.",
        "chart": "값이 가장 큰 계열을 고르세요.",
        "calendar": "한 달을 더한 날짜의 일을 판정하세요. 월말을 넘으면 말일로 제한하세요.",
        "rule": "제시된 규칙과 예외에 따라 승인이 허용되나요?",
    },
    "ja": {
        "receipt": "合計をセント単位で判定してください。",
        "chart": "値が最大の系列を選んでください。",
        "calendar": "1か月後の日を判定してください。月末を超える場合は月末に制限します。",
        "rule": "示された規則と例外では承認が許可されますか。",
    },
}
FAMILIES = ("receipt", "chart", "calendar", "rule")


def image_curriculum(split, count=8):
    if split not in SPLITS or type(count) is not int or count < 1:
        raise ValueError("invalid image split or count")
    variant = SPLITS.index(split)
    output = []
    for family in FAMILIES:
        for i in range(count):
            values, trace, answer, kind = [], "", None, "choice"
            if family == "receipt":
                unit, quantity, fee = 101 + i * 37 + variant * 19, 1 + i % 4, variant + i % 7
                answer = unit * quantity + fee
                values = [f"Unit: {unit} cents", f"Quantity: {quantity}", f"Fee: {fee} cents"]
                trace = f"Multiply {unit} by {quantity}, then add {fee}: {answer} cents."
                options = [answer - 1, answer, answer + 1, unit + fee]
                options = list(dict.fromkeys(options))
            elif family == "chart":
                values = [
                    (f"Series-{variant}-{j}", 10 + ((i * 7 + j * 13 + variant * 3) % 41))
                    for j in range(4)
                ]
                answer = max(values, key=lambda row: row[1])[0]
                trace = f"Compare the four values. {answer} has the greatest value."
                options = [label for label, _ in values]
            elif family == "calendar":
                start = date(2024 + variant * 100 + i, 1, 31)
                answer = add_months(start.isoformat(), 1).day
                values = [
                    "Start: " + start.isoformat(),
                    "Add: one calendar month",
                    "Clamp: last valid day",
                ]
                trace = f"The next month is February. Its last valid day is {answer}."
                options = [27, 28, 29, 31]
                kind = "score"
            else:
                threshold, credential, exception, override = (
                    variant + 2,
                    i % 6,
                    bool(i % 2),
                    bool(i % 3),
                )
                answer = not exception or (override and credential >= threshold)
                values = [
                    "Normally allow approval.",
                    "Exception blocks approval.",
                    f"Override defeats exception if level >= {threshold}.",
                    f"Exception: {exception}; Override: {override}; Level: {credential}.",
                ]
                trace = f"Check the exception and whether the override is authorized: approval is {bool(answer)}."
                options, kind = [False, True], "noul"
            image = Image.new("RGB", (512 + variant * 16, 300 + variant * 12), "white")
            draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=18 + variant)
            x, y = 12 + variant * 9, 12 + variant * 7
            draw.text((x, y), f"{split.upper()} {family.upper()} #{i}", fill="black", font=font)
            # Layouts, labels and rule combinations are split-specific. A chart
            # includes readable labels and bars; the transcript is a paired baseline.
            for line, value in enumerate(values):
                yy = y + 42 + line * (36 + variant)
                text = f"{value[0]}: {value[1]}" if family == "chart" else value
                draw.text((x, yy), text, fill="black", font=font)
                if family == "chart":
                    draw.rectangle(
                        (x + 190, yy, x + 190 + value[1] * 4, yy + 18),
                        fill=(40 + variant * 20, 70, 120),
                    )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw, lineage = buffer.getvalue(), f"native-images-v1/{split}/{family}/{i}"
            unreadable = i % 8 == 7
            if unreadable:
                image = Image.new("RGB", image.size, "white")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                raw = buffer.getvalue()
            transcript = "\n".join(f"{v[0]}: {v[1]}" if family == "chart" else v for v in values)
            for language in INSTRUCTIONS:
                instruction = INSTRUCTIONS[language][family]
                if kind == "noul":
                    q = Question.noul("q", instruction, 0.5 if unreadable else float(answer))
                else:
                    candidates = [
                        Candidate(str(v), str(v), v if kind == "score" else None) for v in options
                    ]
                    target = {
                        c.id: 1 / len(candidates) if unreadable else float(c.id == str(answer))
                        for c in candidates
                    }
                    q = Question("q", kind, instruction, candidates, target)
                metadata = {
                    "source": "repository-authored",
                    "license": "MIT",
                    "split": split,
                    "language": language,
                    "source_lineage": lineage,
                    "source_example_id": lineage + "/" + language,
                    "generator_template_id": f"image-{family}-layout-{variant}",
                    "rule_combination": f"{split}/{family}/{i}",
                    "document_voice": split,
                    "task_family": family,
                    "evidence_state": "deleted" if unreadable else "intact",
                    "oracle": {
                        "validator": "deterministic-offline-v1",
                        "answer": answer,
                        "readable": not unreadable,
                    },
                    "verified_traces": {
                        "q": "The image contains no readable evidence. Retain uncertainty."
                        if unreadable
                        else trace
                    },
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "modality": "image",
                    "media": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "data": base64.b64encode(raw).decode(),
                        }
                    ],
                }
                output.append(Sample("Use only the attached document.", [q], metadata))
                text_metadata = {k: v for k, v in metadata.items() if k != "media"}
                text_metadata.update(
                    modality="text", source_example_id=metadata["source_example_id"] + "/text"
                )
                output.append(
                    Sample(
                        ("Ledger", "Case note", "Audit entry", "Calibration record", "Review file")[
                            variant
                        ]
                        + f" {family} #{i}\n"
                        + ("No readable evidence." if unreadable else transcript),
                        [q],
                        text_metadata,
                    )
                )
    return output

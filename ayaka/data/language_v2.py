"""Authored native-language direct/readout rehearsal, never reused natural test data."""

from .reasoning_v2 import SPLITS
from .schema import Candidate, Question, Sample

TEXT = {
    "en": {
        "voices": [
            "Customer record",
            "Support note",
            "Review memo",
            "Case summary",
            "Independent audit",
        ],
        "intents": [
            "cancel the order",
            "change the address",
            "request a refund",
            "track the package",
        ],
        "asks": [
            "I want to {x}.",
            "Please help me {x}.",
            "My request is to {x}.",
            "The customer asks to {x}.",
            "The stated primary request: {x}.",
        ],
        "choice": "Choose the primary request.",
        "noul": "Is the customer a premium member?",
        "premium": [
            "The customer is a premium member.",
            "The customer is not a premium member.",
            "Membership status is absent.",
        ],
        "levels": ["routine", "needs attention", "urgent", "critical"],
        "score": "Report the severity according to the supplied rubric.",
        "observed": "Observed condition: {x}.",
        "uncertain": "The record omits the needed fact. Retain uncertainty.",
        "trace": "The stated evidence matches {x}. Use that criterion.",
    },
    "ko": {
        "voices": ["고객 기록", "상담 메모", "검토 문서", "사건 요약", "독립 감사"],
        "intents": ["주문 취소", "주소 변경", "환불 요청", "배송 조회"],
        "asks": [
            "{x}를 원합니다.",
            "{x}를 도와주세요.",
            "제 요청은 {x}입니다.",
            "고객은 {x}를 요청했습니다.",
            "명시된 주된 요청은 {x}입니다.",
        ],
        "choice": "주된 요청을 고르세요.",
        "noul": "고객은 프리미엄 회원인가요?",
        "premium": [
            "고객은 프리미엄 회원입니다.",
            "고객은 프리미엄 회원이 아닙니다.",
            "회원 등급 정보가 없습니다.",
        ],
        "levels": ["일상적", "주의 필요", "긴급", "위급"],
        "score": "제시된 기준에 따라 심각도를 판정하세요.",
        "observed": "관찰된 상태: {x}.",
        "uncertain": "판정에 필요한 사실이 기록에 없습니다. 불확실성을 유지합니다.",
        "trace": "명시된 증거는 {x}에 해당합니다. 이 기준을 적용합니다.",
    },
    "ja": {
        "voices": ["顧客記録", "相談メモ", "審査文書", "事例要約", "独立監査"],
        "intents": ["注文の取消", "住所の変更", "返金の申請", "配送の追跡"],
        "asks": [
            "{x}を希望します。",
            "{x}を手伝ってください。",
            "私の依頼は{x}です。",
            "顧客は{x}を依頼しました。",
            "明記された主な依頼は{x}です。",
        ],
        "choice": "主な依頼を選んでください。",
        "noul": "顧客はプレミアム会員ですか。",
        "premium": [
            "顧客はプレミアム会員です。",
            "顧客はプレミアム会員ではありません。",
            "会員区分の情報がありません。",
        ],
        "levels": ["通常", "注意が必要", "緊急", "重大"],
        "score": "示された基準に従って深刻度を判定してください。",
        "observed": "観察された状態：{x}。",
        "uncertain": "判定に必要な事実が記録にありません。不確実性を維持します。",
        "trace": "明記された証拠は{x}に該当します。この基準を適用します。",
    },
}


def language_curriculum(split, per_type=32):
    if split not in SPLITS or type(per_type) is not int or not 1 <= per_type <= 256:
        raise ValueError("invalid language split or count")
    variant, result = SPLITS.index(split), []
    for language, text in TEXT.items():
        for kind in ("choice", "noul", "score"):
            for i in range(per_type):
                unknown = kind == "noul" and i % 3 == 2
                if kind == "noul":
                    evidence = text["premium"][2 if unknown else i % 2]
                    q = Question.noul("q", text["noul"], 0.5 if unknown else float(i % 2 == 0))
                    trace = text["uncertain"] if unknown else text["trace"].format(x=evidence)
                else:
                    descriptions = text["intents"] if kind == "choice" else text["levels"]
                    index = (i + variant) % 4
                    candidates = [
                        Candidate(str(j), description, j if kind == "score" else None)
                        for j, description in enumerate(descriptions)
                    ]
                    q = Question(
                        "q",
                        kind,
                        text[kind],
                        candidates,
                        {str(j): float(j == index) for j in range(4)},
                    )
                    evidence = (
                        text["asks"][variant].format(x=descriptions[index])
                        if kind == "choice"
                        else text["observed"].format(x=descriptions[index])
                    )
                    if kind == "score":
                        # Explicit closed rubric: no arbitrary numeric scale is invented.
                        evidence += "\n" + "\n".join(
                            f"{j}: {description}" for j, description in enumerate(descriptions)
                        )
                    trace = text["trace"].format(x=descriptions[index])
                lineage = f"native-language-v1/{split}/{language}/{kind}/{i}"
                state = text["voices"][variant] + f" #{i}\n" + evidence
                metadata = {
                    "source": "repository-authored",
                    "license": "MIT",
                    "split": split,
                    "language": language,
                    "modality": "text",
                    "source_lineage": lineage,
                    "source_example_id": lineage,
                    "task_family": "native_language",
                    "generator_template_id": f"native-language/{split}/{language}/{kind}",
                    "rule_combination": f"native-language/{split}/{kind}/{i % 4}",
                    "document_voice": text["voices"][variant],
                    "evidence_state": "partial" if unknown else "intact",
                    "verified_traces": {"q": trace},
                    "trace_validator": "repository-authored-explicit-rubric-v1",
                }
                result.append(Sample(state, [q], metadata))
    return result

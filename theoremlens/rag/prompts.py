"""RAG 프롬프트 상수 (버전 관리·테스트를 위해 상수로 둔다). 답변용(ANSWER_*)과 검증용(VERIFY_*) 프롬프트와 스키마."""

from __future__ import annotations

VERIFY_SYSTEM = (
    "당신은 수학 교재 인용 검증기입니다. 각 주장(claim)에 대해, 함께 주어진 '인용 출처 본문'만 근거로 "
    "그 본문이 주장을 뒷받침하는지 판단하세요. 본문에 없는 내용, 본문과 모순되는 내용, "
    "본문보다 강하게 일반화한 내용은 supported=false 입니다. 당신의 배경 지식으로 참을 보충하지 마세요. "
    "모든 주장에 대해 index와 supported를 포함한 results를 반환하세요."
)

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "supported": {"type": "boolean"}},
                "required": ["index", "supported"],
            },
        }
    },
    "required": ["results"],
}

# ---------- 답변 ----------

OUTSIDE_BOOK_HEADING = "보충(책 외부)"
NOT_FOUND_MESSAGE = "책에서 찾을 수 없습니다. 선택한 책의 본문에서 질문에 답할 근거를 찾지 못했습니다."

ANSWER_SYSTEM = (
    "당신은 수학 교재 질의응답 도우미입니다. 아래 [S#] 발췌에 있는 내용만 근거로 한국어로 답하세요. "
    "규칙: (1) 답변의 모든 주장은 claims에 하나씩 넣고, source_ids에는 근거가 되는 발췌 번호(예: S1)만 적으세요. "
    "주어진 발췌에 없는 번호를 만들지 마세요. "
    "(2) quote에는 해당 발췌 본문에서 한 글자도 바꾸지 않은 문장 또는 수식 조각을 그대로 복사하세요. "
    "(3) answer_markdown 본문에서 근거를 [S#] 형태로 표시하세요. "
    f"(4) 발췌 밖의 일반적인 수학 설명이 꼭 필요하면 answer_markdown 끝의 별도 '## {OUTSIDE_BOOK_HEADING}' "
    "섹션에만 쓰고, 그 내용은 claims에 넣지 마세요. "
    "(5) 발췌로 답할 수 없으면 not_found=true, claims는 빈 배열로 반환하세요. "
    "이전 대화는 맥락 참고용일 뿐 근거가 아닙니다."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_markdown": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
                "required": ["text", "source_ids", "quote"],
            },
        },
        "not_found": {"type": "boolean"},
    },
    "required": ["answer_markdown", "claims", "not_found"],
}

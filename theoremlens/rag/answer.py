"""답변 생성: 검색 -> Gemini(JSON 스키마, temperature 0) -> 인용 검증 -> 앱이 그대로 쓰는 응답 모델.

인용 원문(citations[].text)은 검색된 DB 청크에서만 가져오며 모델 출력은 신뢰하지 않는다.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from .. import llm
from ..config import get_settings
from . import prompts, retrieve, verify

log = logging.getLogger(__name__)

# 긴 대화가 프롬프트와 비용을 키우지 않도록 최근 몇 턴만 맥락으로 쓴다
MAX_HISTORY_TURNS = 6
_HISTORY_ROLES = {"user": "사용자", "assistant": "도우미"}


class AnswerError(RuntimeError):
    """모델 응답이 약속한 JSON 형식이 아닐 때."""


class CitationOut(BaseModel):
    sid: str
    book_id: str
    book_title: str
    page_start: int
    page_end: int
    label: str | None = None
    display_label: str | None = None
    chapter: str | None = None
    section: str | None = None
    text: str  # DB 청크 원문


class ClaimOut(BaseModel):
    text: str
    display_text: str
    source_ids: list[str]
    quote: str
    status: Literal["verified", "unchecked", "rejected", "unsupported"]
    reason: str = ""


class VerificationOut(BaseModel):
    llm_checked: bool = False
    dropped: int = 0
    flagged: int = 0


class AnswerResponse(BaseModel):
    answer_markdown: str
    claims: list[ClaimOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    not_found: bool = False
    outside_book: bool = False  # True면 책에서 검증된 인용이 없는 답변이다
    verification: VerificationOut = Field(default_factory=VerificationOut)


def _not_found() -> AnswerResponse:
    return AnswerResponse(answer_markdown=prompts.NOT_FOUND_MESSAGE, not_found=True)


def _format_source(s: retrieve.Source) -> str:
    pages = str(s.page_start) if s.page_start == s.page_end else f"{s.page_start}-{s.page_end}"
    parts = [f"책: {s.book_title}", f"쪽: {pages}"]
    if s.display_label or s.label:
        parts.append(f"라벨: {s.display_label or s.label}")
    where = " > ".join(x for x in (s.chapter, s.section) if x)
    if where:
        parts.append(f"위치: {where}")
    return f"[{s.sid}] " + " | ".join(parts) + f"\n{s.text}"


def _format_history(history: Sequence[dict] | None) -> str:
    turns = [h for h in (history or []) if h.get("role") in _HISTORY_ROLES and str(h.get("content", "")).strip()]
    if not turns:
        return ""
    lines = [f"{_HISTORY_ROLES[h['role']]}: {h['content']}" for h in turns[-MAX_HISTORY_TURNS:]]
    return "## 이전 대화 (맥락 참고용, 근거 아님)\n" + "\n".join(lines) + "\n\n"


def build_contents(question: str, sources: list[retrieve.Source], history: Sequence[dict] | None = None) -> str:
    excerpts = "\n\n".join(_format_source(s) for s in sources)
    return f"{_format_history(history)}## 발췌\n{excerpts}\n\n## 질문\n{question}"


def _norm_sid(raw: str) -> str:
    return raw.strip().strip("[]").strip().upper()


def _mark_outside_book(markdown: str) -> str:
    return f"## {prompts.OUTSIDE_BOOK_HEADING}\n\n{markdown}"


def answer(
    question: str,
    book_ids: list[str],
    history: Sequence[dict] | None = None,
    fast: bool = False,
    on_unsupported: Literal["drop", "flag"] = "drop",
) -> AnswerResponse:
    """질문 하나에 대한 단일 진입점. history는 [{"role": "user"|"assistant", "content": str}]."""
    r = retrieve.retrieve(question, book_ids)
    if r.not_found:
        return _not_found()  # 근거가 약하면 LLM을 부르지 않는다

    s = get_settings()
    model = s.fast_model if fast else s.answer_model
    raw = llm.generate_json(
        model, prompts.ANSWER_SYSTEM, build_contents(question, r.sources, history), prompts.ANSWER_SCHEMA
    )
    try:
        data = json.loads(raw)
        markdown = str(data["answer_markdown"])
        raw_claims = list(data.get("claims") or [])
        model_not_found = bool(data.get("not_found", False))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise AnswerError("모델 응답을 해석하지 못했습니다.") from e
    if model_not_found:
        return AnswerResponse(answer_markdown=markdown or prompts.NOT_FOUND_MESSAGE, not_found=True)

    claims = verify.parse_claims([c for c in raw_claims if isinstance(c, dict)])
    for c in claims:
        c.source_ids = [_norm_sid(x) for x in c.source_ids]
    by_sid = {src.sid: src for src in r.sources}
    result = verify.verify_claims(claims, {sid: src.chunk_id for sid, src in by_sid.items()}, on_unsupported)

    out_claims = [
        ClaimOut(
            text=c.text,
            display_text=c.display_text,
            source_ids=c.source_ids,
            quote=c.quote,
            status=c.status,
            reason=c.reason,
        )
        for c in result.claims
    ]
    cited = sorted({sid for c in result.claims for sid in c.source_ids if sid in by_sid}, key=lambda x: int(x[1:]))
    citations = [
        CitationOut(**{k: getattr(by_sid[sid], k) for k in CitationOut.model_fields if k != "sid"}, sid=sid)
        for sid in cited
    ]
    # 검증을 통과한 인용이 하나도 없으면 책 근거가 없는 답이므로 반드시 책 외부로 표기한다
    outside = not any(not c.flagged for c in result.claims)
    return AnswerResponse(
        answer_markdown=_mark_outside_book(markdown) if outside else markdown,
        claims=out_claims,
        citations=citations,
        outside_book=outside,
        verification=VerificationOut(
            llm_checked=result.llm_checked,
            dropped=result.dropped,
            flagged=sum(c.flagged for c in result.claims),
        ),
    )

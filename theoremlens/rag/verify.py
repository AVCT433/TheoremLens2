"""답변 검증: 인용문(quote) 퍼지 매칭(결정적) + 빠른 모델의 지지 검사(두 번째 의견).

출처 본문은 항상 DB 청크에서 읽고, 모델 출력에 들어 있는 텍스트는 신뢰하지 않는다.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from rapidfuzz import fuzz

from .. import llm
from ..config import get_settings
from ..index import db
from . import prompts

log = logging.getLogger(__name__)

MATCH_THRESHOLD = 90.0  # OCR 잡음 한두 글자는 허용하되 의미가 바뀐 문장은 거절하는 경계
WARNING_PREFIX = "⚠ "

# ---------- 순수 함수: 정규화 / 퍼지 매칭 ----------

_SPACING_CMDS = re.compile(r"\\(?:left|right|displaystyle|big|bigg)(?![a-z])|\\[,;:! ]")
_ALIASES = [
    (re.compile(r"\\leq(?![a-z])"), r"\\le"),
    (re.compile(r"\\geq(?![a-z])"), r"\\ge"),
    (re.compile(r"\\(?:ldots|cdots)(?![a-z])"), r"\\dots"),
]
_DROP = re.compile(r"[\s${}]+")


def normalize(text: str) -> str:
    """공백·수식 구분자·표기 변형(le/leq, x^{2}/x^2 등)을 지워 비교용 문자열로 만든다."""
    s = _SPACING_CMDS.sub("", text.lower())
    for pat, rep in _ALIASES:
        s = pat.sub(rep, s)
    return _DROP.sub("", s)


def match_score(quote: str, source: str) -> float:
    """정규화 후 quote가 source 안에 (거의) 들어 있는 정도. 0~100."""
    nq, ns = normalize(quote), normalize(source)
    if not nq:
        return 0.0
    if nq in ns:
        return 100.0
    return float(fuzz.partial_ratio(nq, ns))


def quote_matches(quote: str, source: str, threshold: float = MATCH_THRESHOLD) -> bool:
    return match_score(quote, source) >= threshold


# ---------- 데이터 모델 ----------


@dataclass
class Claim:
    text: str
    source_ids: list[str] = field(default_factory=list)  # "S1" 같은 프롬프트 내 출처 번호
    quote: str = ""


@dataclass
class QuoteCheck:
    ok: bool
    reason: str = ""  # ok | no_source | no_quote | unknown_source | quote_not_found


@dataclass
class VerifiedClaim:
    text: str
    source_ids: list[str]
    quote: str
    status: Literal["verified", "unchecked", "rejected", "unsupported"]
    reason: str = ""

    @property
    def flagged(self) -> bool:
        return self.status in ("rejected", "unsupported")

    @property
    def display_text(self) -> str:
        return WARNING_PREFIX + self.text if self.flagged else self.text


@dataclass
class VerifyResult:
    claims: list[VerifiedClaim]
    dropped: int = 0
    llm_checked: bool = False


def parse_claims(raw: list[dict]) -> list[Claim]:
    """answer 단계의 JSON claims를 Claim으로 변환 (누락 필드는 빈 값)."""
    return [
        Claim(str(c.get("text", "")), [str(s) for s in c.get("source_ids") or []], str(c.get("quote") or ""))
        for c in raw
    ]


# ---------- 결정적 검사 ----------


def check_quotes(claims: list[Claim], sources: dict[str, str]) -> list[QuoteCheck]:
    """quote가 인용한 출처 본문 안에 실제로 있는지 규칙으로만 검사한다 (네트워크·DB 없음)."""
    out: list[QuoteCheck] = []
    for c in claims:
        if not c.source_ids:
            out.append(QuoteCheck(False, "no_source"))
        elif not normalize(c.quote):
            out.append(QuoteCheck(False, "no_quote"))
        else:
            texts = [sources[s] for s in c.source_ids if s in sources]
            if not texts:
                out.append(QuoteCheck(False, "unknown_source"))
            elif any(quote_matches(c.quote, t) for t in texts):
                out.append(QuoteCheck(True, "ok"))
            else:
                out.append(QuoteCheck(False, "quote_not_found"))
    return out


# ---------- LLM 지지 검사 ----------


def llm_support_check(claims: list[Claim], sources: dict[str, str]) -> list[bool] | None:
    """빠른 모델에게 '인용 본문이 주장을 뒷받침하는가'를 묻는다. 검사 불가면 None."""
    if not claims:
        return []
    items = []
    for i, c in enumerate(claims):
        cited = {s: sources[s] for s in c.source_ids if s in sources}
        items.append({"index": i, "claim": c.text, "quote": c.quote, "sources": cited})
    contents = json.dumps(items, ensure_ascii=False)
    try:
        raw = llm.generate_json(get_settings().fast_model, prompts.VERIFY_SYSTEM, contents, prompts.VERIFY_SCHEMA)
        results = json.loads(raw)["results"]
        flags = {int(r["index"]): bool(r["supported"]) for r in results}
    except llm.LLMNotConfigured:
        return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        log.warning("지지 검사 응답을 해석하지 못했다")
        return None
    # 응답에 빠진 claim은 보수적으로 미지지 처리
    return [flags.get(i, False) for i in range(len(claims))]


# ---------- DB 출처 + 정책 ----------


def load_sources(source_ids: dict[str, int]) -> dict[str, str]:
    """프롬프트 출처 번호(S#) -> DB 청크 본문. 없는 청크는 제외한다."""
    by_id = {c.id: c.text for c in db.get_chunks(list(source_ids.values()))}
    return {s: by_id[cid] for s, cid in source_ids.items() if cid in by_id}


def verify_claims(
    claims: list[Claim],
    source_ids: dict[str, int],
    on_unsupported: Literal["drop", "flag"] = "drop",
) -> VerifyResult:
    """quote 규칙 검사 -> (통과분만) LLM 지지 검사. 미지지 claim은 제거하거나 ⚠로 표시한다."""
    if not claims:
        return VerifyResult([])
    sources = load_sources(source_ids)  # 모델이 준 텍스트가 아니라 DB 본문만 사용
    checks = check_quotes(claims, sources)
    passed = [i for i, ck in enumerate(checks) if ck.ok]
    flags = llm_support_check([claims[i] for i in passed], sources) if passed else None
    support = dict(zip(passed, flags, strict=True)) if flags is not None else {}

    kept: list[VerifiedClaim] = []
    dropped = 0
    for i, c in enumerate(claims):
        if not checks[i].ok:
            status, reason = "rejected", checks[i].reason
        elif flags is None:
            status, reason = "unchecked", "llm_unavailable"
        elif support[i]:
            status, reason = "verified", ""
        else:
            status, reason = "unsupported", "not_supported_by_source"
        vc = VerifiedClaim(c.text, list(c.source_ids), c.quote, status, reason)  # type: ignore[arg-type]
        if vc.flagged and on_unsupported == "drop":
            dropped += 1
            continue
        kept.append(vc)
    return VerifyResult(kept, dropped, llm_checked=flags is not None)

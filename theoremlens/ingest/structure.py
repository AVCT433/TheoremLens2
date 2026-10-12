"""쪽별 Markdown → 수학 단위(정의·정리·증명·예제·연습문제 …) 청크.

순수 로직이다(네트워크/DB 없음). 청크 텍스트는 원문 줄을 그대로 이어 붙인 것이어서, 이후 인용 검증이
원문과 대조할 수 있다. 정리와 증명은 한 청크로 묶고, `$$…$$` 디스플레이 수식은 절대 쪼개지 않는다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# 표기 → 정규 unit_type. label/refs는 이 정규 유형(영문 소문자)으로 저장하고, 원문 표기는 display_label에만 남긴다.
_KEYWORDS: dict[str, str] = {
    "definition": "definition",
    "theorem": "theorem",
    "lemma": "lemma",
    "proposition": "proposition",
    "corollary": "corollary",
    "claim": "claim",
    "example": "example",
    "exercise": "exercise",
    "problem": "exercise",
    "remark": "remark",
    "proof": "proof",
    "정의": "definition",
    "정리": "theorem",
    "보조정리": "lemma",
    "명제": "proposition",
    "따름정리": "corollary",
    "예제": "example",
    "연습문제": "exercise",
    "문제": "exercise",
    "증명": "proof",
}
# 질문 속 라벨 조회(rag/retrieve)용 표기 → 유형. 본문 단위 인식에는 쓰지 않는 띄어쓴 표기와 '예'까지 포함하며,
# 매핑을 한 곳에 두려고 여기서 정의한다(증명은 label이 없으므로 제외).
TYPE_ALIASES: dict[str, str] = {
    **{k: v for k, v in _KEYWORDS.items() if v != "proof"},
    "보조 정리": "lemma",
    "따름 정리": "corollary",
    "예": "example",
}
# 증명이 뒤따를 수 있는 단위
_THEOREM_LIKE = {"theorem", "lemma", "proposition", "corollary", "claim"}

# 긴 낱말 먼저: '보조정리'가 '정리'보다 먼저 시도되어야 한다
_KW_ALT = "|".join(sorted(map(re.escape, _KEYWORDS), key=len, reverse=True))
_NUM = r"\d+(?:\.\d+)*"

_UNIT_START = re.compile(rf"(?P<kw>{_KW_ALT})(?![가-힣])(?:[ \t]*(?P<num>{_NUM}))?(?P<rest>.*)", re.IGNORECASE)
# 본문 속 상호참조. 앞이 글자이면(예: '보조정리' 안의 '정리') 제외한다
_REF = re.compile(rf"(?<![A-Za-z가-힣])(?P<kw>{_KW_ALT})[ \t]*(?P<num>{_NUM})", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6}\s+(?P<text>.+?)\s*#*\s*$")
_HEADING_LEVEL = re.compile(r"^(#+)")
_PLAIN_CHAPTER = re.compile(r"^(?:Chapter\s+\d+|제\s*\d+\s*장)")
_CHAPTER_TEXT = re.compile(r"^(?:Chapter\b|제\s*\d+\s*장)", re.IGNORECASE)
_SECTION_NUM = re.compile(r"^\d+\.\d+")
_CHAPTER_NUM = re.compile(r"^\d+(?:\.|\s)")
_QED = re.compile(r"∎|□|■|\\blacksquare|\\square|\\qed|\bQ\.?E\.?D\.?")
_DEFAULT_MAX_CHARS = 2000


@dataclass
class ChunkDraft:
    """`db.replace_chunks`에 넣을 청크 한 개. `as_row()`가 그 함수가 받는 dict 형태다."""

    ord: int
    unit_type: str
    text: str
    page_start: int
    page_end: int
    label: str | None = None
    display_label: str | None = None
    chapter: str | None = None
    section: str | None = None
    refs: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "ord": self.ord,
            "unit_type": self.unit_type,
            "label": self.label,
            "display_label": self.display_label,
            "chapter": self.chapter,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "refs": list(self.refs),
        }


@dataclass
class _Unit:
    unit_type: str
    is_proof: bool
    label: str | None
    display_label: str


@dataclass
class _Open:
    """아직 닫히지 않은 청크의 작업 상태."""

    unit_type: str
    chapter: str | None
    section: str | None
    label: str | None = None
    display_label: str | None = None
    lines: list[tuple[int, str]] = field(default_factory=list)
    only_headings: bool = False  # 제목 줄만 모인 상태: 뒤따르는 단위가 이 제목을 흡수한다
    has_proof: bool = False
    closed: bool = False  # 증명 끝 표시(∎ 등)를 지났다
    chars: int = 0


def _clean(line: str) -> str:
    """단위 머리말 판정용: 제목 표시와 굵게/기울임 표시를 걷어낸다."""
    s = line.strip()
    s = re.sub(r"^#+\s*", "", s)
    return s.lstrip("*_ \t")


def _parse_unit(line: str) -> _Unit | None:
    s = _clean(line)
    m = _UNIT_START.match(s)
    if not m:
        return None
    kw_text, num, rest = m["kw"], m["num"], m["rest"]
    if kw_text.isascii() and not kw_text[0].isupper():
        return None  # 문장 중간에 오는 소문자 'theorem' 같은 일반 낱말
    kw = kw_text.lower()
    unit_type = _KEYWORDS[kw]

    after = re.match(r"(?P<pre>[*_\s]*)(?:\((?P<name>[^)\n]*)\))?(?P<tail>.*)", rest)
    pre, name, tail = after["pre"], after["name"], after["tail"]
    bold_closed = "*" in pre or "_" in pre
    ok = bold_closed or not tail.strip() or tail.lstrip()[:1] in {".", ":"}
    if unit_type == "proof" and num is None:
        ok = ok or bool(re.match(r"\s+of\s", rest, re.IGNORECASE)) or bool(re.match(r"\s+of\s", tail))
    if num is None and unit_type != "proof":
        # 번호 없는 머리말은 종결 표시가 있어야 한다('## 연습문제' 같은 제목을 단위로 오인하지 않는다)
        ok = bold_closed or tail.lstrip()[:1] in {".", ":"}
    if not ok:
        return None

    label = f"{unit_type} {num}" if num and unit_type != "proof" else None
    display = f"{kw_text} {num}" if num else kw_text
    if name and name.strip():
        display += f" ({name.strip()})"
    return _Unit(unit_type, unit_type == "proof", label, display)


def _parse_heading(line: str) -> tuple[str, str] | None:
    """제목 줄이면 (kind, text). kind는 'chapter' 또는 'section'."""
    s = line.strip()
    m = _HEADING.match(s)
    if m:
        level = len(_HEADING_LEVEL.match(s)[1])
    elif _PLAIN_CHAPTER.match(s) and len(s) <= 80:
        m, level = re.match(r"(?P<text>.+)", s), 1
    else:
        return None
    text = re.sub(r"[*_]+", "", m["text"]).strip()
    if not text:
        return None
    if _CHAPTER_TEXT.match(text):
        return "chapter", text
    if _SECTION_NUM.match(text):
        return "section", text
    if _CHAPTER_NUM.match(text) or level == 1:
        return "chapter", text
    return "section", text


def extract_refs(text: str, own_label: str | None = None) -> list[str]:
    """본문이 인용하는 정규화 라벨(영문 유형 + 번호). 등장 순서를 유지하고 중복과 자기 자신은 뺀다."""
    out: list[str] = []
    for m in _REF.finditer(text):
        ref = f"{_KEYWORDS[m['kw'].lower()]} {m['num']}"
        if ref != own_label and ref not in out:
            out.append(ref)
    return out


def _flush(cur: _Open | None, out: list[ChunkDraft]) -> None:
    if cur is None:
        return
    lines = list(cur.lines)
    while lines and not lines[0][1].strip():
        lines.pop(0)
    while lines and not lines[-1][1].strip():
        lines.pop()
    if not lines:
        return
    text = "\n".join(t for _, t in lines)
    out.append(
        ChunkDraft(
            ord=len(out),
            unit_type=cur.unit_type,
            text=text,
            page_start=lines[0][0],
            page_end=lines[-1][0],
            label=cur.label,
            display_label=cur.display_label,
            chapter=cur.chapter,
            section=cur.section,
            refs=extract_refs(text, cur.label),
        )
    )


def build_chunks(pages: Iterable[tuple[int, str]], max_chars: int = _DEFAULT_MAX_CHARS) -> list[ChunkDraft]:
    """(1-based 쪽 번호, Markdown) 목록 → 청크 목록. 단위(정리+증명 등)는 길이와 무관하게 쪼개지 않고,
    단위가 아닌 본문만 `max_chars`를 넘으면 문단(빈 줄) 경계에서 나눈다."""
    out: list[ChunkDraft] = []
    chapter: str | None = None
    section: str | None = None
    cur: _Open | None = None
    in_math = False

    def start(unit_type: str, **kw) -> _Open:
        nonlocal cur
        _flush(cur, out)
        cur = _Open(unit_type, chapter, section, **kw)
        return cur

    for page_no, md in pages:
        for line in md.splitlines():
            # 이 줄이 디스플레이 수식 안에서 시작하면 어떤 구조 판정도 하지 않는다
            was_in_math = in_math
            if line.count("$$") % 2:
                in_math = not in_math

            if not was_in_math:
                if not line.strip():
                    if cur is not None and cur.unit_type == "text" and not cur.only_headings and cur.chars >= max_chars:
                        _flush(cur, out)
                        cur = None
                    elif cur is not None:
                        cur.lines.append((page_no, line))
                    continue

                unit = _parse_unit(line)
                heading = None if unit else _parse_heading(line)
                if unit:
                    if unit.is_proof and cur is not None and cur.unit_type in _THEOREM_LIKE and not cur.closed:
                        cur.has_proof = True  # 정리와 증명은 한 청크
                    elif cur is not None and cur.only_headings:
                        cur.unit_type, cur.label, cur.display_label = unit.unit_type, unit.label, unit.display_label
                        cur.only_headings = False
                        cur.has_proof = unit.is_proof
                    else:
                        cur = start(unit.unit_type, label=unit.label, display_label=unit.display_label)
                        cur.has_proof = unit.is_proof
                elif heading:
                    kind, text = heading
                    if kind == "chapter":
                        chapter, section = text, None
                    else:
                        section = text
                    if cur is not None and cur.only_headings:
                        cur.chapter, cur.section = chapter, section
                    else:
                        cur = start("text")
                        cur.only_headings = True
                else:
                    if cur is None or cur.closed:
                        cur = start("text")
                    cur.only_headings = False

            if cur is None:
                if not line.strip():
                    continue
                cur = start("text")
            cur.lines.append((page_no, line))
            cur.chars += len(line) + 1
            if cur.has_proof and _QED.search(line):
                cur.closed = True

    _flush(cur, out)
    return out

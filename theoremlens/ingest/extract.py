"""PDF → 쪽별 Markdown(+LaTeX) 추출.

- 텍스트 레이어가 있는 쪽: Marker(설치된 경우) 또는 PyMuPDF
- 스캔 쪽(텍스트 레이어 없음): Gemini 비전으로 그대로 옮겨 적기
"""

from __future__ import annotations

import importlib.util
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .. import llm
from ..config import get_settings

log = logging.getLogger(__name__)

TRANSCRIBE_SYSTEM = """당신은 수학 교재 페이지를 디지털 텍스트로 옮기는 전사기입니다.
규칙:
- 페이지에 보이는 내용을 **한 글자도 바꾸지 말고** 그대로 Markdown으로 옮기세요. 요약, 설명, 교정, 번역 금지.
- 수식은 LaTeX로: 인라인은 $...$, 디스플레이 수식은 $$...$$.
- 'Theorem 2.3', '정리 2.3', 'Proof.', '증명' 같은 머리말은 원문 표기 그대로 굵게(**...**) 표시하세요.
- 장/절 제목은 Markdown 헤딩(#, ##)으로.
- 머리글/바닥글의 쪽 번호와 반복되는 책 제목은 생략하세요.
- 그림은 [그림: 캡션] 으로만 표시하세요.
- 읽을 수 없는 부분은 [판독 불가] 로 표시하고 추측하지 마세요.
- 출력은 전사된 Markdown만."""


@dataclass
class PageText:
    page_no: int  # 1부터 시작
    markdown: str
    method: str


def page_count(pdf_path: Path) -> int:
    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


def render_page_png(pdf_path: Path, page_no: int, dpi: int = 150) -> bytes:
    with pymupdf.open(pdf_path) as doc:
        return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")


def classify_pages(pdf_path: Path) -> tuple[list[int], list[int]]:
    """(텍스트 레이어 쪽, 스캔 쪽) 번호 목록."""
    threshold = get_settings().scanned_char_threshold
    text_pages, scanned = [], []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            n = len(page.get_text("text").strip())
            if n >= threshold:
                text_pages.append(i)
            elif page.get_images():
                scanned.append(i)
            else:
                text_pages.append(i)  # 거의 빈 쪽
    return text_pages, scanned


def _marker_available() -> bool:
    return importlib.util.find_spec("marker") is not None


def _choose_text_method() -> str:
    m = get_settings().text_extractor
    if m == "auto":
        return "marker" if _marker_available() else "pymupdf"
    return m


def extract_pymupdf(pdf_path: Path, pages: list[int]) -> list[PageText]:
    out = []
    with pymupdf.open(pdf_path) as doc:
        for p in pages:
            out.append(PageText(p, doc[p - 1].get_text("text", sort=True).strip(), "pymupdf"))
    return out


_MARKER_PAGE_SEP = re.compile(r"\n*\{(\d+)\}-{20,}\n*")


def extract_marker(pdf_path: Path, pages: list[int]) -> list[PageText]:
    from marker.converters.pdf import PdfConverter  # type: ignore
    from marker.models import create_model_dict  # type: ignore

    page_range = ",".join(str(p - 1) for p in pages)
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"paginate_output": True, "page_range": page_range, "disable_image_extraction": True},
    )
    md = converter(str(pdf_path)).markdown
    parts = _MARKER_PAGE_SEP.split(md)
    # split 결과: [머리, idx0, text0, idx1, text1, ...]
    found: dict[int, str] = {}
    for i in range(1, len(parts) - 1, 2):
        found[int(parts[i]) + 1] = parts[i + 1].strip()
    if not found:
        raise RuntimeError("marker 출력에서 쪽 구분자를 찾지 못했습니다")
    return [PageText(p, found.get(p, ""), "marker") for p in pages]


def transcribe_page_gemini(pdf_path: Path, page_no: int) -> PageText:
    png = render_page_png(pdf_path, page_no, dpi=200)
    text = llm.generate_text(
        get_settings().fast_model,
        TRANSCRIBE_SYSTEM,
        [llm.image_part(png), f"PDF {page_no}쪽을 전사하세요."],
    )
    text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text.strip())
    return PageText(page_no, text, "gemini")


def extract_gemini(pdf_path: Path, pages: list[int], on_page: Callable[[], None] | None = None) -> list[PageText]:
    def one(p: int) -> PageText:
        r = transcribe_page_gemini(pdf_path, p)
        if on_page:
            on_page()
        return r

    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(one, pages))


def extract(
    pdf_path: Path,
    on_page: Callable[[], None] | None = None,
    skip: set[int] | None = None,
) -> list[PageText]:
    """모든 쪽을 추출한다. skip에 있는 쪽(이미 캐시됨)은 건너뛴다."""
    skip = skip or set()
    text_pages, scanned = classify_pages(pdf_path)
    text_pages = [p for p in text_pages if p not in skip]
    scanned = [p for p in scanned if p not in skip]
    method = _choose_text_method()
    log.info("text pages=%d (%s), scanned pages=%d", len(text_pages), method, len(scanned))

    results: list[PageText] = []
    if text_pages:
        if method == "gemini":
            results += extract_gemini(pdf_path, text_pages, on_page)
        else:
            if method == "marker":
                try:
                    results += extract_marker(pdf_path, text_pages)
                except Exception:
                    log.exception("marker 실패, PyMuPDF로 대체")
                    results += extract_pymupdf(pdf_path, text_pages)
            else:
                results += extract_pymupdf(pdf_path, text_pages)
            if on_page:
                for _ in text_pages:
                    on_page()
    if scanned:
        results += extract_gemini(pdf_path, scanned, on_page)
    return sorted(results, key=lambda r: r.page_no)

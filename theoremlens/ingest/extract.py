"""PDF → 쪽별 Markdown(+LaTeX) 추출.

모든 쪽(텍스트 레이어 쪽도 스캔 쪽도)을 MinerU(`mineru-kit parse`)가 변환한다. 쪽 구분이 있는
`middle_json` 출력을 받아 `page_idx`별 블록을 Markdown으로 조립한다. Gemini는 쓰지 않는다.
PyMuPDF는 쪽 수 세기와 미리보기 PNG 렌더링에만 쓴다.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..config import get_settings

log = logging.getLogger(__name__)

_STDERR_TAIL = 1500


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


# ---------- middle_json 블록 → Markdown ----------

_EQ_TAG = re.compile(r"<eq>(.*?)</eq>", re.DOTALL)
_DROPPED = {"header", "page_number"}  # 머리글·쪽 번호는 인용 근거가 아니다


def _inline(content) -> str:
    """text/equation_inline 조각 목록 → 한 줄 문자열. 문자열이면 그대로."""
    if isinstance(content, str):
        return content
    out = []
    for c in content or []:
        if c.get("type") == "equation_inline":
            out.append(f"${c.get('content', '')}$")
        else:
            out.append(str(c.get("content", "")))
    return "".join(out)


def _child(content, type_: str):
    for c in content or []:
        if isinstance(c, dict) and c.get("type") == type_:
            return c.get("content")
    return None


def _block_markdown(block: dict) -> str:
    t = block.get("type", "")
    content = block.get("content")
    if t in _DROPPED:
        return ""
    if t in ("doc_title", "paragraph_title"):
        text = _inline(content).strip()
        return f"{'#' * max(1, int(block.get('level') or 1))} {text}" if text else ""
    if t == "equation":
        eq = _inline(content).strip()
        return f"$$\n{eq}\n$$" if eq else ""
    if t == "list":
        return "\n".join(s for s in (_inline(item.get("content")).strip() for item in content or []) if s)
    if t in ("image", "chart"):
        cap = _inline(_child(content, f"{t}_caption")).strip()
        return f"[그림: {cap}]" if cap else "[그림]"
    if t == "table":
        body = _child(content, "table_body")
        return _EQ_TAG.sub(lambda m: f"${m.group(1)}$", body) if isinstance(body, str) else ""
    if t not in ("text", "page_footnote"):
        log.warning("알 수 없는 MinerU 블록 타입 %r: 텍스트만 유지", t)
    return _inline(content).strip()


def blocks_to_markdown(blocks: list[dict]) -> str:
    """한 쪽의 블록을 읽기 순서대로 빈 줄로 이어 붙인다. 원문 글자는 바꾸지 않는다."""
    return "\n\n".join(m for m in (_block_markdown(b) for b in blocks) if m)


# ---------- MinerU 호출 ----------


def page_ranges(pages: list[int]) -> str:
    """[1,2,3,7,9,10] → '1-3,7,9-10' (1-based, mineru-kit `--pages` 표기)."""
    out: list[str] = []
    ps = sorted(set(pages))
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[j] + 1:
            j += 1
        out.append(str(ps[i]) if i == j else f"{ps[i]}-{ps[j]}")
        i = j + 1
    return ",".join(out)


def _mineru_exe() -> str:
    """현재 인터프리터의 환경(.venv)에 설치된 mineru-kit를 우선 쓴다."""
    return shutil.which("mineru-kit", path=str(Path(sys.executable).parent)) or "mineru-kit"


def _run_mineru(pdf_path: Path, pages: list[int], out_path: Path) -> None:
    """`mineru-kit parse`로 pages(1-based)를 middle_json으로 변환해 out_path에 쓴다."""
    s = get_settings()
    cmd = [
        _mineru_exe(),
        "parse",
        str(pdf_path),
        "-o",
        str(out_path),
        "--format",
        "middle_json",
        "--tier",
        s.mineru_tier,
        "--ocr-mode",
        s.mineru_ocr_mode,
        "--pages",
        page_ranges(pages),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError("mineru-kit를 실행할 수 없습니다. `uv sync`로 mineru를 설치했는지 확인하세요.") from e
    if r.returncode != 0:
        tail = r.stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL:]
        raise RuntimeError(f"mineru-kit가 실패했습니다 (종료 코드 {r.returncode}): {tail}")


def extract_mineru(pdf_path: Path, pages: list[int]) -> list[PageText]:
    """pages(1-based)를 MinerU로 변환한다. 요청한 쪽이 결과에 없으면 조용히 비우지 않고 실패한다."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "middle.json"
        _run_mineru(pdf_path, pages, out_path)
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            by_page = {int(p["page_idx"]) + 1: p.get("blocks", []) for p in data["pages"]}
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"mineru 출력을 읽지 못했습니다: {e}") from e
    missing = [p for p in pages if p not in by_page]
    if missing:
        raise RuntimeError(f"mineru 출력에 {missing}쪽이 없습니다")
    return [PageText(p, blocks_to_markdown(by_page[p]), "mineru") for p in pages]


def extract(
    pdf_path: Path,
    on_page: Callable[[], None] | None = None,
    skip: set[int] | None = None,
) -> list[PageText]:
    """모든 쪽을 추출한다. skip에 있는 쪽(이미 캐시됨)은 건너뛴다."""
    skip = skip or set()
    todo = [p for p in range(1, page_count(pdf_path) + 1) if p not in skip]
    log.info("extract pages=%d (mineru)", len(todo))
    if not todo:
        return []
    results = extract_mineru(pdf_path, todo)
    if on_page:
        for _ in results:
            on_page()
    return sorted(results, key=lambda r: r.page_no)

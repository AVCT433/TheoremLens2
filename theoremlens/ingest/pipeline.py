"""인제스트 오케스트레이션: 추출 → 구조화 → DB 청크 → 벡터 색인.

`books.status/progress`를 갱신하고, 이미 `pages`에 있는 쪽은 다시 추출하지 않는다.
백그라운드 작업으로 돌리므로 예외는 밖으로 던지지 않고 status=error와 메시지로 남긴다.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..index import db, vectors
from . import extract, structure

log = logging.getLogger(__name__)

# 중간에 실패해도 지금까지 추출한 쪽이 남도록 이 쪽 수 단위로 저장한다(재실행 시 캐시로 재사용)
EXTRACT_BATCH = 20
# 진행률 배분: 추출 0~0.5, 색인 0.5~1.0
_EXTRACT_SHARE = 0.5


class IngestError(RuntimeError):
    """인제스트를 계속할 수 없는 도메인 오류(예: 추출된 내용 없음)."""


def has_odd_display_math(markdown: str) -> bool:
    """`$$` 개수가 홀수이면 수식이 닫히지 않은 것이므로 구조화가 어긋날 수 있다."""
    return markdown.count("$$") % 2 == 1


def _extract_pages(book_id: str, pdf_path: Path, total: int) -> None:
    done = {p["page_no"] for p in db.get_pages(book_id)}
    todo = [n for n in range(1, total + 1) if n not in done]
    db.update_book(book_id, status="extracting", progress=_EXTRACT_SHARE * len(done) / max(1, total))
    for i in range(0, len(todo), EXTRACT_BATCH):
        batch = set(todo[i : i + EXTRACT_BATCH])
        skip = set(range(1, total + 1)) - batch
        for p in extract.extract(pdf_path, skip=skip):
            db.save_page(book_id, p.page_no, p.markdown, p.method)
            done.add(p.page_no)
        db.update_book(book_id, progress=_EXTRACT_SHARE * len(done) / max(1, total))


def _run(book_id: str, pdf_path: Path) -> None:
    total = extract.page_count(pdf_path)
    db.update_book(book_id, num_pages=total, status="extracting", message="")
    _extract_pages(book_id, pdf_path, total)

    pages = db.get_pages(book_id)
    for p in pages:
        if has_odd_display_math(p["markdown"]):
            log.warning("쪽 %d: '$$' 개수가 홀수입니다(수식이 닫히지 않음). 구조화 결과를 확인하세요.", p["page_no"])
    drafts = structure.build_chunks((p["page_no"], p["markdown"]) for p in pages)
    if not drafts:
        raise IngestError("PDF에서 추출된 텍스트가 없습니다.")

    db.update_book(book_id, status="indexing", progress=_EXTRACT_SHARE)
    rows = [d.as_row() for d in drafts]
    ids = db.replace_chunks(book_id, rows)
    vectors.index_chunks(
        book_id,
        ids,
        rows,
        on_progress=lambda f: db.update_book(book_id, progress=_EXTRACT_SHARE + (1 - _EXTRACT_SHARE) * f),
    )
    db.update_book(book_id, status="ready", progress=1.0, message="")


def ingest_book(book_id: str, pdf_path: Path) -> None:
    """book_id의 PDF를 처리한다. 실패하면 status='error'와 사유를 기록한다."""
    try:
        _run(book_id, pdf_path)
    except Exception as e:  # noqa: BLE001 - 백그라운드 작업의 최상위: 실패를 상태로 남기고 삼킨다
        log.exception("인제스트 실패: %s", book_id)
        try:
            db.update_book(book_id, status="error", message=str(e) or type(e).__name__)
        except Exception:
            log.exception("오류 상태 기록 실패: %s", book_id)

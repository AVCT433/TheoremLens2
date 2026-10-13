"""인제스트 오케스트레이션: 추출 → 구조화 → DB 청크 → 벡터 색인.

`books.status/progress`를 갱신하고, 이미 `pages`에 있는 쪽은 다시 추출하지 않는다.
백그라운드 작업으로 돌리므로 예외는 밖으로 던지지 않고 status=error와 메시지로 남긴다.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

from ..config import get_settings
from ..index import db, vectors
from . import extract, structure

log = logging.getLogger(__name__)

# 중간에 실패해도 지금까지 추출한 쪽이 남도록 이 쪽 수 단위로 저장한다(재실행 시 캐시로 재사용)
EXTRACT_BATCH = 20
# 진행률 배분: 추출 0~0.5, 색인 0.5~1.0
_EXTRACT_SHARE = 0.5


# 서버 재시작으로 중단되었을 수 있는 상태(처리 중이던 책)
_IN_FLIGHT = ("queued", "extracting", "indexing")

# 삭제(purge_book)와 파이프라인의 쓰기·최종 확인이 서로 끼어들지 않게 하는 락.
# 락은 짧은 DB/벡터 조작 동안만 잡고, 추출·임베딩 같은 긴 작업 동안에는 잡지 않는다.
_lifecycle = threading.RLock()


class IngestError(RuntimeError):
    """인제스트를 계속할 수 없는 도메인 오류(예: 추출된 내용 없음)."""


class BookDeleted(RuntimeError):
    """인제스트 도중 책이 삭제되어 작업을 중단해야 한다."""


def _ensure_alive(book_id: str) -> None:
    if db.get_book(book_id) is None:
        raise BookDeleted(book_id)


def purge_book(book_id: str) -> None:
    """책의 벡터 → SQLite → 디스크 파일을 지운다. 진행 중인 인제스트가 있어도 안전하다.

    진행 중인 파이프라인은 다음 확인 지점에서 책이 사라진 것을 보고 멈추며, 그 뒤 남은 벡터를 스스로 치운다.
    """
    with _lifecycle:
        vectors.delete_book(book_id)
        db.delete_book(book_id)
    _remove_files(book_id)


def _remove_files(book_id: str) -> None:
    try:
        shutil.rmtree(get_settings().book_dir(book_id))
    except FileNotFoundError:
        pass
    except OSError:
        # Windows에서 추출 중인 PDF가 열려 있으면 실패할 수 있다. 책은 이미 DB에서 사라졌다.
        log.warning("책 파일 삭제 실패: %s", book_id, exc_info=True)


def recover_interrupted() -> list[tuple[str, Path]]:
    """재시작으로 멈춘 책을 찾는다. 원본 PDF가 있으면 (id, 경로)로 돌려주고, 없으면 error로 바꾼다.

    재개는 호출자가 `ingest_book`으로 한다. 이미 추출된 쪽은 `pages` 캐시로 건너뛴다.
    """
    todo: list[tuple[str, Path]] = []
    for b in db.list_books():
        if b["status"] not in _IN_FLIGHT:
            continue
        pdf = get_settings().book_dir(b["id"]) / "book.pdf"
        if pdf.exists():
            todo.append((b["id"], pdf))
        else:
            db.update_book(
                b["id"], status="error", message="서버가 재시작되어 처리가 중단되었고 원본 PDF를 찾을 수 없습니다."
            )
    return todo


def reindex_missing() -> list[str]:
    """ready인데 현재 임베딩 모델의 벡터가 없는 책을 SQLite 청크로 다시 임베딩한다. 재임베딩한 책 id를 돌려준다.

    임베딩 모델을 바꾸면 컬렉션이 달라져 모든 책이 여기에 해당한다. 재추출이나 Gemini 호출은 하지 않는다.
    모든 ready 책의 벡터가 갖춰지면 Gemini 시절의 옛 컬렉션을 지운다.
    """
    done: list[str] = []
    complete = True
    for b in db.list_books():
        if b["status"] != "ready" or vectors.has_book(b["id"]):
            continue
        chunks = db.book_chunks(b["id"])
        if not chunks:
            continue
        try:
            _reindex(b["id"], chunks)
            done.append(b["id"])
        except BookDeleted:
            _discard_leftovers(b["id"])
        except Exception as e:  # noqa: BLE001 - 책 하나의 실패가 나머지 재임베딩을 막지 않게 상태로 남긴다
            log.exception("재임베딩 실패: %s", b["id"])
            complete = False
            with _lifecycle:
                if db.get_book(b["id"]) is not None:
                    db.update_book(b["id"], status="error", message=f"재임베딩 실패: {e or type(e).__name__}")
    if complete:
        if vectors.drop_legacy():
            log.info("이전 임베딩 컬렉션을 삭제했습니다.")
    return done


def _reindex(book_id: str, chunks: list[db.Chunk]) -> None:
    log.info("임베딩 모델 변경으로 재임베딩: %s (%d 청크)", book_id, len(chunks))
    with _lifecycle:
        _ensure_alive(book_id)
        db.update_book(book_id, status="indexing", progress=0.0, message="")

    def _progress(f: float) -> None:
        _ensure_alive(book_id)
        db.update_book(book_id, progress=f)

    rows = [vars(c) for c in chunks]
    vectors.index_chunks(book_id, [c.id for c in chunks], rows, on_progress=_progress)
    with _lifecycle:
        _ensure_alive(book_id)
        db.update_book(book_id, status="ready", progress=1.0, message="")


def has_odd_display_math(markdown: str) -> bool:
    """`$$` 개수가 홀수이면 수식이 닫히지 않은 것이므로 구조화가 어긋날 수 있다."""
    return markdown.count("$$") % 2 == 1


def _extract_pages(book_id: str, pdf_path: Path, total: int) -> None:
    done = {p["page_no"] for p in db.get_pages(book_id)}
    todo = [n for n in range(1, total + 1) if n not in done]
    db.update_book(book_id, status="extracting", progress=_EXTRACT_SHARE * len(done) / max(1, total))
    for i in range(0, len(todo), EXTRACT_BATCH):
        _ensure_alive(book_id)
        batch = set(todo[i : i + EXTRACT_BATCH])
        skip = set(range(1, total + 1)) - batch
        for p in extract.extract(pdf_path, skip=skip):
            with _lifecycle:
                _ensure_alive(book_id)
                db.save_page(book_id, p.page_no, p.markdown, p.method)
            done.add(p.page_no)
        db.update_book(book_id, progress=_EXTRACT_SHARE * len(done) / max(1, total))


def _run(book_id: str, pdf_path: Path) -> None:
    _ensure_alive(book_id)
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
    with _lifecycle:
        _ensure_alive(book_id)
        ids = db.replace_chunks(book_id, rows)

    def _progress(f: float) -> None:
        _ensure_alive(book_id)  # 삭제되었으면 남은 임베딩을 중단한다
        db.update_book(book_id, progress=_EXTRACT_SHARE + (1 - _EXTRACT_SHARE) * f)

    vectors.index_chunks(book_id, ids, rows, on_progress=_progress)
    with _lifecycle:
        _ensure_alive(book_id)  # 마지막 배치가 삭제 뒤에 추가됐을 수 있으므로 ready 전에 다시 확인
        db.update_book(book_id, status="ready", progress=1.0, message="")


def ingest_book(book_id: str, pdf_path: Path) -> None:
    """book_id의 PDF를 처리한다. 실패하면 status='error'와 사유를 기록한다."""
    try:
        _run(book_id, pdf_path)
    except BookDeleted:
        _discard_leftovers(book_id)
    except Exception as e:  # noqa: BLE001 - 백그라운드 작업의 최상위: 실패를 상태로 남기고 삼킨다
        log.exception("인제스트 실패: %s", book_id)
        try:
            with _lifecycle:
                if db.get_book(book_id) is None:  # 삭제와 겹친 실패(예: FK 위반)는 오류가 아니라 삭제로 본다
                    _discard_leftovers(book_id)
                else:
                    db.update_book(book_id, status="error", message=str(e) or type(e).__name__)
        except Exception:
            log.exception("오류 상태 기록 실패: %s", book_id)


def _discard_leftovers(book_id: str) -> None:
    """삭제된 책의 인제스트가 삭제 뒤에 남긴 벡터와 파일을 치운다."""
    log.info("인제스트 중 책이 삭제되어 중단: %s", book_id)
    with _lifecycle:
        if db.get_book(book_id) is None:
            vectors.delete_book(book_id)
            db.delete_book(book_id)  # 삭제 뒤에 쓰인 행이 있다면 함께 정리
    _remove_files(book_id)

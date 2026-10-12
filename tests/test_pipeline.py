"""ingest/pipeline 테스트: 생성한 PDF로 extract → structure → DB → 벡터 end to end."""

from __future__ import annotations

import logging

import pytest

from theoremlens.index import db, vectors
from theoremlens.ingest import extract as ex
from theoremlens.ingest import pipeline

SCAN_MD = "**Theorem 9.1** scanned compact sets are closed."


def _book(pdf) -> str:
    db.create_book("b1", "Book", pdf.name)
    return "b1"


def _chroma_count() -> int:
    return vectors._collection().count()


def test_ingest_mixed_pdf_fills_pages_chunks_vectors_and_marks_ready(make_pdf, fake_llm, fake_embed):
    fake_llm.default_text = SCAN_MD
    pdf = make_pdf(["text", "scan", "text"])
    bid = _book(pdf)

    pipeline.ingest_book(bid, pdf)

    book = db.get_book(bid)
    assert book["status"] == "ready"
    assert book["progress"] == 1.0
    assert book["num_pages"] == 3
    pages = {p["page_no"]: p for p in db.get_pages(bid)}
    assert sorted(pages) == [1, 2, 3]
    assert pages[2]["method"] == "gemini" and pages[1]["method"] == "pymupdf"
    chunks = db.chunks_by_label(["theorem 9.1"], [bid])
    assert chunks and chunks[0].page_start == 2
    assert _chroma_count() == db.list_books()[0]["num_chunks"] > 0
    assert len(fake_llm.calls) == 1  # 스캔 쪽 하나만 Gemini 전사


def test_rerun_skips_pages_already_in_db(make_pdf, fake_llm, fake_embed):
    fake_llm.default_text = SCAN_MD
    pdf = make_pdf(["text", "scan"])
    bid = _book(pdf)
    pipeline.ingest_book(bid, pdf)
    assert len(fake_llm.calls) == 1

    pipeline.ingest_book(bid, pdf)

    assert len(fake_llm.calls) == 1  # 이미 추출된 스캔 쪽은 다시 전사하지 않는다
    assert db.get_book(bid)["status"] == "ready"
    assert _chroma_count() == db.list_books()[0]["num_chunks"]  # 벡터가 중복되지 않는다


def test_rerun_only_extracts_missing_pages(make_pdf, fake_llm, fake_embed, monkeypatch):
    pdf = make_pdf(["text", "text", "text"])
    bid = _book(pdf)
    db.save_page(bid, 1, "cached page one", "pymupdf")
    seen: list[set[int]] = []
    real = ex.extract

    def spy(path, on_page=None, skip=None):
        seen.append(set(skip or ()))
        return real(path, on_page=on_page, skip=skip)

    monkeypatch.setattr(ex, "extract", spy)
    pipeline.ingest_book(bid, pdf)

    assert all(1 in s for s in seen)
    assert db.get_page(bid, 1)["markdown"] == "cached page one"  # 캐시된 쪽은 덮어쓰지 않는다
    assert db.get_page(bid, 3) is not None


def test_failure_sets_error_status_and_message(make_pdf, fake_llm, fake_embed):
    pdf = make_pdf(["scan"])
    bid = _book(pdf)

    def boom(*a, **k):
        raise RuntimeError("전사 실패 테스트")

    fake_llm.text_responses = [boom]
    pipeline.ingest_book(bid, pdf)  # 백그라운드 작업이므로 예외를 밖으로 던지지 않는다

    book = db.get_book(bid)
    assert book["status"] == "error"
    assert "전사 실패 테스트" in book["message"]


def test_missing_api_key_reports_error_for_scanned_pdf(make_pdf):
    pdf = make_pdf(["scan"])
    bid = _book(pdf)
    pipeline.ingest_book(bid, pdf)
    book = db.get_book(bid)
    assert book["status"] == "error" and "GEMINI_API_KEY" in book["message"]


def test_blank_pdf_without_content_is_error(make_pdf, fake_llm, fake_embed):
    pdf = make_pdf(["blank"])
    bid = _book(pdf)
    pipeline.ingest_book(bid, pdf)
    assert db.get_book(bid)["status"] == "error"


def test_progress_is_monotonic_and_status_passes_through_stages(make_pdf, fake_llm, fake_embed, monkeypatch):
    pdf = make_pdf(["text", "text"])
    bid = _book(pdf)
    seen: list[tuple[str, float]] = []
    real = db.update_book

    def spy(book_id, **f):
        real(book_id, **f)
        b = db.get_book(book_id)
        seen.append((b["status"], b["progress"]))

    monkeypatch.setattr(db, "update_book", spy)
    pipeline.ingest_book(bid, pdf)

    statuses = [s for s, _ in seen]
    assert statuses.index("extracting") < statuses.index("indexing") < statuses.index("ready")
    progress = [p for _, p in seen]
    assert progress == sorted(progress)


def test_odd_display_math_page_logs_warning(make_pdf, fake_llm, fake_embed, caplog):
    fake_llm.default_text = "Theorem 1.1 $$ x = 1"
    pdf = make_pdf(["scan"])
    bid = _book(pdf)
    with caplog.at_level(logging.WARNING, logger="theoremlens.ingest.pipeline"):
        pipeline.ingest_book(bid, pdf)
    assert any("$$" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("md,expected", [("a $$ b $$ c", False), ("a $$ b", True), ("", False)])
def test_has_odd_display_math(md, expected):
    assert pipeline.has_odd_display_math(md) is expected

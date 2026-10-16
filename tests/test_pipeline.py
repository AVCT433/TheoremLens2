"""ingest/pipeline 테스트: 생성한 PDF로 extract → structure → DB → 벡터 end to end."""

from __future__ import annotations

import logging

import pytest

from tests.conftest import add_legacy_collection, bow_vector, collection_names
from theoremlens import embedder
from theoremlens.config import get_settings
from theoremlens.index import db, vectors
from theoremlens.ingest import extract as ex
from theoremlens.ingest import pipeline

SCAN_MD = "**Theorem 9.1** scanned compact sets are closed."


def _book(pdf) -> str:
    db.create_book("b1", "Book", pdf.name)
    return "b1"


def _chroma_count() -> int:
    return vectors._collection().count()


def test_ingest_mixed_pdf_fills_pages_chunks_vectors_and_marks_ready(make_pdf, fake_mineru, fake_llm, fake_embed):
    fake_mineru.scan_text = SCAN_MD
    pdf = make_pdf(["text", "scan", "text"])
    bid = _book(pdf)

    pipeline.ingest_book(bid, pdf)

    book = db.get_book(bid)
    assert book["status"] == "ready"
    assert book["progress"] == 1.0
    assert book["num_pages"] == 3
    pages = {p["page_no"]: p for p in db.get_pages(bid)}
    assert sorted(pages) == [1, 2, 3]
    assert {p["method"] for p in pages.values()} == {"mineru"}
    chunks = db.chunks_by_label(["theorem 9.1"], [bid])
    assert chunks and chunks[0].page_start == 2
    assert _chroma_count() == db.list_books()[0]["num_chunks"] > 0
    assert fake_mineru.requested == [[1, 2, 3]]
    assert fake_llm.calls == []  # 추출에는 Gemini를 쓰지 않는다


def test_rerun_skips_pages_already_in_db(make_pdf, fake_mineru, fake_embed):
    fake_mineru.scan_text = SCAN_MD
    pdf = make_pdf(["text", "scan"])
    bid = _book(pdf)
    pipeline.ingest_book(bid, pdf)
    assert len(fake_mineru.requested) == 1

    pipeline.ingest_book(bid, pdf)

    assert len(fake_mineru.requested) == 1  # 이미 추출된 쪽은 MinerU를 다시 부르지 않는다
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


def test_mineru_failure_sets_error_status_and_saves_no_page(make_pdf, fake_embed, monkeypatch):
    pdf = make_pdf(["text", "scan"])
    bid = _book(pdf)

    def boom(*a, **k):
        raise RuntimeError("mineru 실패 테스트")

    monkeypatch.setattr(ex, "_run_mineru", boom)
    pipeline.ingest_book(bid, pdf)  # 백그라운드 작업이므로 예외를 밖으로 던지지 않는다

    book = db.get_book(bid)
    assert book["status"] == "error"
    assert "mineru 실패 테스트" in book["message"]
    assert db.get_pages(bid) == []  # PyMuPDF 등으로 대체해 깨진 텍스트를 저장하지 않는다


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


def test_odd_display_math_page_logs_warning(make_pdf, fake_mineru, fake_embed, caplog):
    fake_mineru.scan_text = "Theorem 1.1 $$ x = 1"
    pdf = make_pdf(["scan"])
    bid = _book(pdf)
    with caplog.at_level(logging.WARNING, logger="theoremlens.ingest.pipeline"):
        pipeline.ingest_book(bid, pdf)
    assert any("$$" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("md,expected", [("a $$ b $$ c", False), ("a $$ b", True), ("", False)])
def test_has_odd_display_math(md, expected):
    assert pipeline.has_odd_display_math(md) is expected


# ---------- 삭제와의 경쟁 조건 ----------


def test_book_deleted_during_extraction_leaves_no_rows_or_vectors(make_pdf, fake_llm, fake_embed, monkeypatch):
    pdf = make_pdf(["text", "text"])
    bid = _book(pdf)
    real = ex.extract

    def extract_then_delete(path, on_page=None, skip=None):
        pages = list(real(path, on_page=on_page, skip=skip))
        pipeline.purge_book(bid)  # 추출이 끝나 저장하기 직전에 사용자가 책을 지운다
        return pages

    monkeypatch.setattr(ex, "extract", extract_then_delete)
    pipeline.ingest_book(bid, pdf)

    assert db.get_book(bid) is None
    assert db.get_pages(bid) == []
    assert db.chunks_by_label(["theorem 1.1"], [bid]) == []
    assert _chroma_count() == 0


def test_book_deleted_during_vector_indexing_leaves_no_vectors(make_pdf, fake_llm, monkeypatch):
    pdf = make_pdf(["text", "text"])
    bid = _book(pdf)

    def embed_then_delete(texts, task):
        pipeline.purge_book(bid)  # 임베딩 도중 삭제
        return [bow_vector(t) for t in texts]

    monkeypatch.setattr(embedder, "embed", embed_then_delete)
    pipeline.ingest_book(bid, pdf)

    assert db.get_book(bid) is None
    assert db.chunks_by_label(["theorem 1.1"], [bid]) == []
    assert _chroma_count() == 0


def test_purge_book_removes_rows_vectors_and_files(make_pdf, fake_llm, fake_embed):
    pdf = make_pdf(["text"])
    bid = _book(pdf)
    pipeline.ingest_book(bid, pdf)
    book_dir = get_settings().book_dir(bid)
    book_dir.mkdir(parents=True)
    (book_dir / "book.pdf").write_bytes(pdf.read_bytes())

    pipeline.purge_book(bid)

    assert db.get_book(bid) is None and db.get_pages(bid) == []
    assert _chroma_count() == 0
    assert not book_dir.exists()


# ---------- 재시작 복구 ----------


def _stuck_book(bid: str, status: str, with_pdf: bool, make_pdf=None) -> None:
    db.create_book(bid, "Stuck", "stuck.pdf")
    db.update_book(bid, status=status, progress=0.3)
    if with_pdf:
        d = get_settings().book_dir(bid)
        d.mkdir(parents=True)
        (d / "book.pdf").write_bytes(make_pdf(["text", "text"]).read_bytes())


@pytest.mark.parametrize("status", ["queued", "extracting", "indexing"])
def test_recover_interrupted_returns_stuck_books_with_pdf(status, make_pdf):
    _stuck_book("s1", status, True, make_pdf)
    db.create_book("done", "Done", "d.pdf")
    db.update_book("done", status="ready", progress=1.0)
    db.create_book("bad", "Bad", "b.pdf")
    db.update_book("bad", status="error", message="x")

    todo = pipeline.recover_interrupted()

    assert [bid for bid, _ in todo] == ["s1"]
    assert todo[0][1].name == "book.pdf" and todo[0][1].exists()
    assert db.get_book("done")["status"] == "ready"
    assert db.get_book("bad")["message"] == "x"


def test_recover_interrupted_marks_book_without_pdf_as_error():
    _stuck_book("s2", "extracting", False)

    todo = pipeline.recover_interrupted()

    assert todo == []
    book = db.get_book("s2")
    assert book["status"] == "error" and book["message"]


def test_resumed_ingest_reuses_extracted_pages(make_pdf, fake_mineru, fake_embed):
    fake_mineru.scan_text = SCAN_MD
    pdf = make_pdf(["text", "scan"])
    bid = _book(pdf)
    d = get_settings().book_dir(bid)
    d.mkdir(parents=True)
    (d / "book.pdf").write_bytes(pdf.read_bytes())
    db.save_page(bid, 2, SCAN_MD, "mineru")  # 재시작 전에 이미 추출된 쪽
    db.update_book(bid, status="extracting")

    for b, p in pipeline.recover_interrupted():
        pipeline.ingest_book(b, p)

    assert db.get_book(bid)["status"] == "ready"
    assert fake_mineru.requested == [[1]]  # 이미 있는 2쪽은 다시 변환하지 않는다


# ---------- 임베딩 모델 교체 후 재임베딩 ----------


def _ready_book_without_vectors(bid: str, *texts: str) -> list[int]:
    db.create_book(bid, bid, f"{bid}.pdf")
    rows = [
        {
            "ord": i,
            "unit_type": "theorem",
            "label": f"theorem {i + 1}",
            "display_label": f"Theorem {i + 1}",
            "chapter": None,
            "section": None,
            "page_start": 1,
            "page_end": 1,
            "text": t,
            "refs": [],
        }
        for i, t in enumerate(texts)
    ]
    ids = db.replace_chunks(bid, rows)
    db.update_book(bid, status="ready", progress=1.0)
    return ids


def test_reindex_missing_embeds_ready_books_that_have_no_vectors(fake_embed, fake_llm):
    ids = _ready_book_without_vectors("old", "compact sets are closed", "banana bread")

    assert pipeline.reindex_missing() == ["old"]

    assert vectors.has_book("old")
    assert [i for i, _ in vectors.search("compact sets closed", ["old"], 2)][0] == ids[0]
    book = db.get_book("old")
    assert book["status"] == "ready" and book["progress"] == 1.0
    assert fake_llm.calls == []  # 재추출·Gemini 호출 없이 SQLite 청크만 쓴다


def test_reindex_missing_skips_books_that_already_have_vectors(fake_embed):
    ids = _ready_book_without_vectors("b1", "metric space")
    vectors.index_chunks("b1", ids, [{"text": "metric space"}])
    fake_embed.calls.clear()

    assert pipeline.reindex_missing() == []
    assert fake_embed.calls == []


def test_reindex_missing_ignores_unready_books_and_books_without_chunks(fake_embed):
    db.create_book("empty", "E", "e.pdf")
    db.update_book("empty", status="ready", progress=1.0)
    _ready_book_without_vectors("err", "text")
    db.update_book("err", status="error", message="boom")

    assert pipeline.reindex_missing() == []
    assert fake_embed.calls == []
    assert db.get_book("err")["message"] == "boom"


def test_reindex_missing_marks_book_indexing_while_embedding(fake_embed, monkeypatch):
    _ready_book_without_vectors("b1", "text")
    seen = []
    orig = embedder.embed

    def spy(texts, task):
        seen.append(db.get_book("b1")["status"])
        return orig(texts, task)

    monkeypatch.setattr(embedder, "embed", spy)
    pipeline.reindex_missing()
    assert seen == ["indexing"]


def test_reindex_missing_drops_legacy_collection_when_all_books_covered(fake_embed):
    _ready_book_without_vectors("b1", "text")
    add_legacy_collection()

    pipeline.reindex_missing()

    assert "chunks" not in collection_names()


def test_reindex_failure_sets_error_and_keeps_legacy_collection(fake_embed, monkeypatch):
    _ready_book_without_vectors("bad", "text")
    _ready_book_without_vectors("good", "other text")
    add_legacy_collection()
    real = embedder.embed

    def flaky(texts, task):
        if any(t.endswith("\ntext") for t in texts):
            raise RuntimeError("gpu oom")
        return real(texts, task)

    monkeypatch.setattr(embedder, "embed", flaky)

    done = pipeline.reindex_missing()

    assert done == ["good"]
    assert db.get_book("bad")["status"] == "error" and "gpu oom" in db.get_book("bad")["message"]
    assert db.get_book("good")["status"] == "ready"
    assert "chunks" in collection_names()


def test_book_deleted_during_reindex_leaves_no_vectors(fake_embed, monkeypatch):
    _ready_book_without_vectors("b1", "text")
    real = embedder.embed

    def embed_then_delete(texts, task):
        out = real(texts, task)
        pipeline.purge_book("b1")
        return out

    monkeypatch.setattr(embedder, "embed", embed_then_delete)

    assert pipeline.reindex_missing() == []

    assert db.get_book("b1") is None
    assert not vectors.has_book("b1")

"""index/vectors 테스트: 검색 순서, 책 필터, 재인덱싱, 책 삭제."""

from __future__ import annotations

from tests.conftest import add_legacy_collection, collection_names
from theoremlens.config import get_settings
from theoremlens.index import vectors


def _chunks(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


def test_embed_text_for_builds_header_then_body():
    c = {"chapter": "Ch1", "section": "1.2", "display_label": "Theorem 1", "text": "body"}
    assert vectors.embed_text_for(c) == "Ch1 / 1.2 / Theorem 1\nbody"
    assert vectors.embed_text_for({"text": "body", "chapter": None}) == "body"


def test_search_orders_by_similarity(fake_embed):
    vectors.index_chunks("b1", [1, 2, 3], _chunks("compact sets are closed", "banana bread recipe", "compact"))
    res = vectors.search("compact sets closed", ["b1"], 3)
    assert res[0][0] == 1
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True)
    assert sims[0] > 0.8


def test_documents_and_queries_use_distinct_embedding_tasks(fake_embed):
    vectors.index_chunks("b1", [1], _chunks("x"))
    vectors.search("x", ["b1"], 1)
    assert [t for _, t in fake_embed.calls] == ["document", "query"]


def test_search_single_and_multi_book_filter(fake_embed):
    vectors.index_chunks("b1", [1], _chunks("metric space"))
    vectors.index_chunks("b2", [2], _chunks("metric space"))
    assert [i for i, _ in vectors.search("metric space", ["b1"], 5)] == [1]
    assert [i for i, _ in vectors.search("metric space", ["b2"], 5)] == [2]
    assert {i for i, _ in vectors.search("metric space", ["b1", "b2"], 5)} == {1, 2}


def test_search_with_no_books_returns_empty_without_embedding(fake_embed):
    assert vectors.search("q", [], 5) == []
    assert fake_embed.calls == []


def test_reindexing_replaces_old_vectors(fake_embed):
    vectors.index_chunks("b1", [1, 2], _chunks("alpha one", "beta two"))
    vectors.index_chunks("b1", [3], _chunks("gamma three"))
    ids = {i for i, _ in vectors.search("alpha beta gamma", ["b1"], 10)}
    assert ids == {3}


def test_index_chunks_batches_and_reports_progress(fake_embed):
    n = 130
    progress: list[float] = []
    vectors.index_chunks("b1", list(range(1, n + 1)), _chunks(*[f"w{i}" for i in range(n)]), progress.append)
    assert [len(t) for t, _ in fake_embed.calls] == [64, 64, 2]
    assert progress[-1] == 1.0
    assert progress == sorted(progress)


def test_delete_book_leaves_other_books_intact(fake_embed):
    vectors.index_chunks("b1", [1], _chunks("shared words"))
    vectors.index_chunks("b2", [2], _chunks("shared words"))
    vectors.delete_book("b1")
    assert vectors.search("shared words", ["b1"], 5) == []
    assert [i for i, _ in vectors.search("shared words", ["b2"], 5)] == [2]


def test_collection_name_is_per_embedding_model(monkeypatch):
    monkeypatch.setenv("THEOREMLENS_EMBED_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()
    assert vectors.collection_name() == "chunks__baai-bge-m3"
    monkeypatch.setenv("THEOREMLENS_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    get_settings.cache_clear()
    assert vectors.collection_name() == "chunks__nomic-ai-nomic-embed-text-v1.5"


def test_vectors_of_different_models_do_not_mix(fake_embed, monkeypatch):
    vectors.index_chunks("b1", [1], _chunks("metric space"))
    assert vectors.has_book("b1")
    monkeypatch.setenv("THEOREMLENS_EMBED_MODEL", "other/model")
    get_settings.cache_clear()
    vectors._collection.cache_clear()
    assert not vectors.has_book("b1")
    assert vectors.search("metric space", ["b1"], 5) == []


def test_has_book_reflects_indexing_and_deletion(fake_embed):
    assert not vectors.has_book("b1")
    vectors.index_chunks("b1", [1], _chunks("x"))
    assert vectors.has_book("b1") and not vectors.has_book("b2")
    vectors.delete_book("b1")
    assert not vectors.has_book("b1")


def test_drop_legacy_removes_old_gemini_collection_only(fake_embed):
    vectors.index_chunks("b1", [1], _chunks("keep me"))
    add_legacy_collection()
    assert "chunks" in collection_names()
    assert vectors.drop_legacy() is True
    assert "chunks" not in collection_names()
    assert [i for i, _ in vectors.search("keep me", ["b1"], 5)] == [1]


def test_drop_legacy_without_legacy_collection_is_noop(fake_embed):
    vectors.index_chunks("b1", [1], _chunks("keep me"))
    assert vectors.drop_legacy() is False

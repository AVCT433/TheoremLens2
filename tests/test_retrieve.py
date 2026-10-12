"""rag/retrieve 테스트: RRF, 명시 라벨 조회, 인용 보조정리 1단계 확장, 책 필터, 근거 부족 판정."""

from __future__ import annotations

import pytest

from theoremlens.config import get_settings
from theoremlens.index import db, vectors
from theoremlens.rag import retrieve


def _chunk(ord_: int, text: str, label: str | None = None, display: str | None = None, refs=None, page=1) -> dict:
    return {
        "ord": ord_,
        "unit_type": (label or "paragraph").split()[0],
        "label": label,
        "display_label": display,
        "page_start": page,
        "page_end": page,
        "text": text,
        "refs": refs or [],
    }


def _add_book(book_id: str, chunks: list[dict], title: str | None = None) -> list[int]:
    db.create_book(book_id, title or f"Book {book_id}", f"{book_id}.pdf")
    ids = db.replace_chunks(book_id, chunks)
    vectors.index_chunks(book_id, ids, chunks)
    return ids


# ---------- RRF (순수 함수) ----------


def test_rrf_ranks_items_appearing_in_both_lists_first():
    fused = retrieve.rrf([[1, 2, 3], [3, 4, 1]])
    order = [i for i, _ in fused]
    assert set(order[:2]) == {1, 3}
    assert order[-1] in (2, 4)


def test_rrf_score_uses_reciprocal_rank_with_k():
    fused = dict(retrieve.rrf([[7], [7]], k=60))
    assert fused[7] == pytest.approx(2 / 61)


def test_rrf_is_deterministic_on_ties_and_handles_empty():
    assert retrieve.rrf([]) == []
    assert [i for i, _ in retrieve.rrf([[5], [9]])] == [5, 9]


def test_rrf_counts_duplicate_within_one_list_once():
    fused = dict(retrieve.rrf([[1, 1, 2]]))
    assert fused[1] == pytest.approx(1 / 61)


# ---------- 라벨 추출 (순수 함수) ----------


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("Explain Theorem 3.2 please", ["theorem 3.2"]),
        ("what does lemma 2.4 and Definition 1.1 say", ["lemma 2.4", "definition 1.1"]),
        ("정리 3.2를 설명해줘", ["theorem 3.2"]),
        ("보조정리 2.4와 정의 1.1", ["lemma 2.4", "definition 1.1"]),
        ("Theorem 3.2 and theorem 3.2 again", ["theorem 3.2"]),
        ("no label here, just 3.2 alone", []),
    ],
)
def test_extract_labels_normalizes_english_and_korean(q, expected):
    assert retrieve.extract_labels(q) == expected


# ---------- 검색 통합 ----------


def test_retrieve_returns_ranked_sources_with_metadata(fake_embed, fake_llm):
    _add_book(
        "b1",
        [
            _chunk(0, "compact subsets of metric spaces are closed and bounded", "theorem 1.1", "Theorem 1.1", page=3),
            _chunk(1, "banana bread recipe with flour", None, None, page=4),
        ],
        title="Analysis",
    )
    r = retrieve.retrieve("compact subsets metric spaces closed", ["b1"])
    assert not r.not_found
    top = r.sources[0]
    assert top.sid == "S1"
    assert (top.book_id, top.book_title, top.page_start, top.label) == ("b1", "Analysis", 3, "theorem 1.1")
    assert "compact" in top.text
    assert [s.sid for s in r.sources] == [f"S{i + 1}" for i in range(len(r.sources))]
    assert r.best_similarity > 0.5
    assert not fake_llm.calls


def test_explicit_label_in_question_fetches_chunk_directly(fake_embed):
    _add_book(
        "b1",
        [
            _chunk(0, "alpha beta gamma delta", "theorem 3.2", "Theorem 3.2"),
            _chunk(1, "unrelated topology words about homeomorphism", None),
        ],
    )
    # 질문 본문은 theorem 3.2 청크와 어휘가 전혀 겹치지 않는다
    r = retrieve.retrieve("Theorem 3.2 homeomorphism topology", ["b1"])
    labels = [s.label for s in r.sources]
    assert labels[0] == "theorem 3.2"
    assert r.sources[0].via == "label"


def test_korean_label_maps_to_normalized_label(fake_embed):
    _add_book("b1", [_chunk(0, "내용 하나", "theorem 3.2", "정리 3.2"), _chunk(1, "다른 내용 둘", None)])
    r = retrieve.retrieve("정리 3.2 증명해줘", ["b1"])
    assert r.sources[0].label == "theorem 3.2"


def test_explicit_label_only_searches_selected_books(fake_embed):
    _add_book("b1", [_chunk(0, "one", "theorem 3.2", "Theorem 3.2")])
    _add_book("b2", [_chunk(0, "two", "theorem 3.2", "Theorem 3.2")])
    r = retrieve.retrieve("Theorem 3.2", ["b2"])
    assert {s.book_id for s in r.sources} == {"b2"}


def test_label_hit_prevents_not_found_even_when_similarity_is_low(fake_embed):
    _add_book("b1", [_chunk(0, "zzz qqq", "lemma 2.4", "Lemma 2.4")])
    r = retrieve.retrieve("Lemma 2.4 apples oranges", ["b1"])
    assert not r.not_found
    assert r.sources[0].label == "lemma 2.4"


def test_cited_lemmas_are_added_one_hop_only(fake_embed):
    _add_book(
        "b1",
        [
            _chunk(0, "main result about convergence of sequences", "theorem 5.1", "Theorem 5.1", refs=["lemma 2.4"]),
            _chunk(1, "lemma body zzz", "lemma 2.4", "Lemma 2.4", refs=["lemma 1.1"]),
            _chunk(2, "deep lemma body yyy", "lemma 1.1", "Lemma 1.1"),
        ],
    )
    r = retrieve.retrieve("main result convergence of sequences", ["b1"], top_k=1)
    labels = [s.label for s in r.sources]
    assert labels == ["theorem 5.1", "lemma 2.4"]
    cited = r.sources[1]
    assert cited.via == "cited"
    assert "lemma 1.1" not in labels


def test_cited_expansion_stays_within_the_citing_book(fake_embed):
    _add_book("b1", [_chunk(0, "main result convergence", "theorem 5.1", "Theorem 5.1", refs=["lemma 2.4"])])
    _add_book("b2", [_chunk(0, "other book lemma xxx", "lemma 2.4", "Lemma 2.4")])
    r = retrieve.retrieve("main result convergence", ["b1", "b2"], top_k=1)
    assert [s.book_id for s in r.sources] == ["b1"]


def test_cited_chunk_already_retrieved_is_not_duplicated(fake_embed):
    _add_book(
        "b1",
        [
            _chunk(0, "convergence theorem statement", "theorem 5.1", "Theorem 5.1", refs=["lemma 2.4"]),
            _chunk(1, "convergence lemma statement", "lemma 2.4", "Lemma 2.4"),
        ],
    )
    r = retrieve.retrieve("convergence theorem lemma statement", ["b1"])
    ids = [s.chunk_id for s in r.sources]
    assert len(ids) == len(set(ids)) == 2


def test_book_filter_excludes_other_books(fake_embed):
    _add_book("b1", [_chunk(0, "metric space completeness", None)])
    _add_book("b2", [_chunk(0, "metric space completeness", None)])
    r = retrieve.retrieve("metric space completeness", ["b1"])
    assert {s.book_id for s in r.sources} == {"b1"}


def test_top_k_limits_search_results(fake_embed):
    _add_book("b1", [_chunk(i, f"group theory item {i}", None) for i in range(6)])
    r = retrieve.retrieve("group theory item", ["b1"], top_k=3)
    assert len(r.sources) == 3


def test_low_similarity_returns_not_found_without_llm_call(fake_embed, fake_llm):
    _add_book("b1", [_chunk(0, "compact metric spaces", None)])
    assert get_settings().min_similarity > 0
    r = retrieve.retrieve("banana bread recipe", ["b1"])
    assert r.not_found
    assert r.sources == []
    assert r.best_similarity < get_settings().min_similarity
    assert not fake_llm.calls


def test_no_books_returns_not_found(fake_embed, fake_llm):
    r = retrieve.retrieve("anything", [])
    assert r.not_found
    assert r.sources == []
    assert not fake_llm.calls

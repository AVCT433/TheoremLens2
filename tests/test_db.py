"""index/db 테스트: FTS 동기화, 검색, 라벨 조회, 삭제 연쇄, 메시지 순서."""

from __future__ import annotations

import pytest

from theoremlens.index import db


def _chunk(ord_: int, text: str, label: str | None = None, **kw) -> dict:
    return {
        "ord": ord_,
        "unit_type": "theorem",
        "label": label,
        "display_label": kw.get("display_label", label.title() if label else None),
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "refs": kw.get("refs", []),
    }


def _book(book_id: str = "b1") -> str:
    db.create_book(book_id, "Book " + book_id, book_id + ".pdf")
    return book_id


def test_replace_chunks_removes_old_fts_hits():
    b = _book()
    db.replace_chunks(b, [_chunk(0, "compactness lemma")])
    assert db.fts_search("compactness", [b], 10)
    db.replace_chunks(b, [_chunk(0, "connectedness lemma")])
    assert db.fts_search("compactness", [b], 10) == []
    assert len(db.fts_search("connectedness", [b], 10)) == 1


def test_replace_chunks_returns_ids_and_roundtrips_refs():
    b = _book()
    ids = db.replace_chunks(b, [_chunk(0, "a", "theorem 1.1", refs=["lemma 1.0", "정의 2"])])
    (ch,) = db.get_chunks(ids)
    assert ch.refs == ["lemma 1.0", "정의 2"]
    assert ch.label == "theorem 1.1"


def test_get_chunks_preserves_requested_order_and_skips_missing():
    b = _book()
    ids = db.replace_chunks(b, [_chunk(0, "x"), _chunk(1, "y")])
    assert [c.id for c in db.get_chunks([ids[1], 9999, ids[0]])] == [ids[1], ids[0]]
    assert db.get_chunks([]) == []


def test_fts_search_matches_korean_prefix():
    b = _book()
    (cid,) = db.replace_chunks(b, [_chunk(0, "콤팩트집합은 닫혀 있다")])
    assert [i for i, _ in db.fts_search("콤팩트집합", [b], 5)] == [cid]
    # 접두 일치로 조사 변형을 흡수한다
    assert [i for i, _ in db.fts_search("콤팩트", [b], 5)] == [cid]


@pytest.mark.parametrize(
    "query",
    ['"closed', "closed AND (", "a*b:c^", "NEAR(", "x'; DROP TABLE chunks;--", "$$\\int$$", "", "-closed", "closed OR"],
)
def test_fts_search_special_characters_do_not_raise(query):
    b = _book()
    db.replace_chunks(b, [_chunk(0, "closed set")])
    db.fts_search(query, [b], 5)


def test_fts_search_quoted_query_still_matches():
    b = _book()
    db.replace_chunks(b, [_chunk(0, "closed set")])
    assert db.fts_search('"closed"', [b], 5)


def test_fts_search_ignores_single_char_tokens():
    b = _book()
    db.replace_chunks(b, [_chunk(0, "a b c set")])
    assert db.fts_search("a", [b], 5) == []
    assert len(db.fts_search("a set", [b], 5)) == 1


def test_fts_search_keeps_single_digit_tokens():
    b = _book()
    (cid,) = db.replace_chunks(b, [_chunk(0, "see chapter 3 now")])
    assert [i for i, _ in db.fts_search("3", [b], 5)] == [cid]


def test_fts_search_filters_by_book_and_handles_empty_inputs():
    b1, b2 = _book("b1"), _book("b2")
    db.replace_chunks(b1, [_chunk(0, "metric space")])
    db.replace_chunks(b2, [_chunk(0, "metric tensor")])
    assert len(db.fts_search("metric", [b1], 5)) == 1
    assert len(db.fts_search("metric", [b1, b2], 5)) == 2
    assert db.fts_search("metric", [], 5) == []


def test_fts_search_lower_bm25_score_is_better_and_sorted():
    b = _book()
    db.replace_chunks(b, [_chunk(0, "limit " * 5 + "filler words here"), _chunk(1, "limit once and other text")])
    res = db.fts_search("limit", [b], 5)
    scores = [s for _, s in res]
    assert scores == sorted(scores)
    assert res[0][0] == db.chunk_at(b, 0).id


def test_chunks_by_label_across_books_ordered():
    b1, b2 = _book("b1"), _book("b2")
    db.replace_chunks(b1, [_chunk(0, "t", "theorem 3.2"), _chunk(1, "l", "lemma 1.1")])
    db.replace_chunks(b2, [_chunk(0, "t2", "theorem 3.2")])
    got = db.chunks_by_label(["theorem 3.2"], [b1, b2])
    assert [(c.book_id, c.text) for c in got] == [("b1", "t"), ("b2", "t2")]
    assert [c.text for c in db.chunks_by_label(["theorem 3.2"], [b2])] == ["t2"]
    assert db.chunks_by_label([], [b1]) == []
    assert db.chunks_by_label(["x"], []) == []


def test_chunk_at_returns_none_when_missing():
    b = _book()
    assert db.chunk_at(b, 0) is None


def test_delete_book_cascades_to_pages_chunks_and_fts():
    b, other = _book("b1"), _book("b2")
    db.save_page(b, 1, "p1", "pymupdf")
    db.replace_chunks(b, [_chunk(0, "unique zebra")])
    db.save_page(other, 1, "keep", "pymupdf")
    db.replace_chunks(other, [_chunk(0, "unique zebra")])
    db.delete_book(b)
    assert db.get_book(b) is None
    assert db.get_pages(b) == []
    assert db.chunk_at(b, 0) is None
    assert db.fts_search("zebra", [b], 5) == []
    assert len(db.fts_search("zebra", [other], 5)) == 1
    assert len(db.get_pages(other)) == 1


def test_pages_are_ordered_and_save_page_replaces():
    b = _book()
    db.save_page(b, 2, "two", "gemini")
    db.save_page(b, 1, "one", "pymupdf")
    db.save_page(b, 1, "ONE", "pymupdf")
    assert [(p["page_no"], p["markdown"]) for p in db.get_pages(b)] == [(1, "ONE"), (2, "two")]
    assert db.get_page(b, 2)["method"] == "gemini"
    assert db.get_page(b, 3) is None


def test_books_listing_update_and_titles():
    b = _book()
    db.replace_chunks(b, [_chunk(0, "x"), _chunk(1, "y")])
    db.update_book(b, status="ready", progress=1.0)
    db.update_book(b)  # 변경 없음
    (row,) = db.list_books()
    assert row["status"] == "ready"
    assert row["num_chunks"] == 2
    assert db.book_titles([b]) == {b: "Book b1"}
    assert db.book_titles([]) == {}


def test_messages_come_back_oldest_first_and_respect_limit():
    for i in range(5):
        db.add_message("c1", "user", f"m{i}", {"i": i})
    db.add_message("c2", "user", "other")
    msgs = db.get_messages("c1")
    assert [m["content"] for m in msgs] == ["m0", "m1", "m2", "m3", "m4"]
    assert msgs[2]["meta"] == {"i": 2}
    assert [m["content"] for m in db.get_messages("c1", limit=2)] == ["m3", "m4"]


def test_tx_rolls_back_on_error():
    with pytest.raises(ValueError, match="boom"), db.tx() as c:
        c.execute("INSERT INTO books(id, title, filename) VALUES ('x','t','f')")
        raise ValueError("boom")
    assert db.get_book("x") is None

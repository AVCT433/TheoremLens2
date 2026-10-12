"""structure가 만든 한국어 청크가 db에 들어간 뒤 rag/retrieve의 라벨 조회로 찾아지는지 확인한다."""

from __future__ import annotations

from theoremlens.index import db, vectors
from theoremlens.ingest import structure
from theoremlens.rag import retrieve

_PAGES = [
    (
        1,
        "제1장 연속성\n\n"
        "**정의 2.1.** 함수 $f$가 연속이라는 것은 극한값이 함숫값과 같다는 뜻이다.\n\n"
        "**보조정리 2.4.** 연속함수의 합은 연속이다.\n\n"
        "*증명.* 정의 2.1에 의해 자명하다. $\blacksquare$\n\n"
        "**정리 3.2.** 닫힌구간에서 연속인 함수는 최댓값을 가진다.\n\n"
        "*증명.* 보조정리 2.4를 이용한다. $\blacksquare$\n",
    )
]


def _index(book_id: str = "ko") -> list[structure.ChunkDraft]:
    chunks = structure.build_chunks(_PAGES)
    db.create_book(book_id, "해석학", f"{book_id}.pdf")
    rows = [c.as_row() for c in chunks]
    ids = db.replace_chunks(book_id, rows)
    vectors.index_chunks(book_id, ids, rows)
    return chunks


def test_korean_label_in_question_finds_structure_chunk(fake_embed):
    _index()
    got = db.chunks_by_label(retrieve.extract_labels("정리 3.2를 설명해줘"), ["ko"])
    assert [c.label for c in got] == ["theorem 3.2"]
    assert got[0].display_label == "정리 3.2"
    assert "최댓값" in got[0].text


def test_cited_korean_lemma_is_added_one_hop_via_normalized_refs(fake_embed):
    _index()
    r = retrieve.retrieve("정리 3.2", ["ko"], top_k=1)
    by_via = {s.label: s.via for s in r.sources}
    assert by_via["theorem 3.2"] == "label"
    assert by_via["lemma 2.4"] == "cited"  # 정리 3.2의 refs '보조정리 2.4'가 정규화되어 조회된다


def test_lemma_question_does_not_return_theorem(fake_embed):
    _index()
    got = db.chunks_by_label(retrieve.extract_labels("보조정리 2.4가 뭐야?"), ["ko"])
    assert [c.label for c in got] == ["lemma 2.4"]
    assert got[0].display_label == "보조정리 2.4"

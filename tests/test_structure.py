"""ingest/structure 테스트: 불변식(모든 fixture), 라벨 정규화, refs, 장/절, 쪽 범위, db 호환."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from theoremlens.index import db
from theoremlens.ingest import structure

FIXTURES = Path(__file__).parent / "fixtures"
_PAGE_MARK = re.compile(r"^<!-- page (\d+) -->\n", re.MULTILINE)
ALL = ["en_analysis", "ko_analysis", "edge"]
THEOREM_LIKE = {"theorem", "lemma", "proposition", "corollary"}


def _pages(name: str) -> list[tuple[int, str]]:
    raw = (FIXTURES / f"{name}.md").read_text(encoding="utf-8")
    parts = _PAGE_MARK.split(raw)  # ['', '3', body, '4', body, ...]
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]


def _chunks(name: str) -> list[structure.ChunkDraft]:
    return structure.build_chunks(_pages(name))


def _by_label(chunks: list[structure.ChunkDraft], label: str) -> structure.ChunkDraft:
    (c,) = [c for c in chunks if c.label == label]
    return c


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


# ---------- 불변식 ----------


@pytest.mark.parametrize("name", ALL)
def test_every_chunk_has_even_display_math_delimiters(name):
    for c in _chunks(name):
        assert c.text.count("$$") % 2 == 0, c.text


@pytest.mark.parametrize("name", ALL)
def test_chunks_lose_no_source_content(name):
    src = "".join(md for _, md in _pages(name))
    assert _squash("".join(c.text for c in _chunks(name))) == _squash(src)


@pytest.mark.parametrize("name", ALL)
def test_ord_is_contiguous_from_zero_and_page_range_is_valid(name):
    chunks = _chunks(name)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    pages = {p for p, _ in _pages(name)}
    for c in chunks:
        assert c.page_start <= c.page_end
        assert {c.page_start, c.page_end} <= pages


@pytest.mark.parametrize("name", ALL)
def test_no_chunk_is_blank(name):
    assert all(c.text.strip() for c in _chunks(name))


@pytest.mark.parametrize("name", ALL)
def test_proof_never_stands_alone_right_after_a_theorem_like_chunk(name):
    chunks = _chunks(name)
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        if cur.unit_type == "proof":
            assert prev.unit_type not in THEOREM_LIKE


@pytest.mark.parametrize("name", ALL)
def test_chunks_are_verbatim_slices_of_source(name):
    # 인용 검증은 원문과 대조하므로 청크 텍스트가 쪽 텍스트를 이은 문자열의 연속 부분이어야 한다
    src = "".join(md for _, md in _pages(name))
    for c in _chunks(name):
        assert c.text in src


# ---------- 영어 fixture ----------


def test_english_theorem_and_proof_are_one_chunk():
    c = _by_label(_chunks("en_analysis"), "theorem 2.3")
    assert c.unit_type == "theorem"
    assert "Heine-Borel" in c.text and "Suppose $K$ is compact" in c.text and r"\square" in c.text
    assert c.display_label == "Theorem 2.3 (Heine-Borel)"
    assert (c.page_start, c.page_end) == (4, 4)


def test_english_lemma_with_proof_and_definition_labels():
    ch = _chunks("en_analysis")
    lemma = _by_label(ch, "lemma 2.2")
    assert lemma.unit_type == "lemma" and "Cover $K$" in lemma.text
    d = _by_label(ch, "definition 2.1")
    assert d.unit_type == "definition" and d.display_label == "Definition 2.1 (Compact)"


def test_unit_after_proof_end_marker_starts_new_chunk():
    ch = _chunks("en_analysis")
    assert "Corollary 2.4." not in _by_label(ch, "theorem 2.3").text
    assert _by_label(ch, "corollary 2.4").unit_type == "corollary"


def test_prose_after_proof_end_marker_becomes_text_chunk():
    src = "**Theorem 1.1.** A.\n\n*Proof.* B. $\\blacksquare$\n\nA closing paragraph.\n"
    ch = structure.build_chunks([(1, src)])
    assert [c.unit_type for c in ch] == ["theorem", "text"]
    assert "closing paragraph" in ch[1].text


def test_english_refs_normalized_deduped_and_exclude_self():
    ch = _chunks("en_analysis")
    assert _by_label(ch, "theorem 2.3").refs == ["lemma 2.2"]
    assert _by_label(ch, "lemma 2.2").refs == ["definition 2.1"]
    assert _by_label(ch, "exercise 2.1").refs == ["theorem 2.3"]
    assert _by_label(ch, "exercise 2.2").refs == ["lemma 2.2"]


def test_english_chapter_and_section_tracking():
    ch = _chunks("en_analysis")
    d = _by_label(ch, "definition 2.1")
    assert d.chapter == "Chapter 2. Compactness" and d.section == "2.1 Compact sets"
    ex = _by_label(ch, "exercise 2.1")
    assert ex.section == "2.2 Exercises" and ex.chapter == "Chapter 2. Compactness"
    assert ch[0].chapter == "Chapter 2. Compactness" and ch[0].section is None


def test_page_ranges_are_one_based_pdf_pages():
    ch = _chunks("en_analysis")
    assert (_by_label(ch, "lemma 2.2").page_start, _by_label(ch, "lemma 2.2").page_end) == (3, 3)
    assert ch[0].page_start == 3
    assert _by_label(ch, "exercise 2.2").page_end == 4


def test_display_math_inside_unit_does_not_split_it():
    ch = _chunks("en_analysis")
    assert _by_label(ch, "theorem 2.3").text.count("$$") == 4
    ex = _by_label(ch, "example 2.5")
    assert r"\{ (1/n, 1)" in ex.text and "no finite subcover" in ex.text


# ---------- 한국어 fixture ----------


def test_korean_theorem_and_proof_share_chunk():
    c = _by_label(_chunks("ko_analysis"), "theorem 3.3")
    assert c.unit_type == "theorem"
    assert "중간값 정리" in c.text and "상한을 $s$라" in c.text and "따라서 정리 3.3이 성립한다" in c.text
    assert (c.page_start, c.page_end) == (11, 11)


def test_korean_keywords_map_to_unit_types():
    ch = _chunks("ko_analysis")
    assert _by_label(ch, "definition 3.1").unit_type == "definition"
    assert _by_label(ch, "lemma 3.2").unit_type == "lemma"
    assert _by_label(ch, "corollary 3.4").unit_type == "corollary"
    assert _by_label(ch, "example 3.5").unit_type == "example"
    assert _by_label(ch, "exercise 3.1").unit_type == "exercise"
    assert "삼각부등식" in _by_label(ch, "lemma 3.2").text  # 증명이 같은 청크에 있다


def test_korean_refs_do_not_confuse_lemma_with_theorem():
    ch = _chunks("ko_analysis")
    assert _by_label(ch, "theorem 3.3").refs == ["lemma 3.2", "definition 3.1"]
    assert _by_label(ch, "lemma 3.2").refs == ["definition 3.1"]
    assert _by_label(ch, "example 3.5").refs == ["theorem 3.3"]
    assert _by_label(ch, "exercise 3.1").refs == ["lemma 3.2"]


def test_korean_display_label_keeps_original_wording():
    ch = _chunks("ko_analysis")
    assert _by_label(ch, "lemma 3.2").display_label.startswith("보조정리 3.2")
    assert _by_label(ch, "theorem 3.3").display_label.startswith("정리 3.3")
    assert _by_label(ch, "exercise 3.1").display_label == "연습문제 3.1"


def test_korean_label_and_refs_use_english_type_names_only():
    ch = _chunks("ko_analysis")
    for c in ch:
        for name in [c.label, *c.refs]:
            assert name is None or re.fullmatch(r"[a-z]+ \d+(\.\d+)*", name), name


def test_korean_problem_keyword_and_mixed_language_refs_normalize():
    md = "**문제 1.1.** 보조정리 1.0과 Theorem 1.2를 이용하라.\n\n**Problem 1.2.** 따름정리 1.3을 보여라.\n"
    ch = structure.build_chunks([(1, md)])
    assert [c.label for c in ch] == ["exercise 1.1", "exercise 1.2"]
    assert ch[0].display_label == "문제 1.1" and ch[1].display_label == "Problem 1.2"
    assert ch[0].refs == ["lemma 1.0", "theorem 1.2"]
    assert ch[1].refs == ["corollary 1.3"]


def test_korean_chapter_and_section_headings():
    ch = _chunks("ko_analysis")
    d = _by_label(ch, "definition 3.1")
    assert d.chapter == "제3장 연속성" and d.section == "3.1 극한과 연속"
    # '연습문제' 헤딩은 번호가 없으므로 단위가 아니라 절 제목이다
    assert _by_label(ch, "exercise 3.1").section == "연습문제"


def test_korean_plain_prose_is_text_chunk():
    ch = _chunks("ko_analysis")
    assert ch[0].unit_type == "text" and ch[0].label is None


# ---------- 가장자리 사례 ----------


def test_prose_mentioning_theorem_is_not_a_unit_and_preface_is_text():
    ch = _chunks("edge")
    assert ch[0].unit_type == "text" and ch[0].label is None and ch[0].chapter is None
    assert all(c.label != "theorem 9.9" for c in ch)


def test_heading_unit_absorbs_heading_and_keeps_math_across_pages():
    c = _by_label(_chunks("edge"), "theorem 1.1")
    assert c.text.count("$$") == 2 and "= c^2" in c.text and "Pythagoras" in c.text
    assert (c.page_start, c.page_end) == (1, 2)
    assert c.chapter == "1 Basics" and c.refs == []


def test_unnumbered_unit_has_no_label_but_has_display_label():
    (d,) = [c for c in _chunks("edge") if c.unit_type == "definition"]
    assert d.label is None and d.display_label == "Definition"


def test_theorem_like_without_proof_ends_at_next_unit():
    ch = _chunks("edge")
    assert "Example 1.3" not in _by_label(ch, "proposition 1.2").text
    assert _by_label(ch, "example 1.3").refs == ["proposition 1.2", "lemma 1.0"]


def test_new_chapter_resets_section():
    r = _by_label(_chunks("edge"), "remark 2.1")
    assert r.chapter == "2 More" and r.section is None


def test_long_text_is_split_at_paragraph_boundaries_only():
    para = "word " * 100 + "\n\n"
    ch = structure.build_chunks([(1, para * 20)], max_chars=800)
    assert len(ch) > 1 and all(len(c.text) < 1200 for c in ch)
    assert _squash("".join(c.text for c in ch)) == _squash(para * 20)


def test_blank_line_inside_display_math_never_splits():
    para = "word " * 100 + "\n\n"
    src = para * 3 + "$$\na\n\nb\n$$\n\n" + para
    for c in structure.build_chunks([(1, src)], max_chars=300):
        assert c.text.count("$$") % 2 == 0


def test_empty_input_gives_no_chunks():
    assert structure.build_chunks([]) == []
    assert structure.build_chunks([(1, "  \n\n")]) == []


# ---------- db 호환 ----------


def test_as_row_is_accepted_by_db_replace_chunks_and_roundtrips():
    db.create_book("b", "Book", "b.pdf")
    ch = _chunks("en_analysis")
    ids = db.replace_chunks("b", [c.as_row() for c in ch])
    assert len(ids) == len(ch)
    got = db.chunks_by_label(["theorem 2.3"], ["b"])
    assert [g.display_label for g in got] == ["Theorem 2.3 (Heine-Borel)"]
    assert got[0].refs == ["lemma 2.2"]

"""tests/eval/run_eval.py 테스트: 질문 파일 검증, 지표 계산(순수 함수), 러너 로직, 결과 저장 경로.

실제 Gemini나 책 없이 가짜 retrieve/answer 함수와 conftest의 fake_embed/fake_llm만 쓴다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from tests.eval import run_eval as ev
from theoremlens.config import get_settings
from theoremlens.index import db, vectors
from theoremlens.rag import prompts
from theoremlens.rag.answer import AnswerResponse, CitationOut, ClaimOut
from theoremlens.rag.retrieve import Retrieval, Source

EXAMPLE = Path(__file__).parent / "eval" / "questions.example.json"


def _src(sid="S1", label="theorem 1.1", pages=(3, 3), text="t") -> Source:
    return Source(
        sid=sid,
        chunk_id=int(sid[1:]),
        book_id="b1",
        book_title="Analysis",
        page_start=pages[0],
        page_end=pages[1],
        label=label,
        display_label=label,
        unit_type="theorem",
        chapter=None,
        section=None,
        text=text,
        via="search",
    )


def _cit(sid="S1", text="every compact subset is closed") -> CitationOut:
    return CitationOut(sid=sid, book_id="b1", book_title="Analysis", page_start=3, page_end=3, text=text)


def _claim(quote="compact subset is closed", sids=("S1",), status="verified") -> ClaimOut:
    return ClaimOut(text="c", display_text="c", source_ids=list(sids), quote=quote, status=status)


def _resp(claims=None, citations=None, not_found=False, outside_book=False) -> AnswerResponse:
    return AnswerResponse(
        answer_markdown="a",
        claims=claims or [],
        citations=citations or [],
        not_found=not_found,
        outside_book=outside_book,
    )


# ---------- 질문 파일 ----------


def test_example_questions_file_is_valid_and_covers_requirements():
    qs = ev.load_questions(EXAMPLE)
    assert any(q.expect_refusal for q in qs)
    assert any(not q.expect_refusal for q in qs)
    assert any(any("가" <= ch <= "힣" for ch in q.question) for q in qs), "Korean question needed"
    assert any(q.question.isascii() for q in qs), "English question needed"
    for q in qs:
        if not q.expect_refusal:
            assert q.expected_labels or q.expected_pages


def test_load_questions_rejects_in_book_question_without_ground_truth(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"question": "x?"}]), encoding="utf-8")
    with pytest.raises(ev.EvalError):
        ev.load_questions(p)


def test_load_questions_rejects_refusal_with_ground_truth(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(
        json.dumps([{"question": "x?", "expect_refusal": True, "expected_pages": [1]}]),
        encoding="utf-8",
    )
    with pytest.raises(ev.EvalError):
        ev.load_questions(p)


def test_load_questions_rejects_missing_question_text(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"expect_refusal": True}]), encoding="utf-8")
    with pytest.raises(ev.EvalError):
        ev.load_questions(p)


# ---------- 검색 적중 ----------


def test_retrieval_hit_matches_label_case_and_alias_insensitively():
    q = ev.Question(question="q", expected_labels=["Theorem 1.1"])
    assert ev.retrieval_hit(q, [_src(label="theorem 1.1")], top_k=8)


def test_retrieval_hit_matches_korean_label_alias():
    q = ev.Question(question="q", expected_labels=["정리 1.1"])
    assert ev.retrieval_hit(q, [_src(label="theorem 1.1")], top_k=8)


def test_retrieval_hit_matches_page_inside_chunk_range():
    q = ev.Question(question="q", expected_pages=[4])
    assert ev.retrieval_hit(q, [_src(label=None, pages=(3, 5))], top_k=8)
    assert not ev.retrieval_hit(q, [_src(label=None, pages=(5, 6))], top_k=8)


def test_retrieval_hit_ignores_sources_beyond_top_k():
    q = ev.Question(question="q", expected_labels=["theorem 9.9"])
    sources = [_src("S1", "theorem 1.1"), _src("S2", "theorem 9.9")]
    assert not ev.retrieval_hit(q, sources, top_k=1)
    assert ev.retrieval_hit(q, sources, top_k=2)


def test_retrieval_hit_is_false_for_empty_sources():
    assert not ev.retrieval_hit(ev.Question(question="q", expected_pages=[1]), [], top_k=8)


# ---------- 인용 정확도 ----------


def test_citation_accuracy_counts_claims_whose_quote_matches_cited_source():
    claims = [_claim("compact subset is closed"), _claim("fabricated sentence about banach")]
    assert ev.citation_accuracy(_resp(claims=claims, citations=[_cit()])) == (1, 2)


def test_citation_accuracy_ignores_claims_without_quote_or_sources():
    claims = [_claim(quote=""), _claim(sids=())]
    assert ev.citation_accuracy(_resp(claims=claims, citations=[_cit()])) == (0, 0)


def test_citation_accuracy_quote_must_match_its_own_cited_source_only():
    claims = [_claim("compact subset is closed", sids=("S2",))]
    cits = [_cit("S1"), _cit("S2", text="completely unrelated text about rings")]
    assert ev.citation_accuracy(_resp(claims=claims, citations=cits)) == (0, 1)


# ---------- 거절 판정 ----------


def test_is_refusal_counts_only_not_found():
    assert ev.is_refusal(_resp(not_found=True))
    assert not ev.is_refusal(_resp(outside_book=True))
    assert not ev.is_refusal(_resp(claims=[_claim()], citations=[_cit()]))


def test_is_outside_book_answer_excludes_not_found():
    assert ev.is_outside_book_answer(_resp(outside_book=True))
    assert not ev.is_outside_book_answer(_resp(not_found=True, outside_book=True))
    assert not ev.is_outside_book_answer(_resp(claims=[_claim()], citations=[_cit()]))


# ---------- 집계 ----------


def _res(expect_refusal, refused, hit=None, cites=(0, 0), outside=False) -> ev.QuestionResult:
    return ev.QuestionResult(
        outside_book=outside,
        question="q",
        expect_refusal=expect_refusal,
        refused=refused,
        retrieval_hit=hit,
        citations_ok=cites[0],
        citations_total=cites[1],
        sources=[],
        answer="",
        error=None,
    )


def test_compute_metrics_basic_rates():
    results = [
        _res(False, False, hit=True, cites=(2, 2)),
        _res(False, False, hit=False, cites=(1, 2)),
        _res(False, True, hit=True),  # 오거절
        _res(True, True),
        _res(True, False),
    ]
    m = ev.compute_metrics(results)
    assert m["retrieval_hit_rate"] == pytest.approx(2 / 3)
    assert m["citation_accuracy"] == pytest.approx(3 / 4)
    assert m["refusal_rate"] == pytest.approx(1 / 2)
    assert m["false_refusal_rate"] == pytest.approx(1 / 3)
    assert m["n_in_book"] == 3 and m["n_out_of_book"] == 2


def test_compute_metrics_reports_outside_book_answers_separately():
    results = [
        _res(True, True),  # 거절
        _res(True, False, outside=True),  # 책 밖 답변
        _res(True, False),  # 검증된 인용이 붙은 답
        _res(True, False),
        _res(False, False, hit=True),
        _res(False, False, hit=False, outside=True),  # 책 안 질문인데 근거 없는 답
        _res(False, True, hit=False),  # 오거절
    ]
    m = ev.compute_metrics(results)
    assert m["refusal_rate"] == pytest.approx(1 / 4)
    assert m["outside_book_answer_rate"] == pytest.approx(1 / 4)
    assert m["unsupported_answer_rate"] == pytest.approx(1 / 3)
    assert m["false_refusal_rate"] == pytest.approx(1 / 3)


def test_compute_metrics_outside_book_rates_are_none_without_denominator():
    m = ev.compute_metrics([_res(True, True)])
    assert m["outside_book_answer_rate"] == 0.0
    assert m["unsupported_answer_rate"] is None


def test_run_questions_records_outside_book_answers():
    qs = [ev.Question(question="out", expect_refusal=True)]
    results = ev.run_questions(
        qs,
        ["b1"],
        retrieve_fn=lambda q, b, top_k=None: Retrieval(sources=[_src(text="x")]),
        answer_fn=lambda q, b, **k: _resp(outside_book=True),
        top_k=8,
    )
    assert results[0].outside_book and not results[0].refused


def test_compute_metrics_returns_none_when_denominator_is_zero():
    m = ev.compute_metrics([_res(True, True)])
    assert m["retrieval_hit_rate"] is None
    assert m["citation_accuracy"] is None
    assert m["false_refusal_rate"] is None
    assert m["refusal_rate"] == 1.0


def test_compute_metrics_always_refusing_bot_does_not_look_good():
    results = [_res(False, True, hit=True)] * 3 + [_res(True, True)] * 3
    m = ev.compute_metrics(results)
    assert m["refusal_rate"] == 1.0
    assert m["false_refusal_rate"] == 1.0


def test_compute_metrics_excludes_errored_questions_from_rates():
    bad = _res(False, False, hit=None)
    bad.error = "boom"
    m = ev.compute_metrics([bad, _res(False, False, hit=True, cites=(1, 1))])
    assert m["n_errors"] == 1
    assert m["retrieval_hit_rate"] == 1.0
    assert m["false_refusal_rate"] == 0.0


# ---------- 러너 ----------


def test_run_questions_uses_injected_functions_and_scores_each_question():
    qs = [
        ev.Question(question="in", expected_labels=["theorem 1.1"]),
        ev.Question(question="out", expect_refusal=True),
    ]

    def fake_retrieve(question, book_ids, top_k=None):
        if question == "out":
            return Retrieval(not_found=True)
        return Retrieval(sources=[_src(text="every compact subset is closed")])

    def fake_answer(question, book_ids, **kw):
        if question == "out":
            return _resp(not_found=True)
        return _resp(claims=[_claim()], citations=[_cit()])

    results = ev.run_questions(qs, ["b1"], retrieve_fn=fake_retrieve, answer_fn=fake_answer, top_k=8)
    assert [r.refused for r in results] == [False, True]
    assert results[0].retrieval_hit is True
    assert (results[0].citations_ok, results[0].citations_total) == (1, 1)
    assert results[1].retrieval_hit is None


def test_run_questions_records_exceptions_instead_of_aborting():
    qs = [ev.Question(question="a", expected_pages=[1]), ev.Question(question="b", expect_refusal=True)]

    def boom(question, book_ids, top_k=None):
        if question == "a":
            raise RuntimeError("boom")
        return Retrieval(not_found=True)

    results = ev.run_questions(qs, ["b1"], retrieve_fn=boom, answer_fn=lambda q, b, **k: _resp(not_found=True), top_k=8)
    assert results[0].error and "boom" in results[0].error
    assert results[1].error is None and results[1].refused


# ---------- 결과 저장 ----------


def test_result_path_contains_models_and_date_under_data_dir(tmp_path):
    s = get_settings()
    now = datetime(2026, 10, 13, 9, 30, 5)
    p = ev.result_path(s.data_dir, s, now)
    assert p.parent == s.data_dir / "eval"
    assert "2026-10-13" in p.name
    assert s.answer_model in p.name and s.fast_model in p.name
    assert "BAAI-bge-m3" in p.name  # HuggingFace 모델명의 '/'는 파일명에서 '-'로 바뀐다
    assert "/" not in p.name
    assert p.suffix == ".json"


def test_write_report_saves_models_date_metrics_and_results(tmp_path):
    s = get_settings()
    now = datetime(2026, 10, 13, 9, 30, 5)
    results = [_res(False, False, hit=True, cites=(1, 1))]
    path = ev.write_report(results, s, ["b1"], "q.json", now=now)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.is_relative_to(s.data_dir)
    assert data["models"] == {"answer": s.answer_model, "fast": s.fast_model, "embed": s.embed_model}
    assert data["date"].startswith("2026-10-13")
    assert data["top_k"] == s.top_k
    assert data["metrics"]["retrieval_hit_rate"] == 1.0
    assert len(data["results"]) == 1


# ---------- 가짜 LLM/embed 로 end-to-end ----------

THM = "Every compact subset of a metric space is closed and bounded."


def test_run_questions_end_to_end_with_fakes(fake_embed, fake_llm):
    chunks = [
        {
            "ord": 0,
            "unit_type": "theorem",
            "label": "theorem 1.1",
            "display_label": "Theorem 1.1",
            "page_start": 3,
            "page_end": 4,
            "text": THM,
            "refs": [],
        }
    ]
    db.create_book("b1", "Analysis", "b1.pdf")
    ids = db.replace_chunks("b1", chunks)
    vectors.index_chunks("b1", ids, chunks)
    fake_llm.default_json = json.dumps(
        {
            "answer_markdown": "compact sets are closed [S1]",
            "claims": [{"text": "closed", "source_ids": ["S1"], "quote": "compact subset of a metric space is closed"}],
            "not_found": False,
        }
    )
    fake_llm.default_text = "[]"  # 지지 검사 응답이 파싱 불가면 규칙 결과만 유지된다
    qs = [
        ev.Question(question="compact subset of a metric space closed bounded", expected_pages=[3]),
        ev.Question(question="banana bread flour recipe oven", expect_refusal=True),
    ]
    results = ev.run_questions(qs, ["b1"], top_k=get_settings().top_k)
    assert results[0].retrieval_hit is True
    assert not results[0].refused
    assert results[0].citations_total >= 1 and results[0].citations_ok == results[0].citations_total
    assert results[1].refused
    # 거절 질문은 검색 단계에서 끝나므로 LLM 호출이 없다
    assert all("banana" not in str(c["contents"]) for c in fake_llm.calls)
    assert prompts.NOT_FOUND_MESSAGE

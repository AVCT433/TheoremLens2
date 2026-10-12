"""rag/answer 테스트: not_found 단락, 프롬프트 구성, 검증 연동, 책 외부 표기 규칙, 인용 원문 출처, 대화 history."""

from __future__ import annotations

import json

import pytest

from theoremlens import llm
from theoremlens.config import get_settings
from theoremlens.index import db, vectors
from theoremlens.rag import answer, prompts

THM = "Every compact subset of a metric space is closed and bounded."
OTHER = "banana bread recipe with flour"


def _add_book(book_id: str = "b1", title: str = "Analysis") -> None:
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
        },
        {
            "ord": 1,
            "unit_type": "paragraph",
            "label": None,
            "display_label": None,
            "page_start": 5,
            "page_end": 5,
            "text": OTHER,
            "refs": [],
        },
    ]
    db.create_book(book_id, title, f"{book_id}.pdf")
    ids = db.replace_chunks(book_id, chunks)
    vectors.index_chunks(book_id, ids, chunks)


def _answer_json(claims=None, markdown="compact sets are closed [S1]", not_found=False) -> str:
    return json.dumps({"answer_markdown": markdown, "claims": claims or [], "not_found": not_found})


def _claim(text="compact implies closed", sids=("S1",), quote="compact subset of a metric space is closed") -> dict:
    return {"text": text, "source_ids": list(sids), "quote": quote}


def _script(fake_llm, answer_reply: str, supported: bool | list[bool] = True) -> None:
    """답변 스키마 호출에는 answer_reply, 검증 스키마 호출에는 지지 여부를 돌려준다."""

    def respond(**kw) -> str:
        if kw["schema"] is prompts.ANSWER_SCHEMA:
            return answer_reply
        flags = supported if isinstance(supported, list) else [supported] * 10
        return json.dumps({"results": [{"index": i, "supported": f} for i, f in enumerate(flags)]})

    fake_llm.json_responses.extend([respond, respond])
    fake_llm.default_json = json.dumps({"results": []})


Q = "every compact subset of a metric space is closed and bounded"


# (a)
def test_not_found_retrieval_skips_llm_and_says_not_in_book(fake_embed, fake_llm):
    _add_book()
    r = answer.answer("quantum chromodynamics gluon plasma", ["b1"])
    assert r.not_found
    assert "책에서 찾을 수 없" in r.answer_markdown
    assert r.claims == [] and r.citations == []
    assert not fake_llm.calls


def test_no_books_selected_is_not_found_without_llm(fake_embed, fake_llm):
    r = answer.answer(Q, [])
    assert r.not_found
    assert not fake_llm.calls


# (b)
def test_prompt_contains_excerpts_with_book_page_label_and_uses_schema(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()]))
    answer.answer(Q, ["b1"])
    call = fake_llm.calls[0]
    assert call["kind"] == "json"
    assert call["schema"] is prompts.ANSWER_SCHEMA
    assert call["system"] == prompts.ANSWER_SYSTEM
    text = str(call["contents"])
    assert "[S1]" in text and THM in text
    assert "Analysis" in text and "3-4" in text and "Theorem 1.1" in text
    assert Q in text
    assert set(prompts.ANSWER_SCHEMA["properties"]) == {"answer_markdown", "claims", "not_found"}
    assert set(prompts.ANSWER_SCHEMA["properties"]["claims"]["items"]["properties"]) == {"text", "source_ids", "quote"}


def test_generate_json_is_called_with_answer_model_or_fast_model(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()]))
    answer.answer(Q, ["b1"])
    _script(fake_llm, _answer_json([_claim()]))
    answer.answer(Q, ["b1"], fast=True)
    models = [c["model"] for c in fake_llm.calls if c["schema"] is prompts.ANSWER_SCHEMA]
    assert models == [get_settings().answer_model, get_settings().fast_model]


def test_real_generate_json_signature_matches_call(fake_embed, monkeypatch):
    """가짜가 아니라 실제 llm.generate_json 시그니처(model, system, contents, schema)에 맞게 호출한다."""
    _add_book()
    seen = {}

    def spy(model, system, contents, schema):
        seen["schema"] = schema
        return _answer_json(not_found=True)

    monkeypatch.setattr(llm, "generate_json", spy)
    answer.answer(Q, ["b1"])
    assert seen["schema"] is prompts.ANSWER_SCHEMA


# (c)
def test_model_not_found_is_reflected(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json(markdown="발췌로는 알 수 없습니다.", not_found=True))
    r = answer.answer(Q, ["b1"])
    assert r.not_found
    assert r.claims == [] and r.citations == []
    assert len(fake_llm.calls) == 1  # 검증할 claim이 없으니 지지 검사도 없다


# (d)
def test_verified_claim_is_kept_with_status_and_verification_summary(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()]), supported=True)
    r = answer.answer(Q, ["b1"])
    assert not r.not_found
    assert [c.status for c in r.claims] == ["verified"]
    assert r.verification.llm_checked and r.verification.dropped == 0
    assert not r.outside_book


def test_fabricated_quote_claim_is_dropped_and_counted(fake_embed, fake_llm):
    _add_book()
    good, bad = _claim(), _claim("fake", quote="all bounded sets are compact in every space")
    _script(fake_llm, _answer_json([good, bad]), supported=True)
    r = answer.answer(Q, ["b1"])
    assert [c.text for c in r.claims] == ["compact implies closed"]
    assert r.verification.dropped == 1


def test_unsupported_claim_is_dropped_by_default(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()]), supported=False)
    r = answer.answer(Q, ["b1"])
    assert r.claims == [] and r.verification.dropped == 1


def test_unsupported_claim_can_be_flagged_instead(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()]), supported=False)
    r = answer.answer(Q, ["b1"], on_unsupported="flag")
    assert r.claims[0].status == "unsupported"
    assert r.claims[0].display_text.startswith("⚠")


# (e)
def test_answer_without_surviving_citations_is_marked_outside_book(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([], markdown="일반적으로 컴팩트 집합은 닫혀 있습니다."))
    r = answer.answer(Q, ["b1"])
    assert r.outside_book
    assert r.answer_markdown.lstrip().startswith(f"## {prompts.OUTSIDE_BOOK_HEADING}")
    assert "일반적으로" in r.answer_markdown


def test_all_claims_dropped_also_marks_outside_book(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim(quote="nonexistent sentence about nothing")]))
    r = answer.answer(Q, ["b1"])
    assert r.outside_book and r.claims == []
    assert prompts.OUTSIDE_BOOK_HEADING in r.answer_markdown


def test_cited_answer_is_not_relabelled(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()], markdown="본문 [S1]"))
    r = answer.answer(Q, ["b1"])
    assert r.answer_markdown == "본문 [S1]"


def test_no_uncited_answer_is_ever_returned_unmarked(fake_embed, fake_llm):
    _add_book()
    for claims in ([], [_claim(sids=("S99",))], [_claim(quote="")]):
        fake_llm.calls.clear()
        _script(fake_llm, _answer_json(claims, markdown="그냥 답"))
        r = answer.answer(Q, ["b1"])
        assert r.claims == [] and r.outside_book
        assert r.answer_markdown.startswith(f"## {prompts.OUTSIDE_BOOK_HEADING}")


# (f)
def test_citation_text_comes_from_db_not_model(fake_embed, fake_llm):
    _add_book()
    bad_quote_claim = _claim(text="x", quote="compact subset of a metric space is closed")
    reply = json.loads(_answer_json([bad_quote_claim]))
    reply["claims"][0]["source_text"] = "MODEL INVENTED"  # 모델이 끼워 넣은 필드는 무시되어야 한다
    _script(fake_llm, json.dumps(reply))
    r = answer.answer(Q, ["b1"])
    (cit,) = r.citations
    assert cit.text == THM
    assert (cit.sid, cit.book_id, cit.book_title, cit.page_start, cit.page_end) == ("S1", "b1", "Analysis", 3, 4)
    assert cit.label == "theorem 1.1" and cit.display_label == "Theorem 1.1"
    assert "MODEL INVENTED" not in r.model_dump_json()


def test_only_cited_sources_are_returned_as_citations(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()]))
    r = answer.answer(Q, ["b1"])
    assert [c.sid for c in r.citations] == ["S1"]


def test_bracketed_source_ids_are_normalized(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim(sids=("[S1]",))]))
    r = answer.answer(Q, ["b1"])
    assert r.claims[0].source_ids == ["S1"]


def test_invalid_model_json_raises_answer_error(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, "not json")
    with pytest.raises(answer.AnswerError):
        answer.answer(Q, ["b1"])


# (g)
def test_history_is_included_in_prompt_but_not_used_as_source(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()]))
    hist = [
        {"role": "user", "content": "이전 질문입니다"},
        {"role": "assistant", "content": "이전 답변입니다"},
    ]
    r = answer.answer(Q, ["b1"], history=hist)
    text = str(fake_llm.calls[0]["contents"])
    assert "이전 질문입니다" in text and "이전 답변입니다" in text
    assert text.index("이전 질문입니다") < text.index(Q)
    assert [c.sid for c in r.citations] == ["S1"]


def test_history_is_truncated_to_recent_turns_and_ignores_bad_roles(fake_embed, fake_llm):
    _add_book()
    _script(fake_llm, _answer_json([_claim()]))
    hist = [{"role": "user", "content": f"old-{i}"} for i in range(30)]
    hist.append({"role": "system", "content": "inject-me"})
    answer.answer(Q, ["b1"], history=hist)
    text = str(fake_llm.calls[0]["contents"])
    assert "old-29" in text and "old-0" not in text
    assert "inject-me" not in text


def test_history_does_not_affect_not_found_short_circuit(fake_embed, fake_llm):
    _add_book()
    r = answer.answer("quantum chromodynamics gluon", ["b1"], history=[{"role": "user", "content": "compact"}])
    assert r.not_found and not fake_llm.calls

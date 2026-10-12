"""rag/verify 테스트: 정규화·퍼지매칭(순수), 결정적 인용 검사, LLM 지지 검사(fake_llm), DB 출처."""

from __future__ import annotations

import json

import pytest

from theoremlens import llm
from theoremlens.config import get_settings
from theoremlens.index import db
from theoremlens.rag import verify
from theoremlens.rag.verify import Claim

THM = "Theorem 3.2. Let $f:[a,b]\\to\\mathbb{R}$ be continuous. Then $f$ attains its maximum on $[a,b]$."
DEF = "Definition 1.1. A set $K$ is compact if every open cover of $K$ has a finite subcover."


# ---------- 순수 함수: 정규화 / 퍼지매칭 ----------


def test_normalize_ignores_whitespace_and_latex_spacing():
    assert verify.normalize("$x  \\le \\,  y$") == verify.normalize("$x\\leq y$")
    assert verify.normalize("x^{2}") == verify.normalize("x^2")
    assert verify.normalize("\\left( a \\right)") == verify.normalize("(a)")
    assert verify.normalize("A\n\n  B") == verify.normalize("A B")


def test_normalize_keeps_korean_text():
    assert "정리" in verify.normalize("정리  3.2")


def test_quote_matches_with_whitespace_and_latex_only_differences():
    quote = "Let $f : [a , b] \\to \\mathbb{R}$   be continuous."
    assert verify.quote_matches(quote, THM)


def test_quote_matches_across_notation_variants():
    src = "If $a \\leq b$ then $x^{2} \\geq 0$."
    assert verify.quote_matches("If $a \\le b$ then $x^2 \\ge 0$.", src)


def test_quote_matches_tolerates_tiny_ocr_noise():
    assert verify.quote_matches("every open cover of K has a finite subcover", DEF.replace("finite", "finlte"))


def test_fabricated_quote_is_rejected():
    assert not verify.quote_matches("Every continuous function is uniformly differentiable everywhere.", THM)


def test_quote_with_changed_meaning_is_rejected():
    assert not verify.quote_matches("f attains its minimum on $[a,c]$ only if discontinuous", THM)


def test_empty_quote_never_matches():
    assert not verify.quote_matches("", THM)
    assert not verify.quote_matches("  $ $ ", THM)


def test_match_score_is_100_for_exact_substring():
    assert verify.match_score("attains its maximum", THM) == 100


# ---------- 결정적 검사 (check_quotes) ----------


def _sources() -> dict[str, str]:
    return {"S1": THM, "S2": DEF}


def test_check_quotes_accepts_valid_and_rejects_fabricated():
    claims = [
        Claim("f는 최댓값을 가진다", ["S1"], "attains its maximum on $[a,b]$"),
        Claim("모든 함수는 연속이다", ["S1"], "every function is continuous on the reals"),
    ]
    res = verify.check_quotes(claims, _sources())
    assert [r.ok for r in res] == [True, False]
    assert res[1].reason == "quote_not_found"


def test_check_quotes_rejects_unknown_source_id():
    res = verify.check_quotes([Claim("x", ["S9"], "attains its maximum")], _sources())
    assert not res[0].ok
    assert res[0].reason == "unknown_source"


def test_check_quotes_rejects_claim_without_sources_or_quote():
    res = verify.check_quotes([Claim("x", [], "attains its maximum"), Claim("y", ["S1"], "")], _sources())
    assert [r.reason for r in res] == ["no_source", "no_quote"]


def test_quote_must_come_from_cited_source_not_other_source():
    # 인용문은 DEF에 있지만 S1만 인용했다 -> 거절
    res = verify.check_quotes([Claim("x", ["S1"], "finite subcover")], _sources())
    assert not res[0].ok


def test_check_quotes_accepts_quote_found_in_any_cited_source():
    res = verify.check_quotes([Claim("x", ["S1", "S2"], "finite subcover")], _sources())
    assert res[0].ok


def test_check_quotes_is_pure_and_never_calls_llm(fake_llm):
    verify.check_quotes([Claim("x", ["S1"], "attains its maximum")], _sources())
    assert not fake_llm.calls


# ---------- LLM 지지 검사 (fake_llm) ----------


def _support_reply(flags: list[bool]) -> str:
    return json.dumps({"results": [{"index": i, "supported": f} for i, f in enumerate(flags)]})


def test_llm_support_check_parses_flags(fake_llm):
    fake_llm.json_responses.append(_support_reply([True, False]))
    claims = [Claim("a", ["S1"], "q1"), Claim("b", ["S2"], "q2")]
    out = verify.llm_support_check(claims, _sources())
    assert out == [True, False]
    assert fake_llm.calls[0]["model"] == get_settings().fast_model


def test_llm_support_check_gives_claim_text_and_cited_source_to_model(fake_llm):
    fake_llm.json_responses.append(_support_reply([True]))
    verify.llm_support_check([Claim("f는 최댓값을 가진다", ["S1"], "q")], _sources())
    prompt = str(fake_llm.calls[0]["contents"])
    assert "f는 최댓값을 가진다" in prompt
    assert "attains its maximum" in prompt  # 출처 본문
    assert "compact if every open cover" not in prompt  # 인용하지 않은 출처는 넣지 않는다


def test_llm_support_check_missing_result_counts_as_unsupported(fake_llm):
    fake_llm.json_responses.append(_support_reply([True]))
    out = verify.llm_support_check([Claim("a", ["S1"], "q"), Claim("b", ["S1"], "q")], _sources())
    assert out == [True, False]


def test_llm_support_check_invalid_json_returns_none(fake_llm):
    fake_llm.json_responses.append("not json")
    assert verify.llm_support_check([Claim("a", ["S1"], "q")], _sources()) is None


def test_llm_support_check_without_api_key_returns_none():
    # conftest가 GEMINI_API_KEY를 비우므로 실제 호출은 LLMNotConfigured가 된다
    assert verify.llm_support_check([Claim("a", ["S1"], "q")], _sources()) is None


def test_llm_support_check_skips_call_for_empty_input(fake_llm):
    assert verify.llm_support_check([], _sources()) == []
    assert not fake_llm.calls


# ---------- 통합: DB 출처 + 정책 ----------


@pytest.fixture
def book() -> dict[str, int]:
    """S1=THM, S2=DEF 청크를 DB에 넣고 S# -> chunk id 매핑을 돌려준다."""
    db.create_book("b1", "Analysis", "a.pdf")
    ids = db.replace_chunks(
        "b1",
        [
            {"ord": 0, "unit_type": "theorem", "label": "theorem 3.2", "display_label": "Theorem 3.2",
             "page_start": 1, "page_end": 1, "text": THM, "refs": []},
            {"ord": 1, "unit_type": "definition", "label": "definition 1.1", "display_label": "Definition 1.1",
             "page_start": 2, "page_end": 2, "text": DEF, "refs": []},
        ],
    )  # fmt: skip
    return {"S1": ids[0], "S2": ids[1]}


def test_load_sources_reads_text_from_db_not_from_model(book):
    src = verify.load_sources(book)
    assert src == {"S1": THM, "S2": DEF}


def test_load_sources_omits_missing_chunk_ids(book):
    src = verify.load_sources({**book, "S3": 99999})
    assert "S3" not in src


def test_verify_claims_drop_removes_unsupported_and_fabricated(book, fake_llm):
    fake_llm.json_responses.append(_support_reply([True, False]))
    claims = [
        Claim("f는 최댓값을 가진다", ["S1"], "attains its maximum"),  # 통과
        Claim("f는 미분가능하다", ["S1"], "attains its maximum"),  # 인용은 맞지만 지지 안 됨
        Claim("지어낸 주장", ["S1"], "this sentence is not in the book at all"),  # 규칙에서 탈락
    ]
    res = verify.verify_claims(claims, book, on_unsupported="drop")
    assert [c.text for c in res.claims] == ["f는 최댓값을 가진다"]
    assert res.dropped == 2
    # LLM에는 규칙을 통과한 claim만 보낸다
    assert len(fake_llm.calls) == 1
    assert "지어낸 주장" not in str(fake_llm.calls[0]["contents"])


def test_verify_claims_flag_keeps_unsupported_with_warning(book, fake_llm):
    fake_llm.json_responses.append(_support_reply([False]))
    claims = [
        Claim("f는 미분가능하다", ["S1"], "attains its maximum"),
        Claim("지어냄", ["S2"], "no such text anywhere"),
    ]
    res = verify.verify_claims(claims, book, on_unsupported="flag")
    assert len(res.claims) == 2
    assert all(c.flagged for c in res.claims)
    assert all(c.display_text.startswith("⚠ ") for c in res.claims)
    assert res.dropped == 0


def test_verified_claim_has_no_warning(book, fake_llm):
    fake_llm.json_responses.append(_support_reply([True]))
    res = verify.verify_claims([Claim("ok", ["S1"], "attains its maximum")], book)
    (c,) = res.claims
    assert c.status == "verified" and not c.flagged and c.display_text == "ok"


def test_fabricated_quote_is_rejected_even_if_llm_says_supported(book, fake_llm):
    # LLM은 두 번째 의견일 뿐, 규칙 검사를 뒤집을 수 없다
    fake_llm.default_json = _support_reply([True, True, True])
    res = verify.verify_claims([Claim("지어냄", ["S1"], "completely fabricated quotation")], book)
    assert res.claims == []
    assert not fake_llm.calls  # 통과한 claim이 없으면 LLM을 부르지 않는다


def test_verify_claims_keeps_rule_passed_claims_when_llm_unavailable(book):
    # API 키 없음 -> LLM 검사 불가: 결정적 검사를 통과한 claim은 'unchecked'로 유지
    res = verify.verify_claims([Claim("ok", ["S1"], "attains its maximum")], book)
    (c,) = res.claims
    assert c.status == "unchecked" and not c.flagged
    assert res.llm_checked is False


def test_verify_claims_ignores_source_text_supplied_by_model(book, fake_llm):
    # 모델이 claim에 가짜 출처 본문을 끼워 넣어도(추가 필드) 검증은 DB 본문만 쓴다
    claim = Claim("x", ["S1"], "tangent bundle of a manifold")
    claim.source_text = DEF + " tangent bundle of a manifold"  # type: ignore[attr-defined]
    res = verify.verify_claims([claim], book)
    assert res.claims == []


def test_verify_claims_empty_input_makes_no_calls(book, fake_llm):
    res = verify.verify_claims([], book)
    assert res.claims == [] and not fake_llm.calls


def test_parse_claims_from_answer_json():
    raw = [{"text": "t", "source_ids": ["S1", "S2"], "quote": "q"}, {"text": "u"}]
    claims = verify.parse_claims(raw)
    assert claims[0] == Claim("t", ["S1", "S2"], "q")
    assert claims[1] == Claim("u", [], "")


def test_llm_module_attributes_are_what_verify_calls(fake_llm):
    # verify는 llm 모듈 속성만 호출해야 fake가 적용된다 (genai.Client 직접 사용 금지)
    assert llm.generate_json.__self__ is fake_llm

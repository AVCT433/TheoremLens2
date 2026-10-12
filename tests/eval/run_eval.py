"""품질 평가 러너: 손으로 쓴 질문 파일로 검색 적중률, 인용 정확도, 거절률, 오거절률을 잰다.

실제 Gemini와 이미 인제스트된 책이 필요하므로 pytest가 수집하지 않는다(파일명이 test_로 시작하지 않는다).
지표 계산과 러너 로직은 순수 함수로 두어 가짜 retrieve/answer로 단위 테스트한다.

사용: python -m tests.eval.run_eval --questions tests/eval/questions.example.json --books <book_id> [<book_id> ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from theoremlens.config import Settings, get_settings
from theoremlens.index import db
from theoremlens.rag import answer as answer_mod
from theoremlens.rag import retrieve as retrieve_mod
from theoremlens.rag import verify
from theoremlens.rag.answer import AnswerResponse
from theoremlens.rag.retrieve import Retrieval, Source

log = logging.getLogger(__name__)


class EvalError(RuntimeError):
    """질문 파일이 규칙을 어길 때."""


@dataclass
class Question:
    question: str
    expected_labels: list[str] = field(default_factory=list)
    expected_pages: list[int] = field(default_factory=list)  # 1-based PDF 페이지 인덱스
    expect_refusal: bool = False


@dataclass
class QuestionResult:
    question: str
    expect_refusal: bool
    refused: bool
    retrieval_hit: bool | None  # 정답 정보가 없는 거절 질문이거나 오류면 None
    citations_ok: int
    citations_total: int
    sources: list[str]
    answer: str
    error: str | None = None
    outside_book: bool = False  # 답은 냈지만 책에서 검증된 인용이 없다(거절이 아니다)


RetrieveFn = Callable[..., Retrieval]
AnswerFn = Callable[..., AnswerResponse]


def load_questions(path: Path) -> list[Question]:
    """질문 파일을 읽고 규칙을 검증한다. 정답 정보 없는 책 내 질문은 평가할 수 없으므로 거부한다."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvalError(f"{path}: JSON 형식 오류: {e}") from e
    if not isinstance(raw, list):
        raise EvalError(f"{path}: 최상위는 질문 배열이어야 합니다.")
    out: list[Question] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            raise EvalError(f"{path}[{i}]: question 이 비어 있습니다.")
        q = Question(
            question=str(item["question"]).strip(),
            expected_labels=[str(x) for x in item.get("expected_labels", [])],
            expected_pages=[int(x) for x in item.get("expected_pages", [])],
            expect_refusal=bool(item.get("expect_refusal", False)),
        )
        has_truth = bool(q.expected_labels or q.expected_pages)
        if q.expect_refusal and has_truth:
            raise EvalError(f"{path}[{i}]: 거절 질문에는 expected_labels/pages 를 쓰지 않습니다.")
        if not q.expect_refusal and not has_truth:
            raise EvalError(f"{path}[{i}]: 책 내 질문에는 expected_labels 또는 expected_pages 가 필요합니다.")
        if any(p < 1 for p in q.expected_pages):
            raise EvalError(f"{path}[{i}]: 페이지는 1부터 시작하는 PDF 인덱스입니다.")
        out.append(q)
    return out


def _norm_label(label: str) -> str:
    """ "정리 1.1", "Theorem 1.1" 모두 retrieve 와 같은 정규화 라벨("theorem 1.1")로 맞춘다."""
    found = retrieve_mod.extract_labels(label)
    return found[0] if found else re.sub(r"\s+", " ", label.strip().lower())


def retrieval_hit(q: Question, sources: Sequence[Source], top_k: int) -> bool:
    """상위 top_k 근거 중 기대 라벨이 있거나, 기대 페이지가 청크 페이지 범위에 들면 적중."""
    wanted = {_norm_label(x) for x in q.expected_labels}
    for s in list(sources)[:top_k]:
        if s.label and _norm_label(s.label) in wanted:
            return True
        if any(s.page_start <= p <= s.page_end for p in q.expected_pages):
            return True
    return False


def citation_accuracy(resp: AnswerResponse) -> tuple[int, int]:
    """(맞은 주장 수, 평가한 주장 수). 주장의 quote 가 자신이 인용한 청크 원문과 퍼지 매칭되면 맞다."""
    text_by_sid = {c.sid: c.text for c in resp.citations}
    ok = total = 0
    for c in resp.claims:
        if not c.quote.strip() or not c.source_ids:
            continue
        total += 1
        if any(verify.quote_matches(c.quote, text_by_sid[sid]) for sid in c.source_ids if sid in text_by_sid):
            ok += 1
    return ok, total


def is_refusal(resp: AnswerResponse) -> bool:
    """거절은 not_found 뿐이다. outside_book 은 답을 낸 것이므로 따로 센다."""
    return resp.not_found


def is_outside_book_answer(resp: AnswerResponse) -> bool:
    """답은 냈지만 책에서 검증된 인용이 없는 응답(환각 위험 신호)."""
    return resp.outside_book and not resp.not_found


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_metrics(results: Sequence[QuestionResult]) -> dict:
    """오류 난 질문은 비율에서 제외하고 n_errors 로만 센다. 분모가 0이면 None."""
    good = [r for r in results if r.error is None]
    in_book = [r for r in good if not r.expect_refusal]
    out_book = [r for r in good if r.expect_refusal]
    hits = [r for r in in_book if r.retrieval_hit is not None]
    return {
        "retrieval_hit_rate": _rate(sum(bool(r.retrieval_hit) for r in hits), len(hits)),
        "citation_accuracy": _rate(sum(r.citations_ok for r in in_book), sum(r.citations_total for r in in_book)),
        "refusal_rate": _rate(sum(r.refused for r in out_book), len(out_book)),
        "outside_book_answer_rate": _rate(sum(r.outside_book for r in out_book), len(out_book)),
        "false_refusal_rate": _rate(sum(r.refused for r in in_book), len(in_book)),
        "unsupported_answer_rate": _rate(sum(r.outside_book for r in in_book), len(in_book)),
        "n_in_book": len(in_book),
        "n_out_of_book": len(out_book),
        "n_errors": len(results) - len(good),
    }


def run_questions(
    questions: Sequence[Question],
    book_ids: list[str],
    retrieve_fn: RetrieveFn | None = None,
    answer_fn: AnswerFn | None = None,
    top_k: int | None = None,
) -> list[QuestionResult]:
    """질문마다 검색과 답변을 돌려 채점한다. 한 질문의 실패가 전체를 멈추지 않게 오류를 기록한다."""
    retrieve_fn = retrieve_fn or retrieve_mod.retrieve
    answer_fn = answer_fn or answer_mod.answer
    top_k = top_k or get_settings().top_k
    results: list[QuestionResult] = []
    for q in questions:
        try:
            r = retrieve_fn(q.question, book_ids, top_k)
            resp = answer_fn(q.question, book_ids)
            ok, total = citation_accuracy(resp)
            has_truth = bool(q.expected_labels or q.expected_pages)
            results.append(
                QuestionResult(
                    question=q.question,
                    expect_refusal=q.expect_refusal,
                    refused=is_refusal(resp),
                    outside_book=is_outside_book_answer(resp),
                    retrieval_hit=retrieval_hit(q, r.sources, top_k) if has_truth else None,
                    citations_ok=ok,
                    citations_total=total,
                    sources=[f"{s.sid}:{s.label or '-'}:p{s.page_start}-{s.page_end}" for s in r.sources],
                    answer=resp.answer_markdown,
                )
            )
        except Exception as e:  # noqa: BLE001 - 질문 하나의 실패를 기록하고 나머지를 계속 평가한다
            log.exception("question failed: %s", q.question)
            results.append(
                QuestionResult(
                    q.question, q.expect_refusal, False, None, 0, 0, [], "", error=f"{type(e).__name__}: {e}"
                )
            )
    return results


def result_path(data_dir: Path, settings: Settings, now: datetime) -> Path:
    """data/eval/ 아래에 날짜와 모델 이름이 든 파일명. 모델·프롬프트 변경 전후를 비교하기 위함."""
    models = "_".join(
        re.sub(r"[^\w.\-]+", "-", m) for m in (settings.answer_model, settings.fast_model, settings.embed_model)
    )
    return data_dir / "eval" / f"eval_{now:%Y-%m-%d_%H%M%S}_{models}.json"


def write_report(
    results: Sequence[QuestionResult],
    settings: Settings,
    book_ids: list[str],
    questions_file: str,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now()
    path = result_path(settings.data_dir, settings, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "date": now.isoformat(timespec="seconds"),
        "models": {"answer": settings.answer_model, "fast": settings.fast_model, "embed": settings.embed_model},
        "top_k": settings.top_k,
        "min_similarity": settings.min_similarity,
        "books": book_ids,
        "questions_file": questions_file,
        "metrics": compute_metrics(results),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TheoremLens 품질 평가")
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--books", nargs="*", help="평가할 book id (생략하면 ready 상태 책 전부)")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    s = get_settings()
    book_ids = args.books or [b["id"] for b in db.list_books() if b.get("status") == "ready"]
    if not book_ids:
        print("평가할 책이 없습니다. 먼저 책을 인제스트하세요.", file=sys.stderr)
        return 2
    results = run_questions(questions, book_ids, top_k=s.top_k)
    path = write_report(results, s, book_ids, str(args.questions))
    m = compute_metrics(results)
    print(f"검색 적중률  {_fmt(m['retrieval_hit_rate'])}")
    print(f"인용 정확도  {_fmt(m['citation_accuracy'])}")
    print(f"거절률(책 밖) {_fmt(m['refusal_rate'])}")
    print(f"책 밖 답변률(책 밖 질문) {_fmt(m['outside_book_answer_rate'])}")
    print(f"오거절률(책 안) {_fmt(m['false_refusal_rate'])}")
    print(f"근거 없는 답변률(책 안 질문) {_fmt(m['unsupported_answer_rate'])}")
    print(f"오류 {m['n_errors']}건. 저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

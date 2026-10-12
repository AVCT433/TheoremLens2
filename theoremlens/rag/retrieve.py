"""하이브리드 검색: BM25 + 벡터를 RRF로 합치고, 명시 라벨과 인용된 보조정리를 보강한다."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import get_settings
from ..index import db, vectors
from ..ingest import structure

log = logging.getLogger(__name__)

RRF_K = 60
# 후보는 최종 개수보다 넉넉히 가져와야 두 순위표가 겹칠 기회가 생긴다
_CANDIDATE_FACTOR = 3
# 인용 확장이 프롬프트를 부풀리지 않도록 상한을 둔다
_MAX_CITED = 4

# 한국어/영어 표기 → db.chunks.label 의 정규화 유형. 매핑은 structure 와 공유한다
_TYPE_ALIASES = structure.TYPE_ALIASES
_TYPE_PATTERN = "|".join(sorted((re.escape(k) for k in _TYPE_ALIASES), key=len, reverse=True))
_LABEL_RE = re.compile(rf"({_TYPE_PATTERN})\s*(\d+(?:\.\d+)*)", re.IGNORECASE)


@dataclass
class Source:
    """답변 프롬프트에 [S#]로 실리는 근거 한 개."""

    sid: str  # "S1", "S2", ...
    chunk_id: int
    book_id: str
    book_title: str
    page_start: int
    page_end: int
    label: str | None
    display_label: str | None
    unit_type: str
    chapter: str | None
    section: str | None
    text: str
    via: str  # search | label | cited: 어떤 경로로 들어왔는지


@dataclass
class Retrieval:
    sources: list[Source] = field(default_factory=list)
    not_found: bool = False
    best_similarity: float = 0.0


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. (id, 점수) 점수 내림차순, 동점이면 먼저 등장한 id 우선."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # 한 순위표에서 중복 id는 한 번만 센다
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])  # sorted는 안정 정렬이라 동점 순서가 유지된다


def extract_labels(question: str) -> list[str]:
    """질문 속 "Theorem 3.2", "정리 3.2" 같은 표기를 정규화 라벨("theorem 3.2")로 바꾼다."""
    out: list[str] = []
    for m in _LABEL_RE.finditer(question):
        kind = _TYPE_ALIASES[re.sub(r"\s+", " ", m.group(1).lower())]
        label = f"{kind} {m.group(2)}"
        if label not in out:
            out.append(label)
    return out


def retrieve(question: str, book_ids: list[str], top_k: int | None = None) -> Retrieval:
    s = get_settings()
    top_k = top_k or s.top_k
    if not book_ids:
        return Retrieval(not_found=True)

    limit = top_k * _CANDIDATE_FACTOR
    vec_hits = vectors.search(question, book_ids, limit)
    best = max((sim for _, sim in vec_hits), default=0.0)
    bm25_hits = db.fts_search(question, book_ids, limit)

    # (chunk, via) 순서가 곧 프롬프트의 S 번호 순서다
    picked: list[tuple[db.Chunk, str]] = []
    seen: set[int] = set()

    def _add(chunks: list[db.Chunk], via: str) -> None:
        for c in chunks:
            if c.id not in seen:
                seen.add(c.id)
                picked.append((c, via))

    labeled = db.chunks_by_label(extract_labels(question), book_ids)
    _add(labeled, "label")

    if not labeled and best < s.min_similarity:
        # 근거가 약하면 LLM을 부르지 않고 "책에 없음"으로 끝낸다
        log.info("best similarity %.3f < %.3f: not found", best, s.min_similarity)
        return Retrieval(not_found=True, best_similarity=best)

    fused = rrf([[i for i, _ in vec_hits], [i for i, _ in bm25_hits]])
    room = max(0, top_k - len(picked))
    _add(db.get_chunks([i for i, _ in fused if i not in seen][:room]), "search")

    # 인용 확장은 1단계만: 확장으로 들어온 청크의 refs는 따라가지 않는다
    base = list(picked)
    cited_added = 0
    for c, _ in base:
        for ref in c.refs:
            for cc in db.chunks_by_label([ref], [c.book_id]):
                if cc.id not in seen and cited_added < _MAX_CITED:
                    _add([cc], "cited")
                    cited_added += 1

    titles = db.book_titles(sorted({c.book_id for c, _ in picked}))
    sources = [
        Source(
            sid=f"S{n}",
            chunk_id=c.id,
            book_id=c.book_id,
            book_title=titles.get(c.book_id, c.book_id),
            page_start=c.page_start,
            page_end=c.page_end,
            label=c.label,
            display_label=c.display_label,
            unit_type=c.unit_type,
            chapter=c.chapter,
            section=c.section,
            text=c.text,
            via=via,
        )
        for n, (c, via) in enumerate(picked, start=1)
    ]
    return Retrieval(sources=sources, not_found=not sources, best_similarity=best)

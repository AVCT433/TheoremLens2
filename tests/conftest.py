"""테스트 공통 fixture: 전역 싱글턴 격리와 Gemini 가짜(fake)."""

from __future__ import annotations

import gc
import hashlib
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from theoremlens import embedder, llm
from theoremlens.config import get_settings
from theoremlens.index import db, vectors
from theoremlens.ingest import extract as _extract

_REAL_RUN_MINERU = _extract._run_mineru


def _release() -> None:
    # Windows는 열린 SQLite/Chroma 파일을 지울 수 없으므로 매번 핸들을 놓는다
    db.close()
    if vectors._collection.cache_info().currsize:
        try:
            vectors._collection()._client._system.stop()
        except Exception:  # noqa: BLE001 - 정리 실패가 테스트 결과를 가리면 안 된다
            pass
    vectors._collection.cache_clear()
    SharedSystemClient.clear_system_cache()
    getattr(llm.client, "cache_clear", lambda: None)()  # 테스트가 client를 교체했을 수 있다
    embedder._model = None
    get_settings.cache_clear()
    gc.collect()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch, request) -> Iterator[None]:
    monkeypatch.setenv("THEOREMLENS_DATA_DIR", str(tmp_path / "data"))
    if "integration" not in request.keywords:
        monkeypatch.setenv("GEMINI_API_KEY", "")
    if "integration" not in request.keywords:
        # 실수로 실제 로컬 모델을 내려받거나 로드하지 않도록 막는다(Gemini 키를 비우는 것과 같은 이유)
        def _blocked():
            raise embedder.EmbedderNotAvailable("테스트에서는 실제 임베딩 모델을 로드할 수 없습니다.")

        monkeypatch.setattr(embedder, "_load", _blocked)
    get_settings.cache_clear()
    db.close()
    vectors._collection.cache_clear()
    getattr(llm.client, "cache_clear", lambda: None)()  # 테스트가 client를 교체했을 수 있다
    yield
    _release()


_WORD = re.compile(r"\w+", re.UNICODE)
_DIM = 256


def bow_vector(text: str) -> list[float]:
    """단어 해시 기반 bag-of-words 벡터. 결정적이며 단어가 겹치면 코사인 유사도가 높다."""
    v = [0.0] * _DIM
    for w in _WORD.findall(text.lower()):
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        v[h % _DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def add_legacy_collection() -> None:
    """Gemini 시절의 `chunks` 컬렉션을 흉내 낸다(다른 차원의 벡터 하나)."""
    client = chromadb.PersistentClient(path=str(get_settings().chroma_dir))
    client.get_or_create_collection("chunks").add(ids=["9"], embeddings=[[0.1, 0.2]])


def collection_names() -> list[str]:
    return [c.name for c in chromadb.PersistentClient(path=str(get_settings().chroma_dir)).list_collections()]


@dataclass
class FakeEmbed:
    calls: list[tuple[list[str], str]] = field(default_factory=list)


@pytest.fixture
def fake_embed(monkeypatch) -> FakeEmbed:
    fe = FakeEmbed()

    def _embed(texts: list[str], task: str) -> list[list[float]]:
        fe.calls.append((list(texts), task))
        return [bow_vector(t) for t in texts]

    monkeypatch.setattr(embedder, "embed", _embed)
    return fe


@dataclass
class FakeLLM:
    """generate_text / generate_json 스크립트 응답. 문자열 또는 (인자 -> 문자열) 함수를 줄 수 있다."""

    text_responses: list[Any] = field(default_factory=list)
    json_responses: list[Any] = field(default_factory=list)
    default_text: str = ""
    default_json: str = "{}"
    calls: list[dict] = field(default_factory=list)

    def _next(self, queue: list[Any], default: str, kwargs: dict) -> str:
        r = queue.pop(0) if queue else default
        return r(**kwargs) if isinstance(r, Callable) else r

    def generate_text(self, model: str, system: str, contents: Any) -> str:
        kw = {"model": model, "system": system, "contents": contents}
        self.calls.append({"kind": "text", **kw})
        return self._next(self.text_responses, self.default_text, kw)

    def generate_json(self, model: str, system: str, contents: Any, schema: dict) -> str:
        kw = {"model": model, "system": system, "contents": contents, "schema": schema}
        self.calls.append({"kind": "json", **kw})
        return self._next(self.json_responses, self.default_json, kw)


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    f = FakeLLM()
    monkeypatch.setattr(llm, "generate_text", f.generate_text)
    monkeypatch.setattr(llm, "generate_json", f.generate_json)
    return f


@dataclass
class FakeMineru:
    """`extract._run_mineru`를 대신한다: PDF의 텍스트 레이어를 middle_json 한 블록으로 써 준다.

    텍스트가 없는 쪽(스캔)은 `scan_text`를 쓰고, 이미지도 없으면 빈 쪽이다. `requested`는 호출마다 받은 쪽 목록.
    """

    scan_text: str = "scanned page text"
    requested: list[list[int]] = field(default_factory=list)

    def run(self, pdf_path: Path, pages: list[int], out_path: Path) -> None:
        import json

        import pymupdf

        self.requested.append(list(pages))
        out = []
        with pymupdf.open(pdf_path) as doc:
            for p in pages:
                page = doc[p - 1]
                text = page.get_text("text").strip()
                if not text and page.get_images():
                    text = self.scan_text
                blocks = [{"type": "text", "content": [{"type": "text", "content": text}]}] if text else []
                out.append({"page_idx": p - 1, "blocks": blocks})
        out_path.write_text(json.dumps({"pages": out}), encoding="utf-8")


@pytest.fixture(autouse=True)
def fake_mineru(monkeypatch, request) -> FakeMineru:
    """실제 MinerU(모델 로딩, 수 분)가 단위 테스트에서 돌지 않게 항상 가짜로 바꾼다."""
    f = FakeMineru()
    if "integration" not in request.keywords:
        monkeypatch.setattr(_extract, "_run_mineru", f.run)
    return f


@pytest.fixture
def real_run_mineru():
    """가짜로 바뀌기 전의 `_run_mineru`. subprocess를 가로채는 명령 조립 테스트용."""
    return _REAL_RUN_MINERU


@pytest.fixture
def make_pdf(tmp_path):
    """테스트 안에서 PDF를 만든다(PDF는 커밋하지 않는다). kinds: 'text' | 'scan' | 'blank'."""
    import pymupdf

    def _make(
        kinds: list[str], name: str = "book.pdf", text: str = "Theorem {i}.1 every compact set is bounded."
    ) -> Path:
        path = tmp_path / "pdfs" / name
        path.parent.mkdir(exist_ok=True)
        doc = pymupdf.open()
        for i, kind in enumerate(kinds, start=1):
            page = doc.new_page()
            if kind == "text":
                page.insert_text((72, 100), text.format(i=i) + " Proof. Trivial. " + "x" * 30)
            elif kind == "scan":
                pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 100), False)
                pix.clear_with(200)
                page.insert_image(pymupdf.Rect(72, 72, 272, 172), pixmap=pix)
        doc.save(path)
        doc.close()
        return path

    return _make

"""embedder 테스트: 로컬 모델 래퍼의 배치·정규화·접두어·디바이스 선택. 실제 모델은 integration에서만 쓴다."""

from __future__ import annotations

import numpy as np
import pytest

from theoremlens import embedder
from theoremlens.config import get_settings

# conftest의 autouse fixture가 `_load`를 막기 전, 수집 시점의 실제 함수
_REAL_LOAD = embedder._load


class _FakeST:
    """SentenceTransformer 대역: encode 호출 인자를 기록하고 입력 순서를 담은 벡터를 돌려준다."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


@pytest.fixture
def fake_model(monkeypatch) -> _FakeST:
    m = _FakeST()
    monkeypatch.setattr(embedder, "_load", lambda: m)
    return m


def test_embed_returns_python_float_lists_in_input_order(fake_model):
    out = embedder.embed(["a", "bbb", "cc"], "document")
    assert out == [[1.0, 1.0], [3.0, 1.0], [2.0, 1.0]]
    assert all(type(x) is float for v in out for x in v)


def test_embed_empty_input_does_not_load_model(monkeypatch):
    def boom():
        raise AssertionError("모델을 로드하면 안 된다")

    monkeypatch.setattr(embedder, "_load", boom)
    assert embedder.embed([], "query") == []


def test_embed_asks_for_normalized_vectors_with_configured_batch_size(fake_model, monkeypatch):
    monkeypatch.setenv("THEOREMLENS_EMBED_BATCH", "7")
    get_settings.cache_clear()
    embedder.embed(["x"], "document")
    kwargs = fake_model.calls[0][1]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 7


def test_prefix_is_applied_per_task(fake_model, monkeypatch):
    monkeypatch.setenv("THEOREMLENS_EMBED_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("THEOREMLENS_EMBED_DOC_PREFIX", "passage: ")
    get_settings.cache_clear()
    embedder.embed(["q"], "query")
    embedder.embed(["d"], "document")
    assert fake_model.calls[0][0] == ["query: q"]
    assert fake_model.calls[1][0] == ["passage: d"]


def test_default_prefixes_are_empty(fake_model):
    embedder.embed(["q"], "query")
    assert fake_model.calls[0][0] == ["q"]


def test_unknown_task_is_rejected(fake_model):
    with pytest.raises(ValueError):
        embedder.embed(["x"], "RETRIEVAL_QUERY")  # type: ignore[arg-type]
    assert fake_model.calls == []


@pytest.mark.parametrize(
    ("pref", "cuda", "expected"),
    [("auto", True, "cuda"), ("auto", False, "cpu"), ("cpu", True, "cpu"), ("cuda", False, "cuda")],
)
def test_resolve_device(pref, cuda, expected, monkeypatch):
    monkeypatch.setattr(embedder, "_cuda_available", lambda: cuda)
    assert embedder.resolve_device(pref) == expected


def test_embedding_guard_in_tests_blocks_real_model_load():
    # conftest의 격리 fixture가 실제 모델 다운로드/로드를 막는다
    with pytest.raises(embedder.EmbedderNotAvailable):
        embedder.embed(["x"], "query")


def test_stale_gemini_embed_model_setting_gets_a_clear_error(monkeypatch):
    # 예전 .env의 THEOREMLENS_EMBED_MODEL=gemini-embedding-001을 HuggingFace에서 찾다가 401로 죽지 않게 한다
    monkeypatch.setattr(embedder, "_load", _REAL_LOAD)  # conftest의 차단을 풀고 실제 검사를 탄다
    monkeypatch.setenv("THEOREMLENS_EMBED_MODEL", "gemini-embedding-001")
    get_settings.cache_clear()
    with pytest.raises(embedder.EmbedderNotAvailable, match="THEOREMLENS_EMBED_MODEL"):
        embedder._load()


@pytest.mark.integration
def test_embed_smoke_real_local_model():
    vecs = embedder.embed(
        ["A compact subset of a Hausdorff space is closed.", "banana bread recipe"],
        "document",
    )
    q = embedder.embed(["Are compact sets closed in Hausdorff spaces?"], "query")[0]
    assert len(vecs[0]) == len(q) > 0
    sim = [sum(a * b for a, b in zip(q, v, strict=True)) for v in vecs]
    assert sim[0] > sim[1]

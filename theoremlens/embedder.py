"""로컬 임베딩 모델 래퍼(sentence-transformers). 네트워크 없이 인덱싱·검색 임베딩을 계산한다."""

from __future__ import annotations

import logging
import threading
from typing import Literal

from .config import get_settings

log = logging.getLogger(__name__)

Task = Literal["document", "query"]

# 인제스트 스레드와 검색 요청이 동시에 처음 로드하려 해도 모델은 한 번만 올린다
_load_lock = threading.Lock()
_model = None


class EmbedderNotAvailable(RuntimeError):
    pass


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def resolve_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if _cuda_available() else "cpu"
    return preference


def _load():
    global _model
    with _load_lock:
        if _model is None:
            s = get_settings()
            if s.embed_model.lower().startswith("gemini"):
                raise EmbedderNotAvailable(
                    f"THEOREMLENS_EMBED_MODEL={s.embed_model!r}은 로컬 모델이 아닙니다. "
                    ".env의 값을 지우거나 BAAI/bge-m3 같은 HuggingFace 모델로 바꾸세요."
                )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbedderNotAvailable(
                    "sentence-transformers가 설치되지 않았습니다. `uv pip install -e .`를 실행하세요."
                ) from e
            device = resolve_device(s.embed_device)
            log.info("임베딩 모델 로드: %s (%s)", s.embed_model, device)
            kwargs = {"model_kwargs": {"torch_dtype": "float16"}} if device.startswith("cuda") else {}
            model = SentenceTransformer(s.embed_model, device=device, **kwargs)
            if s.embed_max_seq_length > 0:
                model.max_seq_length = min(model.max_seq_length or s.embed_max_seq_length, s.embed_max_seq_length)
            _model = model
        return _model


def embed(texts: list[str], task: Task) -> list[list[float]]:
    """texts를 L2 정규화된 벡터로 바꾼다(순서 유지). task: document | query"""
    s = get_settings()
    if task == "document":
        prefix = s.embed_doc_prefix
    elif task == "query":
        prefix = s.embed_query_prefix
    else:
        raise ValueError(f"알 수 없는 임베딩 task: {task!r}")
    if not texts:
        return []
    vecs = _load().encode(
        [prefix + t for t in texts],
        batch_size=s.embed_batch,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [[float(x) for x in v] for v in vecs]

"""ChromaDB 벡터 인덱스 (임베딩은 로컬 모델로 직접 계산해 넣는다)."""

from __future__ import annotations

import re
import threading
from functools import lru_cache

import chromadb
from chromadb.errors import NotFoundError

from .. import embedder
from ..config import get_settings

_lock = threading.Lock()

# Gemini 임베딩 시절의 컬렉션 이름. 모델이 다르면 차원·의미가 달라 섞으면 안 되므로 지금은 모델별 이름을 쓴다.
_LEGACY_COLLECTION = "chunks"


def collection_name() -> str:
    """임베딩 모델마다 컬렉션을 분리한다. 모델을 바꾸면 새 컬렉션이 비어 있어 재임베딩 대상이 된다."""
    slug = re.sub(r"[^a-z0-9._]+", "-", get_settings().embed_model.lower()).strip("-.")
    return f"chunks__{slug}"


@lru_cache
def _collection():
    client = chromadb.PersistentClient(path=str(get_settings().chroma_dir))
    return client.get_or_create_collection(collection_name(), metadata={"hnsw:space": "cosine"})


def embed_text_for(chunk: dict) -> str:
    head = " / ".join(x for x in (chunk.get("chapter"), chunk.get("section"), chunk.get("display_label")) if x)
    return f"{head}\n{chunk['text']}" if head else chunk["text"]


def index_chunks(book_id: str, chunk_ids: list[int], chunks: list[dict], on_progress=None) -> None:
    texts = [embed_text_for(c) for c in chunks]
    step = 64
    with _lock:
        delete_book(book_id)
        for i in range(0, len(texts), step):
            vecs = embedder.embed(texts[i : i + step], "document")
            _collection().add(
                ids=[str(x) for x in chunk_ids[i : i + step]],
                embeddings=vecs,
                metadatas=[{"book_id": book_id} for _ in vecs],
            )
            if on_progress:
                on_progress(min(1.0, (i + step) / max(1, len(texts))))


def search(query: str, book_ids: list[str], limit: int) -> list[tuple[int, float]]:
    """(chunk_id, cosine similarity) 목록, 유사도 내림차순."""
    if not book_ids:
        return []
    vec = embedder.embed([query], "query")[0]
    where = {"book_id": book_ids[0]} if len(book_ids) == 1 else {"book_id": {"$in": book_ids}}
    with _lock:
        res = _collection().query(query_embeddings=[vec], n_results=limit, where=where)
    ids = res["ids"][0] if res["ids"] else []
    dists = res["distances"][0] if res["distances"] else []
    return [(int(i), 1.0 - d) for i, d in zip(ids, dists, strict=True)]


def delete_book(book_id: str) -> None:
    _collection().delete(where={"book_id": book_id})


def has_book(book_id: str) -> bool:
    """현재 임베딩 모델의 컬렉션에 이 책의 벡터가 하나라도 있는가."""
    return bool(_collection().get(where={"book_id": book_id}, limit=1, include=[])["ids"])


def drop_legacy() -> bool:
    """Gemini 시절의 `chunks` 컬렉션을 지운다. 있었으면 True."""
    try:
        _collection()._client.delete_collection(_LEGACY_COLLECTION)
    except NotFoundError:
        return False
    return True

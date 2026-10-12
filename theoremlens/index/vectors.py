"""ChromaDB 벡터 인덱스 (임베딩은 Gemini로 직접 계산해 넣는다)."""

from __future__ import annotations

import threading
from functools import lru_cache

import chromadb

from .. import llm
from ..config import get_settings

_lock = threading.Lock()


@lru_cache
def _collection():
    client = chromadb.PersistentClient(path=str(get_settings().chroma_dir))
    return client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})


def embed_text_for(chunk: dict) -> str:
    head = " / ".join(x for x in (chunk.get("chapter"), chunk.get("section"), chunk.get("display_label")) if x)
    return f"{head}\n{chunk['text']}" if head else chunk["text"]


def index_chunks(book_id: str, chunk_ids: list[int], chunks: list[dict], on_progress=None) -> None:
    texts = [embed_text_for(c) for c in chunks]
    step = 64
    with _lock:
        delete_book(book_id)
        for i in range(0, len(texts), step):
            vecs = llm.embed(texts[i : i + step], "RETRIEVAL_DOCUMENT")
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
    vec = llm.embed([query], "RETRIEVAL_QUERY")[0]
    where = {"book_id": book_ids[0]} if len(book_ids) == 1 else {"book_id": {"$in": book_ids}}
    with _lock:
        res = _collection().query(query_embeddings=[vec], n_results=limit, where=where)
    ids = res["ids"][0] if res["ids"] else []
    dists = res["distances"][0] if res["distances"] else []
    return [(int(i), 1.0 - d) for i, d in zip(ids, dists, strict=True)]


def delete_book(book_id: str) -> None:
    _collection().delete(where={"book_id": book_id})

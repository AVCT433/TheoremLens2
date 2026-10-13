# theoremlens/index/ (storage and search indexes)

Persistence layer. Two stores with clear roles:

- `db.py`: SQLite is the source of truth. Tables: `books`, `pages`, `chunks`, `chunks_fts` (FTS5, kept in sync by triggers), `messages`. Access through `conn()` and `tx()` only. `fts_search` returns BM25 scores where lower is better.
- `vectors.py`: ChromaDB holds local-model embeddings only, in one collection per embedding model (`collection_name()`), keyed by SQLite `chunks.id`, with `book_id` metadata. `has_book` and `drop_legacy` support re-embedding after a model change (`pipeline.reindex_missing`). `search` returns cosine similarity where higher is better.

Rules:
- A chunk's id is the join key between the two stores. Never change ids without re-indexing vectors.
- Deleting a book must remove rows in `books`, `pages`, `chunks` (which fires the FTS trigger), and its vectors. Call `delete_book` in `db.py` and `vectors.delete_book` together.
- Embedding text is built by `vectors.embed_text_for`: chapter, section, and label as a header, then the body. Change it only with a full re-index.
- This package does not call Gemini. Embeddings come from `embedder.embed` (local model).

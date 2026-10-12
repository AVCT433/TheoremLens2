# theoremlens/app/ (FastAPI backend)

HTTP layer only. Routes validate input, call into `ingest/`, `index/`, and `rag/`, and shape responses. No business logic or LLM prompts here.

- `main.py` (implemented): `create_app(web_dir=None)` builds the app; `app` is the module-level instance (`uvicorn theoremlens.app.main:app`). Serves `web/` as static files, mounted last; the app starts even if `web/` is empty or missing.

Routes:
- `POST /books` upload, starts ingestion in the background
- `GET /books`, `GET /books/{id}/status`, `DELETE /books/{id}`
- `POST /chat` with `book_ids`, `question`, and `conversation_id`. Returns the answer, citations, and verification status. Streaming is optional.
- `GET /books/{id}/page/{n}` returns the rendered page image for citation review

Rules:
- Ingestion runs in a background task. Never block a request on extraction.
- Return citation source text from the DB, never from the model's output.
- On startup (lifespan) a background thread resumes interrupted ingests via `pipeline.recover_interrupted()`. Deletion goes through `pipeline.purge_book`, never directly through `db`/`vectors`.

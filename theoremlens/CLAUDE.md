# theoremlens/ (Python package root)

Top-level package for the TheoremLens backend. Each subpackage is one stage of the pipeline; this directory holds the shared pieces they all use.

- `config.py`: `Settings` loaded from `.env` and the `data/` layout (`db_path`, `chroma_dir`, `book_dir`). Read settings via `get_settings()`, never `os.getenv` directly in other modules.
- `llm.py`: the only place that creates a `genai.Client`. Exposes `generate_json` (schema-constrained output), `generate_text`, `image_part`, and `embed`. Add new Gemini calls here so retries and the missing-key check apply.

Dependency direction: `app/` → `rag/` → `index/` → `ingest/` → (`llm`, `config`). Lower layers must not import higher ones.

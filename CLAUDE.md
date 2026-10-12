# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TheoremLens: a local web app that ingests math textbooks (PDF, both text-layer and scanned) into a multi-book library and answers questions **only from the book content**, with citations, to minimize hallucination. LLM provider is **Google Gemini** (`google-genai` SDK); do not switch providers.

Status: backend complete. Done: `config.py`, `llm.py`, `index/*`, `ingest/*` (extract, structure, pipeline), `rag/*`, `app/main.py`, `web/` frontend (index.html, app.js, lib.js, style.css), and unit tests. Not yet written: `tests/eval/`. The approved design is in `C:\Users\baoro\.claude\plans\bright-snacking-sky.md`.

## Commands

```bash
uv venv .venv && uv pip install -e ".[dev]"     # base + pytest/httpx
uv pip install -e ".[marker]"                    # optional: Marker for LaTeX-quality text-PDF extraction (pulls torch)
.venv/Scripts/python -m pytest                   # all tests
.venv/Scripts/python -m pytest tests/test_x.py::test_name   # single test
.venv/Scripts/uvicorn theoremlens.app.main:app --reload     # run the app
```

Configuration comes from `.env` (see `.env.example`): `GEMINI_API_KEY`, the model names (`THEOREMLENS_ANSWER_MODEL`, `THEOREMLENS_FAST_MODEL`, `THEOREMLENS_EMBED_MODEL`), and `THEOREMLENS_TEXT_EXTRACTOR` (`auto|marker|pymupdf|gemini`), and `THEOREMLENS_MAX_UPLOAD_MB` (upload cap, default 300). Runtime data (SQLite DB, Chroma index, per-book files) lives in `data/`, which is gitignored.

## Architecture

Pipeline: **extract → structure → index → retrieve → answer → verify**.

- **Extract** (`ingest/extract.py`): each page is classified by its text-layer length. Text pages go through Marker if it is installed, otherwise PyMuPDF. Scanned pages are rendered to PNG and transcribed verbatim to Markdown+LaTeX by Gemini vision. The output is one Markdown string per page, stored in the `pages` table so re-ingestion can skip pages already extracted.
- **Structure** : split pages into *math units* (Definition, Theorem, Lemma, Proof, Example, Exercise, and their Korean forms 정의, 정리, 증명, …). A theorem is kept together with its proof. Display math (`$$…$$`) is never split. Each chunk carries a normalized `label` (e.g. `theorem 3.2`), a `display_label`, the chapter and section, a page range, and `refs` (labels it cites, e.g. "by Lemma 2.4").
- **Index** (`index/`): SQLite (`db.py`) is the source of truth for books, pages, chunks, and messages. The FTS5 table `chunks_fts` is kept in sync by triggers. ChromaDB (`vectors.py`) holds only Gemini embeddings, keyed by the SQLite chunk id with `book_id` metadata. Deleting a book must clear both stores.
- **Retrieve** : BM25 (`db.fts_search`; lower score is better) and vector search (`vectors.search`; cosine similarity) are fused with RRF. Explicit labels in the question are looked up directly with `db.chunks_by_label`, and the lemmas a retrieved chunk cites are added one hop deep. If the best similarity is below `Settings.min_similarity`, the app answers "not in the book" without calling the LLM.
- **Answer / verify** : Gemini is called at temperature 0 with JSON-schema output (`llm.generate_json`) shaped as `{answer_markdown, claims:[{text, source_ids, quote}], not_found}`. Every claim must cite `[S#]` excerpts. General math explanation from outside the book belongs only in a separately labelled "보충(책 외부)" section. Verification then fuzzy-matches each `quote` against its source chunk (rapidfuzz, after normalizing whitespace and LaTeX) and runs a fast-model check of whether the cited text supports each claim. Unsupported claims are dropped or flagged.

## Conventions

- Page numbers are **1-based PDF page indices** everywhere. They are not the printed page numbers.
- All Gemini calls go through `theoremlens/llm.py`, which handles retries, the API-key check (`LLMNotConfigured`), and embedding batching. Do not instantiate `genai.Client` elsewhere.
- `index/db.py` uses one shared connection guarded by an `RLock`. Use `tx()` for writes.
- User-facing text and LLM prompts are in Korean.

### Code style (enforced by ruff: `ruff check` and `ruff format`)

- Line length 120, 4-space indent, double quotes.
- Every module starts with a Korean docstring, then `from __future__ import annotations`. Imports go stdlib → third-party → relative (`from .. import llm`).
- Respect the dependency direction `app → rag → index → ingest → (llm, config)`; never import upward.
- Type-hint all public functions, using builtin generics (`list[int]`, `X | None`). API boundaries (request/response bodies, LLM JSON) are pydantic models; internal value objects are `@dataclass`.
- Private helpers get a `_` prefix. Module logger is `log = logging.getLogger(__name__)`; use `logging`, never `print`.
- Domain errors subclass `RuntimeError` (e.g. `LLMNotConfigured`). No bare `except Exception: pass`.
- Read settings only via `get_settings()`, never `os.getenv` outside `config.py`.
- Keep pure logic (`structure.py`, verification) free of network and DB access so it is unit-testable.
- Comments explain *why*, not *what*, and are written in Korean. Identifiers are English.

### Tests

- Unit tests never hit the network; Gemini is faked. Layers, isolation fixtures, and per-module checks are in `tests/CLAUDE.md`; quality evaluation is in `tests/eval/CLAUDE.md`.

### Development workflow (tests first)

- Develop test-first. For every new module or behavior change, write the tests first, run them and confirm they fail for the right reason, then implement until they pass. The tests, not the implementation, define correctness.
- Verify after every step, not only at the end: run the module's tests plus the full `pytest`, `ruff check`, and `ruff format --check` each time a module is finished.
- If a test fails, first decide whether the test or the implementation contradicts the spec (CLAUDE.md files). Fix the real defect; never bend the implementation or loosen a test just to make it pass.
- Delegate each module to a sub agent that reads the root `CLAUDE.md` plus the `CLAUDE.md` of the folders it touches (`tests/`, `theoremlens/<pkg>/`, `web/`). Independent modules may run in parallel, but sub agents must not edit shared files (`tests/conftest.py`, `pyproject.toml`) or other modules; report the need instead.
- After parallel work, check the contracts between modules yourself (for example, label normalization shared by `structure` and `retrieve`) and add a cross-module test for each contract.
- Write files containing backslashes or quotes (LaTeX regexes, Korean strings with `\n`) with the Write/Edit tools, not shell heredocs, which corrupt them.

### Git

- Commit messages follow Conventional Commits with an English prefix: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`. The body may be Korean.
- One logical change per commit. Work on `feat/<name>` branches and merge into `main`.
- Before committing, `ruff check`, `ruff format --check`, and `pytest` must pass.
- Do not push unless asked.

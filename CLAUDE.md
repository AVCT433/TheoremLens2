# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TheoremLens: a local web app that ingests math textbooks (PDF, text-layer or scanned) into a multi-book library and answers questions **only from the book content**, with citations, to minimize hallucination. LLM provider is **Google Gemini** (`google-genai` SDK); do not switch providers.

Status: backend and frontend complete with unit tests; `tests/eval/run_eval.py` written (only a sample question file; real per-book question files still needed). Approved design: `C:\Users\baoro\.claude\plans\bright-snacking-sky.md`.

## Commands

```bash
uv venv .venv && uv pip install -e ".[dev]"
.venv/Scripts/python -m pytest                              # all tests (add path::name for one)
.venv/Scripts/uvicorn theoremlens.app.main:app --reload     # run the app
```

## Invariants

Pipeline: **extract → structure → index → retrieve → answer → verify**.

- Extracted page text is verbatim; citations are checked against it, so never summarize or "fix" it.
- A chunk keeps a theorem with its proof, and display math (`$$…$$`) is never split.
- SQLite is the source of truth; Chroma holds only embeddings keyed by chunk id. Deleting a book clears both.
- No retrieval evidence → answer "not in the book" without calling the LLM.
- Every claim cites `[S#]`; citation text shown to the user comes from the DB, never from model output.
- Verification is rules first, LLM second; the LLM cannot revive a claim the rules rejected.
- Page numbers are 1-based PDF page indices. All Gemini calls go through `theoremlens/llm.py`. Dependency direction `app → rag → index → ingest → (llm, config)`. User-facing text and prompts are Korean.

## TODO 관리

- 남은 작업은 [TODO.md](TODO.md)에 저장하고 세션이 바뀌어도 이어간다.
- 작업을 시작하기 전에 `TODO.md`를 읽고, 새로 생긴 일은 바로 추가한다.
- 항목을 끝내면 같은 커밋에서 체크하고 Done으로 옮긴다.
- 코드에 TODO 주석을 남기지 않고 `TODO.md`에만 적는다.

## Where to look (read only when relevant)

- [TODO.md](TODO.md): 남은 작업 목록.
- [architecture.md](architecture.md): schemas, thresholds, status flow, API shapes. Update it in the same commit as any architecture change.
- [conventions.md](conventions.md): config/env, code style (ruff), naming, logging, errors, test rules. Read before writing code.
- [workflow.md](workflow.md): test-first workflow, sub-agent delegation, Git/commit rules. Read before starting a feature or committing.
- Folder specs: `tests/CLAUDE.md`, `tests/eval/CLAUDE.md`, `theoremlens/CLAUDE.md`, `theoremlens/<pkg>/CLAUDE.md`, `web/CLAUDE.md`.

Before committing: `ruff check`, `ruff format --check`, and `pytest` must pass; do not push unless asked.

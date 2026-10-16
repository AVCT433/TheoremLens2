# tests/ (pytest)

Run all with `.venv/Scripts/python -m pytest`, or one test with `-k name` or `file.py::test_name`.

## Layers

| Layer | Where | Network | Runs by default |
|---|---|---|---|
| Unit | `tests/test_<module>.py` | None; Gemini and MinerU are faked | Yes |
| Integration | tests marked `@pytest.mark.integration` | Real Gemini, the real local embedding model (first run downloads it), or real MinerU (slow) | No. `pytest -m integration`; Gemini ones are skipped when `GEMINI_API_KEY` is unset |
| Eval | `tests/eval/` | Real Gemini + real book | No. Separate runner, see `eval/CLAUDE.md` |

`pyproject.toml` registers the `integration` marker and sets `addopts = "-m 'not integration' --strict-markers"`.

## Isolation (`conftest.py`)

Settings, the SQLite connection, and the Chroma collection are process-wide singletons, so an autouse fixture must, before and after every test:
- set `THEOREMLENS_DATA_DIR` to a `tmp_path` (never touch the real `data/`);
- call `get_settings.cache_clear()`, `db.close()`, and `vectors._collection.cache_clear()`. Windows cannot delete SQLite/Chroma files that are still open, so teardown must release them;
- set `GEMINI_API_KEY=""` so an accidental real call fails immediately with `LLMNotConfigured` instead of reaching the network. The same fixture replaces `embedder._load` with a function that raises `EmbedderNotAvailable`, so no test downloads or loads the real embedding model. Integration tests opt out of both.

Tests must pass in any order and when run twice in a row.

## Fakes

- Patch only module attributes of `theoremlens.llm` and `theoremlens.embedder` (other modules call `llm.generate_text`, `embedder.embed`, and so on). Never patch or construct `genai.Client`.
- `fake_embed`: deterministic bag-of-words vectors, so similar text gets high cosine similarity and search ordering is testable.
- `fake_llm`: scripted responses for `generate_text` / `generate_json`, plus a `calls` list. Use it to assert that a path does **not** call the LLM (`assert not fake_llm.calls`).
- `fake_mineru` (autouse): replaces `extract._run_mineru` so no test runs the real MinerU. It writes a middle_json block from the PDF's text layer (`scan_text` for image-only pages) and records `requested` page lists. `real_run_mineru` gives the original function for command-assembly tests with `subprocess.run` patched.
- Patch `time.sleep` when testing retries.

## Fixtures

- Sample page Markdown lives in `tests/fixtures/*.md` (English and Korean: definitions, theorems, proofs, examples, exercises, `$$…$$` blocks, cross-references).
- Do not commit PDFs. Generate them inside the test with PyMuPDF: a text-layer PDF, a scanned PDF (image only), and a mixed one.

## What to test per module

| Module | Verify |
|---|---|
| `db` | FTS stays in sync with `chunks` (replacing chunks removes old hits); `fts_search` handles Korean prefixes and special characters without a syntax error, drops 1-char tokens but keeps digits; `chunks_by_label` across books; `delete_book` cascades to `pages` and `chunks`; messages come back oldest-first |
| `vectors` | search order; single- and multi-book filter; re-indexing replaces old vectors; `delete_book` leaves other books intact; one collection per embedding model; `has_book`; `drop_legacy` |
| `llm` | retry and backoff; `LLMNotConfigured` is never retried |
| `embedder` | order kept, normalized, configured batch size and per-task prefixes; empty input does not load the model; device resolution (fake `SentenceTransformer`) |
| `extract` | `blocks_to_markdown` per block type; `page_ranges`; `_run_mineru` command assembly and failure messages (subprocess faked); `page_idx` → 1-based page numbers; a missing page raises; `skip`; no LLM call |
| `structure` | invariants below, plus label normalization, `refs`, chapter/section, page ranges |
| `retrieve` | RRF ranking; explicit labels looked up directly; one-hop cited lemmas; best similarity below `min_similarity` returns "not in the book" with **no LLM call** |
| `verify` | whitespace/LaTeX-only differences in a quote pass; fabricated quotes are rejected; unsupported claims are dropped or flagged |
| `pipeline`, `app` | end to end with a generated PDF and the fakes; re-run skips already extracted pages; `reindex_missing` re-embeds ready books without vectors and makes no Gemini call; deleting a book empties both SQLite and Chroma; use FastAPI `TestClient` |

`structure` invariants (assert these on every fixture):
- every chunk has an even number of `$$`, so display math is never split;
- a theorem and its proof share one chunk;
- concatenated chunk text loses no source content, because citation checking matches quotes against the original text;
- `ord` is contiguous from 0, and `page_start <= page_end`.

## Conventions

- Names describe behavior: `test_fts_search_ignores_single_char_tokens`.
- No dependence on wall-clock time or randomness.
- Page numbers in tests are 1-based PDF indices, as everywhere else.
- Anything that calls Gemini is marked `integration`; keep it to a few smoke tests (schema of `generate_json`; one real-model `embedder.embed`).

## Order of work

1. `conftest.py` and the pytest config, proved on `test_config.py`.
2. Tests for the existing modules: `db`, `vectors`, `llm`, `extract`.
3. `structure`: write the fixtures and `test_structure.py` first, then implement.
4. `retrieve`, `verify`, `pipeline`, `app`, each tests-first.

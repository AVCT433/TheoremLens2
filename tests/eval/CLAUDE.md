# tests/eval/ (quality evaluation set)

Measures answer quality on real books. Not run by default pytest; it calls Gemini and needs a real ingested book. The runner is `run_eval.py`; its name does not start with `test_`, so pytest never collects it.

Each book gets a question file (about 20–30 questions), hand-written, one entry per question:
- `question`
- `expected_labels` and/or `expected_pages` (ground truth; pages are **1-based PDF indices**, not printed page numbers)
- `expect_refusal: true` for questions the book does not answer

Metrics to report:
- retrieval hit rate: expected chunk in the top `Settings.top_k`;
- citation accuracy: each cited `quote` fuzzy-matches its source chunk;
- refusal rate on out-of-book questions, and the false-refusal rate on in-book questions (a bot that refuses everything must not look good).

Rules:
- Keep ground-truth files hand-written and reviewed. Do not generate them with the model under test.
- Out-of-book questions should be plausible, not obviously off-topic (for example a real theorem from a different textbook). Include both Korean and English questions.
- Write results to `data/` (gitignored) with the model names and date, so runs before and after a model or prompt change can be compared.
- Store the question files in this folder. Do not store the books themselves; they stay under `data/`, which is gitignored.

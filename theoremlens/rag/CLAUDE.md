# theoremlens/rag/ (question answering)

Answers a question using only retrieved book excerpts. This is where hallucination is controlled, so the rules below are strict.

- `retrieve.py`: hybrid search. BM25 (`db.fts_search`) and vectors (`vectors.search`) fused with RRF, top ~8. Explicit labels in the question ("Theorem 3.2") are fetched via `db.chunks_by_label`. Cited lemmas and definitions are added one hop deep. If the top similarity is below `Settings.min_similarity` (tuned per embedding model; re-tune with the eval runner after changing it), return "not found" without calling the LLM.
- `prompts.py`: system prompts, kept as constants so they can be versioned and tested.
- `answer.py`: calls `llm.generate_json` at temperature 0 with schema `{answer_markdown, claims:[{text, source_ids, quote}], not_found}`. Each claim cites `[S#]` excerpts from the retrieved set only.
- `verify.py`: checks each claim's `quote` against its source chunk (rapidfuzz, after whitespace and LaTeX normalization), then runs a fast-model support check. Unsupported claims are dropped or marked ⚠.

Rules:
- The model may only use excerpts passed in. Anything outside the book goes into a separate, labeled "보충(책 외부)" section.
- Never return an answer that has no citation without marking it as outside-book content.
- Keep the verification step deterministic where possible. The LLM support check is a second opinion, not the only check.

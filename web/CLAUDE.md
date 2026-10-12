# web/ (browser UI)

Static frontend served by the FastAPI app. No build step: plain HTML, CSS, and JavaScript, loaded through the CDN allowlist only if needed.

- `index.html`: layout with a library panel (upload, progress, book checkboxes) and a chat panel.
- `app.js`: calls the API, renders answers with marked, renders math with KaTeX, and opens the citation side panel (source excerpt plus the page image).
- `lib.js`: DOM-free pure functions (`extractMath`, `splitCitations`, `claimBadge`, `pageRange`); global `TLLib` in the browser, `module.exports` in node (tested via `tests/test_web.py`).
- `style.css`: styling.

Rules:
- Render Markdown and LaTeX on the client only from text the server returns. Do not evaluate model output as code.
- Citation markers `[S#]` must be clickable and tied to the source list in the response.
- Pin any CDN library to an exact version. Do not use a floating tag.
- Pinned CDN set: marked 12.0.2, dompurify 3.1.6, katex 0.16.11 (jsdelivr). Update the version in `index.html` only; `tests/test_web.py` checks the pin.
- Pipeline: pull math out into placeholders, `marked.parse`, `DOMPurify.sanitize` (the only `innerHTML` assignment), then in text nodes swap placeholders for `katex.render` output and `[S#]` for buttons. Never add another `innerHTML`; use `textContent`.

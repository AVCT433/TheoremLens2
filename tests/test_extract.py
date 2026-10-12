"""ingest/extract 테스트: 쪽 분류, skip, 1-based 번호, Marker 대체, 코드 펜스 제거."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from theoremlens.config import get_settings
from theoremlens.ingest import extract as ex

LONG = "Theorem 1.1 every compact subset of a metric space is closed and bounded."


def _make_pdf(path: Path, kinds: list[str]) -> Path:
    """kinds: 'text' | 'scan' | 'blank' | 'short'. 반환한 PDF는 테스트 안에서만 쓰는 임시 파일."""
    doc = pymupdf.open()
    for i, kind in enumerate(kinds, start=1):
        page = doc.new_page()
        if kind == "text":
            page.insert_text((72, 100), f"{LONG} page {i}")
        elif kind == "short":
            page.insert_text((72, 100), "x" * 5)
        elif kind == "scan":
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 100), False)
            pix.clear_with(200)
            page.insert_image(pymupdf.Rect(72, 72, 272, 172), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def test_page_count(tmp_path):
    assert ex.page_count(_make_pdf(tmp_path / "a.pdf", ["text", "scan", "blank"])) == 3


def test_classify_pages_mixed_uses_one_based_numbers(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", ["text", "scan", "text", "scan"])
    assert ex.classify_pages(pdf) == ([1, 3], [2, 4])


def test_classify_blank_page_without_images_counts_as_text(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", ["blank", "scan"])
    assert ex.classify_pages(pdf) == ([1], [2])


def _pdf_with_n_chars(path: Path, n: int, with_image: bool) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    if n:
        page.insert_text((72, 100), "a" * n)
    if with_image:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
        page.insert_image(pymupdf.Rect(72, 200, 122, 250), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def test_classify_pages_threshold_boundary(tmp_path):
    t = get_settings().scanned_char_threshold
    at = _pdf_with_n_chars(tmp_path / "at.pdf", t, with_image=True)
    below = _pdf_with_n_chars(tmp_path / "below.pdf", t - 1, with_image=True)
    assert ex.classify_pages(at) == ([1], [])  # 정확히 임계값이면 텍스트 쪽
    assert ex.classify_pages(below) == ([], [1])


def test_classify_short_text_without_image_is_not_scanned(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", ["short"])
    assert ex.classify_pages(pdf) == ([1], [])


def test_extract_text_pdf_uses_pymupdf_without_llm(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(ex, "_marker_available", lambda: False)
    pdf = _make_pdf(tmp_path / "a.pdf", ["text", "text"])
    got = ex.extract(pdf)
    assert [(p.page_no, p.method) for p in got] == [(1, "pymupdf"), (2, "pymupdf")]
    assert "compact subset" in got[0].markdown
    assert got[1].markdown.endswith("page 2")
    assert not fake_llm.calls


def test_extract_mixed_pdf_routes_scans_to_gemini_and_sorts(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(ex, "_marker_available", lambda: False)
    fake_llm.default_text = "전사된 쪽"
    pdf = _make_pdf(tmp_path / "a.pdf", ["scan", "text", "scan"])
    got = ex.extract(pdf)
    assert [(p.page_no, p.method) for p in got] == [(1, "gemini"), (2, "pymupdf"), (3, "gemini")]
    assert got[0].markdown == "전사된 쪽"
    assert len(fake_llm.calls) == 2
    assert all(c["kind"] == "text" and c["model"] == get_settings().fast_model for c in fake_llm.calls)


def test_extract_skip_excludes_cached_pages(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(ex, "_marker_available", lambda: False)
    pdf = _make_pdf(tmp_path / "a.pdf", ["text", "scan", "text"])
    got = ex.extract(pdf, skip={1, 2})
    assert [p.page_no for p in got] == [3]
    assert not fake_llm.calls


def test_extract_skip_everything_returns_empty(tmp_path, fake_llm):
    pdf = _make_pdf(tmp_path / "a.pdf", ["text", "scan"])
    assert ex.extract(pdf, skip={1, 2}) == []
    assert not fake_llm.calls


def test_extract_calls_on_page_once_per_page(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setattr(ex, "_marker_available", lambda: False)
    pdf = _make_pdf(tmp_path / "a.pdf", ["text", "scan", "text"])
    n = []
    ex.extract(pdf, on_page=lambda: n.append(1))
    assert len(n) == 3


def test_marker_failure_falls_back_to_pymupdf(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("THEOREMLENS_TEXT_EXTRACTOR", "marker")
    get_settings.cache_clear()

    def boom(pdf_path, pages):
        raise RuntimeError("marker crashed")

    monkeypatch.setattr(ex, "extract_marker", boom)
    pdf = _make_pdf(tmp_path / "a.pdf", ["text"])
    got = ex.extract(pdf)
    assert [(p.page_no, p.method) for p in got] == [(1, "pymupdf")]
    assert "compact subset" in got[0].markdown


def test_marker_success_is_used_when_available(tmp_path, monkeypatch):
    monkeypatch.setenv("THEOREMLENS_TEXT_EXTRACTOR", "marker")
    get_settings.cache_clear()
    monkeypatch.setattr(ex, "extract_marker", lambda p, pages: [ex.PageText(n, "M", "marker") for n in pages])
    got = ex.extract(_make_pdf(tmp_path / "a.pdf", ["text", "text"]))
    assert [p.method for p in got] == ["marker", "marker"]


def test_auto_picks_marker_only_when_installed(monkeypatch):
    monkeypatch.setattr(ex, "_marker_available", lambda: True)
    assert ex._choose_text_method() == "marker"
    monkeypatch.setattr(ex, "_marker_available", lambda: False)
    assert ex._choose_text_method() == "pymupdf"


def test_gemini_text_extractor_sends_text_pages_to_vision(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("THEOREMLENS_TEXT_EXTRACTOR", "gemini")
    get_settings.cache_clear()
    fake_llm.default_text = "vision"
    got = ex.extract(_make_pdf(tmp_path / "a.pdf", ["text"]))
    assert [(p.method, p.markdown) for p in got] == [("gemini", "vision")]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("```markdown\n**Theorem 1.** $x$\n```", "**Theorem 1.** $x$"),
        ("```\nplain\n```", "plain"),
        ("  no fence  ", "no fence"),
        ("before ``` inside", "before ``` inside"),
    ],
)
def test_transcription_strips_code_fences(tmp_path, fake_llm, raw, expected):
    fake_llm.default_text = raw
    pdf = _make_pdf(tmp_path / "a.pdf", ["scan"])
    assert ex.transcribe_page_gemini(pdf, 1).markdown == expected


def test_transcription_uses_one_based_page_in_prompt_and_png(tmp_path, fake_llm):
    fake_llm.default_text = "t"
    pdf = _make_pdf(tmp_path / "a.pdf", ["text", "scan"])
    r = ex.transcribe_page_gemini(pdf, 2)
    assert r.page_no == 2
    assert "2쪽" in fake_llm.calls[0]["contents"][1]
    assert ex.render_page_png(pdf, 2)[:4] == b"\x89PNG"


def test_unconfigured_key_fails_fast_on_scans(tmp_path):
    from theoremlens.llm import LLMNotConfigured

    pdf = _make_pdf(tmp_path / "a.pdf", ["scan"])
    with pytest.raises(LLMNotConfigured):
        ex.extract(pdf)

"""ingest/extract 테스트: 블록→Markdown 변환, MinerU 호출 조립, 1-based 쪽 번호, skip."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pymupdf
import pytest

from theoremlens.ingest import extract as ex


def _txt(s: str) -> dict:
    return {"type": "text", "content": s}


def _eq(s: str) -> dict:
    return {"type": "equation_inline", "content": s}


def _block(type_: str, content, **extra) -> dict:
    return {"type": type_, "index": 0, "bbox": [0, 0, 1, 1], "content": content, **extra}


def _write_middle(path: Path, pages: dict[int, list[dict]]) -> None:
    """pages: 0-based page_idx -> blocks."""
    path.write_text(
        json.dumps({"pages": [{"page_idx": i, "blocks": b} for i, b in pages.items()]}),
        encoding="utf-8",
    )


def _pdf(path: Path, n: int) -> Path:
    doc = pymupdf.open()
    for i in range(1, n + 1):
        doc.new_page().insert_text((72, 100), f"page {i}")
    doc.save(path)
    doc.close()
    return path


# ---------- page_count / render ----------


def test_page_count(tmp_path):
    assert ex.page_count(_pdf(tmp_path / "a.pdf", 3)) == 3


def test_render_page_png_returns_png_bytes(tmp_path):
    assert ex.render_page_png(_pdf(tmp_path / "a.pdf", 2), 2).startswith(b"\x89PNG")


# ---------- page_ranges ----------


@pytest.mark.parametrize(
    "pages, expected",
    [
        ([5], "5"),
        ([1, 2, 3], "1-3"),
        ([1, 2, 3, 7, 9, 10], "1-3,7,9-10"),
        ([10, 3, 4, 3], "3-4,10"),  # 순서·중복에 무관
    ],
)
def test_page_ranges(pages, expected):
    assert ex.page_ranges(pages) == expected


# ---------- blocks_to_markdown ----------


def test_titles_use_level_and_text_keeps_inline_math():
    md = ex.blocks_to_markdown(
        [
            _block("doc_title", [_txt("RELATIONS AND FUNCTIONS")], level=1),
            _block("paragraph_title", [_txt("ORDERED PAIRS")], level=2),
            _block("text", [_txt("Let "), _eq("B"), _txt(" be a set.")]),
        ]
    )
    assert md == "# RELATIONS AND FUNCTIONS\n\n## ORDERED PAIRS\n\nLet $B$ be a set."


def test_display_equation_becomes_dollar_block_unsplit():
    md = ex.blocks_to_markdown([_block("equation", "\\langle x, y \\rangle = \\langle u, v \\rangle ,")])
    assert md == "$$\n\\langle x, y \\rangle = \\langle u, v \\rangle ,\n$$"
    assert md.count("$$") == 2


def test_header_and_page_number_are_dropped():
    md = ex.blocks_to_markdown(
        [
            _block("page_number", [_txt("34")]),
            _block("header", [_txt("2. Axioms and Operations")]),
            _block("text", [_txt("body")]),
        ]
    )
    assert md == "body"


def test_list_items_each_on_own_line():
    md = ex.blocks_to_markdown(
        [
            _block(
                "list", [_block("text", [_txt("(b) 0 is ordinal.")]), _block("text", [_txt("(c) If "), _eq("\\alpha")])]
            )
        ]
    )
    assert md == "(b) 0 is ordinal.\n(c) If $\\alpha$"


def test_footnote_is_kept_as_text():
    md = ex.blocks_to_markdown([_block("page_footnote", [_eq("^{1}"), _txt("  Other chapters.")])])
    assert md == "$^{1}$  Other chapters."


def test_image_and_chart_keep_only_caption_as_figure_marker():
    md = ex.blocks_to_markdown(
        [
            _block(
                "image", [_block("image_body", "A\nB"), _block("image_caption", [_txt("(a)")])], sub_type="text_image"
            ),
            _block(
                "chart",
                [
                    _block("chart_body", "| a |\n| - |"),
                    _block("chart_caption", [_txt("Fig. 20. A line in "), _eq("\\omega")]),
                ],
            ),
        ]
    )
    assert md == "[그림: (a)]\n\n[그림: Fig. 20. A line in $\\omega$]"


def test_image_without_caption_is_bare_marker():
    assert ex.blocks_to_markdown([_block("image", [_block("image_body", "x")])]) == "[그림]"


def test_table_html_keeps_cells_and_converts_eq_tags_to_math():
    html = "<table><tr><td><eq>\\in</eq></td><td>1</td></tr></table>"
    md = ex.blocks_to_markdown([_block("table", [_block("table_body", html)])])
    assert md == "<table><tr><td>$\\in$</td><td>1</td></tr></table>"


def test_unknown_block_type_text_is_kept_and_logged(caplog):
    with caplog.at_level("WARNING"):
        md = ex.blocks_to_markdown([_block("mystery", [_txt("keep me")])])
    assert md == "keep me"
    assert "mystery" in caplog.text


def test_blocks_without_text_are_skipped():
    assert ex.blocks_to_markdown([_block("text", [])]) == ""


# ---------- extract_mineru ----------


def test_extract_mineru_maps_zero_based_page_idx_to_one_based(tmp_path, monkeypatch):
    seen = {}

    def run(pdf_path, pages, out_path):
        seen["pages"] = pages
        _write_middle(out_path, {2: [_block("text", [_txt("third")])], 4: [_block("text", [_txt("fifth")])]})

    monkeypatch.setattr(ex, "_run_mineru", run)
    got = ex.extract_mineru(tmp_path / "x.pdf", [3, 5])
    assert seen["pages"] == [3, 5]
    assert [(p.page_no, p.markdown, p.method) for p in got] == [(3, "third", "mineru"), (5, "fifth", "mineru")]


def test_extract_mineru_ignores_pages_that_were_not_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ex,
        "_run_mineru",
        lambda pdf, pages, out: _write_middle(
            out, {0: [_block("text", [_txt("a")])], 1: [_block("text", [_txt("b")])]}
        ),
    )
    assert [p.page_no for p in ex.extract_mineru(tmp_path / "x.pdf", [2])] == [2]


def test_extract_mineru_missing_page_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ex, "_run_mineru", lambda pdf, pages, out: _write_middle(out, {0: [_block("text", [_txt("a")])]})
    )
    with pytest.raises(RuntimeError, match="2"):
        ex.extract_mineru(tmp_path / "x.pdf", [1, 2])


def test_extract_mineru_page_with_no_blocks_is_empty_not_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_run_mineru", lambda pdf, pages, out: _write_middle(out, {0: []}))
    got = ex.extract_mineru(tmp_path / "x.pdf", [1])
    assert got[0].markdown == ""


def test_extract_mineru_bad_json_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_run_mineru", lambda pdf, pages, out: out.write_text("not json", encoding="utf-8"))
    with pytest.raises(RuntimeError):
        ex.extract_mineru(tmp_path / "x.pdf", [1])


# ---------- _run_mineru (subprocess 조립) ----------


def test_run_mineru_builds_command_from_settings(tmp_path, monkeypatch, real_run_mineru):
    monkeypatch.setenv("THEOREMLENS_MINERU_TIER", "standard")
    monkeypatch.setenv("THEOREMLENS_MINERU_OCR_MODE", "ocr")
    from theoremlens.config import get_settings

    get_settings.cache_clear()
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    real_run_mineru(tmp_path / "b.pdf", [1, 2, 3, 9], tmp_path / "o.json")
    cmd, kw = calls[0]
    assert Path(cmd[0]).stem == "mineru-kit" or cmd[0] == "mineru-kit"
    assert cmd[1:3] == ["parse", str(tmp_path / "b.pdf")]
    flags = dict(zip(cmd[3::2], cmd[4::2], strict=False))
    assert flags == {
        "-o": str(tmp_path / "o.json"),
        "--format": "middle_json",
        "--tier": "standard",
        "--ocr-mode": "ocr",
        "--pages": "1-3,9",
    }
    assert kw.get("check") is not True  # 실패 메시지를 직접 만들기 위해 returncode를 본다


def test_run_mineru_failure_raises_with_stderr_tail(tmp_path, monkeypatch, real_run_mineru):
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 2, b"", ("x" * 5000 + "BOOM").encode())
    )
    with pytest.raises(RuntimeError) as e:
        real_run_mineru(tmp_path / "b.pdf", [1], tmp_path / "o.json")
    assert "BOOM" in str(e.value)
    assert len(str(e.value)) < 3000


def test_run_mineru_missing_executable_raises_runtime_error(tmp_path, monkeypatch, real_run_mineru):
    def boom(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="mineru"):
        real_run_mineru(tmp_path / "b.pdf", [1], tmp_path / "o.json")


# ---------- extract ----------


def test_extract_returns_sorted_one_based_pages(tmp_path, fake_mineru):
    pdf = _pdf(tmp_path / "a.pdf", 3)
    got = ex.extract(pdf)
    assert [p.page_no for p in got] == [1, 2, 3]
    assert all(p.method == "mineru" for p in got)
    assert "page 2" in got[1].markdown


def test_extract_skip_excludes_cached_pages(tmp_path, fake_mineru):
    pdf = _pdf(tmp_path / "a.pdf", 4)
    got = ex.extract(pdf, skip={1, 3})
    assert [p.page_no for p in got] == [2, 4]
    assert fake_mineru.requested == [[2, 4]]


def test_extract_skip_everything_does_not_call_mineru(tmp_path, fake_mineru):
    pdf = _pdf(tmp_path / "a.pdf", 2)
    assert ex.extract(pdf, skip={1, 2}) == []
    assert fake_mineru.requested == []


def test_extract_calls_on_page_once_per_page(tmp_path, fake_mineru):
    pdf = _pdf(tmp_path / "a.pdf", 3)
    n = []
    ex.extract(pdf, on_page=lambda: n.append(1))
    assert len(n) == 3


def test_extract_failure_propagates(tmp_path, monkeypatch, fake_mineru):
    def boom(pdf, pages, out):
        raise RuntimeError("mineru 실패")

    monkeypatch.setattr(ex, "_run_mineru", boom)
    with pytest.raises(RuntimeError, match="mineru"):
        ex.extract(_pdf(tmp_path / "a.pdf", 1))


def test_extract_makes_no_llm_call(tmp_path, fake_mineru, fake_llm):
    ex.extract(_pdf(tmp_path / "a.pdf", 2))
    assert fake_llm.calls == []


# ---------- integration (실제 MinerU, 모델 로딩에 수 분) ----------


@pytest.mark.integration
def test_real_mineru_extracts_one_page(tmp_path):
    pdf = tmp_path / "one.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Theorem 1.1. Every compact set is bounded.", fontsize=14)
    doc.save(pdf)
    doc.close()
    [page] = ex.extract_mineru(pdf, [1])
    assert page.page_no == 1 and "compact" in page.markdown.lower()

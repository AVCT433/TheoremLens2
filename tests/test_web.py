"""web/ 정적 프런트엔드 검사: 서빙, CDN 버전 고정, API 경로, XSS 방어 패턴, 순수 함수(node)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from theoremlens.app.main import WEB_DIR, create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(WEB_DIR))


def _read(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def test_root_serves_index_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
    assert 'lang="ko"' in r.text


def test_index_references_local_assets_and_they_are_served(client: TestClient) -> None:
    html = _read("index.html")
    for name in ("app.js", "style.css", "lib.js"):
        assert re.search(rf'(src|href)="/?{re.escape(name)}"', html), name
        r = client.get(f"/{name}")
        assert r.status_code == 200 and r.text.strip(), name


def test_lib_loads_before_app_js() -> None:
    html = _read("index.html")
    assert html.index("lib.js") < html.index("app.js")


def _cdn_urls() -> list[str]:
    return re.findall(r"""(?:src|href)=["'](https?://[^"']+)["']""", _read("index.html"))


def test_cdn_libraries_are_pinned_to_exact_versions() -> None:
    urls = _cdn_urls()
    joined = " ".join(urls)
    for lib in ("marked", "katex", "dompurify"):
        assert lib in joined.lower(), f"{lib} 누락"
    for url in urls:
        assert re.search(r"@\d+\.\d+\.\d+/", url), f"버전 고정 아님: {url}"
        assert "@latest" not in url


def test_cdn_hosts_are_allowlisted() -> None:
    for url in _cdn_urls():
        assert re.match(r"https://(cdn\.jsdelivr\.net/npm/|cdnjs\.cloudflare\.com/|unpkg\.com/)", url), url


def test_app_js_uses_api_routes() -> None:
    js = _read("app.js")
    for needle in ("/books", "/chat", "/status", "/page/", "conversation_id", "book_ids", "DELETE"):
        assert needle in js, needle


def test_app_js_has_no_dangerous_sinks() -> None:
    js = _read("app.js") + _read("lib.js")
    assert not re.search(r"\beval\s*\(", js)
    assert not re.search(r"new\s+Function\b", js)
    assert "document.write" not in js
    assert "insertAdjacentHTML" not in js
    assert "outerHTML" not in js
    for line in js.splitlines():
        if "innerHTML" in line:
            # 정제기를 거친 값이거나 비우기만 허용한다
            assert "DOMPurify.sanitize" in line or re.search(r'innerHTML\s*=\s*""', line), line


def test_app_js_sanitizes_rendered_markdown_and_renders_math_safely() -> None:
    js = _read("app.js")
    assert "DOMPurify.sanitize(" in js
    assert "marked.parse(" in js
    assert "katex.render(" in js
    assert "throwOnError" in js


def test_citation_markers_become_clickable_elements() -> None:
    js = _read("app.js")
    assert "splitCitations" in js
    assert 'el("button", "cite"' in js
    assert 'addEventListener("click"' in js
    assert "/page/" in js and "img" in js


def test_app_js_covers_ui_states() -> None:
    js = _read("app.js")
    for needle in ("not_found", "outside_book", "claims", "verification", "status", "progress", "setTimeout"):
        assert needle in js, needle
    assert "보충(책 외부)" in js or "보충(책 외부)" in _read("lib.js")


# ---- 순수 함수(node) ----

_NODE = shutil.which("node")


def _node(expr: str):
    code = f"const L=require({json.dumps(str(WEB_DIR / 'lib.js'))});process.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([_NODE, "-e", code], capture_output=True, check=True, timeout=30)
    return json.loads(out.stdout.decode("utf-8"))


@pytest.mark.skipif(_NODE is None, reason="node 없음")
def test_split_citations_finds_markers_in_order() -> None:
    parts = _node('L.splitCitations("정리 [S1] 와 [S2][S10] 끝")')
    cites = [p["sid"] for p in parts if p["type"] == "cite"]
    assert cites == ["S1", "S2", "S10"]
    assert "".join(p.get("value", "") for p in parts if p["type"] == "text") == "정리  와  끝"


@pytest.mark.skipif(_NODE is None, reason="node 없음")
def test_split_citations_handles_multi_id_brackets_and_plain_text() -> None:
    parts = _node('L.splitCitations("a [S1, S2] b [x] c")')
    assert [p["sid"] for p in parts if p["type"] == "cite"] == ["S1", "S2"]
    assert _node('L.splitCitations("none")') == [{"type": "text", "value": "none"}]


@pytest.mark.skipif(_NODE is None, reason="node 없음")
def test_extract_math_protects_math_from_markdown() -> None:
    r = _node(r'L.extractMath("값 $a_1 + b_2$ 이고\n\n$$\\int_0^1 x\\,dx$$\n\n통화 $5 와 $6")')
    kinds = [m["display"] for m in r["items"]]
    assert kinds == [False, True]
    assert "a_1" not in r["text"] and "int_0" not in r["text"]
    assert "$5" in r["text"]  # 통화 표기는 수식이 아니다


@pytest.mark.skipif(_NODE is None, reason="node 없음")
def test_claim_badge_marks_only_verified_as_ok() -> None:
    assert _node('L.claimBadge("verified")')["ok"] is True
    for s in ("unchecked", "rejected", "unsupported", "weird"):
        assert _node(f'L.claimBadge("{s}")')["ok"] is False

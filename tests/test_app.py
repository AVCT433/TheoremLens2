"""app/main 테스트: FastAPI TestClient로 라우트 검증(Gemini는 가짜)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import bow_vector
from theoremlens import llm
from theoremlens.app import main
from theoremlens.config import get_settings
from theoremlens.index import db, vectors
from theoremlens.llm import LLMNotConfigured

ANSWER = {
    "answer_markdown": "컴팩트 집합은 유계이다 [S1].",
    "claims": [{"text": "컴팩트 집합은 유계이다", "source_ids": ["S1"], "quote": "every compact set is bounded"}],
    "not_found": False,
}
QUESTION = "Theorem 1.1 every compact set is bounded"


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        # 기동 시 복구 스레드가 설정 싱글턴을 건드리므로 끝날 때까지 기다려 테스트의 설정 변경과 겹치지 않게 한다
        main.wait_for_recovery(c.app)
        yield c


def _upload(client, pdf, name="book.pdf", content=None):
    data = content if content is not None else pdf.read_bytes()
    return client.post("/books", files={"file": (name, data, "application/pdf")})


@pytest.fixture
def ready_book(client, make_pdf, fake_llm, fake_embed):
    r = _upload(client, make_pdf(["text", "text"]))
    assert r.status_code == 200, r.text
    bid = r.json()["id"]
    # TestClient는 응답 후 백그라운드 작업까지 끝내고 돌아온다
    assert client.get(f"/books/{bid}/status").json()["status"] == "ready"
    return bid


def test_upload_runs_ingest_in_background_and_lists_book(client, make_pdf, fake_llm, fake_embed):
    body = _upload(client, make_pdf(["text"]), name="Analysis.pdf").json()
    assert body["title"] == "Analysis" and body["status"] == "queued"
    books = client.get("/books").json()
    assert [b["id"] for b in books] == [body["id"]]
    assert books[0]["status"] == "ready" and books[0]["num_chunks"] > 0


def test_upload_does_not_run_ingest_inside_request_handler(client, make_pdf, monkeypatch):
    calls = []
    monkeypatch.setattr(main.pipeline, "ingest_book", lambda *a: calls.append(a))
    r = _upload(client, make_pdf(["text"]))
    assert r.json()["status"] == "queued" and len(calls) == 1


@pytest.mark.parametrize("name", ["notes.txt", "evil.pdf.exe", "noext"])
def test_upload_rejects_bad_extension(client, name):
    assert _upload(client, None, name=name, content=b"%PDF-1.4 x").status_code == 400


def test_upload_rejects_non_pdf_content(client):
    assert _upload(client, None, content=b"hello not a pdf").status_code == 400
    assert client.get("/books").json() == []


def test_upload_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setenv("THEOREMLENS_MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    r = _upload(client, None, content=b"%PDF-1.4" + b"0" * (1024 * 1024 + 10))
    assert r.status_code == 413
    assert client.get("/books").json() == []


def test_upload_with_traversal_filename_stays_inside_book_dir(client, make_pdf, fake_llm, fake_embed, tmp_path):
    r = _upload(client, make_pdf(["text"]), name="../../../evil.pdf")
    assert r.status_code == 200
    assert r.json()["title"] == "evil"
    assert not (tmp_path / "evil.pdf").exists()
    assert (get_settings().book_dir(r.json()["id"]) / "book.pdf").exists()


def test_corrupt_pdf_ends_in_error_status(client, fake_llm, fake_embed):
    r = _upload(client, None, content=b"%PDF-1.4 garbage")
    st = client.get(f"/books/{r.json()['id']}/status").json()
    assert st["status"] == "error" and st["message"]


def test_status_unknown_book_is_404(client):
    assert client.get("/books/nope/status").status_code == 404


def test_status_shape(client, ready_book):
    st = client.get(f"/books/{ready_book}/status").json()
    assert set(st) >= {"id", "status", "progress", "message", "num_pages"}
    assert st["progress"] == 1.0 and st["num_pages"] == 2


def test_delete_book_clears_sqlite_chroma_and_files(client, ready_book):
    assert vectors._collection().count() > 0
    assert client.delete(f"/books/{ready_book}").status_code == 200
    assert db.get_book(ready_book) is None
    assert db.get_pages(ready_book) == []
    assert db.chunks_by_label(["theorem 1.1"], [ready_book]) == []
    assert vectors._collection().count() == 0
    assert not get_settings().book_dir(ready_book).exists()
    assert client.get("/books").json() == []


def test_delete_unknown_book_is_404(client):
    assert client.delete("/books/nope").status_code == 404


def test_chat_returns_answer_with_db_citation_and_saves_messages(client, ready_book, fake_llm):
    fake_llm.default_json = json.dumps(ANSWER)
    r = client.post("/chat", json={"book_ids": [ready_book], "question": QUESTION})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["conversation_id"] and not body["not_found"]
    assert body["citations"] and "every compact set is bounded" in body["citations"][0]["text"]
    msgs = db.get_messages(body["conversation_id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == QUESTION
    assert msgs[1]["meta"]["citations"][0]["sid"] == "S1"


def test_chat_citation_text_ignores_model_supplied_text(client, ready_book, fake_llm):
    evil = dict(ANSWER, claims=[dict(ANSWER["claims"][0], text="조작", quote="FABRICATED SOURCE TEXT")])
    fake_llm.default_json = json.dumps(evil)
    body = client.post("/chat", json={"book_ids": [ready_book], "question": QUESTION}).json()
    assert all("FABRICATED" not in c["text"] for c in body["citations"])


def test_chat_follow_up_passes_history_to_model(client, ready_book, fake_llm):
    fake_llm.default_json = json.dumps(ANSWER)
    first = client.post("/chat", json={"book_ids": [ready_book], "question": QUESTION}).json()
    client.post(
        "/chat",
        json={
            "book_ids": [ready_book],
            "question": "Theorem 1.1 every compact set is bounded again",
            "conversation_id": first["conversation_id"],
        },
    )
    assert "컴팩트 집합은 유계이다" in fake_llm.calls[-1]["contents"]  # 이전 답변이 맥락으로 들어간다
    assert len(db.get_messages(first["conversation_id"])) == 4


def test_chat_not_found_does_not_call_llm(client, ready_book, fake_llm):
    r = client.post("/chat", json={"book_ids": [ready_book], "question": "zzzz qqqq wwww"})
    assert r.status_code == 200 and r.json()["not_found"] is True
    assert not fake_llm.calls


def test_chat_unknown_book_is_404(client, fake_llm):
    assert client.post("/chat", json={"book_ids": ["nope"], "question": "q?"}).status_code == 404


def test_chat_book_not_ready_is_409(client):
    db.create_book("b1", "T", "t.pdf")
    assert client.post("/chat", json={"book_ids": ["b1"], "question": "q?"}).status_code == 409


@pytest.mark.parametrize(
    "payload",
    [{"book_ids": [], "question": "q"}, {"book_ids": ["a"], "question": "  "}, {"book_ids": ["a"]}, {"question": "q"}],
)
def test_chat_validates_body(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_chat_rejects_bad_conversation_id(client, ready_book):
    r = client.post("/chat", json={"book_ids": [ready_book], "question": "q?", "conversation_id": "../x"})
    assert r.status_code == 422


def test_chat_without_api_key_is_503(client, ready_book, monkeypatch):
    def no_key(*a, **k):
        raise LLMNotConfigured("GEMINI_API_KEY가 설정되지 않았습니다.")

    monkeypatch.setattr(llm, "generate_json", no_key)
    r = client.post("/chat", json={"book_ids": [ready_book], "question": QUESTION})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
    assert db.get_messages("x") == []


def test_chat_malformed_model_json_is_502_and_saves_nothing(client, ready_book, fake_llm):
    fake_llm.default_json = "not json"
    r = client.post("/chat", json={"book_ids": [ready_book], "question": QUESTION, "conversation_id": "c1"})
    assert r.status_code == 502
    assert db.get_messages("c1") == []


def test_page_image_is_png(client, ready_book):
    r = client.get(f"/books/{ready_book}/page/1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


@pytest.mark.parametrize("n", [0, 3, -1])
def test_page_image_out_of_range_is_404(client, ready_book, n):
    assert client.get(f"/books/{ready_book}/page/{n}").status_code == 404


def test_page_image_unknown_book_is_404(client):
    assert client.get("/books/nope/page/1").status_code == 404


def test_static_mount_works_with_empty_web_dir(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    with TestClient(main.create_app(web_dir=web)) as c:
        assert c.get("/books").status_code == 200
        assert c.get("/").status_code == 404


def test_static_serves_index_when_present(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    with TestClient(main.create_app(web_dir=web)) as c:
        assert "hi" in c.get("/").text
        assert c.get("/..%2f..%2fsecret").status_code == 404


def test_app_starts_without_web_dir(tmp_path):
    with TestClient(main.create_app(web_dir=tmp_path / "missing")) as c:
        assert c.get("/books").status_code == 200


# ---------- 재시작 복구 / 삭제 경쟁 ----------


def test_startup_resumes_books_stuck_in_processing(make_pdf, fake_llm, fake_embed):
    pdf = make_pdf(["text", "text"])
    bid = "stuck1"
    db.create_book(bid, "Stuck", "stuck.pdf")
    db.update_book(bid, status="extracting", progress=0.2)
    book_dir = get_settings().book_dir(bid)
    book_dir.mkdir(parents=True)
    (book_dir / "book.pdf").write_bytes(pdf.read_bytes())

    with TestClient(main.create_app(web_dir=None)) as c:
        main.wait_for_recovery(c.app)
        st = c.get(f"/books/{bid}/status").json()

    assert st["status"] == "ready" and st["progress"] == 1.0


def test_startup_marks_stuck_book_without_pdf_as_error():
    db.create_book("stuck2", "Stuck", "stuck.pdf")
    db.update_book("stuck2", status="indexing")

    with TestClient(main.create_app(web_dir=None)) as c:
        main.wait_for_recovery(c.app)
        st = c.get("/books/stuck2/status").json()

    assert st["status"] == "error" and st["message"]


def test_startup_does_not_touch_ready_or_error_books(monkeypatch):
    calls = []
    monkeypatch.setattr(main.pipeline, "ingest_book", lambda *a: calls.append(a))
    db.create_book("r", "R", "r.pdf")
    db.update_book("r", status="ready", progress=1.0)
    db.create_book("e", "E", "e.pdf")
    db.update_book("e", status="error", message="boom")

    with TestClient(main.create_app(web_dir=None)) as c:
        main.wait_for_recovery(c.app)

    assert calls == []
    assert db.get_book("r")["status"] == "ready" and db.get_book("e")["message"] == "boom"


def test_delete_during_ingest_leaves_nothing_behind(client, make_pdf, fake_llm, monkeypatch):
    state = {}

    def embed_then_delete(texts, task):
        if "bid" in state and not state.get("deleted"):
            state["deleted"] = True
            assert client.delete(f"/books/{state['bid']}").status_code == 200
        return [bow_vector(t) for t in texts]

    monkeypatch.setattr(llm, "embed", embed_then_delete)
    real = main.pipeline.ingest_book

    def ingest(bid, path):
        state["bid"] = bid
        real(bid, path)

    monkeypatch.setattr(main.pipeline, "ingest_book", ingest)
    r = _upload(client, make_pdf(["text", "text"]))
    bid = r.json()["id"]

    assert state["deleted"]
    assert db.get_book(bid) is None
    assert db.chunks_by_label(["theorem 1.1"], [bid]) == []
    assert vectors._collection().count() == 0
    assert not get_settings().book_dir(bid).exists()

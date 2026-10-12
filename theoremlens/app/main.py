"""FastAPI 앱: 업로드·서재·삭제·채팅·쪽 이미지 라우트와 web/ 정적 서빙. 로직은 하위 계층에 위임한다."""

from __future__ import annotations

import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ..config import ROOT, get_settings
from ..index import db, vectors
from ..ingest import extract, pipeline
from ..llm import LLMNotConfigured
from ..rag import answer as rag_answer

log = logging.getLogger(__name__)

WEB_DIR = ROOT / "web"
_PDF_MAGIC = b"%PDF-"
_CHUNK = 1024 * 1024
_CONV_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_BOOKS_PER_CHAT = 20


class BookOut(BaseModel):
    id: str
    title: str
    filename: str
    num_pages: int = 0
    status: str
    progress: float = 0.0
    message: str = ""
    created_at: str | None = None
    num_chunks: int | None = None


class BookStatus(BaseModel):
    id: str
    status: str
    progress: float
    message: str
    num_pages: int


class ChatRequest(BaseModel):
    book_ids: list[str] = Field(min_length=1, max_length=_MAX_BOOKS_PER_CHAT)
    question: str = Field(max_length=4000)
    conversation_id: str | None = None

    @field_validator("question")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("질문이 비어 있습니다")
        return v

    @field_validator("conversation_id")
    @classmethod
    def _safe_id(cls, v: str | None) -> str | None:
        if v is not None and not _CONV_ID.match(v):
            raise ValueError("conversation_id 형식이 올바르지 않습니다")
        return v


class ChatResponse(rag_answer.AnswerResponse):
    conversation_id: str


def _title_from(filename: str) -> str:
    # 클라이언트가 준 파일명은 표시용으로만 쓰고, 디렉터리 부분은 버린다(경로 순회 방지)
    name = Path(filename.replace("\\", "/")).name
    return Path(name).stem or "book"


def _require_book(book_id: str) -> dict:
    book = db.get_book(book_id)
    if book is None:
        raise HTTPException(404, "책을 찾을 수 없습니다.")
    return book


def _save_upload(file: UploadFile, dest: Path, max_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as out:
        head = file.file.read(len(_PDF_MAGIC))
        if head != _PDF_MAGIC:
            raise HTTPException(400, "PDF 파일이 아닙니다.")
        out.write(head)
        size = len(head)
        while chunk := file.file.read(_CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, "파일이 너무 큽니다.")
            out.write(chunk)


def create_app(web_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="TheoremLens")

    @app.post("/books", response_model=BookOut)
    def upload_book(background: BackgroundTasks, file: Annotated[UploadFile, File()]) -> BookOut:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(400, "PDF 파일(.pdf)만 올릴 수 있습니다.")
        s = get_settings()
        book_id = uuid.uuid4().hex
        pdf_path = s.book_dir(book_id) / "book.pdf"  # 저장 경로는 서버가 정한다
        try:
            _save_upload(file, pdf_path, s.max_upload_mb * 1024 * 1024)
        except HTTPException:
            shutil.rmtree(s.book_dir(book_id), ignore_errors=True)
            raise
        db.create_book(book_id, _title_from(filename), Path(filename.replace("\\", "/")).name)
        background.add_task(pipeline.ingest_book, book_id, pdf_path)
        return BookOut(**_require_book(book_id))

    @app.get("/books", response_model=list[BookOut])
    def list_books() -> list[dict]:
        return db.list_books()

    @app.get("/books/{book_id}/status", response_model=BookStatus)
    def book_status(book_id: str) -> dict:
        return _require_book(book_id)

    @app.delete("/books/{book_id}")
    def delete_book(book_id: str) -> dict:
        _require_book(book_id)
        vectors.delete_book(book_id)
        db.delete_book(book_id)
        try:
            shutil.rmtree(get_settings().book_dir(book_id))
        except FileNotFoundError:
            pass
        except OSError:
            log.warning("책 파일 삭제 실패: %s", book_id, exc_info=True)
        return {"deleted": book_id}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        for bid in dict.fromkeys(req.book_ids):
            if _require_book(bid)["status"] != "ready":
                raise HTTPException(409, "아직 처리가 끝나지 않은 책이 있습니다.")
        cid = req.conversation_id or uuid.uuid4().hex
        history = db.get_messages(cid)
        try:
            resp = rag_answer.answer(req.question, list(dict.fromkeys(req.book_ids)), history=history)
        except LLMNotConfigured as e:
            raise HTTPException(503, str(e)) from e
        except rag_answer.AnswerError as e:
            raise HTTPException(502, str(e)) from e
        db.add_message(cid, "user", req.question, {"book_ids": req.book_ids})
        db.add_message(cid, "assistant", resp.answer_markdown, resp.model_dump())
        return ChatResponse(**resp.model_dump(), conversation_id=cid)

    @app.get("/books/{book_id}/page/{page_no}")
    def page_image(book_id: str, page_no: int) -> Response:
        book = _require_book(book_id)
        pdf_path = get_settings().book_dir(book_id) / "book.pdf"
        if not 1 <= page_no <= (book["num_pages"] or 0) or not pdf_path.exists():
            raise HTTPException(404, "쪽을 찾을 수 없습니다.")
        return Response(extract.render_page_png(pdf_path, page_no), media_type="image/png")

    # API 라우트 뒤에 마운트해야 정적 파일이 API를 가리지 않는다. 비어 있거나 없어도 앱은 뜬다.
    directory = WEB_DIR if web_dir is None else web_dir
    if directory.is_dir():
        app.mount("/", StaticFiles(directory=directory, html=True), name="web")
    return app


app = create_app()

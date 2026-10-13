"""SQLite 저장소: 책, 페이지, 청크(FTS5), 참조표, 대화 기록."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    num_pages INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued | extracting | indexing | ready | error
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS pages (
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page_no INTEGER NOT NULL,               -- 1부터 시작하는 PDF 쪽 번호
    markdown TEXT NOT NULL,
    method TEXT NOT NULL,                   -- marker | pymupdf | gemini
    PRIMARY KEY (book_id, page_no)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    unit_type TEXT NOT NULL,
    label TEXT,                             -- 정규화된 라벨, 예: 'theorem 3.2'
    display_label TEXT,                     -- 원문 표기, 예: 'Theorem 3.2'
    chapter TEXT,
    section TEXT,
    page_start INTEGER,
    page_end INTEGER,
    text TEXT NOT NULL,
    refs TEXT DEFAULT '[]'                  -- 본문에서 참조하는 정규화 라벨 목록(JSON)
);
CREATE INDEX IF NOT EXISTS idx_chunks_label ON chunks(book_id, label);
CREATE INDEX IF NOT EXISTS idx_chunks_ord ON chunks(book_id, ord);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, display_label, content='chunks', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, display_label) VALUES (new.id, new.text, coalesce(new.display_label, ''));
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, display_label)
    VALUES ('delete', old.id, old.text, coalesce(old.display_label, ''));
END;
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            c = sqlite3.connect(get_settings().db_path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA journal_mode = WAL")
            c.executescript(SCHEMA)
            _conn = c
        return _conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    with _lock:
        c = conn()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None


@dataclass
class Chunk:
    id: int
    book_id: str
    ord: int
    unit_type: str
    label: str | None
    display_label: str | None
    chapter: str | None
    section: str | None
    page_start: int
    page_end: int
    text: str
    refs: list[str]

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Chunk:
        d = dict(r)
        d["refs"] = json.loads(d.get("refs") or "[]")
        return cls(**d)


# ---------- books ----------


def create_book(book_id: str, title: str, filename: str) -> None:
    with tx() as c:
        c.execute("INSERT INTO books(id, title, filename) VALUES (?,?,?)", (book_id, title, filename))


def update_book(book_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with tx() as c:
        c.execute(f"UPDATE books SET {cols} WHERE id=?", (*fields.values(), book_id))


def list_books() -> list[dict]:
    with _lock:
        rows = (
            conn()
            .execute(
                "SELECT b.*, (SELECT count(*) FROM chunks WHERE book_id=b.id) AS num_chunks "
                "FROM books b ORDER BY created_at DESC"
            )
            .fetchall()
        )
    return [dict(r) for r in rows]


def get_book(book_id: str) -> dict | None:
    with _lock:
        r = conn().execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
    return dict(r) if r else None


def book_titles(book_ids: list[str]) -> dict[str, str]:
    if not book_ids:
        return {}
    q = ",".join("?" * len(book_ids))
    with _lock:
        rows = conn().execute(f"SELECT id, title FROM books WHERE id IN ({q})", book_ids).fetchall()
    return {r["id"]: r["title"] for r in rows}


def delete_book(book_id: str) -> None:
    with tx() as c:
        c.execute("DELETE FROM chunks WHERE book_id=?", (book_id,))
        c.execute("DELETE FROM pages WHERE book_id=?", (book_id,))
        c.execute("DELETE FROM books WHERE id=?", (book_id,))


# ---------- pages ----------


def save_page(book_id: str, page_no: int, markdown: str, method: str) -> None:
    with tx() as c:
        c.execute(
            "INSERT OR REPLACE INTO pages(book_id, page_no, markdown, method) VALUES (?,?,?,?)",
            (book_id, page_no, markdown, method),
        )


def get_pages(book_id: str) -> list[dict]:
    with _lock:
        rows = (
            conn()
            .execute("SELECT page_no, markdown, method FROM pages WHERE book_id=? ORDER BY page_no", (book_id,))
            .fetchall()
        )
    return [dict(r) for r in rows]


def get_page(book_id: str, page_no: int) -> dict | None:
    with _lock:
        r = (
            conn()
            .execute("SELECT page_no, markdown, method FROM pages WHERE book_id=? AND page_no=?", (book_id, page_no))
            .fetchone()
        )
    return dict(r) if r else None


# ---------- chunks ----------


def replace_chunks(book_id: str, chunks: list[dict]) -> list[int]:
    ids: list[int] = []
    with tx() as c:
        c.execute("DELETE FROM chunks WHERE book_id=?", (book_id,))
        for ch in chunks:
            cur = c.execute(
                "INSERT INTO chunks(book_id, ord, unit_type, label, display_label, chapter, section,"
                " page_start, page_end, text, refs) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    book_id,
                    ch["ord"],
                    ch["unit_type"],
                    ch.get("label"),
                    ch.get("display_label"),
                    ch.get("chapter"),
                    ch.get("section"),
                    ch["page_start"],
                    ch["page_end"],
                    ch["text"],
                    json.dumps(ch.get("refs", []), ensure_ascii=False),
                ),
            )
            ids.append(cur.lastrowid)
    return ids


def get_chunks(ids: list[int]) -> list[Chunk]:
    if not ids:
        return []
    q = ",".join("?" * len(ids))
    with _lock:
        rows = conn().execute(f"SELECT * FROM chunks WHERE id IN ({q})", ids).fetchall()
    by_id = {r["id"]: Chunk.from_row(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def book_chunks(book_id: str) -> list[Chunk]:
    """책의 모든 청크를 ord 순서로. 임베딩을 다시 만들 때 쓴다."""
    with _lock:
        rows = conn().execute("SELECT * FROM chunks WHERE book_id=? ORDER BY ord", (book_id,)).fetchall()
    return [Chunk.from_row(r) for r in rows]


def chunks_by_label(labels: list[str], book_ids: list[str]) -> list[Chunk]:
    if not labels or not book_ids:
        return []
    ql = ",".join("?" * len(labels))
    qb = ",".join("?" * len(book_ids))
    with _lock:
        rows = (
            conn()
            .execute(
                f"SELECT * FROM chunks WHERE label IN ({ql}) AND book_id IN ({qb}) ORDER BY book_id, ord",
                (*labels, *book_ids),
            )
            .fetchall()
        )
    return [Chunk.from_row(r) for r in rows]


def chunk_at(book_id: str, ord_: int) -> Chunk | None:
    with _lock:
        r = conn().execute("SELECT * FROM chunks WHERE book_id=? AND ord=?", (book_id, ord_)).fetchone()
    return Chunk.from_row(r) if r else None


_FTS_TOKEN = re.compile(r"\w+", re.UNICODE)


def fts_search(query: str, book_ids: list[str], limit: int) -> list[tuple[int, float]]:
    """BM25 검색. (chunk_id, score) 목록. sqlite bm25는 점수가 낮을수록 관련도가 높다."""
    tokens = [t for t in _FTS_TOKEN.findall(query) if len(t) > 1 or t.isdigit()]
    if not tokens or not book_ids:
        return []
    # 토큰을 따옴표로 감싸 FTS 문법 오류를 막고, 접두 일치로 한국어 조사 변형을 흡수
    match = " OR ".join(f'"{t}"*' for t in tokens)
    qb = ",".join("?" * len(book_ids))
    with _lock:
        rows = (
            conn()
            .execute(
                "SELECT c.id, bm25(chunks_fts) AS s FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                f"WHERE chunks_fts MATCH ? AND c.book_id IN ({qb}) ORDER BY s LIMIT ?",
                (match, *book_ids, limit),
            )
            .fetchall()
        )
    return [(r["id"], r["s"]) for r in rows]


# ---------- messages ----------


def add_message(conversation_id: str, role: str, content: str, meta: dict | None = None) -> None:
    with tx() as c:
        c.execute(
            "INSERT INTO messages(conversation_id, role, content, meta) VALUES (?,?,?,?)",
            (conversation_id, role, content, json.dumps(meta or {}, ensure_ascii=False)),
        )


def get_messages(conversation_id: str, limit: int = 50) -> list[dict]:
    with _lock:
        rows = (
            conn()
            .execute(
                "SELECT role, content, meta FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            )
            .fetchall()
        )
    out = [{"role": r["role"], "content": r["content"], "meta": json.loads(r["meta"])} for r in rows]
    return list(reversed(out))

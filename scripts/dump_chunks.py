"""책 한 권의 청크를 사람이 읽기 좋은 Markdown으로 내보낸다(구조화 결과 점검용).

사용법:
    python scripts/dump_chunks.py                  # 책 목록
    python scripts/dump_chunks.py BOOK_ID          # 표준 출력
    python scripts/dump_chunks.py BOOK_ID -o out.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from theoremlens.index import db


def _header(c: db.Chunk) -> str:
    pages = f"{c.page_start}" if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
    title = c.display_label or "(라벨 없음)"
    return f"## [{c.ord}] {c.unit_type} · {title} · p.{pages}"


def render(book: dict, chunks: list[db.Chunk]) -> str:
    out = [f"# {book['title']} ({book['id']})", "", f"청크 {len(chunks)}개, 상태 {book['status']}", ""]
    for c in chunks:
        out.append(_header(c))
        meta = [f"chunk id: {c.id}", f"label: {c.label}", f"chapter: {c.chapter}", f"section: {c.section}"]
        if c.refs:
            meta.append("refs: " + ", ".join(c.refs))
        meta.append(f"{len(c.text)}자")
        out += ["", "> " + " | ".join(meta), "", c.text, "", "---", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book_id", nargs="?", help="생략하면 책 목록을 출력한다")
    ap.add_argument("-o", "--output", type=Path, help="저장할 파일(생략 시 표준 출력)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 인코딩에서 한글·수식이 깨지지 않게
    if not args.book_id:
        for b in db.list_books():
            print(f"{b['id']}\t{b['status']}\t청크 {b['num_chunks']}\t{b['title']}")
        return 0

    book = db.get_book(args.book_id)
    if book is None:
        print(f"책을 찾을 수 없습니다: {args.book_id}", file=sys.stderr)
        return 1
    text = render(book, db.book_chunks(args.book_id))
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"저장: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""MinerU로 PDF 일부 쪽만 변환하는 스모크 스크립트.

사용법: python smoke_mineru.py <pdf> <출력.md> [--tier standard]
MinerU는 프로젝트 의존성이 아니므로 격리 환경에서 실행한다(smoke_mineru.sh 참고).
"""

import argparse
import subprocess
import time
from pathlib import Path

import fitz  # PyMuPDF

PAGE_RANGE = [44, 45, 46]  # 0-based (PDF 45~47쪽)


def slice_pdf(src: str, dst: Path, pages: list[int]) -> None:
    with fitz.open(src) as doc, fitz.open() as out:
        for p in pages:
            out.insert_pdf(doc, from_page=p, to_page=p)
        out.save(dst)


def main(pdf_path: str, out_path: str, tier: str) -> None:
    out = Path(out_path + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # with tempfile.TemporaryDirectory() as tmp:
    #     part = Path(tmp) / "part.pdf"
    #     slice_pdf(pdf_path, part, PAGE_RANGE)
    t0 = time.time()
    subprocess.run(
        [
            "mineru-kit",
            "parse",
            str(pdf_path),
            "-o",
            str(out),
            "--tier",
            tier,
            "--format",
            "middle_json",
            "--pages",
            "44-46",
        ],
        check=True,
    )
    md = out.read_text(encoding="utf-8")
    print("convert", round(time.time() - t0, 1), "chars", len(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--tier", default="standard", choices=["flash", "basic", "standard", "advanced"])
    a = ap.parse_args()
    main(a.pdf, a.out, a.tier)

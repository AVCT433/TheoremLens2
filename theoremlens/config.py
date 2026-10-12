"""환경 변수 기반 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    answer_model: str = field(default_factory=lambda: os.getenv("THEOREMLENS_ANSWER_MODEL", "gemini-2.5-pro"))
    fast_model: str = field(default_factory=lambda: os.getenv("THEOREMLENS_FAST_MODEL", "gemini-2.5-flash"))
    embed_model: str = field(default_factory=lambda: os.getenv("THEOREMLENS_EMBED_MODEL", "gemini-embedding-001"))
    text_extractor: str = field(default_factory=lambda: os.getenv("THEOREMLENS_TEXT_EXTRACTOR", "auto"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("THEOREMLENS_DATA_DIR", ROOT / "data")))

    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("THEOREMLENS_MAX_UPLOAD_MB", "300")))

    # 검색 파라미터
    top_k: int = 8
    # RRF 점수 기준 근거 부족 판정에 쓰는 임베딩 코사인 유사도 하한
    min_similarity: float = 0.55
    # 페이지에 이 글자 수보다 적으면 스캔 페이지로 간주
    scanned_char_threshold: int = 40

    @property
    def db_path(self) -> Path:
        return self.data_dir / "theoremlens.sqlite"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def book_dir(self, book_id: str) -> Path:
        return self.data_dir / "books" / book_id


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s

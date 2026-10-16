"""Gemini 클라이언트 래퍼(생성·이미지 전사). 임베딩은 `embedder.py`가 로컬 모델로 처리한다."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from google import genai
from google.genai import types

from .config import get_settings


class LLMNotConfigured(RuntimeError):
    pass


@lru_cache
def client() -> genai.Client:
    key = get_settings().api_key
    if not key:
        raise LLMNotConfigured("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return genai.Client(api_key=key)


def _retry(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except LLMNotConfigured:
            raise
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2**i)


def generate_json(model: str, system: str, contents: Any, schema: dict) -> str:
    """JSON 스키마를 강제한 생성. 응답 텍스트(JSON)를 반환."""
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=schema,
    )
    return _retry(lambda: client().models.generate_content(model=model, contents=contents, config=cfg)).text


def generate_text(model: str, system: str, contents: Any) -> str:
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=0)
    return _retry(lambda: client().models.generate_content(model=model, contents=contents, config=cfg)).text or ""

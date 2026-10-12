"""Gemini 클라이언트 래퍼."""

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


def image_part(data: bytes, mime: str = "image/png") -> types.Part:
    return types.Part.from_bytes(data=data, mime_type=mime)


def embed(texts: list[str], task: str) -> list[list[float]]:
    """task: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY"""
    s = get_settings()
    out: list[list[float]] = []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        res = _retry(
            lambda batch=batch: client().models.embed_content(
                model=s.embed_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task),
            )
        )
        out.extend(e.values for e in res.embeddings)
    return out

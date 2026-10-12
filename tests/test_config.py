"""config와 conftest 격리 fixture 검증."""

from __future__ import annotations

from pathlib import Path

import pytest

from theoremlens import llm
from theoremlens.config import get_settings
from theoremlens.index import db

from .conftest import bow_vector


def test_data_dir_is_isolated_to_tmp_path(tmp_path):
    s = get_settings()
    assert s.data_dir == tmp_path / "data"
    assert s.data_dir.is_dir()


def test_derived_paths_live_under_data_dir(tmp_path):
    s = get_settings()
    assert s.db_path == tmp_path / "data" / "theoremlens.sqlite"
    assert s.chroma_dir == tmp_path / "data" / "chroma"
    assert s.book_dir("b1") == tmp_path / "data" / "books" / "b1"


def test_api_key_is_blank_so_llm_fails_fast():
    assert get_settings().api_key == ""
    with pytest.raises(llm.LLMNotConfigured):
        llm.generate_text("m", "s", "hi")


def test_settings_are_cached_until_cleared():
    assert get_settings() is get_settings()


def test_env_overrides_models(monkeypatch):
    monkeypatch.setenv("THEOREMLENS_FAST_MODEL", "my-fast")
    get_settings.cache_clear()
    assert get_settings().fast_model == "my-fast"


def test_db_never_touches_real_data_dir(tmp_path):
    db.create_book("b", "t", "f.pdf")
    assert Path(get_settings().db_path).parent == tmp_path / "data"


def test_bow_vector_is_deterministic_and_similarity_ordered():
    a, b, c = bow_vector("compact set"), bow_vector("compact set closed"), bow_vector("banana")
    assert a == bow_vector("compact set")
    dot = lambda x, y: sum(p * q for p, q in zip(x, y, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c)


def test_fake_llm_records_calls(fake_llm):
    fake_llm.text_responses.append("hello")
    assert llm.generate_text("m", "s", "x") == "hello"
    assert fake_llm.calls[0]["kind"] == "text"

"""llm 래퍼 테스트: 재시도/백오프, 미설정 키."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from theoremlens import llm


class _FakeModels:
    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.generate_calls = 0

    def generate_content(self, model, contents, config):
        self.generate_calls += 1
        if self.generate_calls <= self.fail_times:
            raise RuntimeError("transient")
        return SimpleNamespace(text="ok")


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    rec: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", rec.append)
    return rec


def _use(monkeypatch, models: _FakeModels) -> None:
    # genai.Client를 만들지 않고 llm.client 모듈 속성만 교체한다
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(models=models))


def test_generate_text_retries_with_exponential_backoff(monkeypatch, sleeps):
    m = _FakeModels(fail_times=3)
    _use(monkeypatch, m)
    assert llm.generate_text("m", "s", "hi") == "ok"
    assert m.generate_calls == 4
    assert sleeps == [1, 2, 4]


def test_generate_text_gives_up_after_four_attempts(monkeypatch, sleeps):
    m = _FakeModels(fail_times=99)
    _use(monkeypatch, m)
    with pytest.raises(RuntimeError, match="transient"):
        llm.generate_text("m", "s", "hi")
    assert m.generate_calls == 4
    assert len(sleeps) == 3


def test_generate_json_returns_text_and_retries(monkeypatch, sleeps):
    m = _FakeModels(fail_times=1)
    _use(monkeypatch, m)
    assert llm.generate_json("m", "s", "hi", {"type": "object"}) == "ok"
    assert m.generate_calls == 2


def test_not_configured_is_never_retried(sleeps):
    with pytest.raises(llm.LLMNotConfigured):
        llm.generate_text("m", "s", "hi")
    assert sleeps == []


def test_retry_helper_does_not_retry_not_configured(sleeps):
    calls = []

    def fn():
        calls.append(1)
        raise llm.LLMNotConfigured("no key")

    with pytest.raises(llm.LLMNotConfigured):
        llm._retry(fn)
    assert calls == [1]
    assert sleeps == []

"""Tests for the LLM client protocol, offline double and HTTP adapter.

The HTTP adapter is exercised with a stubbed ``urlopen`` so the suite stays
fully offline and deterministic.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from agent_guardrails.llm import LLMError, OpenAICompatLLM, ScriptedLLM


# --------------------------------------------------------------------------
# ScriptedLLM
# --------------------------------------------------------------------------
def test_scripted_llm_replays_in_order():
    client = ScriptedLLM(["one", "two"])
    assert client.complete([{"role": "user", "content": "a"}]) == "one"
    assert client.complete([{"role": "user", "content": "b"}]) == "two"


def test_scripted_llm_raises_when_exhausted():
    with pytest.raises(LLMError):
        ScriptedLLM([]).complete([])


def test_scripted_llm_records_prompts():
    client = ScriptedLLM(["x"])
    client.complete([{"role": "system", "content": "s"}])
    assert client.prompts == [[{"role": "system", "content": "s"}]]


def test_scripted_llm_copies_response_list():
    source = ["a"]
    client = ScriptedLLM(source)
    client.complete([])
    assert source == ["a"], "constructor must not alias the caller's list"


# --------------------------------------------------------------------------
# OpenAICompatLLM construction
# --------------------------------------------------------------------------
def test_requires_base_url():
    with pytest.raises(LLMError):
        OpenAICompatLLM(base_url="", model="m")


def test_requires_model():
    with pytest.raises(LLMError):
        OpenAICompatLLM(base_url="https://example.com", model="")


def test_base_url_trailing_slash_is_normalised():
    assert OpenAICompatLLM("https://x.test/", "m").base_url == "https://x.test"


# --------------------------------------------------------------------------
# OpenAICompatLLM transport (stubbed)
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _install(monkeypatch, handler):
    monkeypatch.setattr("urllib.request.urlopen", handler)


def test_successful_completion_returns_content(monkeypatch):
    captured = {}

    def handler(request, timeout=None):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
        )

    _install(monkeypatch, handler)
    client = OpenAICompatLLM("https://api.test", "gpt-x", api_key="sk-test")
    assert client.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert captured["url"] == "https://api.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-x"
    assert captured["body"]["temperature"] == 0.0


def test_no_api_key_omits_authorization_header(monkeypatch):
    captured = {}

    def handler(request, timeout=None):
        captured["auth"] = request.get_header("Authorization")
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    _install(monkeypatch, handler)
    OpenAICompatLLM("https://api.test", "m", api_key=None).complete([])
    assert captured["auth"] is None


def test_http_400_is_not_retried(monkeypatch):
    calls = []

    def handler(request, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", {}, io.BytesIO(b"{}")
        )

    _install(monkeypatch, handler)
    client = OpenAICompatLLM("https://api.test", "m", max_retries=3, retry_backoff_seconds=0)
    with pytest.raises(LLMError) as excinfo:
        client.complete([])
    assert "HTTP 400" in str(excinfo.value)
    assert len(calls) == 1, "4xx is a client error and must not be retried"


def test_http_429_is_retried_then_succeeds(monkeypatch):
    calls = []

    def handler(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many", {}, io.BytesIO(b"{}")
            )
        return _FakeResponse({"choices": [{"message": {"content": "recovered"}}]})

    _install(monkeypatch, handler)
    client = OpenAICompatLLM("https://api.test", "m", max_retries=3, retry_backoff_seconds=0)
    assert client.complete([]) == "recovered"
    assert len(calls) == 2


def test_http_500_exhausts_retries(monkeypatch):
    calls = []

    def handler(request, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(
            request.full_url, 503, "Unavailable", {}, io.BytesIO(b"{}")
        )

    _install(monkeypatch, handler)
    client = OpenAICompatLLM("https://api.test", "m", max_retries=2, retry_backoff_seconds=0)
    with pytest.raises(LLMError) as excinfo:
        client.complete([])
    assert "HTTP 503" in str(excinfo.value)
    assert len(calls) == 2


def test_transport_error_is_wrapped(monkeypatch):
    def handler(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    _install(monkeypatch, handler)
    with pytest.raises(LLMError) as excinfo:
        OpenAICompatLLM("https://api.test", "m").complete([])
    assert "transport error" in str(excinfo.value)


def test_invalid_json_response_is_reported(monkeypatch):
    class BadResponse:
        def read(self):
            return b"<html>not json</html>"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    _install(monkeypatch, lambda request, timeout=None: BadResponse())
    with pytest.raises(LLMError) as excinfo:
        OpenAICompatLLM("https://api.test", "m").complete([])
    assert "invalid JSON" in str(excinfo.value)


def test_missing_choices_is_reported(monkeypatch):
    _install(monkeypatch, lambda request, timeout=None: _FakeResponse({"choices": []}))
    with pytest.raises(LLMError) as excinfo:
        OpenAICompatLLM("https://api.test", "m").complete([])
    assert "no choices" in str(excinfo.value)


def test_non_string_content_is_reported(monkeypatch):
    _install(
        monkeypatch,
        lambda request, timeout=None: _FakeResponse(
            {"choices": [{"message": {"content": {"nested": True}}}]}
        ),
    )
    with pytest.raises(LLMError) as excinfo:
        OpenAICompatLLM("https://api.test", "m").complete([])
    assert "not a string" in str(excinfo.value)

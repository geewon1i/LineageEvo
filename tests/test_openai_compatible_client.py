import json

import pytest

from lineage_evo.config import LLMConfig
from lineage_evo.llm import OpenAICompatibleLLMClient


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_client_builds_request_and_parses_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": '{"factor": "rank(close)"}'}}]})

    monkeypatch.setattr("lineage_evo.llm.openai_compatible.request.urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        LLMConfig(base_url="https://example.test/v1", api_key="key", model="model", timeout_seconds=3)
    )

    response = client.complete(system_prompt="sys", user_prompt="user")

    assert response.content == '{"factor": "rank(close)"}'
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 3
    assert captured["body"]["model"] == "model"
    assert captured["body"]["messages"][0]["content"] == "sys"


def test_openai_compatible_client_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("LINEAGEEVO_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LINEAGEEVO_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LINEAGEEVO_LLM_API_KEY"):
        OpenAICompatibleLLMClient(LLMConfig(api_key=None, model=None))


def test_openai_compatible_client_retries_read_timeout(monkeypatch):
    calls = {"count": 0}

    class TimeoutThenSuccessResponse(FakeHTTPResponse):
        def read(self):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("The read operation timed out")
            return super().read()

    def fake_urlopen(req, timeout):
        return TimeoutThenSuccessResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("lineage_evo.llm.openai_compatible.request.urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient(
        LLMConfig(
            base_url="https://example.test/v1",
            api_key="key",
            model="model",
            timeout_seconds=3,
            max_retry=2,
            retry_wait_seconds=0,
        )
    )

    response = client.complete(system_prompt="sys", user_prompt="user")

    assert response.content == "ok"
    assert calls["count"] == 2

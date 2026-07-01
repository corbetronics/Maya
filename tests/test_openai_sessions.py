"""OpenAI session client tests."""

import json
from io import BytesIO
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from backend.openai_sessions import (
    LOCAL_DEVELOPMENT_SAFETY_IDENTIFIER,
    MissingOpenAIAPIKeyError,
    MissingOpenAISafetyIdentifierError,
    OpenAISessionClient,
)


class FakeResponse:
    """Minimal context manager response for urlopen tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the JSON payload to return."""
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        """Enter the fake response context."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the fake response context."""

    def read(self) -> bytes:
        """Return the encoded fake JSON body."""
        return json.dumps(self.payload).encode("utf-8")


def test_openai_session_client_requires_api_key_only_when_called(monkeypatch) -> None:
    """Confirm the client can be constructed without credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAISessionClient()

    with pytest.raises(MissingOpenAIAPIKeyError):
        client.create_ephemeral_session({"model": "gpt-realtime-2"})


def test_openai_session_client_posts_session_config(monkeypatch) -> None:
    """Confirm the client posts the wrapped session config and returns OpenAI JSON."""
    captured: dict[str, Any] = {}

    def fake_urlopen(session_request: Request, timeout: float) -> FakeResponse:
        captured["url"] = session_request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(session_request.header_items())
        captured["body"] = json.loads((session_request.data or b"{}").decode("utf-8"))
        return FakeResponse({"client_secret": {"value": "ephemeral-test-secret"}})

    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_SAFETY_IDENTIFIER", "hashed-user-id")
    monkeypatch.setattr("backend.openai_sessions.request.urlopen", fake_urlopen)
    session_config = {
        "type": "realtime",
        "model": "gpt-realtime-2",
        "audio": {"output": {"voice": "alloy"}},
        "instructions": "Project Maya instructions",
    }

    response = OpenAISessionClient().create_ephemeral_session(session_config)

    assert response == {"client_secret": {"value": "ephemeral-test-secret"}}
    assert captured["url"] == "https://api.openai.com/v1/realtime/client_secrets"
    assert captured["timeout"] == 10.0
    assert captured["headers"]["Authorization"] == "Bearer test-api-key"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["Openai-safety-identifier"] == "hashed-user-id"
    assert captured["body"] == {"session": session_config}
    assert "modalities" not in captured["body"]["session"]


def test_openai_session_client_uses_development_safety_identifier(monkeypatch) -> None:
    """Confirm development has a local-only safety identifier fallback."""
    captured: dict[str, Any] = {}

    def fake_urlopen(session_request: Request, timeout: float) -> FakeResponse:
        _ = timeout
        captured["headers"] = dict(session_request.header_items())
        return FakeResponse({"client_secret": {"value": "ephemeral-test-secret"}})

    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.delenv("OPENAI_SAFETY_IDENTIFIER", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setattr("backend.openai_sessions.request.urlopen", fake_urlopen)

    OpenAISessionClient().create_ephemeral_session({"type": "realtime"})

    assert (
        captured["headers"]["Openai-safety-identifier"]
        == LOCAL_DEVELOPMENT_SAFETY_IDENTIFIER
    )


def test_openai_session_client_requires_safety_identifier_outside_development(
    monkeypatch,
) -> None:
    """Confirm non-development environments require a configured safety identifier."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.delenv("OPENAI_SAFETY_IDENTIFIER", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(MissingOpenAISafetyIdentifierError):
        OpenAISessionClient().create_ephemeral_session({"type": "realtime"})


def test_openai_session_client_logs_rejected_status_and_body_without_secrets(
    monkeypatch,
    caplog,
) -> None:
    """Confirm OpenAI HTTP rejections log only response status and body."""

    def fake_urlopen(session_request: Request, timeout: float) -> FakeResponse:
        _ = session_request
        _ = timeout
        raise HTTPError(
            url="https://api.openai.com/v1/realtime/client_secrets",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=BytesIO(b'{"error":"missing route"}'),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    monkeypatch.setenv("OPENAI_SAFETY_IDENTIFIER", "safe-user-id")
    monkeypatch.setattr("backend.openai_sessions.request.urlopen", fake_urlopen)

    with pytest.raises(HTTPError):
        OpenAISessionClient().create_ephemeral_session({"type": "realtime"})

    assert "status=404" in caplog.text
    assert '{"error":"missing route"}' in caplog.text
    assert "secret-api-key" not in caplog.text
    assert "safe-user-id" not in caplog.text

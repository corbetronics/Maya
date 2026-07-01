"""Top-level backend endpoint tests."""

from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.openai_sessions import OpenAISessionClient


def test_root_health_endpoint_returns_ok() -> None:
    """Confirm the top-level health endpoint is available."""
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_maya_session_config_endpoint_returns_realtime_config() -> None:
    """Confirm the Maya endpoint returns a local realtime session config."""
    client = TestClient(create_app())

    response = client.get("/maya/session-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "gpt-realtime-2"
    assert payload["audio"]["output"]["voice"] == "alloy"
    assert "voice" not in payload
    assert "audio" in payload["modalities"]
    assert "instructions" in payload
    assert "## Constitution" in payload["instructions"]
    assert "turn_detection" in payload
    assert "input_audio_transcription" in payload


def test_maya_ephemeral_session_endpoint_returns_openai_response(monkeypatch) -> None:
    """Confirm the endpoint builds config and returns the mocked OpenAI response."""
    captured: dict[str, Any] = {}

    def fake_create_ephemeral_session(
        self: OpenAISessionClient,
        session_config: dict[str, Any],
    ) -> dict[str, Any]:
        captured["client"] = self
        captured["session_config"] = session_config
        return {"client_secret": {"value": "mock-secret"}}

    monkeypatch.setattr(
        OpenAISessionClient,
        "create_ephemeral_session",
        fake_create_ephemeral_session,
    )
    client = TestClient(create_app())

    response = client.post("/maya/ephemeral-session")

    assert response.status_code == 200
    assert response.json() == {"client_secret": {"value": "mock-secret"}}
    assert captured["session_config"]["model"] == "gpt-realtime-2"
    assert captured["session_config"]["audio"]["output"]["voice"] == "alloy"
    assert "voice" not in captured["session_config"]
    assert "## Constitution" in captured["session_config"]["instructions"]


def test_maya_ephemeral_session_missing_api_key_returns_clear_500(monkeypatch) -> None:
    """Confirm the endpoint reports missing OpenAI credentials clearly."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app())

    response = client.post("/maya/ephemeral-session")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "OPENAI_API_KEY is required to create a session.",
    }


def test_maya_ephemeral_session_openai_http_error_returns_safe_502(monkeypatch) -> None:
    """Confirm OpenAI HTTP rejections become useful gateway errors."""

    def fake_create_ephemeral_session(
        self: OpenAISessionClient,
        session_config: dict[str, Any],
    ) -> dict[str, Any]:
        _ = self
        _ = session_config
        raise HTTPError(
            url="https://api.openai.com/v1/realtime/client_secrets",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        OpenAISessionClient,
        "create_ephemeral_session",
        fake_create_ephemeral_session,
    )
    client = TestClient(create_app())

    response = client.post("/maya/ephemeral-session")

    assert response.status_code == 502
    assert response.json() == {
        "detail": "OpenAI rejected ephemeral session creation: 404",
    }


def test_frontend_catch_all_serves_index_for_non_api_paths(tmp_path: Path) -> None:
    """Confirm production frontend routing serves React for application paths."""
    index_path = tmp_path / "index.html"
    index_path.write_text("<html><body>Project Maya</body></html>", encoding="utf-8")
    client = TestClient(create_app(frontend_dist=tmp_path))

    response = client.get("/producer")

    assert response.status_code == 200
    assert "Project Maya" in response.text


def test_frontend_catch_all_does_not_intercept_maya_api_paths(tmp_path: Path) -> None:
    """Confirm unknown Maya API paths are not rewritten to the React app."""
    index_path = tmp_path / "index.html"
    index_path.write_text("<html><body>Project Maya</body></html>", encoding="utf-8")
    client = TestClient(create_app(frontend_dist=tmp_path))

    response = client.get("/maya/not-found")

    assert response.status_code == 404


def test_cors_is_configurable_for_local_development(monkeypatch) -> None:
    """Confirm CORS can be configured without exposing it broadly in production."""
    monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:5173"]')
    client = TestClient(create_app())

    response = client.options(
        "/maya/session-config",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_is_same_origin_by_default_in_production(monkeypatch) -> None:
    """Confirm production does not allow cross-origin requests by default."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    client = TestClient(create_app())

    response = client.get("/health", headers={"Origin": "https://example.com"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers

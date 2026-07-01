"""Realtime session configuration tests."""

from pathlib import Path

from backend.realtime_session import RealtimeSessionConfig, build_session_config
from brain import CharacterEngine, ConstitutionLoader


def test_session_config_instructions_include_mayas_constitution(tmp_path: Path) -> None:
    """Confirm session instructions include the raw Constitution content."""
    constitution_text = "# Realtime Constitution\n\nStay in podcast guest mode.\n"
    constitution_path = tmp_path / "constitution.md"
    constitution_path.write_text(constitution_text, encoding="utf-8")
    maya = CharacterEngine(
        constitution_loader=ConstitutionLoader(path=constitution_path),
    ).create_maya()

    config = build_session_config(maya)

    assert constitution_text in config["instructions"]


def test_session_config_modalities_include_audio() -> None:
    """Confirm audio is enabled by default."""
    maya = CharacterEngine().create_maya()

    config = build_session_config(maya)

    assert "audio" in config["modalities"]


def test_session_config_voice_is_configurable() -> None:
    """Confirm callers can choose a different Realtime voice."""
    maya = CharacterEngine().create_maya()
    session_config = RealtimeSessionConfig(voice="verse")

    config = build_session_config(maya, session_config=session_config)

    assert config["audio"]["output"]["voice"] == "verse"
    assert "voice" not in config


def test_session_config_does_not_require_api_key(monkeypatch) -> None:
    """Confirm building config does not depend on OpenAI credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    maya = CharacterEngine().create_maya()

    config = build_session_config(maya)

    assert config["model"] == "gpt-realtime-2"


def test_session_config_serializes_current_webrtc_shape() -> None:
    """Confirm session config matches the current Realtime WebRTC session shape."""
    maya = CharacterEngine().create_maya()

    config = build_session_config(maya)

    assert config["type"] == "realtime"
    assert config["model"] == "gpt-realtime-2"
    assert config["audio"] == {"output": {"voice": "alloy"}}
    assert "voice" not in config

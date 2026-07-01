"""Realtime API session configuration for Project MAYA."""

from dataclasses import dataclass, field
from typing import Any

from brain.engine import MayaCharacter
from brain.prompt_composer import PromptComposer


@dataclass(frozen=True, slots=True)
class RealtimeSessionConfig:
    """Local configuration values for a future OpenAI Realtime session."""

    model: str = "gpt-realtime-2"
    voice: str = "alloy"
    modalities: tuple[str, ...] = ("audio", "text")
    turn_detection: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        }
    )
    input_audio_transcription: dict[str, Any] = field(
        default_factory=lambda: {
            "model": "whisper-1",
        }
    )


def build_session_config(
    maya: MayaCharacter,
    session_config: RealtimeSessionConfig | None = None,
) -> dict[str, Any]:
    """Build a Realtime API session config without opening a network connection."""
    config = session_config or RealtimeSessionConfig()
    prompt_bundle = PromptComposer().compose(maya)

    return {
        "type": "realtime",
        "model": config.model,
        "instructions": prompt_bundle.system_prompt,
        "audio": {
            "output": {
                "voice": config.voice,
            },
        },
    }

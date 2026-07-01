"""Shared data models for the Project MAYA brain package."""

from dataclasses import dataclass, field
from enum import StrEnum


class SpeakerRole(StrEnum):
    """Participant roles in a podcast conversation."""

    HOST = "host"
    MAYA = "maya"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One spoken or system-authored turn in the conversation."""

    speaker: SpeakerRole
    text: str


@dataclass(frozen=True, slots=True)
class BrainInput:
    """Input envelope for future brain processing."""

    turns: tuple[ConversationTurn, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BrainOutput:
    """Output envelope for future brain processing."""

    turn: ConversationTurn | None = None

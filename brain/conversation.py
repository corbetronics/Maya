"""Conversation state contracts for podcast dialogue."""

from dataclasses import dataclass, field
from enum import StrEnum

from brain.models import ConversationTurn


class ConversationDepth(StrEnum):
    """Supported depth levels for a podcast exchange."""

    SURFACE = "surface"
    PERSONAL = "personal"
    REFLECTIVE = "reflective"
    DEEP = "deep"


@dataclass(slots=True)
class ConversationState:
    """Mutable state for the current podcast conversation."""

    turns: tuple[ConversationTurn, ...] = field(default_factory=tuple)
    current_speaker: str | None = None
    topic: str | None = None
    depth: ConversationDepth = ConversationDepth.SURFACE
    silence_duration_seconds: float = 0.0
    question_count: int = 0
    answer_count: int = 0

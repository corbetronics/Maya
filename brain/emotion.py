"""Emotion state contracts for Maya's future expression layer."""

from dataclasses import dataclass


@dataclass(slots=True)
class ConversationEmotion:
    """Mutable emotional state tracked across a live conversation."""

    confidence: float = 0.5
    warmth: float = 0.5
    energy: float = 0.5
    reflection: float = 0.5
    humour: float = 0.5


@dataclass(frozen=True, slots=True)
class EmotionState:
    """A neutral description of emotional tone for a response."""

    label: str = "neutral"
    intensity: float = 0.0

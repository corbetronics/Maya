"""Reflection contracts for future self-review behavior."""

from dataclasses import dataclass, field
from enum import StrEnum


class ThinkingState(StrEnum):
    """Named internal thinking modes Maya may occupy during conversation."""

    IDLE = "idle"
    THINKING = "thinking"
    REMEMBERING = "remembering"
    UNCERTAIN = "uncertain"
    EXCITED = "excited"


@dataclass(frozen=True, slots=True)
class ReflectionNote:
    """A recorded observation about a conversation moment."""

    summary: str
    tags: tuple[str, ...] = field(default_factory=tuple)

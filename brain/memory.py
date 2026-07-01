"""Memory contracts for future long-term context."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from brain.models import ConversationTurn


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A durable fact or observation Maya may recall later."""

    key: str
    value: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class WorkingMemory:
    """In-memory conversation context collected during one live podcast session."""

    conversation_history: list[ConversationTurn] = field(default_factory=list)
    facts_learned: dict[str, str] = field(default_factory=dict)
    callbacks: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    speaker_names: dict[str, str] = field(default_factory=dict)
    timestamps: list[datetime] = field(default_factory=list)

    def remember_turn(
        self,
        turn: ConversationTurn,
        occurred_at: datetime | None = None,
    ) -> None:
        """Store a conversation turn and its timestamp in insertion order."""
        self.conversation_history.append(turn)
        self.timestamps.append(occurred_at or datetime.now(UTC))

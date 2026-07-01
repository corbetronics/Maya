"""Deterministic curiosity engine for follow-up question decisions."""

from dataclasses import dataclass, field

from brain.models import ConversationTurn, SpeakerRole


QUESTION_STARTERS = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "could",
    "would",
    "can",
    "do",
    "does",
    "did",
    "is",
    "are",
)

DEPTH_SIGNALS = (
    "because",
    "changed",
    "felt",
    "realized",
    "remember",
    "surprised",
    "struggled",
    "learned",
    "decided",
    "noticed",
    "important",
    "meaning",
)

FOLLOW_UP_PROMPTS = (
    "What made that stand out to you?",
    "How did that change the way you think about it?",
    "What happened next?",
    "What do you wish people understood about that?",
)


@dataclass(frozen=True, slots=True)
class CuriosityState:
    """Topics Maya may explore later in a conversation."""

    open_questions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CuriosityDecision:
    """Result of deciding whether Maya should ask a follow-up question."""

    should_ask: bool
    score: float
    reason: str
    question: str | None = None


@dataclass(frozen=True, slots=True)
class CuriosityEngine:
    """Pure-Python rules engine for Maya's follow-up question behavior."""

    ask_threshold: float = 0.62
    recent_maya_question_window: int = 3

    def decide(
        self,
        turns: tuple[ConversationTurn, ...],
        state: CuriosityState | None = None,
    ) -> CuriosityDecision:
        """Decide whether Maya should ask a follow-up question."""
        latest_host_turn = self._latest_host_turn(turns)
        if latest_host_turn is None:
            return CuriosityDecision(
                should_ask=False,
                score=0.0,
                reason="No host turn is available.",
            )

        if self._recent_maya_question_count(turns) >= 2:
            return CuriosityDecision(
                should_ask=False,
                score=0.0,
                reason="Maya has asked enough recent questions.",
            )

        text = latest_host_turn.text.strip()
        score = self._score_host_turn(text)
        if state and state.open_questions:
            score += 0.12

        bounded_score = min(score, 1.0)
        if bounded_score < self.ask_threshold:
            return CuriosityDecision(
                should_ask=False,
                score=bounded_score,
                reason="The latest host turn does not need a follow-up.",
            )

        return CuriosityDecision(
            should_ask=True,
            score=bounded_score,
            reason="The latest host turn invites deeper exploration.",
            question=self._select_question(text, state),
        )

    def _latest_host_turn(self, turns: tuple[ConversationTurn, ...]) -> ConversationTurn | None:
        """Return the newest host-authored turn, if one exists."""
        for turn in reversed(turns):
            if turn.speaker == SpeakerRole.HOST and turn.text.strip():
                return turn
        return None

    def _recent_maya_question_count(self, turns: tuple[ConversationTurn, ...]) -> int:
        """Count recent Maya turns that already asked questions."""
        recent_turns = turns[-self.recent_maya_question_window :]
        return sum(
            1
            for turn in recent_turns
            if turn.speaker == SpeakerRole.MAYA and self._looks_like_question(turn.text)
        )

    def _score_host_turn(self, text: str) -> float:
        """Score how much the latest host turn invites curiosity."""
        words = text.split()
        lower_text = text.lower()
        score = 0.0

        if len(words) >= 8:
            score += 0.22
        if len(words) >= 18:
            score += 0.18
        depth_signal_count = sum(1 for signal in DEPTH_SIGNALS if signal in lower_text)
        score += min(depth_signal_count * 0.14, 0.42)
        if self._looks_like_question(text):
            score -= 0.2
        if text.endswith("..."):
            score += 0.12
        if any(char.isdigit() for char in text):
            score += 0.08

        return max(score, 0.0)

    def _looks_like_question(self, text: str) -> bool:
        """Return whether text appears to ask a question."""
        stripped_text = text.strip()
        if stripped_text.endswith("?"):
            return True
        first_word = stripped_text.split(maxsplit=1)[0].lower() if stripped_text else ""
        return first_word in QUESTION_STARTERS

    def _select_question(self, text: str, state: CuriosityState | None) -> str:
        """Select a deterministic follow-up question."""
        if state and state.open_questions:
            return state.open_questions[0]

        lower_text = text.lower()
        if "because" in lower_text or "meaning" in lower_text:
            return FOLLOW_UP_PROMPTS[1]
        if "struggled" in lower_text or "felt" in lower_text:
            return FOLLOW_UP_PROMPTS[3]
        if "remember" in lower_text or "realized" in lower_text:
            return FOLLOW_UP_PROMPTS[0]
        return FOLLOW_UP_PROMPTS[2]

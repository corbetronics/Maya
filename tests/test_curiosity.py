"""Curiosity engine tests."""

from brain.curiosity import CuriosityEngine, CuriosityState
from brain.models import ConversationTurn, SpeakerRole


def test_curiosity_engine_asks_for_deeper_host_turn() -> None:
    """Confirm the engine asks when a host turn has enough depth signals."""
    engine = CuriosityEngine()
    turns = (
        ConversationTurn(
            speaker=SpeakerRole.HOST,
            text="I realized because of that trip that the work had a different meaning.",
        ),
    )

    decision = engine.decide(turns)

    assert decision.should_ask is True
    assert decision.question is not None


def test_curiosity_engine_uses_open_question_first() -> None:
    """Confirm queued open questions take priority when Maya asks."""
    engine = CuriosityEngine()
    turns = (
        ConversationTurn(
            speaker=SpeakerRole.HOST,
            text="I remember the moment because it changed how I approached everything after.",
        ),
    )
    state = CuriosityState(open_questions=("What did that teach you?",))

    decision = engine.decide(turns, state)

    assert decision.should_ask is True
    assert decision.question == "What did that teach you?"


def test_curiosity_engine_holds_back_after_recent_maya_questions() -> None:
    """Confirm the engine avoids stacking too many Maya questions."""
    engine = CuriosityEngine()
    turns = (
        ConversationTurn(speaker=SpeakerRole.MAYA, text="What happened next?"),
        ConversationTurn(speaker=SpeakerRole.MAYA, text="How did that feel?"),
        ConversationTurn(
            speaker=SpeakerRole.HOST,
            text="I realized because of it that the whole project had changed.",
        ),
    )

    decision = engine.decide(turns)

    assert decision.should_ask is False

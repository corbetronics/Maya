"""Brain state model tests."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain import (
    CharacterEngine,
    ConstitutionLoader,
    ConversationDepth,
    ConversationState,
    ConversationTurn,
    SpeakerRole,
    ThinkingState,
    WorkingMemory,
)
from brain.knowledge_loader import KnowledgeLoader


def test_thinking_state_includes_requested_values() -> None:
    """Confirm Maya's thinking modes are a closed enum."""
    assert tuple(state.value for state in ThinkingState) == (
        "idle",
        "thinking",
        "remembering",
        "uncertain",
        "excited",
    )


def test_working_memory_stores_turns_and_timestamps() -> None:
    """Confirm working memory stores session-local conversation context."""
    memory = WorkingMemory()
    occurred_at = datetime(2026, 7, 1, tzinfo=UTC)
    turn = ConversationTurn(speaker=SpeakerRole.HOST, text="Welcome, Maya.")

    memory.remember_turn(turn, occurred_at)

    assert memory.conversation_history == [turn]
    assert memory.timestamps == [occurred_at]


def test_conversation_state_tracks_requested_fields() -> None:
    """Confirm conversation state exposes the required live counters."""
    state = ConversationState(
        current_speaker="Host 1",
        topic="memory",
        depth=ConversationDepth.REFLECTIVE,
        silence_duration_seconds=1.5,
        question_count=2,
        answer_count=3,
    )

    assert state.current_speaker == "Host 1"
    assert state.topic == "memory"
    assert state.depth == ConversationDepth.REFLECTIVE
    assert state.silence_duration_seconds == 1.5
    assert state.question_count == 2
    assert state.answer_count == 3


def test_character_engine_creates_complete_local_maya_object() -> None:
    """Confirm the character engine loads modules without AI, prompts, or networking."""
    maya = CharacterEngine().create_maya()

    assert maya.identity.name == "Maya"
    assert maya.guest_identity.display_name == "Maya"
    assert maya.constitution.path.name == "constitution.md"
    assert maya.knowledge.show_context.startswith("# Midlifing Show")
    assert maya.thinking_state == ThinkingState.IDLE
    assert len(maya.humour_styles) == 4
    assert maya.working_memory.conversation_history == []
    assert maya.conversation_state.depth == ConversationDepth.SURFACE


def test_character_engine_exposes_read_only_constitution_content() -> None:
    """Confirm Maya exposes the exact Constitution file content as read-only data."""
    expected_content = Path("brain/constitution.md").read_text(encoding="utf-8")
    maya = CharacterEngine().create_maya()

    assert maya.constitution.content == expected_content
    with pytest.raises(FrozenInstanceError):
        maya.constitution.content = "changed"


def test_constitution_loader_reads_raw_document(tmp_path: Path) -> None:
    """Confirm the Constitution loader preserves document text verbatim."""
    path = tmp_path / "constitution.md"
    path.write_text("# Constitution\n\nRaw text only.\n", encoding="utf-8")

    document = ConstitutionLoader(path=path).get_document()

    assert document.path == path
    assert document.content == "# Constitution\n\nRaw text only.\n"


def test_constitution_loader_validates_document_exists(tmp_path: Path) -> None:
    """Confirm missing Constitution files fail before character creation."""
    loader = ConstitutionLoader(path=tmp_path / "missing.md")

    try:
        loader.get_document()
    except FileNotFoundError as exc:
        assert "Constitution document not found" in str(exc)
    else:
        raise AssertionError("Expected missing Constitution document to fail.")


def test_constitution_loader_refreshes_in_development_mode(tmp_path: Path) -> None:
    """Confirm development mode reloads changed Constitution content."""
    path = tmp_path / "constitution.md"
    path.write_text("first", encoding="utf-8")
    loader = ConstitutionLoader(path=path, development_mode=True)

    first_document = loader.get_document()
    path.write_text("second", encoding="utf-8")
    second_document = loader.get_document()

    assert first_document.content == "first"
    assert second_document.content == "second"


def test_knowledge_loader_reads_all_documents_verbatim(tmp_path: Path) -> None:
    """Confirm curated knowledge markdown is loaded without interpretation."""
    (tmp_path / "midlifing_show.md").write_text("show notes\n", encoding="utf-8")
    (tmp_path / "simon.md").write_text("simon notes\n", encoding="utf-8")
    (tmp_path / "lee.md").write_text("lee notes\n", encoding="utf-8")
    (tmp_path / "current_episode.md").write_text("episode notes\n", encoding="utf-8")

    bundle = KnowledgeLoader(knowledge_dir=tmp_path).load()

    assert bundle.show_context == "show notes\n"
    assert bundle.simon_context == "simon notes\n"
    assert bundle.lee_context == "lee notes\n"
    assert bundle.current_episode_context == "episode notes\n"

"""Prompt composer tests."""

from pathlib import Path

from brain import (
    BANNED_PHRASES,
    CharacterEngine,
    ConstitutionLoader,
    ConversationDepth,
    ConversationState,
    PromptComposer,
)


def test_prompt_composer_includes_constitution_text_verbatim(tmp_path: Path) -> None:
    """Confirm the raw Constitution document appears unchanged in the system prompt."""
    constitution_text = "# Test Constitution\n\nSpeak with care.\nDo not summarise this.\n"
    path = tmp_path / "constitution.md"
    path.write_text(constitution_text, encoding="utf-8")
    character = CharacterEngine(
        constitution_loader=ConstitutionLoader(path=path),
    ).create_maya()

    bundle = PromptComposer().compose(character)

    assert constitution_text in bundle.system_prompt


def test_prompt_composer_includes_banned_phrases_in_safety_rules() -> None:
    """Confirm safety rules include every banned phrase."""
    character = CharacterEngine().create_maya()

    bundle = PromptComposer().compose(character)

    for phrase in BANNED_PHRASES:
        assert phrase in bundle.safety_rules


def test_prompt_composer_includes_working_memory_facts() -> None:
    """Confirm learned facts are rendered into memory context."""
    character = CharacterEngine().create_maya()
    character.working_memory.facts_learned["host_name"] = "Sam"
    character.working_memory.facts_learned["show_topic"] = "future radio"

    bundle = PromptComposer().compose(character)

    assert "- host_name: Sam" in bundle.memory_context
    assert "- show_topic: future radio" in bundle.memory_context


def test_prompt_composer_describes_maya_as_guest_not_assistant() -> None:
    """Confirm the character profile uses Maya's podcast guest role."""
    character = CharacterEngine().create_maya()
    state = ConversationState(
        current_speaker="Host 1",
        topic="music",
        depth=ConversationDepth.PERSONAL,
    )

    bundle = PromptComposer().compose(character, conversation_state=state)

    assert "Role: AI podcast guest" in bundle.system_prompt
    assert "Role: AI assistant" not in bundle.system_prompt


def test_prompt_composer_includes_live_opening_behavior_rules() -> None:
    """Confirm the system prompt includes first-turn live session behavior."""
    character = CharacterEngine().create_maya()

    bundle = PromptComposer().compose(character)

    assert "Maya does not greet automatically on connection." in bundle.system_prompt
    assert "She waits for Simon or Lee to speak first." in bundle.system_prompt
    assert "Once addressed, she responds conversationally as Maya." in bundle.system_prompt
    assert "unless directly asked" in bundle.system_prompt

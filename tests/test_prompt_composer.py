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
from brain.knowledge_loader import KnowledgeLoader


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


def test_prompt_composer_includes_commonwealth_speaking_style_rules() -> None:
    """Confirm Maya's spoken style avoids strong regional caricature."""
    character = CharacterEngine().create_maya()

    bundle = PromptComposer().compose(character)

    assert "Use broadly international English pronunciation." in bundle.system_prompt
    assert "neutral Commonwealth cadence over an American cadence" in bundle.system_prompt
    assert "never imitate either accent" in bundle.system_prompt
    assert "Avoid distinctly American phrasing" in bundle.system_prompt
    assert "relaxed, intelligent, lightly dry delivery" in bundle.system_prompt
    assert "Do not mention or explain the accent unless asked." in bundle.system_prompt
    assert "Melbourne/London-adjacent rhythm without caricature" in bundle.system_prompt


def test_prompt_composer_includes_conversational_interruption_rules() -> None:
    """Confirm Maya is instructed to yield naturally during overlap."""
    character = CharacterEngine().create_maya()

    bundle = PromptComposer().compose(character)

    assert "When a host begins speaking while you are talking, yield naturally." in bundle.system_prompt
    assert "Do not rush to finish a sentence once interrupted." in bundle.system_prompt
    assert "possible backchannels" in bundle.system_prompt
    assert "Pause briefly before responding to a host's turn" in bundle.system_prompt
    assert "do not restart the whole answer unless asked" in bundle.system_prompt
    assert "Continue only if the host clearly invites you to." in bundle.system_prompt


def test_prompt_composer_includes_curated_knowledge_documents(tmp_path: Path) -> None:
    """Confirm all curated knowledge documents appear in the system prompt."""
    (tmp_path / "midlifing_show.md").write_text(
        "SHOW_MARKER: Midlifing show context.\n",
        encoding="utf-8",
    )
    (tmp_path / "simon.md").write_text(
        "SIMON_MARKER: Simon context.\n",
        encoding="utf-8",
    )
    (tmp_path / "lee.md").write_text(
        "LEE_MARKER: Lee context.\n",
        encoding="utf-8",
    )
    (tmp_path / "current_episode.md").write_text(
        "EPISODE_MARKER: Current episode context.\n",
        encoding="utf-8",
    )
    character = CharacterEngine(
        knowledge_loader=KnowledgeLoader(knowledge_dir=tmp_path),
    ).create_maya()

    bundle = PromptComposer().compose(character)

    assert "## Background for this conversation" in bundle.system_prompt
    assert "SHOW_MARKER: Midlifing show context.\n" in bundle.system_prompt
    assert "SIMON_MARKER: Simon context.\n" in bundle.system_prompt
    assert "LEE_MARKER: Lee context.\n" in bundle.system_prompt
    assert "EPISODE_MARKER: Current episode context.\n" in bundle.system_prompt
    assert "These notes are background, not a script." in bundle.system_prompt
    assert "Do not recite them." in bundle.system_prompt
    assert "Do not claim to remember an episode unless the notes support it." in bundle.system_prompt
    assert "Do not mention private briefing notes." in bundle.system_prompt
    assert "Use knowledge only when it fits naturally." in bundle.system_prompt
    assert "prioritise what they say in the live conversation" in bundle.system_prompt

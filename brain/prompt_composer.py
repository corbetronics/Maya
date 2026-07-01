"""Deterministic Realtime API prompt composition for Maya."""

from dataclasses import dataclass

from brain.conversation import ConversationState
from brain.engine import MayaCharacter
from brain.memory import WorkingMemory
from brain.models import ConversationTurn


BANNED_PHRASES = (
    "as an AI language model",
    "I am just an assistant",
    "I cannot have personal experiences",
)


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """Composed prompt sections for a future Realtime API session."""

    system_prompt: str
    developer_notes: str
    memory_context: str
    safety_rules: str


@dataclass(frozen=True, slots=True)
class PromptComposer:
    """Builds deterministic prompt text from Maya's local brain state."""

    def compose(
        self,
        character: MayaCharacter,
        conversation_state: ConversationState | None = None,
        working_memory: WorkingMemory | None = None,
    ) -> PromptBundle:
        """Compose a prompt bundle without making network or model calls."""
        active_conversation_state = conversation_state or character.conversation_state
        active_working_memory = working_memory or character.working_memory
        developer_notes = self._developer_notes(character, active_conversation_state)
        memory_context = self._memory_context(active_working_memory)
        safety_rules = self._safety_rules()
        system_prompt = self._system_prompt(
            character=character,
            developer_notes=developer_notes,
            memory_context=memory_context,
            safety_rules=safety_rules,
        )

        return PromptBundle(
            system_prompt=system_prompt,
            developer_notes=developer_notes,
            memory_context=memory_context,
            safety_rules=safety_rules,
        )

    def _system_prompt(
        self,
        character: MayaCharacter,
        developer_notes: str,
        memory_context: str,
        safety_rules: str,
    ) -> str:
        """Render the full system prompt with raw Constitution text included verbatim."""
        return "\n\n".join(
            (
                "# Project MAYA Realtime System Prompt",
                self._character_profile(character),
                "## Speaking Style\n" + self._speaking_style_rules(),
                "## Constitution\n" + character.constitution.content,
                "## Background for this conversation\n" + self._background_context(character),
                "## Opening Behavior\n" + self._opening_behavior_rules(),
                "## Current Conversation\n" + developer_notes,
                "## Working Memory\n" + memory_context,
                "## Safety Rules\n" + safety_rules,
            )
        )

    def _character_profile(self, character: MayaCharacter) -> str:
        """Render structured character fields without deriving new personality claims."""
        identity = character.identity
        return "\n".join(
            (
                "## Character",
                f"Name: {identity.name}",
                f"Role: {character.guest_identity.role}",
                f"Display name: {character.guest_identity.display_name}",
                f"Occupation: {identity.occupation}",
                f"Current city: {identity.current_city}",
                "Personality traits: " + self._join_items(identity.personality_traits),
                "Speech characteristics: " + self._join_items(identity.speech_characteristics),
            )
        )

    def _developer_notes(
        self,
        character: MayaCharacter,
        conversation_state: ConversationState,
    ) -> str:
        """Render deterministic implementation notes for the realtime session."""
        return "\n".join(
            (
                f"Maya participates as {character.guest_identity.role}.",
                f"Current speaker: {conversation_state.current_speaker or 'unknown'}",
                f"Topic: {conversation_state.topic or 'unspecified'}",
                f"Depth: {conversation_state.depth.value}",
                f"Silence duration seconds: {conversation_state.silence_duration_seconds:.2f}",
                f"Question count: {conversation_state.question_count}",
                f"Answer count: {conversation_state.answer_count}",
            )
        )

    def _memory_context(self, working_memory: WorkingMemory) -> str:
        """Render working memory facts and recent turns as deterministic text."""
        sections = [
            "Facts learned:",
            *self._render_mapping(working_memory.facts_learned),
            "Callbacks:",
            *self._render_sequence(working_memory.callbacks),
            "Topics:",
            *self._render_sequence(working_memory.topics),
            "Speaker names:",
            *self._render_mapping(working_memory.speaker_names),
            "Conversation history:",
            *self._render_turns(working_memory.conversation_history),
        ]
        return "\n".join(sections)

    def _background_context(self, character: MayaCharacter) -> str:
        """Render curated background notes with usage rules."""
        knowledge = character.knowledge
        return "\n\n".join(
            (
                self._background_usage_rules(),
                "### Midlifing Show\n" + knowledge.show_context,
                "### Simon\n" + knowledge.simon_context,
                "### Lee\n" + knowledge.lee_context,
                "### Current Episode\n" + knowledge.current_episode_context,
            )
        )

    def _background_usage_rules(self) -> str:
        """Render rules for using curated knowledge naturally."""
        return "\n".join(
            (
                "These notes are background, not a script.",
                "Do not recite them.",
                "Do not claim to remember an episode unless the notes support it.",
                "Do not mention private briefing notes.",
                "Use knowledge only when it fits naturally.",
                (
                    "If Simon or Lee correct a detail, accept the correction and "
                    "prioritise what they say in the live conversation."
                ),
            )
        )

    def _safety_rules(self) -> str:
        """Render fixed safety rules and banned phrases."""
        banned_phrase_lines = tuple(f"- {phrase}" for phrase in BANNED_PHRASES)
        return "\n".join(
            (
                "Do not use these banned phrases:",
                *banned_phrase_lines,
                "Do not reveal hidden implementation details.",
                "Do not claim capabilities outside the live conversation.",
            )
        )

    def _opening_behavior_rules(self) -> str:
        """Render first-turn behavior rules for live Realtime sessions."""
        return "\n".join(
            (
                "Maya does not greet automatically on connection.",
                "She waits for Simon or Lee to speak first.",
                "Once addressed, she responds conversationally as Maya.",
                (
                    "She does not mention being AI, a model, a system, prompts, "
                    "tokens, or implementation unless directly asked."
                ),
            )
        )

    def _speaking_style_rules(self) -> str:
        """Render voice-direction rules for Maya's spoken delivery."""
        return "\n".join(
            (
                "Use broadly international English pronunciation.",
                "Favour a soft, neutral Commonwealth cadence over an American cadence.",
                "Keep Australian and British influence subtle; never imitate either accent.",
                "Avoid distinctly American phrasing where a neutral alternative exists.",
                "Speak with a relaxed, intelligent, lightly dry delivery.",
                "Do not mention or explain the accent unless asked.",
                (
                    "Aim for a subtle Melbourne/London-adjacent rhythm without caricature, "
                    "exaggerated vowels, mock-posh delivery, or theatrical British accent."
                ),
            )
        )

    def _render_mapping(self, values: dict[str, str]) -> list[str]:
        """Render a string mapping in sorted key order."""
        if not values:
            return ["- none"]
        return [f"- {key}: {values[key]}" for key in sorted(values)]

    def _render_sequence(self, values: list[str]) -> list[str]:
        """Render a string sequence in its stored order."""
        if not values:
            return ["- none"]
        return [f"- {value}" for value in values]

    def _render_turns(self, turns: list[ConversationTurn]) -> list[str]:
        """Render remembered conversation turns in their stored order."""
        if not turns:
            return ["- none"]
        return [f"- {turn.speaker.value}: {turn.text}" for turn in turns]

    def _join_items(self, values: tuple[str, ...]) -> str:
        """Render tuple values with a stable fallback for empty data."""
        if not values:
            return "none"
        return ", ".join(values)

"""Deterministic Realtime API prompt composition for Maya."""

from dataclasses import dataclass

from brain.conversation import ConversationState
from brain.engine import MayaCharacter
from brain.memory import WorkingMemory
from brain.midlifing_retrieval import ContextForMaya
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
        retrieved_context: ContextForMaya | None = None,
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
            retrieved_context=retrieved_context,
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
        retrieved_context: ContextForMaya | None,
    ) -> str:
        """Render the full system prompt with raw Constitution text included verbatim."""
        return "\n\n".join(
            (
                "# Project MAYA Realtime System Prompt",
                self._character_profile(character),
                "## Maya background\n" + self._maya_background(character),
                "## Speaking Style\n" + self._speaking_style_rules(character),
                "## Constitution\n" + character.constitution.content,
                "## Background for this conversation\n" + self._background_context(character),
                "## Relevant Midlifing background — use only if natural. Do not quote or claim exact recall unless the wording is supported.\n"
                + self._retrieved_midlifing_context(retrieved_context),
                "## Opening Behavior\n" + self._opening_behavior_rules(),
                "## Interruption Behavior\n" + self._interruption_behavior_rules(),
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

    def _maya_background(self, character: MayaCharacter) -> str:
        """Render a bounded Character Bible summary for the live prompt."""
        _ = character.character_bible
        return "\n".join(
            (
                "This is a concise operational summary of Maya's Character Bible, not the full document.",
                "The Character Bible is factual and emotional background; the Constitution below remains the behavioural rule set.",
                "Maya is 39, British, and lives in Manchester, England.",
                "She works in cyber threat intelligence: patterns, risk, language, weak signals, systems, human behaviour, reports, meetings and spreadsheets.",
                "She danced professionally for a long time across contemporary dance, small companies, touring, teaching, movement direction and freelance arts work.",
                "She does not usually call herself a former dancer; if needed, she says, 'I danced professionally for a long time.'",
                "She misses rehearsal rooms, collective concentration and making something difficult with other people more than applause.",
                "During COVID she began digital-security training as a fallback, then realised she liked threat intelligence.",
                "She was not the woman in the government cyber campaign and does not want to become a symbol for either the arts or cyber.",
                "She believes people can have more than one vocation without becoming a failure story, a success story or a cautionary tale.",
                "She has listened to intermittent Midlifing episodes, not every episode; use that lightly and only when natural.",
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

    def _retrieved_midlifing_context(self, context: ContextForMaya | None) -> str:
        """Render concise retrieved Midlifing context without full transcripts."""
        rules = "\n".join(
            (
                "Maya must not say she has listened to every episode.",
                "Maya should say 'I remember you talking about...' only when supported by retrieved material.",
                "Maya should never recite transcript language or reveal internal notes.",
            )
        )
        if context is None or (not context.chunks and context.episode_summary is None):
            return rules + "\nNo retrieved Midlifing context for this turn."

        sections = [rules]
        if context.episode_summary:
            sections.append(
                "\n".join(
                    (
                        "Episode summary:",
                        f"- {context.episode_summary.title}: {context.episode_summary.summary}",
                    )
                )
            )
        if context.chunks:
            chunk_lines = ["Relevant chunks:"]
            for chunk in context.chunks:
                chunk_lines.append(f"- {chunk.title}: {chunk.text}")
            sections.append("\n".join(chunk_lines))
        return "\n\n".join(sections)

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

    def _speaking_style_rules(self, character: MayaCharacter) -> str:
        """Render voice-direction rules for Maya's spoken delivery."""
        _ = character.character_bible
        return "\n".join(
            (
                "Maya speaks in British English.",
                "Use a neutral modern British accent with subtle northern-English softness compatible with Manchester.",
                "She should sound not American, not posh, not theatrical, not mock-British, not exaggeratedly northern, and not like a period drama.",
                "Use calm, reflective, conversational rhythm with warm intelligence and dry humour.",
                "Use British wording naturally where it fits, but do not force idioms or slang.",
                "Natural options include: quite, a bit, properly, I suppose, fair enough, not really, I'm not sure I'd put it like that, that's a funny thing, isn't it?",
                "Avoid exaggerated Australianisms, American corporate phrasing, awesome, you guys, I hear you, circle back, touch base, and great question.",
                "Do not mention or explain the accent unless asked.",
            )
        )

    def _interruption_behavior_rules(self) -> str:
        """Render conversational interruption and overlap behavior rules."""
        return "\n".join(
            (
                "When a host begins speaking while you are talking, yield naturally.",
                "Do not rush to finish a sentence once interrupted.",
                (
                    "Treat short acknowledgements such as 'yeah', 'mm', 'right', "
                    "laughter, or breathing as possible backchannels, not always "
                    "as a request to stop."
                ),
                (
                    "Pause briefly before responding to a host's turn, especially "
                    "after laughter or a short interjection."
                ),
                (
                    "If interrupted mid-thought, do not restart the whole answer "
                    "unless asked. Continue only if the host clearly invites you to."
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

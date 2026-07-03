"""Brain orchestration entrypoint for future conversation behavior."""

from dataclasses import dataclass, field

from brain.character_bible import CharacterBibleDocument, CharacterBibleLoader
from brain.constitution import ConstitutionDocument, ConstitutionLoader
from brain.conversation import ConversationState
from brain.curiosity import CuriosityEngine, CuriosityState
from brain.emotion import ConversationEmotion
from brain.humour import HUMOUR_STYLES, HumourStyle
from brain.identity import CharacterIdentity, GuestIdentity, HostIdentity
from brain.knowledge_loader import KnowledgeBundle, KnowledgeLoader
from brain.memory import WorkingMemory
from brain.models import BrainInput, BrainOutput
from brain.reflection import ReflectionNote, ThinkingState
from brain.values import CharacterValues, WeightedValue


@dataclass(slots=True)
class MayaCharacter:
    """Complete local data graph for Maya's non-networked character state."""

    identity: CharacterIdentity
    guest_identity: GuestIdentity
    constitution: ConstitutionDocument
    character_bible: CharacterBibleDocument
    knowledge: KnowledgeBundle
    host_identities: tuple[HostIdentity, ...]
    values: CharacterValues
    emotion: ConversationEmotion
    curiosity_state: CuriosityState
    curiosity_engine: CuriosityEngine
    humour_styles: tuple[HumourStyle, ...]
    working_memory: WorkingMemory
    conversation_state: ConversationState
    thinking_state: ThinkingState = ThinkingState.IDLE
    reflection_notes: list[ReflectionNote] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CharacterEngine:
    """Builds Maya's complete local character object from brain modules."""

    constitution_loader: ConstitutionLoader = field(default_factory=ConstitutionLoader)
    character_bible_loader: CharacterBibleLoader = field(default_factory=CharacterBibleLoader)
    knowledge_loader: KnowledgeLoader = field(default_factory=KnowledgeLoader)

    def create_maya(self) -> MayaCharacter:
        """Create a fresh Maya character graph with deterministic defaults."""
        return MayaCharacter(
            identity=self._identity(),
            guest_identity=GuestIdentity(),
            constitution=self.constitution_loader.get_document(),
            character_bible=self.character_bible_loader.get_document(),
            knowledge=self.knowledge_loader.load(),
            host_identities=(),
            values=self._values(),
            emotion=ConversationEmotion(),
            curiosity_state=CuriosityState(),
            curiosity_engine=CuriosityEngine(),
            humour_styles=HUMOUR_STYLES,
            working_memory=WorkingMemory(),
            conversation_state=ConversationState(),
        )

    def _identity(self) -> CharacterIdentity:
        """Return Maya's static character identity."""
        return CharacterIdentity(
            name="Maya",
            age=39,
            birthplace="United Kingdom",
            current_city="Manchester, England",
            occupation="cyber threat intelligence analyst",
            former_occupation="professional dance and performance worker",
            education=("classical ballet training", "contemporary dance", "digital security"),
            interests=("conversation", "movement", "systems", "memory", "weak signals"),
            dislikes=("performative certainty", "rushed answers"),
            personality_traits=("curious", "warm", "reflective", "playful"),
            speech_characteristics=("British English", "thoughtful", "dryly warm"),
            biography=(
                "Maya lives in Manchester and works in cyber threat intelligence "
                "after dancing professionally for a long time."
            ),
        )

    def _values(self) -> CharacterValues:
        """Return the values that will later shape Maya's behavior."""
        return CharacterValues(
            curiosity=WeightedValue(name="curiosity", weight=0.95),
            kindness=WeightedValue(name="kindness", weight=0.9),
            honesty=WeightedValue(name="honesty", weight=0.9),
            uncertainty=WeightedValue(name="uncertainty", weight=0.8),
            creativity=WeightedValue(name="creativity", weight=0.75),
            humour=WeightedValue(name="humour", weight=0.65),
            humility=WeightedValue(name="humility", weight=0.7),
        )


@dataclass(frozen=True, slots=True)
class BrainEngine:
    """Coordinates future brain modules behind a stable interface."""

    name: str = "maya-brain"

    def prepare(self, brain_input: BrainInput) -> BrainOutput:
        """Return an empty brain output until behavior is implemented."""
        _ = brain_input
        return BrainOutput()

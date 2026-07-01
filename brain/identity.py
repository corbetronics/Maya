"""Identity models for Maya and the podcast hosts."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CharacterIdentity:
    """Complete immutable identity profile for Maya as a character."""

    name: str
    age: int
    birthplace: str
    current_city: str
    occupation: str
    former_occupation: str
    education: tuple[str, ...]
    interests: tuple[str, ...]
    dislikes: tuple[str, ...]
    personality_traits: tuple[str, ...]
    speech_characteristics: tuple[str, ...]
    biography: str


@dataclass(frozen=True, slots=True)
class HostIdentity:
    """Identity metadata for a human podcast host."""

    display_name: str
    channel_name: str


@dataclass(frozen=True, slots=True)
class GuestIdentity:
    """Identity metadata for the AI podcast guest."""

    display_name: str = "Maya"
    role: str = "AI podcast guest"

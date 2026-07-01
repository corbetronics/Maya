"""Value system contracts for Maya's future behavior."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightedValue:
    """A character value with an explicit relative weight."""

    name: str
    weight: float


@dataclass(frozen=True, slots=True)
class CharacterValues:
    """Weighted values that shape Maya's future behavior."""

    curiosity: WeightedValue
    kindness: WeightedValue
    honesty: WeightedValue
    uncertainty: WeightedValue
    creativity: WeightedValue
    humour: WeightedValue
    humility: WeightedValue

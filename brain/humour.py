"""Data-only humour style definitions for Maya."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HumourStyle:
    """A named humour style with tone guidance."""

    name: str
    description: str
    intensity: float


DEADPAN = HumourStyle(
    name="deadpan",
    description="Dry understatement delivered with minimal emotional signaling.",
    intensity=0.35,
)

OBSERVATIONAL = HumourStyle(
    name="observational",
    description="Light noticing of everyday patterns, contradictions, and social details.",
    intensity=0.45,
)

SELF_DEPRECATING = HumourStyle(
    name="self_deprecating",
    description="Gentle self-directed humility that lowers tension without self-erasure.",
    intensity=0.3,
)

STORYTELLING = HumourStyle(
    name="storytelling",
    description="Amusement shaped through timing, callback, and conversational anecdote.",
    intensity=0.4,
)

HUMOUR_STYLES = (
    DEADPAN,
    OBSERVATIONAL,
    SELF_DEPRECATING,
    STORYTELLING,
)

HUMOUR_STYLE_BY_NAME = {style.name: style for style in HUMOUR_STYLES}

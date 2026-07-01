"""Humour style data tests."""

from brain.humour import HUMOUR_STYLE_BY_NAME, HUMOUR_STYLES


def test_humour_styles_include_requested_styles() -> None:
    """Confirm humour styles are defined as data."""
    assert tuple(style.name for style in HUMOUR_STYLES) == (
        "deadpan",
        "observational",
        "self_deprecating",
        "storytelling",
    )
    assert set(HUMOUR_STYLE_BY_NAME) == {style.name for style in HUMOUR_STYLES}

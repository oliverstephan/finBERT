"""Very small language guess (German vs. English) based on function words, no model download needed."""

import re

_DE = {
    "der",
    "die",
    "das",
    "und",
    "nicht",
    "mit",
    "für",
    "ist",
    "sind",
    "wird",
    "werden",
    "auf",
    "des",
    "den",
    "dem",
    "ein",
    "eine",
    "einer",
    "im",
    "zum",
    "zur",
    "sich",
    "auch",
    "nach",
    "bei",
    "aus",
    "wurde",
    "über",
    "gegenüber",
    "vorjahr",
    "unternehmen",
    "gesellschaft",
    "prognose",
    "umsatz",
    "deutlich",
}
_EN = {
    "the",
    "and",
    "not",
    "with",
    "for",
    "is",
    "are",
    "will",
    "be",
    "on",
    "of",
    "a",
    "an",
    "in",
    "to",
    "its",
    "has",
    "have",
    "was",
    "were",
    "from",
    "by",
    "at",
    "year",
    "company",
    "said",
    "that",
    "this",
}
_WORD = re.compile(r"[a-zäöüß]+")


def detect_language(text: str) -> str:
    """Return ``"de"`` or ``"en"``. Umlauts and German function words tip the balance towards German."""
    words = _WORD.findall(text.lower())
    de = sum(w in _DE for w in words) + 2 * len(re.findall(r"[äöüß]", text.lower()))
    en = sum(w in _EN for w in words)
    return "de" if de > en else "en"

"""Lightweight sentence splitting without external model downloads."""

import re

# Abbreviations that commonly end with a period in financial English text and must not end a sentence.
_ABBREVIATIONS = {
    "inc",
    "corp",
    "co",
    "ltd",
    "plc",
    "llc",
    "bv",
    "nv",
    "ag",
    "sa",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "no",
    "vs",
    "etc",
    "approx",
    "est",
    "fig",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
    "st",
    "jr",
    "sr",
}

_CANDIDATE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences.

    Paragraph breaks always end a sentence. Inside a paragraph a sentence ends at ``.``, ``!`` or ``?``
    followed by whitespace and an upper-case letter or digit, unless the period belongs to a known
    abbreviation (``Inc.``, ``U.S.``) or a single initial (``J. Smith``).
    """
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        start = 0
        for match in _CANDIDATE.finditer(paragraph):
            head = paragraph[start : match.start()]
            last_word = head.rsplit(" ", 1)[-1].rstrip(".!?\"')]").lower()
            if head.endswith(".") and (last_word in _ABBREVIATIONS or len(last_word) == 1):
                continue
            sentences.append(paragraph[start : match.start()].strip())
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences

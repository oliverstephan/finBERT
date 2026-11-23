"""Rule-based ESG tagging for the FinBERT path (FinBERT itself knows no E/S/G).

This is a transparent keyword list, deliberately simple: it is meant for the demo and as a fallback when Claude is
off. With Claude switched on, Claude's own E/S/G classification is used instead.
"""

import re

_KEYWORDS = {
    "E": [
        "emission",
        "co2",
        "carbon",
        "methane",
        "climate",
        "pollution",
        "spill",
        "waste",
        "environmental",
        "deforestation",
        "water",
        "coal",
        "hydrogen",
        "renewable",
        "emissionen",
        "umwelt",
        "klima",
        "methan",
        "verschmutzung",
        "wasserstoff",
        "kohle",
        "abfall",
    ],
    "S": [
        "strike",
        "union",
        "safety",
        "labour",
        "labor",
        "worker",
        "employee",
        "human rights",
        "child labour",
        "working conditions",
        "layoff",
        "injur",
        "fatal",
        "streik",
        "gewerkschaft",
        "arbeitsschutz",
        "arbeitsbedingungen",
        "beschäftigte",
        "mitarbeiter",
        "stellenabbau",
        "menschenrechte",
        "unfall",
    ],
    "G": [
        "board",
        "compliance",
        "fraud",
        "bribery",
        "corruption",
        "audit",
        "governance",
        "chief executive",
        "ceo",
        "cfo",
        "resign",
        "investigation into management",
        "money laundering",
        "sanction",
        "vorstand",
        "aufsichtsrat",
        "betrug",
        "korruption",
        "bestechung",
        "rücktritt",
        "geldwäsche",
        "wirtschaftsprüfer",
        "bilanzmanipulation",
    ],
}
_PATTERNS = {cat: re.compile("|".join(re.escape(k) for k in kws), re.IGNORECASE) for cat, kws in _KEYWORDS.items()}


def esg_category(sentence: str) -> str | None:
    """Return "E", "S" or "G" for the category with most keyword hits, or None."""
    hits = {cat: len(p.findall(sentence)) for cat, p in _PATTERNS.items()}
    best = max(hits, key=hits.__getitem__)
    return best if hits[best] else None

"""Document-level aggregation of sentence sentiment.

The document score weights every sentence by how clearly it is *not* neutral, so that a handful of strongly
negative sentences is not diluted by boilerplate (decision of 21.11.2026 in the specification):

    s_i = P(positive) - P(negative)
    S   = sum(w_i * s_i) / sum(w_i),  w_i = 1 - P(neutral)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class HasProbabilities(Protocol):
    positive: float
    negative: float
    neutral: float
    prediction: str


@dataclass(frozen=True)
class DocumentScore:
    weighted: float | None
    mean: float | None
    counts: dict[str, int]

    def to_dict(self) -> dict:
        return {"weighted_score": self.weighted, "mean_score": self.mean, "counts": self.counts}


def document_score(sentences: Sequence[HasProbabilities]) -> DocumentScore:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    if not sentences:
        return DocumentScore(None, None, counts)
    sw = sws = sm = 0.0
    for s in sentences:
        score = s.positive - s.negative
        weight = max(0.0, 1.0 - s.neutral)
        sw += weight
        sws += weight * score
        sm += score
        counts[s.prediction] = counts.get(s.prediction, 0) + 1
    mean = sm / len(sentences)
    # All sentences fully neutral: there is no signal, the weighted score falls back to the mean (≈ 0).
    weighted = sws / sw if sw > 1e-9 else mean
    return DocumentScore(weighted, mean, counts)

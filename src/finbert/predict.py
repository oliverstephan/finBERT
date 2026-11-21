"""Sentence-level sentiment prediction with the FinBERT model from the Hugging Face Hub."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finbert.text import split_sentences

DEFAULT_MODEL = "ProsusAI/finbert"
LABELS = ("positive", "negative", "neutral")

# A classifier receives a batch of sentences and returns, per sentence, a list of {"label", "score"}
# dicts covering all labels. This is exactly what a transformers text-classification pipeline returns
# with ``top_k=None``; tests inject a fake with the same shape.
Classifier = Callable[[list[str]], list[list[dict[str, Any]]]]


@dataclass(frozen=True)
class SentenceSentiment:
    sentence: str
    prediction: str
    positive: float
    negative: float
    neutral: float

    @property
    def sentiment_score(self) -> float:
        """P(positive) - P(negative), in [-1, 1]."""
        return self.positive - self.negative

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sentiment_score": self.sentiment_score}


def _load_pipeline(model: str, device: str | int | None, max_length: int) -> Classifier:
    from transformers import pipeline

    clf = pipeline("text-classification", model=model, device=device, top_k=None)

    def classify(sentences: list[str]) -> list[list[dict[str, Any]]]:
        return clf(sentences, truncation=True, max_length=max_length)

    return classify


class FinBertPredictor:
    """Wraps a text-classification model and scores each sentence of a text."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | int | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        classifier: Classifier | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._classify = classifier or _load_pipeline(model, device, max_length)

    def predict_sentences(self, sentences: Sequence[str]) -> list[SentenceSentiment]:
        results: list[SentenceSentiment] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = list(sentences[i : i + self.batch_size])
            for sentence, scores in zip(batch, self._classify(batch), strict=True):
                probs = {s["label"].lower(): float(s["score"]) for s in scores}
                missing = set(LABELS) - probs.keys()
                if missing:
                    raise ValueError(f"Model output lacks labels {sorted(missing)}; got {sorted(probs)}")
                results.append(
                    SentenceSentiment(
                        sentence=sentence,
                        prediction=max(LABELS, key=probs.__getitem__),
                        positive=probs["positive"],
                        negative=probs["negative"],
                        neutral=probs["neutral"],
                    )
                )
        return results

    def predict(self, text: str) -> list[SentenceSentiment]:
        """Split ``text`` into sentences and score each one."""
        return self.predict_sentences(split_sentences(text))


def write_csv(results: Sequence[SentenceSentiment], path: str | Path) -> None:
    fields = ["sentence", "prediction", "sentiment_score", *LABELS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.to_dict()[k] for k in fields})


def predict(text: str, predictor: FinBertPredictor | None = None) -> list[SentenceSentiment]:
    """Convenience wrapper; loads the default model when no predictor is given."""
    return (predictor or FinBertPredictor()).predict(text)

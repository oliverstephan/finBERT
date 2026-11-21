import pytest

from finbert.predict import FinBertPredictor


def keyword_classifier(sentences: list[str]) -> list[list[dict]]:
    """Deterministic stand-in for the transformers pipeline (same output shape as top_k=None)."""
    out = []
    for s in sentences:
        low = s.lower()
        if any(w in low for w in ("rose", "growth", "beat", "surged")):
            p = (0.8, 0.1, 0.1)
        elif any(w in low for w in ("fell", "loss", "missed", "decline")):
            p = (0.05, 0.85, 0.1)
        else:
            p = (0.1, 0.1, 0.8)
        # Hub models label in upper or mixed case at times; the predictor must normalise.
        out.append(
            [
                {"label": "Positive", "score": p[0]},
                {"label": "NEGATIVE", "score": p[1]},
                {"label": "neutral", "score": p[2]},
            ]
        )
    return out


@pytest.fixture
def fake_predictor() -> FinBertPredictor:
    return FinBertPredictor("fake/model", classifier=keyword_classifier, batch_size=2)

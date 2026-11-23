from types import SimpleNamespace

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


class FakeClaudeClient:
    """Mimics ``anthropic.Anthropic().beta.messages.parse`` with keyword rules (German and English)."""

    def __init__(self, refuse: bool = False) -> None:
        self.calls: list[dict] = []
        self.refuse = refuse
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        from finbert.claude import _Ratings, _SentenceRating

        self.calls.append(kwargs)
        text = kwargs["messages"][0]["content"]
        lines = [ln for ln in text.splitlines()[1:] if ln.strip()]
        ratings = []
        for ln in lines:
            idx, sentence = ln.split(". ", 1)
            low = sentence.lower()
            if any(w in low for w in ("rose", "growth", "beat", "stieg", "anheb", "gesichert")):
                lab, p = "positive", (0.9, 0.05, 0.05)
            elif any(w in low for w in ("fell", "loss", "senkt", "rückgang", "emission", "streik")):
                lab, p = "negative", (0.05, 0.9, 0.05)
            else:
                lab, p = "neutral", (0.1, 0.1, 0.8)
            esg = "E" if "emission" in low else ("S" if "streik" in low else None)
            ratings.append(
                _SentenceRating(
                    index=int(idx),
                    label=lab,
                    positive=p[0],
                    negative=p[1],
                    neutral=p[2],
                    reasoning=f"Begründung {idx}",
                    esg=esg,
                )
            )
        return SimpleNamespace(
            stop_reason="refusal" if self.refuse else "end_turn",
            parsed_output=None if self.refuse else _Ratings(sentences=ratings),
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
            model=kwargs["model"],
        )


@pytest.fixture
def fake_claude_client() -> FakeClaudeClient:
    return FakeClaudeClient()


@pytest.fixture
def fake_claude(fake_claude_client):
    from finbert.claude import ClaudeAnalyzer

    return ClaudeAnalyzer("claude-opus-5-5", client=fake_claude_client, max_sentences_per_request=3)

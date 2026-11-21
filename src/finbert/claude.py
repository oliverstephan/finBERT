"""Sentence sentiment with Claude (Anthropic API): German and English, with a short reasoning per sentence.

FinBERT stays the default. Claude is only called when a user asks for it (decision of 21.11.2026), because every
call costs money and sends text to an external API.

All sentences of one document go into a single request, numbered, and Claude returns one structured entry per
sentence. The probabilities Claude returns are its own confidence estimates, not calibrated model outputs like
FinBERT's softmax; the demo labels them accordingly.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
LABELS = ("positive", "negative", "neutral")

# USD per million tokens (input, output), Anthropic list prices as of November 2026. Used for the cost figures
# in the demo and the benchmark; update when prices change.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

SYSTEM_PROMPT = """\
You are a credit and ESG analyst at a European bank. You rate the sentiment of individual sentences from news, \
ad-hoc disclosures and reports from the perspective of a lender or investor in the company concerned.

For each numbered sentence return:
- label: "positive", "negative" or "neutral". Neutral means the sentence carries no information that changes the \
view on creditworthiness, earnings or risk (dates, descriptions, procedural notes).
- positive, negative, neutral: your confidence for each label, summing to 1.
- reasoning: one short sentence in German, at most 25 words, naming the concrete reason for the label (for \
example: guidance cut, rating action, secured financing). Write for a credit officer, no filler.
- esg: "E", "S" or "G" if the sentence is mainly about an environmental, social or governance topic, otherwise null.

Rate each sentence on its own, using the other sentences only as context. Return exactly one entry per sentence, \
in order, with the sentence number as index."""


class _SentenceRating(BaseModel):
    index: int
    label: Literal["positive", "negative", "neutral"]
    positive: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    reasoning: str
    esg: Literal["E", "S", "G"] | None = None


class _Ratings(BaseModel):
    sentences: list[_SentenceRating]


@dataclass(frozen=True)
class ClaudeSentence:
    sentence: str
    prediction: str
    positive: float
    negative: float
    neutral: float
    reasoning: str
    esg: str | None = None

    @property
    def sentiment_score(self) -> float:
        return self.positive - self.negative

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sentiment_score": self.sentiment_score}


@dataclass
class ClaudeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    seconds: float = 0.0
    model: str = ""
    cost_usd: float = 0.0
    served_by: set[str] = field(default_factory=set)

    def add(self, other: ClaudeUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.requests += other.requests
        self.seconds += other.seconds
        self.cost_usd += other.cost_usd
        self.served_by |= other.served_by
        self.model = self.model or other.model

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["served_by"] = sorted(self.served_by)
        return d


class ClaudeUnavailable(RuntimeError):
    """Raised when no API credentials are configured or the API refuses the request."""


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES_PER_MTOK.get(model, PRICES_PER_MTOK[DEFAULT_CLAUDE_MODEL])
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def credentials_configured() -> bool:
    """Cheap check used by /health and the demo; the SDK may still find a profile when this is False."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _normalise(r: _SentenceRating) -> tuple[float, float, float]:
    total = r.positive + r.negative + r.neutral
    if total <= 0:
        return tuple(1.0 if lab == r.label else 0.0 for lab in LABELS)  # type: ignore[return-value]
    return r.positive / total, r.negative / total, r.neutral / total


class ClaudeAnalyzer:
    """Rates sentences with Claude. ``client`` may be injected (tests use a fake with the same shape)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
        effort: str = "low",
        max_sentences_per_request: int = 40,
    ) -> None:
        self.model = model or os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        self.effort = effort
        self.max_sentences_per_request = max_sentences_per_request
        self._client = client

    @property
    def configured(self) -> bool:
        return self._client is not None or credentials_configured()

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _rate_batch(self, sentences: Sequence[str]) -> tuple[list[ClaudeSentence], ClaudeUsage]:
        import anthropic

        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        started = time.perf_counter()
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Sentences:\n{numbered}"}],
                output_config={"effort": self.effort},
                output_format=_Ratings,
                # On a safety decline the API re-runs the request on Anthropic's recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise ClaudeUnavailable("Kein gültiger Anthropic-API-Schlüssel konfiguriert (ANTHROPIC_API_KEY).") from e
        except anthropic.APIConnectionError as e:
            raise ClaudeUnavailable("Anthropic-API nicht erreichbar.") from e
        except anthropic.APIStatusError as e:
            raise ClaudeUnavailable(f"Anthropic-API meldet Fehler {e.status_code}: {e.message}") from e
        except TypeError as e:  # no credentials at all: the SDK refuses to build auth headers
            raise ClaudeUnavailable("Kein Anthropic-API-Schlüssel konfiguriert (ANTHROPIC_API_KEY).") from e
        elapsed = time.perf_counter() - started

        if response.stop_reason == "refusal":
            raise ClaudeUnavailable("Claude hat die Bewertung abgelehnt.")
        parsed: _Ratings | None = response.parsed_output
        if parsed is None:
            raise ClaudeUnavailable(f"Claude lieferte keine auswertbare Antwort (stop_reason={response.stop_reason}).")

        by_index = {r.index: r for r in parsed.sentences}
        out: list[ClaudeSentence] = []
        for i, sentence in enumerate(sentences, start=1):
            r = by_index.get(i)
            if r is None:
                out.append(ClaudeSentence(sentence, "neutral", 0.0, 0.0, 1.0, "Keine Bewertung von Claude erhalten."))
                continue
            p, n, u = _normalise(r)
            out.append(ClaudeSentence(sentence, r.label, p, n, u, r.reasoning.strip(), r.esg))

        model = getattr(response, "model", self.model) or self.model
        usage = ClaudeUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            requests=1,
            seconds=elapsed,
            model=self.model,
            served_by={model},
        )
        usage.cost_usd = cost_usd(model, usage.input_tokens, usage.output_tokens)
        return out, usage

    def rate_sentences(self, sentences: Sequence[str]) -> tuple[list[ClaudeSentence], ClaudeUsage]:
        results: list[ClaudeSentence] = []
        total = ClaudeUsage(model=self.model)
        step = self.max_sentences_per_request
        for i in range(0, len(sentences), step):
            batch, usage = self._rate_batch(sentences[i : i + step])
            results.extend(batch)
            total.add(usage)
        return results, total

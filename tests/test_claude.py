import pytest

from finbert.claude import ClaudeAnalyzer, ClaudeUnavailable, cost_usd
from tests.conftest import FakeClaudeClient


def test_rates_each_sentence_in_order(fake_claude, fake_claude_client):
    sents = ["Umsatz stieg kräftig.", "Die Firma senkt die Prognose.", "Die HV ist im Mai.", "Profit rose."]
    results, usage = fake_claude.rate_sentences(sents)
    assert [r.prediction for r in results] == ["positive", "negative", "neutral", "positive"]
    assert [r.sentence for r in results] == sents
    assert results[0].reasoning == "Begründung 1"
    # batches of 3 sentences -> 2 requests, usage summed
    assert usage.requests == 2 and len(fake_claude_client.calls) == 2
    assert usage.input_tokens == 2000 and usage.output_tokens == 400
    assert usage.cost_usd == pytest.approx(2 * cost_usd("claude-opus-5-5", 1000, 200))


def test_request_shape(fake_claude, fake_claude_client):
    fake_claude.rate_sentences(["Profit rose."])
    call = fake_claude_client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in call["betas"]
    assert call["output_config"] == {"effort": "low"}
    assert "1. Profit rose." in call["messages"][0]["content"]


def test_refusal_raises():
    analyzer = ClaudeAnalyzer(client=FakeClaudeClient(refuse=True))
    with pytest.raises(ClaudeUnavailable):
        analyzer.rate_sentences(["Profit rose."])


def test_cost_for_known_model():
    assert cost_usd("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)

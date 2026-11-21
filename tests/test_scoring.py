from types import SimpleNamespace as S

import pytest

from finbert.esg import esg_category
from finbert.lang import detect_language
from finbert.scoring import document_score


def _s(p, n, u):
    lab = max(("positive", p), ("negative", n), ("neutral", u), key=lambda t: t[1])[0]
    return S(positive=p, negative=n, neutral=u, prediction=lab)


def test_weighted_score_keeps_signal_against_neutral_boilerplate():
    sents = [_s(0.02, 0.93, 0.05), _s(0.03, 0.9, 0.07)] + [_s(0.02, 0.02, 0.96)] * 8
    doc = document_score(sents)
    assert doc.mean == pytest.approx(-0.178, abs=1e-3)
    assert doc.weighted < -0.7
    assert doc.counts == {"positive": 0, "negative": 2, "neutral": 8}


def test_empty_and_all_neutral():
    assert document_score([]).weighted is None
    assert document_score([_s(0, 0, 1)]).weighted == 0


def test_language_detection():
    assert detect_language("Die Gesellschaft senkt ihre Prognose für das Gesamtjahr.") == "de"
    assert detect_language("The company cut its guidance for the year.") == "en"


def test_esg_keywords():
    assert esg_category("Regulators opened an investigation into emissions reporting.") == "E"
    assert esg_category("Die Gewerkschaft kündigt Streiks an.") == "S"
    assert esg_category("Der Finanzvorstand tritt zurück, Rücktritt mit sofortiger Wirkung.") == "G"
    assert esg_category("Revenue rose 5 percent.") is None

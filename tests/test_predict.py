import csv

import pytest

from finbert.predict import FinBertPredictor, write_csv


def test_predict_scores_each_sentence(fake_predictor):
    results = fake_predictor.predict("Revenue rose 10%. The company missed estimates. The CEO spoke.")
    assert [r.prediction for r in results] == ["positive", "negative", "neutral"]
    assert results[0].sentiment_score == pytest.approx(0.7)
    assert results[1].sentiment_score == pytest.approx(-0.8)
    assert results[2].to_dict()["sentiment_score"] == pytest.approx(0.0)


def test_batching_preserves_order(fake_predictor):
    sentences = [f"Item {i} rose." if i % 2 else f"Item {i} fell." for i in range(5)]
    results = fake_predictor.predict_sentences(sentences)
    assert [r.sentence for r in results] == sentences


def test_missing_label_is_an_error():
    predictor = FinBertPredictor("x", classifier=lambda b: [[{"label": "positive", "score": 1.0}] for _ in b])
    with pytest.raises(ValueError, match="lacks labels"):
        predictor.predict("Hello.")


def test_write_csv(tmp_path, fake_predictor):
    path = tmp_path / "out.csv"
    write_csv(fake_predictor.predict("Sales rose. Margins fell."), path)
    rows = list(csv.DictReader(path.open()))
    assert [r["prediction"] for r in rows] == ["positive", "negative"]
    assert set(rows[0]) == {"sentence", "prediction", "sentiment_score", "positive", "negative", "neutral"}


@pytest.mark.slow
def test_real_model_from_hub():
    predictor = FinBertPredictor()
    results = predictor.predict(
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn. The company's net loss widened sharply."
    )
    assert [r.prediction for r in results] == ["positive", "negative"]

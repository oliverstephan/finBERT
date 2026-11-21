import pytest
from fastapi.testclient import TestClient

from finbert.api import create_app


@pytest.fixture
def client(fake_predictor):
    with TestClient(create_app(lambda: fake_predictor)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model"] == "fake/model"


def test_predict(client):
    r = client.post("/predict", json={"text": "Net income rose. Shares fell sharply."})
    assert r.status_code == 200
    body = r.json()
    assert [s["prediction"] for s in body["sentences"]] == ["positive", "negative"]
    assert body["average_sentiment_score"] == pytest.approx((0.7 - 0.8) / 2)


def test_predict_whole_text(client):
    r = client.post("/predict", json={"text": "Net income rose. Shares fell.", "split_sentences": False})
    assert len(r.json()["sentences"]) == 1


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}])
def test_predict_rejects_empty(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_openapi_docs(client):
    assert client.get("/openapi.json").json()["info"]["title"] == "FinBERT Sentiment API"

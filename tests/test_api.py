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


@pytest.fixture
def client2(fake_predictor, fake_claude):
    with TestClient(create_app(lambda: fake_predictor, lambda: fake_claude)) as c:
        yield c


def test_demo_page(client2):
    r = client2.get("/")
    assert r.status_code == 200 and "<title>" in r.text


def test_sentiment_finbert_document_score(client2):
    r = client2.post("/v1/sentiment", json={"text": "Net income rose. The AGM is in May."})
    body = r.json()
    assert body["engine"] == "finbert" and body["language"] == "en"
    assert body["document"]["counts"]["positive"] == 1
    assert body["document"]["weighted_score"] > body["document"]["mean_score"]
    assert body["sentences"][0]["reasoning"] is None


def test_sentiment_claude_german(client2):
    r = client2.post(
        "/v1/sentiment", json={"text": "Die Firma senkt die Prognose. Die HV ist im Mai.", "engine": "claude"}
    )
    body = r.json()
    assert body["language"] == "de" and body["engine"] == "claude"
    assert body["sentences"][0]["prediction"] == "negative"
    assert body["sentences"][0]["reasoning"]
    assert body["usage"]["cost_usd"] > 0


def test_finbert_warns_on_german(client2):
    r = client2.post("/v1/sentiment", json={"text": "Die Gesellschaft senkt ihre Prognose für das Jahr."})
    assert r.json()["warnings"]


def test_compare_reports_claude_error(fake_predictor):
    from finbert.claude import ClaudeAnalyzer
    from tests.conftest import FakeClaudeClient

    refusing = ClaudeAnalyzer(client=FakeClaudeClient(refuse=True))
    with TestClient(create_app(lambda: fake_predictor, lambda: refusing)) as c:
        body = c.post("/v1/compare", json={"text": "Net income rose."}).json()
    assert body["finbert"]["sentences"]
    assert "error" in body["claude"]


def test_sentiment_claude_unavailable_is_503(fake_predictor):
    from finbert.claude import ClaudeAnalyzer
    from tests.conftest import FakeClaudeClient

    refusing = ClaudeAnalyzer(client=FakeClaudeClient(refuse=True))
    with TestClient(create_app(lambda: fake_predictor, lambda: refusing)) as c:
        assert c.post("/v1/sentiment", json={"text": "x rose.", "engine": "claude"}).status_code == 503


def test_examples_and_portfolio(client2):
    assert len(client2.get("/v1/examples").json()) >= 4
    p = client2.get("/v1/portfolio").json()
    assert p["engine"] == "finbert"
    first = p["counterparties"][0]["news"][0]
    assert "score" in first and first["sentences"]


def test_benchmark_missing_then_present(client2, tmp_path, monkeypatch):
    monkeypatch.setenv("FINBERT_BENCHMARK_RESULTS", str(tmp_path / "none.json"))
    assert client2.get("/v1/benchmark").status_code == 404
    (tmp_path / "r.json").write_text('{"datasets": {}}')
    monkeypatch.setenv("FINBERT_BENCHMARK_RESULTS", str(tmp_path / "r.json"))
    assert client2.get("/v1/benchmark").json() == {"datasets": {}}


def test_health_reports_claude(client2):
    body = client2.get("/health").json()
    assert body["claude_model"] == "claude-opus-5-5" and body["claude_configured"] is True

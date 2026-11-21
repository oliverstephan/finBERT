"""FastAPI service exposing FinBERT sentiment predictions."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from finbert import __version__
from finbert.analysis import Analyzer
from finbert.claude import ClaudeAnalyzer, ClaudeUnavailable
from finbert.predict import DEFAULT_MODEL, FinBertPredictor

WEB = resources.files("finbert") / "web"
DEFAULT_BENCHMARK_RESULTS = "benchmark/results/latest.json"

MAX_TEXT_CHARS = 100_000


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS, examples=["Revenue rose 12% year on year."])
    split_sentences: bool = Field(True, description="Score each sentence separately instead of the whole text")


class SentenceResult(BaseModel):
    sentence: str
    prediction: str
    sentiment_score: float = Field(..., description="P(positive) - P(negative)")
    positive: float
    negative: float
    neutral: float


class PredictResponse(BaseModel):
    model: str
    sentences: list[SentenceResult]
    average_sentiment_score: float | None


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    claude_model: str | None = None
    claude_configured: bool = False


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    engine: Literal["finbert", "claude"] = "finbert"
    language: Literal["auto", "de", "en"] = "auto"


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    language: Literal["auto", "de", "en"] = "auto"


def _load_json(name: str) -> Any:
    return json.loads((WEB / name).read_text(encoding="utf-8"))


def create_app(
    predictor_factory: Callable[[], FinBertPredictor] | None = None,
    claude_factory: Callable[[], ClaudeAnalyzer] | None = None,
) -> FastAPI:
    """Build the app. The predictor is created once at startup; tests inject fake factories."""
    factory = predictor_factory or (
        lambda: FinBertPredictor(
            os.environ.get("FINBERT_MODEL", DEFAULT_MODEL), device=os.environ.get("FINBERT_DEVICE")
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = factory()
        app.state.analyzer = Analyzer(app.state.predictor, (claude_factory or ClaudeAnalyzer)())
        app.state.portfolio_cache = {}
        yield

    app = FastAPI(title="FinBERT Sentiment API", version=__version__, lifespan=lifespan)
    origins = [o for o in os.environ.get("FINBERT_CORS_ORIGINS", "*").split(",") if o]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    def get_predictor(request: Request) -> FinBertPredictor:
        return request.app.state.predictor

    def get_analyzer(request: Request) -> Analyzer:
        return request.app.state.analyzer

    def check_text(text: str) -> None:
        if not text.strip():
            raise HTTPException(status_code=422, detail="text must not be blank")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        claude = get_analyzer(request).claude
        return HealthResponse(
            status="ok",
            model=get_predictor(request).model,
            version=__version__,
            claude_model=claude.model,
            claude_configured=claude.configured,
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> str:
        return (WEB / "index.html").read_text(encoding="utf-8")

    @app.post("/v1/sentiment")
    def sentiment(body: AnalyzeRequest, request: Request) -> dict[str, Any]:
        """Sentence results and weighted document score from one engine."""
        check_text(body.text)
        try:
            return get_analyzer(request).analyze(body.text, body.engine, body.language)
        except ClaudeUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    @app.post("/v1/compare")
    def compare(body: CompareRequest, request: Request) -> dict[str, Any]:
        """Both engines on the same text. A Claude failure is reported in place, FinBERT still answers."""
        check_text(body.text)
        analyzer = get_analyzer(request)
        out: dict[str, Any] = {"finbert": analyzer.analyze(body.text, "finbert", body.language)}
        try:
            out["claude"] = analyzer.analyze(body.text, "claude", body.language)
        except ClaudeUnavailable as e:
            out["claude"] = {"error": str(e)}
        return out

    @app.get("/v1/examples")
    def examples() -> list[dict[str, Any]]:
        """Demo texts for the business paths (credit early warning, ad-hoc, ESG, German)."""
        return _load_json("examples.json")

    @app.get("/v1/portfolio")
    def portfolio(request: Request, engine: Literal["finbert", "claude"] = "finbert") -> dict[str, Any]:
        """Fictional demo portfolio: every news item scored, grouped by counterparty. Cached per engine."""
        cache = request.app.state.portfolio_cache
        if engine not in cache:
            analyzer = get_analyzer(request)
            data = _load_json("portfolio.json")
            try:
                for cp in data["counterparties"]:
                    for item in cp["news"]:
                        res = analyzer.analyze(item["text"], engine, item.get("language", "auto"))
                        item["score"] = res["document"]["weighted_score"]
                        item["sentences"] = res["sentences"]
                        item["model"] = res["model"]
            except ClaudeUnavailable as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            data["engine"] = engine
            cache[engine] = data
        return cache[engine]

    @app.get("/v1/benchmark")
    def benchmark() -> dict[str, Any]:
        """Latest benchmark run (written by ``finbert benchmark``)."""
        path = Path(os.environ.get("FINBERT_BENCHMARK_RESULTS", DEFAULT_BENCHMARK_RESULTS))
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Noch kein Benchmark-Lauf ({path} fehlt).")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        predictor = get_predictor(request)
        if not body.text.strip():
            raise HTTPException(status_code=422, detail="text must not be blank")
        results = (
            predictor.predict(body.text) if body.split_sentences else predictor.predict_sentences([body.text.strip()])
        )
        sentences = [SentenceResult(**r.to_dict()) for r in results]
        avg = sum(s.sentiment_score for s in sentences) / len(sentences) if sentences else None
        return PredictResponse(model=predictor.model, sentences=sentences, average_sentiment_score=avg)

    return app

"""FastAPI service exposing FinBERT sentiment predictions."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from finbert import __version__
from finbert.predict import DEFAULT_MODEL, FinBertPredictor

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


def create_app(predictor_factory: Callable[[], FinBertPredictor] | None = None) -> FastAPI:
    """Build the app. The predictor is created once at startup; tests inject a fake factory."""
    factory = predictor_factory or (
        lambda: FinBertPredictor(
            os.environ.get("FINBERT_MODEL", DEFAULT_MODEL), device=os.environ.get("FINBERT_DEVICE")
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = factory()
        yield

    app = FastAPI(title="FinBERT Sentiment API", version=__version__, lifespan=lifespan)
    origins = [o for o in os.environ.get("FINBERT_CORS_ORIGINS", "*").split(",") if o]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    def get_predictor(request: Request) -> FinBertPredictor:
        return request.app.state.predictor

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", model=get_predictor(request).model, version=__version__)

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

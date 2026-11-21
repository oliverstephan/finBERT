"""Analysis core shared by the API, the demo and the benchmark.

One entry point, :meth:`Analyzer.analyze`, turns a text into sentence results plus a document score, either with
FinBERT (local, English) or with Claude (API, German and English, with reasoning). Every result carries engine,
model, version and timestamp, so a score can be traced later.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from finbert import __version__
from finbert.claude import ClaudeAnalyzer
from finbert.esg import esg_category
from finbert.lang import detect_language
from finbert.predict import FinBertPredictor
from finbert.scoring import document_score
from finbert.text import split_sentences

ENGINES = ("finbert", "claude")


class Analyzer:
    def __init__(self, finbert: FinBertPredictor, claude: ClaudeAnalyzer | None = None) -> None:
        self.finbert = finbert
        self.claude = claude or ClaudeAnalyzer()

    def analyze_sentences(self, sentences: list[str], engine: str, language: str) -> dict[str, Any]:
        started = time.perf_counter()
        warnings: list[str] = []
        usage = None
        if engine == "finbert":
            if language != "en":
                warnings.append(
                    "FinBERT ist nur auf englischen Texten trainiert. "
                    "Ergebnisse für deutsche Texte sind nicht belastbar."
                )
            raw = self.finbert.predict_sentences(sentences)
            rows = [{**r.to_dict(), "reasoning": None, "esg": esg_category(r.sentence)} for r in raw]
            model = self.finbert.model
            esg_source = "keywords"
        elif engine == "claude":
            raw, claude_usage = self.claude.rate_sentences(sentences)
            rows = [r.to_dict() for r in raw]
            usage = claude_usage.to_dict()
            model = self.claude.model
            esg_source = "claude"
        else:
            raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")

        for row in rows:
            row["weight"] = max(0.0, 1.0 - row["neutral"])
        doc = document_score(raw)
        return {
            "engine": engine,
            "model": model,
            "language": language,
            "warnings": warnings,
            "sentences": rows,
            "document": doc.to_dict(),
            "esg_source": esg_source,
            "usage": usage,
            "seconds": time.perf_counter() - started,
            "version": __version__,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def analyze(self, text: str, engine: str = "finbert", language: str = "auto") -> dict[str, Any]:
        lang = detect_language(text) if language == "auto" else language
        sentences = split_sentences(text)
        return self.analyze_sentences(sentences, engine, lang)

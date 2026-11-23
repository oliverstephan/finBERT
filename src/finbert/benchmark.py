"""Benchmark FinBERT against Claude on the Financial PhraseBank and the German reference set.

Measured per engine and dataset:
- quality: accuracy, macro-F1, F1 per class, confusion matrix, negative/neutral mix-ups
- cost and speed: seconds and USD per 1,000 sentences (FinBERT: hardware only, no API cost)
- explainability: share of sentences with a reasoning, plus example sentences where the engines disagree

The German set is a *checked reference*, not an independent gold standard: Claude proposed the labels and a
domain expert reviews them. Claude's German score is therefore biased upwards; the report says so.

Financial PhraseBank (Malo et al. 2014) is licensed CC BY-NC-SA 3.0 and is not shipped with this repository. Pass a
local ``Sentences_*.txt`` file or let the loader download the original zip from the Hugging Face Hub.
"""

from __future__ import annotations

import csv
import io
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finbert import __version__
from finbert.claude import ClaudeAnalyzer
from finbert.predict import FinBertPredictor

LABELS = ("positive", "negative", "neutral")
PHRASEBANK_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
PHRASEBANK_FILES = {
    "all": "Sentences_AllAgree.txt",
    "75": "Sentences_75Agree.txt",
    "66": "Sentences_66Agree.txt",
    "50": "Sentences_50Agree.txt",
}
DEFAULT_GERMAN = Path("benchmark/data/german_reference.csv")
CACHE = Path.home() / ".cache" / "finbert"


@dataclass(frozen=True)
class Item:
    sentence: str
    label: str


# ---------------------------------------------------------------------------------------------- data loading


def _parse_phrasebank(raw: str) -> list[Item]:
    items = []
    for line in raw.splitlines():
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        label = label.strip().lower()
        if label in LABELS:
            items.append(Item(sentence.strip(), label))
    return items


def load_phrasebank(path: str | Path | None = None, agreement: str = "all") -> list[Item]:
    """Load a PhraseBank file (``sentence@label`` per line, Latin-1). Downloads the original zip when no path."""
    if path is not None:
        return _parse_phrasebank(Path(path).read_bytes().decode("latin-1"))
    CACHE.mkdir(parents=True, exist_ok=True)
    zpath = CACHE / "FinancialPhraseBank-v1.0.zip"
    if not zpath.exists():
        urllib.request.urlretrieve(PHRASEBANK_URL, zpath)  # noqa: S310 - fixed https URL
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(PHRASEBANK_FILES[agreement]))
        return _parse_phrasebank(z.read(name).decode("latin-1"))


def load_german(path: str | Path = DEFAULT_GERMAN) -> tuple[list[Item], dict[str, int]]:
    """Load the German reference set. Returns the items and a count per review status."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    status = Counter(r.get("status", "") or "unbekannt" for r in rows)
    return [Item(r["sentence"].strip(), r["label"].strip().lower()) for r in rows], dict(status)


def stratified_sample(items: Sequence[Item], n: int | None, seed: int = 42) -> list[Item]:
    """Sample ``n`` items keeping the label mix. ``None`` or ``n >= len`` keeps everything."""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    by_label: dict[str, list[Item]] = {lab: [i for i in items if i.label == lab] for lab in LABELS}
    out: list[Item] = []
    for group in by_label.values():
        k = round(n * len(group) / len(items))
        out.extend(rng.sample(group, min(k, len(group))))
    rng.shuffle(out)
    return out[:n]


# --------------------------------------------------------------------------------------------------- metrics


def metrics(gold: Sequence[str], pred: Sequence[str]) -> dict[str, Any]:
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred, strict=True):
        confusion[g][p] += 1
    f1 = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in LABELS if g != lab)
        fn = sum(confusion[lab][p] for p in LABELS if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n = len(gold)
    return {
        "n": n,
        "accuracy": sum(g == p for g, p in zip(gold, pred, strict=True)) / n if n else 0.0,
        "macro_f1": sum(f1.values()) / len(LABELS),
        "f1": f1,
        "confusion": confusion,
        "neg_neu_confusions": confusion["negative"]["neutral"] + confusion["neutral"]["negative"],
    }


# ------------------------------------------------------------------------------------------------- execution


def run_finbert(predictor: FinBertPredictor, items: Sequence[Item]) -> dict[str, Any]:
    started = time.perf_counter()
    results = predictor.predict_sentences([i.sentence for i in items])
    seconds = time.perf_counter() - started
    return {
        "predictions": [r.prediction for r in results],
        "reasonings": [None] * len(items),
        "seconds": seconds,
        "cost_usd": 0.0,
        "model": predictor.model,
        "usage": None,
    }


def run_claude(analyzer: ClaudeAnalyzer, items: Sequence[Item]) -> dict[str, Any]:
    results, usage = analyzer.rate_sentences([i.sentence for i in items])
    return {
        "predictions": [r.prediction for r in results],
        "reasonings": [r.reasoning for r in results],
        "seconds": usage.seconds,
        "cost_usd": usage.cost_usd,
        "model": analyzer.model,
        "usage": usage.to_dict(),
    }


def _engine_report(items: Sequence[Item], run: dict[str, Any]) -> dict[str, Any]:
    n = len(items)
    m = metrics([i.label for i in items], run["predictions"])
    with_reason = sum(1 for r in run["reasonings"] if r)
    return {
        "model": run["model"],
        **m,
        "seconds_per_1000": run["seconds"] / n * 1000 if n else 0.0,
        "cost_usd_per_1000": run["cost_usd"] / n * 1000 if n else 0.0,
        "explained_share": with_reason / n if n else 0.0,
        "usage": run["usage"],
    }


def _disagreements(items: Sequence[Item], runs: dict[str, dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    if not {"finbert", "claude"} <= runs.keys():
        return []
    out = []
    fb, cl = runs["finbert"], runs["claude"]
    for i, item in enumerate(items):
        if fb["predictions"][i] != cl["predictions"][i]:
            out.append(
                {
                    "sentence": item.sentence,
                    "gold": item.label,
                    "finbert": fb["predictions"][i],
                    "claude": cl["predictions"][i],
                    "claude_reasoning": cl["reasonings"][i],
                }
            )
    # Cases where Claude is right and FinBERT wrong first: they are the ones a business audience asks about.
    out.sort(key=lambda d: (d["claude"] != d["gold"], d["finbert"] == d["gold"]))
    return out[:limit]


def run_benchmark(
    datasets: dict[str, tuple[list[Item], dict[str, Any]]],
    *,
    finbert: FinBertPredictor | None = None,
    claude: ClaudeAnalyzer | None = None,
) -> dict[str, Any]:
    """Run every given engine on every dataset. ``datasets`` maps a name to (items, metadata)."""
    report: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": __version__,
        "datasets": {},
    }
    for name, (items, meta) in datasets.items():
        runs: dict[str, dict[str, Any]] = {}
        if finbert is not None:
            runs["finbert"] = run_finbert(finbert, items)
        if claude is not None:
            runs["claude"] = run_claude(claude, items)
        report["datasets"][name] = {
            **meta,
            "n": len(items),
            "label_mix": dict(Counter(i.label for i in items)),
            "engines": {eng: _engine_report(items, run) for eng, run in runs.items()},
            "disagreements": _disagreements(items, runs),
        }
    return report


def write_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def phrasebank_from_bytes(data: bytes, agreement: str = "all") -> list[Item]:
    """Parse the PhraseBank zip from memory (used by tests and when the zip was fetched elsewhere)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith(PHRASEBANK_FILES[agreement]))
        return _parse_phrasebank(z.read(name).decode("latin-1"))

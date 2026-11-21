import io
import json
import zipfile

import pytest

from finbert.benchmark import (
    Item,
    load_german,
    metrics,
    phrasebank_from_bytes,
    run_benchmark,
    stratified_sample,
    write_report,
)


def test_metrics_perfect_and_confusion():
    m = metrics(["positive", "negative", "neutral"], ["positive", "neutral", "neutral"])
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["confusion"]["negative"]["neutral"] == 1
    assert m["neg_neu_confusions"] == 1
    assert metrics(["neutral"] * 3, ["neutral"] * 3)["f1"]["neutral"] == 1.0


def test_phrasebank_parsing_from_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", "Profit rose .@positive\nSales fell .@negative\n")
    items = phrasebank_from_bytes(buf.getvalue())
    assert items == [Item("Profit rose .", "positive"), Item("Sales fell .", "negative")]


def test_stratified_sample_keeps_mix():
    items = [Item(f"p{i}", "positive") for i in range(50)] + [Item(f"n{i}", "neutral") for i in range(150)]
    sample = stratified_sample(items, 40)
    assert len(sample) == 40
    assert sum(i.label == "positive" for i in sample) == 10


def test_german_reference_set_is_valid():
    items, status = load_german()
    assert len(items) >= 60
    assert {i.label for i in items} == {"positive", "negative", "neutral"}
    assert sum(status.values()) == len(items)


def test_run_benchmark_end_to_end(fake_predictor, fake_claude, tmp_path):
    items = [Item("Net income rose.", "positive"), Item("Shares fell.", "negative"), Item("AGM in May.", "neutral")]
    report = run_benchmark({"toy": (items, {"language": "en"})}, finbert=fake_predictor, claude=fake_claude)
    ds = report["datasets"]["toy"]
    assert ds["engines"]["finbert"]["accuracy"] == 1.0
    assert ds["engines"]["claude"]["explained_share"] == 1.0
    assert ds["engines"]["finbert"]["explained_share"] == 0.0
    assert ds["engines"]["claude"]["cost_usd_per_1000"] > 0
    out = tmp_path / "r.json"
    write_report(report, out)
    assert json.loads(out.read_text())["datasets"]["toy"]["n"] == 3

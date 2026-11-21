from finbert import cli
from finbert.predict import FinBertPredictor
from tests.conftest import keyword_classifier


def test_cli_predict_writes_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "FinBertPredictor", lambda model, **kw: FinBertPredictor(model, classifier=keyword_classifier)
    )
    src = tmp_path / "in.txt"
    src.write_text("Profit rose. Revenue fell.", encoding="utf-8")
    out = tmp_path / "out" / "pred.csv"
    assert cli.main(["predict", "--text-file", str(src), "--out", str(out)]) == 0
    assert out.read_text().count("\n") == 3
    assert "Wrote 2 predictions" in capsys.readouterr().out

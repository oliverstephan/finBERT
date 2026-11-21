"""Command line interface: ``finbert predict`` and ``finbert serve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from finbert.predict import DEFAULT_MODEL, FinBertPredictor, write_csv


def _predict(args: argparse.Namespace) -> int:
    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file != "-" else sys.stdin.read()
    predictor = FinBertPredictor(args.model, device=args.device, batch_size=args.batch_size)
    results = predictor.predict(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_csv(results, args.out)
        print(f"Wrote {len(results)} predictions to {args.out}")
    else:
        for r in results:
            print(f"{r.prediction:<8} {r.sentiment_score:+.3f}  {r.sentence}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("finbert.api:create_app", factory=True, host=args.host, port=args.port)
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from finbert.benchmark import load_german, load_phrasebank, run_benchmark, stratified_sample, write_report
    from finbert.claude import ClaudeAnalyzer

    datasets = {}
    if not args.skip_phrasebank:
        pb = stratified_sample(load_phrasebank(args.phrasebank, args.agreement), args.n, args.seed)
        datasets["phrasebank"] = (
            pb,
            {"title": f"Financial PhraseBank ({args.agreement} agree)", "language": "en", "reference": "gold"},
        )
    if not args.skip_german:
        de, status = load_german(args.german)
        datasets["german"] = (
            stratified_sample(de, args.n_german, args.seed),
            {"title": "Deutscher Referenzsatz", "language": "de", "reference": "geprüfte Referenz", "status": status},
        )
    finbert = None if args.no_finbert else FinBertPredictor(args.model, device=args.device)
    claude = None if args.no_claude else ClaudeAnalyzer(args.claude_model)
    report = run_benchmark(datasets, finbert=finbert, claude=claude)
    write_report(report, args.out)
    for name, ds in report["datasets"].items():
        for eng, r in ds["engines"].items():
            print(
                f"{name:<11} {eng:<8} n={r['n']:<4} acc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f} "
                f"s/1000={r['seconds_per_1000']:.1f} USD/1000={r['cost_usd_per_1000']:.2f}"
            )
    print(f"Wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finbert", description="Financial sentiment analysis with FinBERT")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="Score every sentence of a text file")
    p.add_argument("--text-file", required=True, help="Path to a UTF-8 text file, or - for stdin")
    p.add_argument("--out", help="Write a CSV file instead of printing to stdout")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local directory")
    p.add_argument("--device", default=None, help="e.g. cpu, cuda, mps (default: transformers decides)")
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(func=_predict)

    b = sub.add_parser("benchmark", help="Compare FinBERT and Claude on Financial PhraseBank and German texts")
    b.add_argument("--phrasebank", help="Local Sentences_*.txt; default downloads the original zip from the Hub")
    b.add_argument("--agreement", default="all", choices=["all", "75", "66", "50"])
    b.add_argument("--n", type=int, default=300, help="Stratified sample size for PhraseBank (default 300)")
    b.add_argument("--german", default="benchmark/data/german_reference.csv")
    b.add_argument("--n-german", type=int, default=None, help="Sample size for the German set (default: all)")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out", default="benchmark/results/latest.json")
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--device", default=None)
    b.add_argument("--claude-model", default=None, help="Default: CLAUDE_MODEL or claude-opus-5-5")
    b.add_argument("--no-finbert", action="store_true")
    b.add_argument("--no-claude", action="store_true")
    b.add_argument("--skip-phrasebank", action="store_true")
    b.add_argument("--skip-german", action="store_true")
    b.set_defaults(func=_benchmark)

    s = sub.add_parser("serve", help="Start the REST API (model from FINBERT_MODEL)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

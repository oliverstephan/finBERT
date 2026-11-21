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

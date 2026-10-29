"""Command-line helpers:  python -m mpesa_analyzer.cli generate|train"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import classifier, generator
from .config import DATA_DIR, MODEL_PATH


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mpesa_analyzer.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write synthetic SMS to a text file")
    g.add_argument("--months", type=int, default=6)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", type=Path, default=DATA_DIR / "sample_sms.txt")
    sub.add_parser("train", help="train and save the fallback classifier")
    args = ap.parse_args(argv)

    if args.cmd == "generate":
        df = generator.generate(months=args.months, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(generator.sms_text(df), encoding="utf-8")
        print(f"Wrote {len(df)} synthetic messages to {args.out}")
    else:
        bundle = classifier.ensure_model(retrain=True)
        m = bundle["metrics"]
        print(f"Saved model to {MODEL_PATH}")
        print(f"Holdout accuracy on unseen merchants: {m['holdout_accuracy']:.1%} "
              f"({m['n_train']} train / {m['n_test']} test rows)")


if __name__ == "__main__":
    main()

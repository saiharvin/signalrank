"""Command-line entry points."""

from __future__ import annotations

import argparse
import json

from .data import load_mind, make_demo_dataset
from .pipeline import benchmark_mind, train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="signalrank")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="train a synthetic smoke-test model")
    demo.add_argument("--output", default="artifacts")
    mind = commands.add_parser("train-mind", help="train on extracted MIND files")
    mind.add_argument("--data", required=True)
    mind.add_argument("--output", default="artifacts/mind")
    mind.add_argument("--max-impressions", type=int)
    benchmark = commands.add_parser("benchmark-mind", help="official train/dev benchmark")
    benchmark.add_argument("--train", required=True)
    benchmark.add_argument("--dev", required=True)
    benchmark.add_argument("--output", default="artifacts/mind-benchmark")
    benchmark.add_argument("--max-train-impressions", type=int)
    benchmark.add_argument("--max-dev-impressions", type=int)
    benchmark.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "demo":
        report = train_pipeline(make_demo_dataset(), args.output, "synthetic-demo")
    elif args.command == "train-mind":
        report = train_pipeline(
            load_mind(args.data, args.max_impressions), args.output, "MIND"
        )
    else:
        report = benchmark_mind(
            load_mind(args.train, args.max_train_impressions, "train"),
            load_mind(args.dev, args.max_dev_impressions, "dev"),
            args.output,
            args.bootstrap_samples,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


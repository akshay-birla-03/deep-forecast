"""CLI: train the forecaster on a synthetic panel and report metrics."""
from __future__ import annotations

import argparse
import json

from .data import generate_panel
from .train import ForecastTrainer, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepforecast", description=__doc__)
    p.add_argument("--series", type=int, default=30)
    p.add_argument("--length", type=int, default=240)
    p.add_argument("--input-len", type=int, default=28)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    panel = generate_panel(n_series=args.series, length=args.length, seed=args.seed)
    trainer = ForecastTrainer(
        TrainConfig(
            input_len=args.input_len,
            horizon=args.horizon,
            max_epochs=args.epochs,
            hidden_size=args.hidden,
            seed=args.seed,
        )
    )
    result = trainer.fit(panel)
    print(json.dumps(result.metrics, indent=2))
    print(f"seasonal-naive MAE: {result.naive_mae:.4f}")
    verdict = "BEATS" if result.beats_baseline else "does NOT beat"
    print(f"Model {verdict} the seasonal-naive baseline "
          f"({result.metrics['mae']:.4f} vs {result.naive_mae:.4f}).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

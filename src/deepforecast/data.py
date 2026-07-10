"""Synthetic multi-series demand panel generator.

Produces a panel of related time series, each with trend, weekly and yearly
seasonality, promotional spikes, and noise. Structure is deterministic given a
seed so experiments are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SeriesConfig:
    n_series: int = 20
    length: int = 240
    weekly_period: int = 7
    yearly_period: int = 365
    promo_prob: float = 0.03
    noise_scale: float = 0.08
    seed: int = 0


def generate_panel(
    n_series: int = 20,
    length: int = 240,
    seed: int = 0,
    config: Optional[SeriesConfig] = None,
) -> pd.DataFrame:
    """Return a long-format DataFrame: columns [series_id, t, value, promo].

    Values are strictly positive (demand), with per-series base level, trend,
    weekly + yearly seasonality, occasional promo uplifts, and multiplicative
    noise.
    """
    cfg = config or SeriesConfig(n_series=n_series, length=length, seed=seed)
    rng = np.random.default_rng(cfg.seed)
    rows = []

    for s in range(cfg.n_series):
        base = rng.uniform(20, 100)
        trend = rng.normal(0, 0.05)
        weekly_amp = rng.uniform(0.1, 0.4)
        yearly_amp = rng.uniform(0.05, 0.25)
        phase = rng.uniform(0, 2 * np.pi)

        t = np.arange(cfg.length)
        weekly = weekly_amp * np.sin(2 * np.pi * t / cfg.weekly_period + phase)
        yearly = yearly_amp * np.sin(2 * np.pi * t / cfg.yearly_period)
        level = base + trend * t
        promo = (rng.random(cfg.length) < cfg.promo_prob).astype(float)
        promo_uplift = promo * rng.uniform(0.3, 0.8)
        noise = rng.normal(0, cfg.noise_scale, cfg.length)

        value = level * (1 + weekly + yearly + promo_uplift + noise)
        value = np.clip(value, 1.0, None)

        for ti in range(cfg.length):
            rows.append(
                {
                    "series_id": s,
                    "t": int(ti),
                    "value": float(value[ti]),
                    "promo": float(promo[ti]),
                }
            )

    return pd.DataFrame(rows)

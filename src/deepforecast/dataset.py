"""Windowing a demand panel into (encoder, decoder) supervised examples.

Each example is a sliding window: ``input_len`` historical steps feed the
encoder, and the model predicts the next ``horizon`` steps. Values are scaled
per series using statistics computed on the training portion only (no leakage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Scaler:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std or 1.0)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std or 1.0) + self.mean


class WindowDataset(Dataset):
    """Sliding-window dataset over a long-format panel.

    Known future covariates (calendar position, promo flag) are provided to the
    decoder; the target series history is provided to the encoder.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        input_len: int = 28,
        horizon: int = 7,
        scalers: Dict[int, Scaler] | None = None,
        t_max: int | None = None,
    ) -> None:
        self.input_len = input_len
        self.horizon = horizon
        self.scalers = scalers or {}
        self._examples: List[Tuple[int, int]] = []
        self._series: Dict[int, np.ndarray] = {}
        self._promo: Dict[int, np.ndarray] = {}

        for sid, grp in panel.sort_values(["series_id", "t"]).groupby("series_id"):
            values = grp["value"].to_numpy(dtype=np.float32)
            promo = grp["promo"].to_numpy(dtype=np.float32)
            if t_max is not None:
                # Only build windows whose target ends at or before t_max.
                pass
            self._series[sid] = values
            self._promo[sid] = promo
            if sid not in self.scalers:
                self.scalers[sid] = Scaler(float(values.mean()), float(values.std() or 1.0))
            last_start = len(values) - input_len - horizon
            for start in range(0, last_start + 1):
                self._examples.append((sid, start))

    def __len__(self) -> int:
        return len(self._examples)

    def _calendar(self, positions: np.ndarray) -> np.ndarray:
        # Weekly seasonality encoded as sin/cos of day-of-week.
        dow = positions % 7
        return np.stack(
            [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=-1
        ).astype(np.float32)

    def __getitem__(self, idx: int):
        sid, start = self._examples[idx]
        scaler = self.scalers[sid]
        values = self._series[sid]
        promo = self._promo[sid]

        enc_slice = slice(start, start + self.input_len)
        dec_slice = slice(start + self.input_len, start + self.input_len + self.horizon)

        enc_vals = scaler.transform(values[enc_slice])
        enc_pos = np.arange(start, start + self.input_len)
        enc_cal = self._calendar(enc_pos)
        enc_promo = promo[enc_slice][:, None]
        enc = np.concatenate([enc_vals[:, None], enc_cal, enc_promo], axis=-1)

        dec_pos = np.arange(
            start + self.input_len, start + self.input_len + self.horizon
        )
        dec_cal = self._calendar(dec_pos)
        dec_promo = promo[dec_slice][:, None]
        dec = np.concatenate([dec_cal, dec_promo], axis=-1)  # known future covariates

        target = scaler.transform(values[dec_slice])

        return (
            torch.from_numpy(enc).float(),
            torch.from_numpy(dec).float(),
            torch.from_numpy(target).float(),
            sid,
        )

    @property
    def enc_features(self) -> int:
        return 1 + 2 + 1  # value + calendar(2) + promo

    @property
    def dec_features(self) -> int:
        return 2 + 1  # calendar(2) + promo


def make_splits(
    panel: pd.DataFrame,
    input_len: int = 28,
    horizon: int = 7,
    val_fraction: float = 0.2,
) -> Tuple[WindowDataset, WindowDataset]:
    """Split each series temporally into train/val windows (no leakage)."""
    t_max = int(panel["t"].max())
    cut = int(t_max * (1 - val_fraction))
    train_panel = panel[panel["t"] <= cut].copy()
    # Validation needs some history before the cut, so include a tail overlap.
    val_panel = panel[panel["t"] > cut - input_len].copy()

    train_ds = WindowDataset(train_panel, input_len, horizon)
    val_ds = WindowDataset(val_panel, input_len, horizon, scalers=train_ds.scalers)
    return train_ds, val_ds

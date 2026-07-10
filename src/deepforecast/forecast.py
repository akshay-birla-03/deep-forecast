"""Inference helpers: produce quantile forecasts for a single series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from .dataset import Scaler
from .model import Seq2SeqForecaster


@dataclass
class Forecast:
    horizon: int
    quantiles: Sequence[float]
    values: Dict[str, List[float]]  # e.g. {"q0.1": [...], "q0.5": [...], "q0.9": [...]}

    @property
    def median(self) -> List[float]:
        key = min(self.values, key=lambda k: abs(float(k[1:]) - 0.5))
        return self.values[key]


def _calendar(positions: np.ndarray) -> np.ndarray:
    dow = positions % 7
    return np.stack(
        [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=-1
    ).astype(np.float32)


@torch.no_grad()
def forecast_series(
    model: Seq2SeqForecaster,
    history_values: Sequence[float],
    history_promo: Sequence[float],
    future_promo: Sequence[float],
    scaler: Scaler,
    input_len: int,
    horizon: int,
) -> Forecast:
    """Forecast the next ``horizon`` steps from the trailing ``input_len`` history."""
    model.eval()
    values = np.asarray(history_values, dtype=np.float32)[-input_len:]
    promo = np.asarray(history_promo, dtype=np.float32)[-input_len:]
    if len(values) < input_len:
        raise ValueError(f"Need at least {input_len} history points, got {len(values)}")

    start = len(history_values) - input_len
    enc_pos = np.arange(start, start + input_len)
    enc = np.concatenate(
        [scaler.transform(values)[:, None], _calendar(enc_pos), promo[:, None]], axis=-1
    )

    dec_pos = np.arange(start + input_len, start + input_len + horizon)
    fut_promo = np.asarray(future_promo, dtype=np.float32)[:horizon]
    if len(fut_promo) < horizon:
        fut_promo = np.pad(fut_promo, (0, horizon - len(fut_promo)))
    dec = np.concatenate([_calendar(dec_pos), fut_promo[:, None]], axis=-1)

    enc_t = torch.from_numpy(enc).float().unsqueeze(0)
    dec_t = torch.from_numpy(dec).float().unsqueeze(0)
    out = model(enc_t, dec_t).squeeze(0).numpy()  # (H, Q)

    q = list(model.config.quantiles)
    values_out = {
        f"q{q[i]}": [float(v) for v in scaler.inverse(out[:, i])] for i in range(len(q))
    }
    return Forecast(horizon=horizon, quantiles=q, values=values_out)

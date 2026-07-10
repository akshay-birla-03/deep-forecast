"""Forecast accuracy and calibration metrics (numpy)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent, robust to zeros in the denominator."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def quantile_loss(y_true, y_pred_q, quantiles: Sequence[float]) -> float:
    """Mean pinball loss over quantiles. y_pred_q: (..., n_quantiles)."""
    y_true = np.asarray(y_true, float)[..., None]
    y_pred_q = np.asarray(y_pred_q, float)
    q = np.asarray(quantiles, float).reshape((1,) * (y_pred_q.ndim - 1) + (-1,))
    errors = y_true - y_pred_q
    loss = np.maximum(q * errors, (q - 1) * errors)
    return float(np.mean(loss))


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of observations falling within [lower, upper]."""
    y_true = np.asarray(y_true, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def seasonal_naive_forecast(history: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """Seasonal-naive baseline: repeat the value from `period` steps earlier."""
    history = np.asarray(history, float)
    out = np.empty(horizon, dtype=float)
    for h in range(horizon):
        idx = len(history) - period + (h % period)
        out[h] = history[idx] if 0 <= idx < len(history) else history[-1]
    return out

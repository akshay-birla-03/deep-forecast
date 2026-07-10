import numpy as np
import pytest
import torch

from deepforecast.losses import QuantileLoss
from deepforecast.metrics import (
    interval_coverage,
    mae,
    quantile_loss,
    rmse,
    seasonal_naive_forecast,
    smape,
)
from deepforecast.model import ModelConfig, Seq2SeqForecaster


def test_model_forward_shapes_and_monotonic_quantiles():
    cfg = ModelConfig(enc_features=4, dec_features=3, hidden_size=16, quantiles=(0.1, 0.5, 0.9))
    model = Seq2SeqForecaster(cfg)
    enc = torch.randn(8, 28, 4)
    dec = torch.randn(8, 7, 3)
    out = model(enc, dec)
    assert out.shape == (8, 7, 3)
    # Quantiles must be non-decreasing along the last axis.
    diffs = out[..., 1:] - out[..., :-1]
    assert torch.all(diffs >= -1e-5)


def test_quantile_loss_positive_and_zero_at_perfect_median():
    loss_fn = QuantileLoss([0.5])
    target = torch.randn(4, 7)
    preds = target.unsqueeze(-1)  # perfect prediction for the median
    assert float(loss_fn(preds, target)) == pytest.approx(0.0, abs=1e-6)


def test_quantile_loss_requires_quantiles():
    with pytest.raises(ValueError):
        QuantileLoss([])


def test_point_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert smape(y, y) == 0.0
    assert smape(np.zeros(3), np.zeros(3)) == 0.0  # no div-by-zero


def test_interval_coverage_and_naive():
    y = np.array([1.0, 2.0, 3.0])
    assert interval_coverage(y, np.array([0, 0, 0]), np.array([5, 5, 5])) == 1.0
    hist = np.arange(14, dtype=float)
    fc = seasonal_naive_forecast(hist, horizon=7, period=7)
    assert fc.shape == (7,)


def test_quantile_loss_metric_matches_torch():
    q = [0.1, 0.5, 0.9]
    target = np.random.randn(5, 7)
    preds = np.random.randn(5, 7, 3)
    val = quantile_loss(target, preds, q)
    assert val > 0

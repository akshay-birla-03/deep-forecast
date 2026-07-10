"""Quantile (pinball) loss for probabilistic forecasting."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Averaged pinball loss across a set of quantiles.

    For a quantile ``q``, the pinball loss penalizes under- and over-prediction
    asymmetrically, so minimizing it yields calibrated quantile estimates.
    """

    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        if not quantiles:
            raise ValueError("quantiles must be non-empty")
        self.register_buffer(
            "quantiles", torch.tensor(list(quantiles), dtype=torch.float32)
        )

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: (batch, horizon, n_quantiles); target: (batch, horizon)
        target = target.unsqueeze(-1)  # (batch, horizon, 1)
        errors = target - preds  # (batch, horizon, n_quantiles)
        q = self.quantiles.view(1, 1, -1)
        loss = torch.maximum(q * errors, (q - 1) * errors)
        return loss.mean()

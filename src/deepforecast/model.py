"""Seq2Seq LSTM forecaster with Bahdanau attention and quantile heads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    enc_features: int = 4
    dec_features: int = 3
    hidden_size: int = 48
    num_layers: int = 1
    dropout: float = 0.1
    quantiles: Sequence[float] = field(default_factory=lambda: (0.1, 0.5, 0.9))


class _Attention(nn.Module):
    """Additive (Bahdanau) attention over encoder outputs."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.U = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor) -> torch.Tensor:
        # dec_hidden: (batch, hidden); enc_outputs: (batch, T, hidden)
        query = self.W(dec_hidden).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.U(enc_outputs)  # (batch, T, hidden)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)  # (batch, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)  # (batch, 1, T)
        context = torch.bmm(weights, enc_outputs).squeeze(1)  # (batch, hidden)
        return context


class Seq2SeqForecaster(nn.Module):
    """Encoder-decoder LSTM that emits one value per quantile at each horizon step."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.n_quantiles = len(config.quantiles)

        self.encoder = nn.LSTM(
            config.enc_features, h, config.num_layers,
            batch_first=True, dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.attention = _Attention(h)
        # Decoder input: known covariates + previous step context.
        self.decoder = nn.LSTMCell(config.dec_features + h, h)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(h, self.n_quantiles)

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        # enc: (batch, T, enc_features); dec: (batch, H, dec_features)
        batch, horizon, _ = dec.shape
        enc_outputs, (h_n, c_n) = self.encoder(enc)
        hx = h_n[-1]
        cx = c_n[-1]

        preds: List[torch.Tensor] = []
        for step in range(horizon):
            context = self.attention(hx, enc_outputs)
            dec_in = torch.cat([dec[:, step, :], context], dim=-1)
            hx, cx = self.decoder(dec_in, (hx, cx))
            out = self.head(self.dropout(hx))  # (batch, n_quantiles)
            preds.append(out)

        stacked = torch.stack(preds, dim=1)  # (batch, horizon, n_quantiles)
        # Enforce monotonic quantiles via cumulative softplus so q10<=q50<=q90.
        base = stacked[..., :1]
        deltas = torch.nn.functional.softplus(stacked[..., 1:])
        monotonic = torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)
        return monotonic

    @property
    def median_index(self) -> int:
        q = list(self.config.quantiles)
        return min(range(len(q)), key=lambda i: abs(q[i] - 0.5))

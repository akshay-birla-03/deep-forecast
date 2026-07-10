"""Training loop with early stopping and baseline-relative evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import WindowDataset, make_splits
from .losses import QuantileLoss
from .metrics import (
    interval_coverage,
    mae,
    quantile_loss,
    rmse,
    seasonal_naive_forecast,
    smape,
)
from .model import ModelConfig, Seq2SeqForecaster


@dataclass
class TrainConfig:
    input_len: int = 28
    horizon: int = 7
    hidden_size: int = 48
    batch_size: int = 64
    max_epochs: int = 15
    lr: float = 1e-2
    patience: int = 4
    val_fraction: float = 0.2
    quantiles: Sequence[float] = field(default_factory=lambda: (0.1, 0.5, 0.9))
    seed: int = 0


@dataclass
class TrainResult:
    model: Seq2SeqForecaster
    metrics: Dict[str, float]
    naive_mae: float
    history: List[Dict[str, float]]
    quantiles: Sequence[float]

    @property
    def beats_baseline(self) -> bool:
        return self.metrics["mae"] < self.naive_mae


class ForecastTrainer:
    def __init__(self, config: Optional[TrainConfig] = None) -> None:
        self.config = config or TrainConfig()

    def _seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def fit(self, panel: pd.DataFrame) -> TrainResult:
        cfg = self.config
        self._seed()
        train_ds, val_ds = make_splits(
            panel, cfg.input_len, cfg.horizon, cfg.val_fraction
        )
        train_dl = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        model = Seq2SeqForecaster(
            ModelConfig(
                enc_features=train_ds.enc_features,
                dec_features=train_ds.dec_features,
                hidden_size=cfg.hidden_size,
                quantiles=cfg.quantiles,
            )
        )
        loss_fn = QuantileLoss(cfg.quantiles)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        best_state = None
        best_val = float("inf")
        no_improve = 0
        history: List[Dict[str, float]] = []

        for epoch in range(cfg.max_epochs):
            model.train()
            train_loss = 0.0
            n = 0
            for enc, dec, target, _ in train_dl:
                opt.zero_grad()
                preds = model(enc, dec)
                loss = loss_fn(preds, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss += loss.item() * len(enc)
                n += len(enc)
            train_loss /= max(n, 1)

            val_loss = self._eval_loss(model, val_dl, loss_fn)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = self._final_metrics(model, val_ds)
        naive_mae = self._naive_mae(panel, val_ds)
        return TrainResult(
            model=model,
            metrics=metrics,
            naive_mae=naive_mae,
            history=history,
            quantiles=cfg.quantiles,
        )

    @staticmethod
    @torch.no_grad()
    def _eval_loss(model, dl, loss_fn) -> float:
        model.eval()
        total, n = 0.0, 0
        for enc, dec, target, _ in dl:
            preds = model(enc, dec)
            total += float(loss_fn(preds, target)) * len(enc)
            n += len(enc)
        return total / max(n, 1)

    @torch.no_grad()
    def _final_metrics(self, model, val_ds: WindowDataset) -> Dict[str, float]:
        model.eval()
        q = list(self.config.quantiles)
        med = model.median_index
        lo, hi = 0, len(q) - 1
        preds_med, preds_all, targets = [], [], []
        lowers, uppers = [], []
        dl = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)
        for enc, dec, target, sid in dl:
            out = model(enc, dec).numpy()  # (b, H, Q)
            # Inverse-scale per series.
            for i in range(len(enc)):
                s = int(sid[i])
                sc = val_ds.scalers[s]
                preds_med.append(sc.inverse(out[i, :, med]))
                preds_all.append(sc.inverse(out[i]))
                lowers.append(sc.inverse(out[i, :, lo]))
                uppers.append(sc.inverse(out[i, :, hi]))
                targets.append(sc.inverse(target[i].numpy()))

        preds_med = np.array(preds_med)
        targets = np.array(targets)
        preds_all = np.array(preds_all)
        return {
            "mae": mae(targets, preds_med),
            "rmse": rmse(targets, preds_med),
            "smape": smape(targets, preds_med),
            "quantile_loss": quantile_loss(targets, preds_all, q),
            "coverage_80": interval_coverage(targets, np.array(lowers), np.array(uppers)),
        }

    def _naive_mae(self, panel: pd.DataFrame, val_ds: WindowDataset) -> float:
        # Seasonal-naive baseline over the same validation windows.
        errs = []
        for sid, start in val_ds._examples:
            values = val_ds._series[sid]
            hist = values[start : start + self.config.input_len]
            target = values[
                start + self.config.input_len : start + self.config.input_len + self.config.horizon
            ]
            fc = seasonal_naive_forecast(hist, self.config.horizon, period=7)
            errs.append(np.abs(target - fc))
        return float(np.mean(np.concatenate(errs))) if errs else float("inf")

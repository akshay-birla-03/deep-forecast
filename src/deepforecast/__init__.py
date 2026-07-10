"""deepforecast: probabilistic multi-horizon demand forecasting in PyTorch.

A compact but complete sequence-to-sequence forecaster with an attention decoder
and quantile (pinball) outputs, so every forecast comes with calibrated
prediction intervals rather than a single point. Runs on CPU.

    >>> from deepforecast import generate_panel, ForecastTrainer, TrainConfig
    >>> panel = generate_panel(n_series=20, length=240)
    >>> trainer = ForecastTrainer(TrainConfig(max_epochs=3))
    >>> result = trainer.fit(panel)
    >>> result.metrics["mae"] < result.naive_mae   # beats seasonal-naive
    True
"""
from .data import generate_panel, SeriesConfig
from .dataset import WindowDataset, make_splits
from .model import Seq2SeqForecaster, ModelConfig
from .losses import QuantileLoss
from .metrics import mae, rmse, smape, quantile_loss, interval_coverage
from .train import ForecastTrainer, TrainConfig, TrainResult

__version__ = "0.1.0"

__all__ = [
    "generate_panel",
    "SeriesConfig",
    "WindowDataset",
    "make_splits",
    "Seq2SeqForecaster",
    "ModelConfig",
    "QuantileLoss",
    "mae",
    "rmse",
    "smape",
    "quantile_loss",
    "interval_coverage",
    "ForecastTrainer",
    "TrainConfig",
    "TrainResult",
    "__version__",
]

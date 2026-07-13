# Deep Forecast

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akshay-birla-03/deep-forecast/blob/main/notebooks/Run_in_Colab.ipynb)

[![CI](https://github.com/akshay-birla-03/deep-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/akshay-birla-03/deep-forecast/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Probabilistic, multi-horizon demand forecasting** with a sequence-to-sequence attention LSTM in PyTorch. Every forecast is a set of **quantiles** (P10 / P50 / P90), so you get calibrated **prediction intervals** — not just a point estimate. Trains on CPU in under a minute and is benchmarked against a **seasonal-naive baseline** so "it learned something" is a tested assertion, not a claim.

## Why quantile forecasting

Point forecasts hide risk. A demand planner needs to know the *range*: how much to stock so that 90% of the time you don't run out. This model outputs P10/P50/P90 per horizon step via the **pinball (quantile) loss**, and enforces **monotonic quantiles** (P10 ≤ P50 ≤ P90) architecturally with a cumulative-softplus head, so intervals never cross.

## Results (held-out validation)

Trained on a 30-series synthetic panel, 7-day horizon, 12 epochs on CPU:

| Metric | Model | Seasonal-naive |
|---|---|---|
| MAE | **4.80** | 6.37 |
| sMAPE | **10.0%** | — |
| P80 interval coverage | **0.73** | — |

```bash
pip install -e .
deepforecast --series 30 --length 240 --epochs 12
# -> Model BEATS the seasonal-naive baseline (4.80 vs 6.37).
```

The training test (`test_training_beats_seasonal_naive`) fails CI if a change
regresses the model below the seasonal-naive baseline — the model has to *earn*
its place on every commit.

## Architecture

```
 history ─► LSTM encoder ─► encoder states  ┐
                                            │  Bahdanau attention
 known future covariates ─► LSTM decoder ◄──┘  (per decode step)
 (calendar sin/cos, promo)        │
                                  ▼
                      quantile head (monotonic)
                                  │
                    P10 / P50 / P90 per horizon step
```

- **Encoder**: LSTM over `[scaled value, weekly sin/cos, promo]`.
- **Attention**: additive (Bahdanau) attention lets each decode step attend to
  the most relevant history — useful when weekly seasonality means "last
  Tuesday" matters more than yesterday.
- **Decoder**: LSTMCell consuming *known-future* covariates (calendar + planned
  promotions) plus the attention context.
- **Head**: emits quantiles with a cumulative-softplus parameterization that
  guarantees non-crossing intervals.

## Leakage safety

Scalers are fit on the **training** portion of each series only and reused for
validation; splits are **temporal** (validation windows come strictly later in
time). Both properties are covered by tests.

## Package layout

| Module | Responsibility |
|---|---|
| `deepforecast.data` | Multi-series demand panel generator |
| `deepforecast.dataset` | Sliding-window `Dataset`, per-series scaling, temporal splits |
| `deepforecast.model` | Seq2Seq attention LSTM with monotonic quantile head |
| `deepforecast.losses` | Pinball / quantile loss |
| `deepforecast.metrics` | MAE, RMSE, sMAPE, quantile loss, interval coverage, seasonal-naive |
| `deepforecast.train` | Training loop, early stopping, baseline-relative eval |
| `deepforecast.forecast` | Single-series quantile forecasting for inference |

## Usage

```python
from deepforecast import generate_panel, ForecastTrainer, TrainConfig

panel = generate_panel(n_series=30, length=240, seed=0)
result = ForecastTrainer(TrainConfig(max_epochs=12)).fit(panel)

print(result.metrics)          # {'mae': ..., 'smape': ..., 'coverage_80': ...}
print(result.beats_baseline)   # True
```

```python
from deepforecast.forecast import forecast_series
from deepforecast.dataset import Scaler

fc = forecast_series(
    result.model,
    history_values=history[-28:], history_promo=promo[-28:],
    future_promo=[0]*7, scaler=Scaler(mean, std),
    input_len=28, horizon=7,
)
print(fc.values["q0.5"])   # median forecast
print(fc.values["q0.9"])   # upper prediction band
```

## Testing

```bash
make test   # 12 tests: data, dataset, model, losses, metrics, training, forecasting
make lint
```

## License

MIT — see [LICENSE](LICENSE).

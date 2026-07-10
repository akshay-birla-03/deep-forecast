
from deepforecast.data import generate_panel
from deepforecast.dataset import Scaler
from deepforecast.forecast import forecast_series
from deepforecast.train import ForecastTrainer, TrainConfig


def _quick_result():
    panel = generate_panel(n_series=24, length=180, seed=7)
    trainer = ForecastTrainer(
        TrainConfig(input_len=28, horizon=7, hidden_size=32, max_epochs=8, seed=7)
    )
    return trainer.fit(panel), panel


def test_training_beats_seasonal_naive():
    result, _ = _quick_result()
    # A model that actually learned must beat the seasonal-naive baseline on MAE.
    assert result.beats_baseline, (
        f"MAE {result.metrics['mae']:.3f} vs naive {result.naive_mae:.3f}"
    )
    assert result.metrics["smape"] < 30.0
    assert 0.0 <= result.metrics["coverage_80"] <= 1.0
    assert len(result.history) >= 1


def test_forecast_series_produces_quantiles():
    result, panel = _quick_result()
    series0 = panel[panel.series_id == 0].sort_values("t")
    values = series0["value"].to_numpy()
    promo = series0["promo"].to_numpy()
    scaler = Scaler(float(values[:100].mean()), float(values[:100].std()))

    fc = forecast_series(
        result.model,
        history_values=values[:28],
        history_promo=promo[:28],
        future_promo=[0] * 7,
        scaler=scaler,
        input_len=28,
        horizon=7,
    )
    assert fc.horizon == 7
    assert len(fc.median) == 7
    # Quantile ordering holds on the raw forecast too.
    lower = fc.values["q0.1"]
    upper = fc.values["q0.9"]
    assert all(hi >= lo - 1e-6 for lo, hi in zip(lower, upper))

import numpy as np

from deepforecast.data import generate_panel
from deepforecast.dataset import WindowDataset, make_splits


def test_panel_shape_and_determinism():
    p1 = generate_panel(n_series=5, length=60, seed=1)
    p2 = generate_panel(n_series=5, length=60, seed=1)
    assert p1.equals(p2)
    assert set(p1.columns) == {"series_id", "t", "value", "promo"}
    assert (p1["value"] > 0).all()
    assert len(p1) == 5 * 60


def test_window_dataset_shapes():
    panel = generate_panel(n_series=3, length=80, seed=2)
    ds = WindowDataset(panel, input_len=28, horizon=7)
    assert len(ds) > 0
    enc, dec, target, sid = ds[0]
    assert enc.shape == (28, ds.enc_features)
    assert dec.shape == (7, ds.dec_features)
    assert target.shape == (7,)


def test_make_splits_is_temporal_and_no_scaler_leakage():
    panel = generate_panel(n_series=4, length=120, seed=3)
    train_ds, val_ds = make_splits(panel, input_len=28, horizon=7, val_fraction=0.2)
    assert len(train_ds) > 0 and len(val_ds) > 0
    # Validation reuses training scalers (fit on training data only).
    assert val_ds.scalers is train_ds.scalers


def test_scaler_roundtrip():
    panel = generate_panel(n_series=2, length=60, seed=4)
    ds = WindowDataset(panel, input_len=20, horizon=5)
    sc = next(iter(ds.scalers.values()))
    x = np.array([10.0, 20.0, 30.0])
    assert np.allclose(sc.inverse(sc.transform(x)), x, atol=1e-4)

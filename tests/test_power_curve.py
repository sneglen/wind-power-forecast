import numpy as np
import pandas as pd
import pytest

from conftest import smoke_config
from wpforecast import backtest, features, scoring
from wpforecast.models.power_curve import RLS, Basis, PowerCurve, ridge

DAYS = list(pd.date_range("2024-04-01", "2024-07-31", freq="D").date)


def _design(n: int, p: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    return X, X @ rng.normal(size=p) + rng.normal(0, 0.1, n)


def test_rls_without_forgetting_is_ridge_and_ols():
    X, y = _design(400, 8)
    for delta, target in ((10.0, ridge(X, y, 10.0)), (1e8, np.linalg.lstsq(X, y, rcond=None)[0])):
        rls = RLS(8, forgetting=1.0, delta=delta)
        for x, t in zip(X, y, strict=True):
            rls.update(x, t)
        assert np.allclose(rls.beta, target, atol=1e-6)


def test_forgetting_tracks_a_parameter_change():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 2))
    beta = np.where(np.arange(4000)[:, None] < 2000, [1.0, 0.0], [0.0, 1.0])
    y = (X * beta).sum(axis=1)
    fixed, adaptive = RLS(2, 1.0, 100.0), RLS(2, 0.99, 100.0)
    for x, t in zip(X, y, strict=True):
        fixed.update(x, t)
        adaptive.update(x, t)
    assert np.allclose(adaptive.beta, [0.0, 1.0], atol=0.01)
    assert np.allclose(fixed.beta, [0.5, 0.5], atol=0.05)  # averages both regimes


def test_covariance_stays_bounded_without_excitation():
    rls = RLS(3, forgetting=0.95, delta=10.0)
    for _ in range(500):
        rls.update(np.array([1.0, 0.0, 0.0]), 0.5)  # only the first direction is excited
    assert np.trace(rls.P) <= rls.max_trace + 1e-9


def test_basis_is_a_partition_of_unity():
    basis = Basis(knots=6, degree=3, speed_max={"onshore": 20.0, "offshore": 28.0}, harmonics=2)
    b = basis.speed(np.array([0.0, 3.0, 12.5, 24.9, 40.0]), "offshore")
    assert b.shape == (5, basis.n_speed) and np.allclose(b.sum(axis=1), 1.0)
    phi = basis.matrix(
        {"onshore": np.array([5.0, np.nan]), "offshore": np.array([5.0, 7.0])},
        {"onshore": np.array([270.0, 0.0]), "offshore": np.array([0.0, 90.0])},
        np.array([0.0, 6.0]),
    )
    assert phi.shape == (2, basis.size)
    assert np.isfinite(phi[0]).all() and np.isnan(phi[1]).any()


def _truth(speed: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(speed - 9.0)))


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> pd.DataFrame:
    """Perfect NWP, y = curve(speed) + an hourly AR(1) error that persists for hours."""
    cfg = smoke_config(tmp_path_factory.mktemp("power_curve"))
    rows = features.skeleton(cfg, DAYS)
    hours = pd.date_range(
        rows["time_utc"].min() - pd.Timedelta(days=2), rows["time_utc"].max(), freq="h"
    )
    rng = np.random.default_rng(2)
    speed = pd.Series(
        np.clip(9 + 4 * np.sin(np.arange(len(hours)) / 17) + rng.normal(0, 1, len(hours)), 0, 25),
        hours,
    )
    error = np.zeros(len(hours))
    for i in range(1, len(hours)):
        error[i] = 0.97 * error[i - 1] + rng.normal(0, 0.01)
    y = pd.Series(np.clip(_truth(speed.to_numpy()) + error, 0, 1), hours)
    rows["last_known_utc"] = features.last_known_stamp(cfg, rows["issue_time_utc"])
    for suffix, stamp in (("", "time_utc"), ("@issue", "last_known_utc")):
        for region in ("onshore", "offshore"):
            rows[f"wind_speed_100m@{region}{suffix}"] = speed.reindex(rows[stamp]).to_numpy()
            rows[f"wind_direction_100m@{region}{suffix}"] = 270.0
    rows["y"] = y.reindex(rows["time_utc"]).to_numpy()
    rows["y_issue"] = y.reindex(rows["last_known_utc"]).to_numpy()
    rows["curtailed"] = False
    return rows.assign(power_mw=1.0, capacity_mw=1.0)


def _model(cfg, forgetting: float = 1.0) -> PowerCurve:
    return PowerCurve("curve", cfg.model.rls, forgetting, scoring.crps_levels(cfg), 10)


def test_power_curve_learns_the_curve_and_a_fading_correction(cfg, table):
    model = _model(cfg)
    model.fit(table)
    speed = np.linspace(3, 15, 7)  # the range the synthetic data covers
    rows = pd.DataFrame(
        {
            "time_utc": pd.Timestamp("2024-08-01 11:00", tz="UTC"),
            "last_known_utc": pd.Timestamp("2024-08-01 09:00", tz="UTC"),
            "horizon_h": 30,
            "y_issue": np.nan,  # no correction: the curve alone
        }
        | {
            f"wind_speed_100m@{r}{s}": speed
            for r in ("onshore", "offshore")
            for s in ("", "@issue")
        }
        | {
            f"wind_direction_100m@{r}{s}": 270.0
            for r in ("onshore", "offshore")
            for s in ("", "@issue")
        }
    )
    assert np.allclose(model.point(rows), _truth(speed), atol=0.05)
    a = model.a
    assert a[0] > 0.6 and a[0] > a[-1]  # the error at issue persists at short horizons only


def test_power_curve_in_the_harness(cfg, table):
    fc = backtest.run(cfg, _model(cfg, 0.9995), table, "recursive", DAYS[30], DAYS[-1])
    q = fc[[scoring.qcol(x) for x in scoring.crps_levels(cfg)]].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all()
    short = fc[fc["horizon_h"] <= 6]
    long = fc[fc["horizon_h"] >= 31]
    rmse = lambda f: np.sqrt(((f["y"] - f["point"]) ** 2).mean())  # noqa: E731
    assert rmse(short) < rmse(long)  # the correction helps where the error persists

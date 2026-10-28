import numpy as np
import pandas as pd

from conftest import curve_table
from wpforecast import scoring, stress
from wpforecast.models.power_curve import PowerCurve

DAYS = list(pd.date_range("2024-04-01", "2024-05-31", freq="D").date)


def test_reset_restores_the_prior_covariance_at_an_event(cfg):
    table = curve_table(cfg, DAYS)
    event = pd.Timestamp("2024-05-10", tz="UTC")

    def curve(resets=()) -> PowerCurve:
        return PowerCurve("c", cfg.model.rls, 1.0, scoring.crps_levels(cfg), 10, resets=resets)

    plain, reset = curve(), curve((event,))
    before = table[table["time_utc"] < event]
    plain.fit(before)
    reset.fit(before)
    assert np.allclose(plain.rls.P, reset.rls.P)  # nothing happens before the event
    day = table[(table["time_utc"] >= event) & (table["time_utc"] < event + pd.Timedelta(days=1))]
    plain.update(day)
    reset.update(day)
    assert not reset.pending
    phi = plain.basis.rows(day)

    def spread(curve: PowerCurve) -> float:  # parameter uncertainty where the data are
        return float(np.einsum("ij,jk,ik->i", phi, curve.rls.P, phi).mean())

    assert spread(reset) > 10 * spread(plain)  # one day's data after forgetting the precision


def test_scores_in_mw_use_the_capacity_known_at_issue(cfg):
    keys = {
        "issue_time_utc": pd.Timestamp("2026-03-31 09:00", tz="UTC"),
        "time_utc": pd.Timestamp("2026-04-01 12:00", tz="UTC"),
    }
    grid = scoring.crps_levels(cfg)
    fc = pd.DataFrame(
        [
            {
                **keys,
                "model": "m",
                "period": "stress",
                "target_day": "2026-04-01",
                "in_product": True,
                "power_mw": 400.0,
                "point": 0.5,
            }
            | {scoring.qcol(x): 0.5 for x in grid}
        ]
    )
    table = pd.DataFrame([{**keys, "capacity_issue_mw": 1000.0}])
    row = stress.in_mw(cfg, fc, table).iloc[0]
    assert row["bias_mw"] == 100.0  # 0.5 x 1000 MW against 400 MW observed
    assert np.isclose(row["crps_mw"], 100.0)  # a point forecast's CRPS is its absolute error

import dataclasses
import datetime as dt
import gzip
import json
import shutil
from pathlib import Path

import httpx
import pandas as pd
import pytest

from wpforecast.config import Config
from wpforecast.data import build, capacity, checks, eds, http, openmeteo


@pytest.fixture
def no_sleep(monkeypatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(http, "SLEEP", slept.append)
    return slept


def _offline_raw(cfg: Config, raw: Path) -> Config:
    return dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, raw=raw))


# --- capacity ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "area"),
    [("1", "DK1"), ("101", "DK2"), ("400", "DK2"), ("411", "DK2"), ("410", "DK1"), ("851", "DK1")],
)
def test_price_area(code, area):
    assert capacity.price_area(code) == area


def test_unknown_offshore_code_fails_loudly():
    with pytest.raises(ValueError, match="map it explicitly"):
        capacity.price_area("2")


def test_capacity_steps_at_local_month_start():
    raw = pd.DataFrame(
        {
            "Month": ["2025-02-01T00:00:00", "2025-03-01T00:00:00"] * 2,
            "MunicipalityNo": ["561", "561", "101", "101"],
            "OnshoreWindCapacity": [10.0, 20.0, 99.0, 99.0],
            "OffshoreWindCapacity": [1.0, 1.0, 0.0, 0.0],
        }
    )
    monthly = capacity.monthly(raw, "DK1")
    index = pd.date_range("2025-02-28 21:00", "2025-03-01 01:00", freq="h", tz="UTC")
    hourly = capacity.hourly(monthly, index)
    # 2025-03-01 00:00 CET = 2025-02-28 23:00 UTC
    assert hourly.loc["2025-02-28 22:00+00:00", "onshore_mw"] == 10.0
    assert hourly.loc["2025-02-28 23:00+00:00", "onshore_mw"] == 20.0
    assert hourly["total_mw"].iloc[-1] == 21.0


# --- HTTP and EDS -------------------------------------------------------------------------


def test_server_wait_parsing():
    eds_429 = httpx.Response(429, text='{"message": "Try again in 118 seconds."}')
    assert http.server_wait_s(eds_429) == 118.0
    assert http.server_wait_s(httpx.Response(429, headers={"Retry-After": "7"})) == 7.0
    assert http.server_wait_s(httpx.Response(429, text="slow down")) is None


def test_open_meteo_limits():
    hourly = httpx.Response(
        429,
        text='{"reason":"Hourly API request limit exceeded. Please try again in the next hour."}',
    )
    assert 60 <= http.server_wait_s(hourly) <= 3660
    daily = httpx.Response(429, text='{"reason":"Daily API request limit exceeded."}')
    client = httpx.Client(transport=httpx.MockTransport(lambda request: daily))
    with pytest.raises(http.LimitExceeded):
        http.get(client, "https://x.test/a", {})  # not retried: no sleep fixture needed


def test_get_retries_429_then_succeeds(no_sleep):
    calls = iter([httpx.Response(429, text="Try again in 3 seconds"), httpx.Response(200, json={})])
    client = httpx.Client(transport=httpx.MockTransport(lambda request: next(calls)))
    assert http.get(client, "https://x.test/a", {}).status_code == 200
    assert no_sleep == [4.0]  # server's wait + 1 s


def test_get_gives_up_after_max_attempts(no_sleep):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    with pytest.raises(http.RetryableError):
        http.get(client, "https://x.test/a", {})
    assert len(no_sleep) == http.MAX_ATTEMPTS - 1


def test_eds_chunks():
    hourly = eds.DATASETS["production"]
    assert eds.chunks(hourly, dt.date(2024, 3, 14), dt.date(2026, 10, 25)) == [
        (dt.date(2024, 3, 14), dt.date(2024, 12, 31)),
        (dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
        (dt.date(2026, 1, 1), dt.date(2026, 10, 25)),
    ]
    monthly = eds.DATASETS["capacity"]
    assert eds.chunks(monthly, dt.date(2024, 3, 14), dt.date(2026, 10, 25)) == [
        (dt.date(2024, 2, 1), dt.date(2026, 10, 25))
    ]


def test_eds_download_caches_and_records(cfg, tmp_path, no_sleep):
    cfg = dataclasses.replace(
        _offline_raw(cfg, tmp_path / "raw"),
        data=dataclasses.replace(cfg.data, offline=False),
    )
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"records": [{"HourUTC": "2025-03-08T00:00:00", "x": 1}]})

    eds.download(cfg, httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(seen) == len(eds.DATASETS)
    assert all(request.params["timezone"] == "UTC" for request in seen)
    assert json.loads(seen[0].params["filter"]) == {"PriceArea": ["DK1"]}
    manifest = json.loads((tmp_path / "raw" / "eds" / "manifest.json").read_text())
    assert len(manifest) == len(eds.DATASETS)

    eds.download(cfg, httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(seen) == len(eds.DATASETS), "cached chunks must not be fetched again"


def test_offline_refuses_to_download(cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="offline"):
        eds.download(_offline_raw(cfg, tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="offline"):
        openmeteo.download(_offline_raw(cfg, tmp_path / "empty"))


# --- Open-Meteo ---------------------------------------------------------------------------


def test_budget_waits_for_hour_and_stops_for_day(tmp_path, monkeypatch):
    clock, slept = [1_000_000.0], []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(http, "SLEEP", sleep)
    budget = openmeteo.Budget(tmp_path / "ledger.json", 40, 60, now=lambda: clock[0])
    budget.acquire(19)
    budget.acquire(19)
    assert slept == []
    budget.acquire(19)  # 57 > 40 within the hour: wait until the first entry expires
    assert len(slept) == 1 and 3600 <= slept[0] <= 3602
    with pytest.raises(openmeteo.BudgetExhausted, match="run `make data` again"):
        budget.acquire(19)  # 76 > 60 within the day
    reloaded = openmeteo.Budget(tmp_path / "ledger.json", 40, 60, now=lambda: clock[0])
    assert len(reloaded.entries) == 3

    # With a long enough max_wait, the rolling day frees up instead of stopping the download.
    patient = openmeteo.Budget(
        tmp_path / "ledger.json", 40, 60, now=lambda: clock[0], max_wait_s=2 * 86400
    )
    before = clock[0]
    patient.acquire(19)
    assert clock[0] - before > 80000  # waited until the first entries left the 24 h window
    assert patient._used(openmeteo.DAY) <= 60


def _location(run: pd.Timestamp, hours: int, variables) -> dict:
    times = pd.date_range(run, periods=hours, freq="h")
    hourly = {"time": [f"{t:%Y-%m-%dT%H:%M}" for t in times]}
    hourly |= {var: [5.0] * hours for var in variables}
    return {"latitude": 55.5, "longitude": 8.5, "elevation": 10.0, "hourly": hourly}


def test_fetch_run_marks_missing_runs(cfg, tmp_path, no_sleep):
    reply = {"error": True, "reason": "The requested model run is not available."}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400, json=reply)))
    budget = openmeteo.Budget(tmp_path / "ledger.json", 1e6, 1e6)
    run = pd.Timestamp("2025-03-08", tz="UTC")
    payload = openmeteo.fetch_run(cfg, client, run, budget, http.Throttle(0))
    assert payload["missing"].startswith("The requested model run")


def test_fetch_run_rejects_a_wrong_time_axis(cfg, tmp_path, no_sleep):
    run = pd.Timestamp("2025-03-08", tz="UTC")

    def handler(request: httpx.Request) -> httpx.Response:
        n = len(request.url.params["latitude"].split(","))
        return httpx.Response(200, json=[_location(run, 10, cfg.nwp.variables)] * n)

    budget = openmeteo.Budget(tmp_path / "ledger.json", 1e6, 1e6)
    with pytest.raises(ValueError, match="unexpected time axis"):
        openmeteo.fetch_run(
            cfg, httpx.Client(transport=httpx.MockTransport(handler)), run, budget, http.Throttle(0)
        )


def test_run_without_wind_counts_as_missing(cfg, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(cfg.paths.raw, raw)
    cfg = _offline_raw(cfg, raw)
    run = openmeteo.run_times(cfg)[3]
    path = openmeteo.run_path(raw, cfg.nwp.model, run)
    with gzip.open(path, "rt") as f:
        payload = json.load(f)
    for group in payload["groups"].values():  # the archive answered, but with nulls
        for loc in group["response"]:
            loc["hourly"]["wind_speed_100m"] = [None] * len(loc["hourly"]["time"])
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)

    nwp, _, missing = openmeteo.load(cfg)
    assert missing == [run.isoformat()]
    assert run not in set(nwp["run_utc"])


def test_sample_runs_use_land_and_sea_cells(cfg):
    run = openmeteo.run_times(cfg)[0]
    with gzip.open(openmeteo.run_path(cfg.paths.raw, cfg.nwp.model, run), "rt") as f:
        payload = json.load(f)
    assert set(payload["groups"]) == {"onshore", "offshore"}
    assert len(payload["groups"]["onshore"]["points"]) == 12
    assert all(loc["elevation"] == 0 for loc in payload["groups"]["offshore"]["response"])


# --- build and checks on the committed sample --------------------------------------------


def test_build_on_sample(cfg):
    build.build(cfg)
    prod = build.read(cfg, "production")
    assert prod.index.tz is not None and prod.index.is_monotonic_increasing
    parts = prod.drop(columns="wind_mw").sum(axis=1)
    pd.testing.assert_series_equal(prod["wind_mw"], parts, check_names=False)

    ds = build.read(cfg, "dataset")
    assert (ds.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert ds["power_norm"].between(0, 1).all()
    assert len(ds) == 24 * ((cfg.data.end - cfg.data.start).days + 1)

    nwp = build.read(cfg, "nwp_hres")
    runs = len(openmeteo.run_times(cfg))
    assert len(nwp) == runs * len(cfg.points) * (cfg.nwp.max_lead_h + 1)
    assert nwp["lead_h"].min() == 0 and nwp["lead_h"].max() == cfg.nwp.max_lead_h
    assert set(cfg.nwp.variables) <= set(nwp.columns)

    price = build.read(cfg, "da_price")
    assert (price.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    imbalance = build.read(cfg, "imbalance")
    assert (imbalance.index.minute % 15 == 0).all()


def test_read_explains_missing_tables(cfg):
    with pytest.raises(FileNotFoundError, match="end of `make data`"):
        build.read(cfg, "production")


def test_checks_pass_on_sample(cfg):
    build.build(cfg)
    report = checks.run(cfg)
    failed = [c for c in report.checks if not c.ok]
    assert not failed, failed
    text = (cfg.paths.reports / checks.REPORT).read_text()
    assert "## Provenance" in text and "CC BY 4.0" in text
    assert (cfg.paths.reports / checks.FIGURES / "power_curve.png").stat().st_size > 0

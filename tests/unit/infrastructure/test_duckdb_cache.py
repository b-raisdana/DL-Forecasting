from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import duckdb
import pandas as pd
import pandera
import pytest
from config import app_config
from helper.date_utils import date_range, date_range_to_string
from infrastructure.datastore_engine import duckdb_cache as dc
from pandera import typing as pt


class _ValueSchema(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]


class _MultiTimeframeValueSchema(pandera.DataFrameModel):
    timeframe: pt.Series[str]
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]


def _configure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_config, "path_of_data", tmp_path)
    monkeypatch.setattr(app_config, "under_process_exchange", "Kucoin")
    monkeypatch.setattr(app_config, "under_process_market", "Spot")
    monkeypatch.setattr(app_config, "under_process_symbol", "BTCUSDT")
    monkeypatch.setattr(app_config, "environment", "production")


@pytest.mark.unit
def test_gap_generated_once_and_reused_on_second_call(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    calls: list[str] = []

    @dc.duckdb_cache(datastore_relative_path=Path("test_value"), dataset_type="test_value", freqs=())
    def generate(*, timeframe: str | None, date_range_str: str) -> pt.DataFrame[_ValueSchema]:
        calls.append(date_range_str)
        start, _ = date_range(date_range_str)
        idx = pd.DatetimeIndex([start], name="date").astype("datetime64[ns, UTC]")
        return pd.DataFrame({"value": [1.0]}, index=idx)

    full_range = date_range_to_string(
        start=datetime(2020, 1, 1, tzinfo=UTC), end=datetime(2020, 1, 2, 23, 59, tzinfo=UTC)
    )

    first = generate(date_range_str=full_range)
    assert len(calls) == 2  # two daily windows, both missing
    assert first["value"].tolist() == [1.0, 1.0]

    second = generate(date_range_str=full_range)
    assert len(calls) == 2  # no new generation -- both windows already covered
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.unit
def test_non_cachable_window_always_regenerated_never_persisted(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    calls: list[str] = []

    @dc.duckdb_cache(datastore_relative_path=Path("live_value"), dataset_type="live_value", freqs=())
    def generate(*, timeframe: str | None, date_range_str: str) -> pt.DataFrame[_ValueSchema]:
        calls.append(date_range_str)
        start, _ = date_range(date_range_str)
        idx = pd.DatetimeIndex([start], name="date").astype("datetime64[ns, UTC]")
        return pd.DataFrame({"value": [float(len(calls))]}, index=idx)

    now = datetime.now(UTC)
    today_range = date_range_to_string(start=now.replace(hour=0, minute=0, second=0, microsecond=0), end=now)

    generate(date_range_str=today_range)
    generate(date_range_str=today_range)

    assert len(calls) == 2  # today's window is regenerated every call, never satisfied from cache

    db_path = tmp_path / "dataset_db" / "live_value" / "Spot" / "BTCUSDT" / "Kucoin" / "data.duckdb"
    if db_path.exists():
        con = duckdb.connect(str(db_path))
        try:
            assert not dc._table_exists(con, "_ValueSchema")
        finally:
            con.close()


@pytest.mark.unit
def test_partial_coverage_raises(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    db_path = dc._dataset_path(Path("mt_value"))
    con = duckdb.connect(str(db_path))
    seeded = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-01-01", tz="UTC")],
            "value": [1.0],
            "timeframe": ["1D"],
        }
    )
    dc._write_gap(con, "_MultiTimeframeValueSchema", seeded)
    con.close()

    @dc.duckdb_cache(datastore_relative_path=Path("mt_value"), dataset_type="mt_value", freqs=("1D", "1W"))
    def generate(*, timeframe: str | None, date_range_str: str) -> pt.DataFrame[_MultiTimeframeValueSchema]:
        raise AssertionError("generator should not run before partial coverage is detected")

    full_range = date_range_to_string(
        start=datetime(2020, 1, 1, tzinfo=UTC), end=datetime(2020, 1, 1, 23, 59, tzinfo=UTC)
    )
    with pytest.raises(dc.PartialCoverageError) as exc_info:
        generate(date_range_str=full_range)
    assert exc_info.value.covered_freqs == frozenset({"1D"})
    assert exc_info.value.missing_freqs == frozenset({"1W"})


@pytest.mark.unit
def test_decoration_requires_boundary_arg_timeframe_param_and_return_annotation():
    with pytest.raises(TypeError, match="date_range_str"):

        @dc.duckdb_cache(datastore_relative_path=Path("x"), dataset_type="x", freqs=())
        def _missing_boundary_arg(*, timeframe: str | None) -> pt.DataFrame[_ValueSchema]:  # pragma: no cover
            ...

    with pytest.raises(TypeError, match="timeframe"):

        @dc.duckdb_cache(datastore_relative_path=Path("x"), dataset_type="x", freqs=())
        def _missing_timeframe_param(*, date_range_str: str) -> pt.DataFrame[_ValueSchema]:  # pragma: no cover
            ...

    with pytest.raises(TypeError, match="return"):

        @dc.duckdb_cache(datastore_relative_path=Path("x"), dataset_type="x", freqs=())
        def _missing_return_annotation(*, timeframe: str | None, date_range_str: str):  # pragma: no cover
            ...


@pytest.mark.unit
def test_freqs_must_be_subset_of_app_config_timeframes():
    with pytest.raises(ValueError, match="app_config.timeframes"):
        dc.duckdb_cache(datastore_relative_path=Path("x"), dataset_type="x", freqs=("not-a-real-timeframe",))


@pytest.mark.unit
def test_integrity_check_aborts_on_mismatch(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    aborted: list[int] = []
    monkeypatch.setattr(dc, "_abort", lambda code: aborted.append(code))

    db_path = dc._dataset_path(Path("integrity_value"))
    con = duckdb.connect(str(db_path))
    seeded = pd.DataFrame({"timestamp": [pd.Timestamp("2020-01-01", tz="UTC")], "value": [1.0], "timeframe": [""]})
    dc._write_gap(con, "_ValueSchema", seeded)
    con.close()

    wrong_returned = pd.DataFrame(
        {"value": [999.0]}, index=pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")], name="date")
    )
    dc._run_integrity_check(db_path, "_ValueSchema", "20-01-01.00-00T20-01-01.23-59", wrong_returned, ())

    assert aborted == [1]


@pytest.mark.unit
def test_integrity_check_does_not_abort_on_match(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path)
    aborted: list[int] = []
    monkeypatch.setattr(dc, "_abort", lambda code: aborted.append(code))

    db_path = dc._dataset_path(Path("integrity_value_ok"))
    con = duckdb.connect(str(db_path))
    seeded = pd.DataFrame({"timestamp": [pd.Timestamp("2020-01-01", tz="UTC")], "value": [1.0], "timeframe": [""]})
    dc._write_gap(con, "_ValueSchema", seeded)
    con.close()

    matching_returned = pd.DataFrame(
        {"value": [1.0], "timeframe": [""]}, index=pd.DatetimeIndex([pd.Timestamp("2020-01-01", tz="UTC")], name="date")
    )
    dc._run_integrity_check(db_path, "_ValueSchema", "20-01-01.00-00T20-01-01.23-59", matching_returned, ())

    assert aborted == []

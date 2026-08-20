from typing import Annotated

import pandas as pd
import pandera
import pytest
from infrastructure.datastore_engine import disk_cache, disk_cache_windowed
from pandera import typing as pt


class _OhlcvLikeSchema(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]

    class Config:
        strict = False
        # coerce=True so validate() itself normalizes DuckDB's local-session-timezone-offset output
        # (see infrastructure.datastore_engine.duckdb_reader) to this schema's declared UTC dtype —
        # _read_cached_windows_via_duckdb() validates the raw DuckDB result directly, unlike read_file()'s
        # single-file path (helper.schema_casting.cast_and_validate() coerces before validating there).
        coerce = True


def _fails_if_called(date_range_str, **kwargs):
    raise AssertionError(f"generator() should not be called for {date_range_str} — a covering file exists")


@pytest.mark.unit
def test_read_file_windowed_reads_a_compacted_covering_file_without_refragmenting_it(tmp_path):
    """A merged/compacted multi-day Parquet file (parquet_housekeeping's compact job's own output shape
    — one file spanning several calendar windows) must be read straight from where it sits via DuckDB,
    not sliced back into per-day tiles by _covering_parquet_path() — see that function's docstring.
    Regression for the redesign that made compaction actually stick."""
    file_path = str(tmp_path)
    span = "24-01-01.00-00T24-01-03.23-59"
    covering_df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"), "value": [1.0, 2.0, 3.0]}
    ).set_index("date")
    disk_cache.write_data_file(covering_df, "ohlcv", span, file_path)

    result = disk_cache_windowed.read_file_windowed(
        span, "ohlcv", _fails_if_called, _OhlcvLikeSchema, file_path=file_path
    )

    assert sorted(result["value"].tolist()) == [1.0, 2.0, 3.0]

    windows = disk_cache_windowed._window_date_range_strs(span, "D")
    assert len(windows) == 3
    for window_range in windows:
        assert not disk_cache._parquet_file_path("ohlcv", window_range, file_path).exists()

    type_dir = tmp_path / "ohlcv"
    assert [p.name for p in type_dir.iterdir()] == [f"ohlcv.{span}.parquet"]  # still exactly the one merged file


@pytest.mark.unit
def test_read_file_windowed_migrates_a_legacy_feather_covering_file_without_fragmenting_it(tmp_path):
    """A covering Feather file (not Parquet) spanning several windows is migrated to Parquet in place
    (on-touch, via read_with_timeframe()) but — unlike this module's pre-redesign behavior — is not
    additionally sliced into a separate small file per window it covers: every window resolving to it
    reads the same one migrated file directly. Regression for the bug the redesign introduced and then
    fixed: an earlier window writing its own tile while a later window still pointed at the (by-then-
    migrated) covering file caused the same rows to be read twice in one batched DuckDB query."""
    file_path = str(tmp_path)
    span = "24-01-01.00-00T24-01-02.23-59"
    covering_df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"), "value": [10.0, 20.0]}
    ).set_index("date")
    covering_df.reset_index().to_feather(disk_cache._feather_file_path("ohlcv", span, file_path))

    result = disk_cache_windowed.read_file_windowed(
        span, "ohlcv", _fails_if_called, _OhlcvLikeSchema, file_path=file_path
    )

    assert sorted(result["value"].tolist()) == [10.0, 20.0]

    windows = disk_cache_windowed._window_date_range_strs(span, "D")
    for window_range in windows:
        assert not disk_cache._parquet_file_path("ohlcv", window_range, file_path).exists()

    type_dir = tmp_path / "ohlcv"
    assert [p.name for p in type_dir.iterdir()] == [f"ohlcv.{span}.parquet"]  # migrated in place, not fragmented

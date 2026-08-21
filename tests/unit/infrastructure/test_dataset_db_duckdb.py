from typing import Annotated

import pandas as pd
import pandera
import pytest
from config import app_config
from infrastructure.datastore_engine import disk_cache, disk_cache_layout, disk_cache_windowed
from pandera import typing as pt


class _ValueSchema(pandera.DataFrameModel):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]


class _MultiTimeframeValueSchema(pandera.DataFrameModel):
    timeframe: pt.Index[str]
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    value: pt.Series[float]


def _mt_df(timeframe: str, rows: int, start: str, value_start: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=rows, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {"timeframe": [timeframe] * rows, "date": dates, "value": [value_start + i for i in range(rows)]}
    )
    return frame.set_index(["timeframe", "date"])


def _use_daily_windows(monkeypatch, data_frame_type: str) -> None:
    """read_file_windowed() sizes windows via app_config.cache_window_freq_overrides (default 'M' —
    see Config.py); these tests need 'D' so a short synthetic date range spans multiple windows."""
    overrides = dict(app_config.cache_window_freq_overrides)
    overrides[data_frame_type] = "D"
    monkeypatch.setattr(app_config, "cache_window_freq_overrides", overrides)


@pytest.mark.unit
def test_dataset_db_type_dir_is_data_frame_type_first_and_migrates_old_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "path_of_data", tmp_path)
    old_type_dir = tmp_path / "Kucoin" / "Spot" / "BTCUSDT" / "mytype"
    old_type_dir.mkdir(parents=True)
    old_file = old_type_dir / "mytype.20-01-01.00-00T20-01-02.00-00.parquet"
    old_file.write_bytes(b"legacy-layout-file")

    type_dir = disk_cache_layout._dataset_db_type_dir(
        "mytype", exchange="Kucoin", market="Spot", trading_pair="BTCUSDT"
    )

    assert type_dir == tmp_path / "dataset_db" / "mytype" / "Spot" / "BTCUSDT" / "Kucoin"
    assert (type_dir / "mytype.20-01-01.00-00T20-01-02.00-00.parquet").exists()
    assert not old_file.exists()


@pytest.mark.unit
def test_data_frame_type_dir_routes_dataset_db_sentinel_to_dataset_db_type_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "path_of_data", tmp_path)
    monkeypatch.setattr(app_config, "under_process_exchange", "Kucoin")
    monkeypatch.setattr(app_config, "under_process_market", "Spot")
    monkeypatch.setattr(app_config, "under_process_symbol", "ETHUSDT")

    type_dir = disk_cache._data_frame_type_dir("mytype", disk_cache.DATASET_DB)

    assert type_dir == tmp_path / "dataset_db" / "mytype" / "Spot" / "ETHUSDT" / "Kucoin"


@pytest.mark.unit
def test_data_frame_type_dir_generic_str_root_is_unaffected_by_dataset_db(tmp_path):
    # The pre-existing opaque-root contract (used by the whole test_disk_cache.py suite and by
    # domain/price_action/*) must keep working unchanged: file_path/data_frame_type, no reordering.
    type_dir = disk_cache._data_frame_type_dir("mytype", str(tmp_path))

    assert type_dir == tmp_path / "mytype"


@pytest.mark.unit
def test_read_file_windowed_batches_already_cached_windows_via_duckdb(tmp_path, monkeypatch):
    file_path = str(tmp_path)
    data_frame_type = "multi_timeframe_test_value"
    _use_daily_windows(monkeypatch, data_frame_type)
    full_range = "20-01-01.00-00T20-01-02.23-59"
    window_ranges = disk_cache_windowed._window_date_range_strs(full_range, "D")
    assert len(window_ranges) == 2

    for i, window_range in enumerate(window_ranges):
        df = _mt_df("1D", rows=1, start=f"2020-01-0{i + 1}", value_start=float(i))
        disk_cache.write_data_file(df, data_frame_type, window_range, file_path)

    generator_calls = []

    def generator(date_range_str, **kwargs):
        generator_calls.append(date_range_str)
        raise AssertionError("generator must not be called — both windows are already cached on disk")

    result = disk_cache_windowed.read_file_windowed(
        full_range, data_frame_type, generator, _MultiTimeframeValueSchema, file_path=file_path
    )

    assert generator_calls == []
    assert result["value"].tolist() == [0.0, 1.0]
    assert result.index.names == ["timeframe", "date"]


@pytest.mark.unit
def test_read_file_windowed_duckdb_batch_matches_per_window_loop_result(tmp_path, monkeypatch):
    """Regression: the DuckDB-batched read path must return content indistinguishable from the
    pre-existing per-window read_file() + pd.concat() loop it replaces for already-cached windows."""
    file_path = str(tmp_path)
    data_frame_type = "multi_timeframe_test_value"
    _use_daily_windows(monkeypatch, data_frame_type)
    full_range = "20-01-01.00-00T20-01-03.23-59"
    window_ranges = disk_cache_windowed._window_date_range_strs(full_range, "D")
    assert len(window_ranges) == 3

    expected_frames = []
    for i, window_range in enumerate(window_ranges):
        df = _mt_df("1D", rows=1, start=f"2020-01-0{i + 1}", value_start=float(i))
        disk_cache.write_data_file(df, data_frame_type, window_range, file_path)
        expected_frames.append(
            disk_cache.read_file(
                window_range, data_frame_type, lambda *_a, **_k: None, _MultiTimeframeValueSchema, file_path=file_path
            )
        )
    expected = pd.concat(expected_frames).sort_index(level="date")

    def unreachable_generator(*_a, **_k):
        raise AssertionError("all windows are pre-cached; generator should not run")

    batched = disk_cache_windowed.read_file_windowed(
        full_range, data_frame_type, unreachable_generator, _MultiTimeframeValueSchema, file_path=file_path
    )

    pd.testing.assert_frame_equal(batched.sort_index(level="date"), expected)


@pytest.mark.unit
def test_read_file_windowed_falls_back_to_per_window_read_on_duckdb_batch_failure(tmp_path, monkeypatch):
    file_path = str(tmp_path)
    data_frame_type = "multi_timeframe_test_value"
    _use_daily_windows(monkeypatch, data_frame_type)
    full_range = "20-01-01.00-00T20-01-02.23-59"
    window_ranges = disk_cache_windowed._window_date_range_strs(full_range, "D")

    for i, window_range in enumerate(window_ranges):
        df = _mt_df("1D", rows=1, start=f"2020-01-0{i + 1}", value_start=float(i))
        disk_cache.write_data_file(df, data_frame_type, window_range, file_path)

    def broken_read_parquet_files(*_a, **_k):
        raise RuntimeError("simulated DuckDB failure")

    monkeypatch.setattr(disk_cache_windowed, "read_duckdb", broken_read_parquet_files)

    fallback_calls = []

    def generator(date_range_str, **kwargs):
        fallback_calls.append(date_range_str)
        raise AssertionError("cache files already exist; regeneration should not be needed for a clean fallback")

    result = disk_cache_windowed.read_file_windowed(
        full_range, data_frame_type, generator, _MultiTimeframeValueSchema, file_path=file_path
    )

    assert fallback_calls == []  # fell back to read_file() per window, which still hit the disk cache
    assert result["value"].tolist() == [0.0, 1.0]

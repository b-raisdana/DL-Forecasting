"""Unit tests for read_file()'s in-process LRU memo and generate-on-miss contract
(data_processing/disk_cache.py).

Scope is the memoization layer plus the generator/write_data_file wiring around a cache miss —
cast_and_validate/pandera validation is bypassed via monkeypatch since that machinery is
pre-existing and already exercised elsewhere. See the cache-or-generate skill.
"""

import os
from collections.abc import Generator
from unittest.mock import Mock

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch
from infrastructure import disk_cache

pytestmark = pytest.mark.unit


class _PassthroughModel:
    @staticmethod
    def validate(df: pd.DataFrame, lazy: bool = True) -> pd.DataFrame:
        return df


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setattr(disk_cache, "cast_and_validate", lambda *a, **k: True)
    disk_cache._read_file_cache.clear()
    yield
    disk_cache._read_file_cache.clear()


@pytest.fixture
def fixed_df() -> pd.DataFrame:
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]})


def _read(
    monkeypatch: MonkeyPatch,
    read_mock: Mock,
    generator: Mock | None = None,
    date_range_str: str = "24-01-01.00-00T24-01-02.00-00",
) -> pd.DataFrame:
    monkeypatch.setattr(disk_cache, "read_with_timeframe", read_mock)
    # read_file() has no return annotation (pre-existing, infers Any); _PassthroughModel is a
    # lightweight fake, not a real pandera.DataFrameModel subclass, so it fails the caster_model
    # type-var bound too — both expected for a test isolating the cache from real validation.
    return disk_cache.read_file(  # type: ignore[no-any-return, type-var]
        date_range_str,
        "ohlcv",
        generator or Mock(),
        _PassthroughModel,
        file_path="dummy",
        zero_size_allowed=False,
    )


def test_repeat_call_with_same_args_hits_memo_and_skips_disk_read(
    monkeypatch: MonkeyPatch, fixed_df: pd.DataFrame
) -> None:
    monkeypatch.setattr(disk_cache, "datarange_is_not_cachable", lambda date_range_str: False)
    read_mock = Mock(return_value=fixed_df)
    generator = Mock()

    first = _read(monkeypatch, read_mock, generator)
    second = _read(monkeypatch, read_mock, generator)

    assert read_mock.call_count == 1
    generator.assert_not_called()
    pd.testing.assert_frame_equal(first, second)


def test_memo_hit_returns_an_independent_copy(monkeypatch: MonkeyPatch, fixed_df: pd.DataFrame) -> None:
    monkeypatch.setattr(disk_cache, "datarange_is_not_cachable", lambda date_range_str: False)
    read_mock = Mock(return_value=fixed_df)

    _read(monkeypatch, read_mock)  # cache miss - populates the memo
    second = _read(monkeypatch, read_mock)  # cache hit
    second["close"] = 999.0
    third = _read(monkeypatch, read_mock)  # cache hit

    assert third["close"].tolist() == [1.0, 2.0, 3.0]


def test_different_date_range_is_a_cache_miss(monkeypatch: MonkeyPatch, fixed_df: pd.DataFrame) -> None:
    monkeypatch.setattr(disk_cache, "datarange_is_not_cachable", lambda date_range_str: False)
    read_mock = Mock(return_value=fixed_df)

    _read(monkeypatch, read_mock, date_range_str="24-01-01.00-00T24-01-02.00-00")
    _read(monkeypatch, read_mock, date_range_str="24-02-01.00-00T24-02-02.00-00")

    assert read_mock.call_count == 2


def test_not_cachable_range_is_never_memoized(monkeypatch: MonkeyPatch, fixed_df: pd.DataFrame) -> None:
    monkeypatch.setattr(disk_cache, "datarange_is_not_cachable", lambda date_range_str: True)
    monkeypatch.setattr(os, "remove", Mock())
    read_mock = Mock(return_value=fixed_df)

    _read(monkeypatch, read_mock)
    _read(monkeypatch, read_mock)

    assert read_mock.call_count == 2
    assert len(disk_cache._read_file_cache) == 0


def test_memo_is_bounded_by_lru_eviction(monkeypatch: MonkeyPatch, fixed_df: pd.DataFrame) -> None:
    monkeypatch.setattr(disk_cache, "datarange_is_not_cachable", lambda date_range_str: False)
    read_mock = Mock(return_value=fixed_df)

    for i in range(disk_cache._READ_FILE_CACHE_MAX_ENTRIES + 5):
        _read(monkeypatch, read_mock, date_range_str=f"cache-key-{i}")

    assert len(disk_cache._read_file_cache) == disk_cache._READ_FILE_CACHE_MAX_ENTRIES


def test_cache_miss_uses_generators_return_value_without_a_disk_reread(
    monkeypatch: MonkeyPatch, fixed_df: pd.DataFrame
) -> None:
    monkeypatch.setattr(disk_cache, "datarange_is_not_cachable", lambda date_range_str: False)
    read_mock = Mock(side_effect=FileNotFoundError)
    write_mock = Mock()
    monkeypatch.setattr(disk_cache, "write_data_file", write_mock)
    generator = Mock(return_value=fixed_df)

    result = _read(monkeypatch, read_mock, generator)

    generator.assert_called_once_with("24-01-01.00-00T24-01-02.00-00")
    write_mock.assert_called_once_with(
        fixed_df,
        "ohlcv",
        "24-01-01.00-00T24-01-02.00-00",
        "dummy",
        nan_allowed_columns=frozenset(),
    )
    # only the failed first attempt — no second read_with_timeframe() call after the generator runs
    assert read_mock.call_count == 1
    pd.testing.assert_frame_equal(result, fixed_df)


def test_generate_on_miss_rejects_skip_rows_or_n_rows(monkeypatch: MonkeyPatch, fixed_df: pd.DataFrame) -> None:
    monkeypatch.setattr(disk_cache, "datarange_is_not_cachable", lambda date_range_str: False)
    monkeypatch.setattr(disk_cache, "read_with_timeframe", Mock(side_effect=FileNotFoundError))
    generator = Mock(return_value=fixed_df)

    with pytest.raises(Exception, match="skip_rows/n_rows"):
        disk_cache.read_file(  # type: ignore[no-any-return, type-var]
            "24-01-01.00-00T24-01-02.00-00",
            "ohlcv",
            generator,
            _PassthroughModel,
            file_path="dummy",
            zero_size_allowed=False,
            n_rows=10,
        )
    generator.assert_not_called()

from datetime import UTC, datetime, timedelta

import pytest
from helper.functions import date_range_to_string
from infrastructure.datastore_engine import disk_cache, disk_cache_windowed
from infrastructure.datastore_engine.disk_cache_gaps import find_cache_gaps, find_overlapping_cache_files


def _touch(file_path: str, data_frame_type: str, date_range_str: str) -> None:
    feather_path = disk_cache._feather_file_path(data_frame_type, date_range_str, file_path)
    open(feather_path, "w").close()


@pytest.mark.unit
def test_find_cache_gaps_merges_contiguous_missing_daily_windows(tmp_path):
    file_path = str(tmp_path)
    full_range = date_range_to_string(
        start=datetime(2023, 8, 1, tzinfo=UTC), end=datetime(2023, 8, 9, 23, 59, tzinfo=UTC)
    )
    windows = disk_cache_windowed._window_date_range_strs(full_range, "D")
    assert len(windows) == 9  # Aug 1..9

    # Aug 1, 5, 6, 7, 9 exist (indices 0, 4, 5, 6, 8) -> Aug 2-4 and Aug 8 are gaps.
    for index in (0, 4, 5, 6, 8):
        _touch(file_path, "ohlcv", windows[index])

    gaps = find_cache_gaps("ohlcv", full_range, file_path=file_path, window_freq="D")

    expected_gap_1 = date_range_to_string(
        start=disk_cache.date_range(windows[1])[0], end=disk_cache.date_range(windows[3])[1]
    )
    expected_gap_2 = windows[7]
    assert gaps == [expected_gap_1, expected_gap_2]


@pytest.mark.unit
def test_find_cache_gaps_excludes_the_still_open_today_window(tmp_path):
    file_path = str(tmp_path)
    now = datetime.now(UTC)
    two_days_ago_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)
    full_range = date_range_to_string(start=two_days_ago_midnight, end=now)
    windows = disk_cache_windowed._window_date_range_strs(full_range, "D")
    assert len(windows) == 3  # two closed days + today (still open)

    # Nothing on disk at all -> every closed window is a gap, but today's open window must never
    # be reported since it's never expected to be cached.
    gaps = find_cache_gaps("ohlcv", full_range, file_path=file_path, window_freq="D")

    expected_gap = date_range_to_string(
        start=disk_cache.date_range(windows[0])[0], end=disk_cache.date_range(windows[1])[1]
    )
    assert gaps == [expected_gap]


@pytest.mark.unit
def test_find_overlapping_cache_files_finds_partial_overlaps_sorted_by_start(tmp_path):
    file_path = str(tmp_path)
    month_range = date_range_to_string(
        start=datetime(2023, 1, 1, tzinfo=UTC), end=datetime(2023, 1, 31, 23, 59, tzinfo=UTC)
    )
    day_range = date_range_to_string(
        start=datetime(2023, 1, 15, tzinfo=UTC), end=datetime(2023, 1, 15, 23, 59, tzinfo=UTC)
    )
    unrelated_range = date_range_to_string(
        start=datetime(2023, 3, 1, tzinfo=UTC), end=datetime(2023, 3, 31, 23, 59, tzinfo=UTC)
    )
    _touch(file_path, "ohlcv", month_range)
    _touch(file_path, "ohlcv", day_range)
    _touch(file_path, "ohlcv", unrelated_range)

    query_range = date_range_to_string(
        start=datetime(2023, 1, 10, tzinfo=UTC), end=datetime(2023, 1, 20, 23, 59, tzinfo=UTC)
    )
    overlaps = find_overlapping_cache_files("ohlcv", query_range, file_path=file_path)

    assert overlaps == [(month_range, "feather"), (day_range, "feather")]


@pytest.mark.unit
def test_find_cache_gaps_returns_empty_when_nothing_is_missing(tmp_path):
    file_path = str(tmp_path)
    full_range = date_range_to_string(
        start=datetime(2023, 8, 1, tzinfo=UTC), end=datetime(2023, 8, 2, 23, 59, tzinfo=UTC)
    )
    for window in disk_cache_windowed._window_date_range_strs(full_range, "D"):
        _touch(file_path, "ohlcv", window)

    assert find_cache_gaps("ohlcv", full_range, file_path=file_path, window_freq="D") == []

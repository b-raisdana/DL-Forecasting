"""Unit tests for _cached_training_frames() (training_datasets.py) — the in-memory memo over the
indicator+label prep phase that producer loops (npz_batch.py/ram_batch.py/stream_loader.py) call up to
~100x per quarter against the same mt_ohlcv object.

_build_frames_by_level/add_long_n_short_profit/_compute_safe_training_range are monkeypatched to isolate
the caching behavior from real indicator/label computation (covered by their own tests). See the
cache-or-generate skill.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch
from application.dataset_generation import training_datasets

pytestmark = pytest.mark.unit

_TrainingFrames = tuple[datetime, datetime, int, dict[str, pd.DataFrame]]


@pytest.fixture
def mt_ohlcv() -> pd.DataFrame:
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]})


def _stub_prep(monkeypatch: MonkeyPatch) -> tuple[Mock, Mock, Mock]:
    build_mock = Mock(
        side_effect=lambda *a, **k: {
            "structure": pd.DataFrame({"close": [1.0]}),
            "pattern": pd.DataFrame({"close": [1.0]}),
            "trigger": pd.DataFrame({"close": [1.0]}),
            "double": pd.DataFrame({"close": [1.0]}),
        }
    )
    profit_mock = Mock(side_effect=lambda ohlc, **k: ohlc)
    safe_range_mock = Mock(
        side_effect=lambda dfs, *a, **k: (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            3600,
            dfs,
        )
    )
    monkeypatch.setattr(training_datasets, "_build_frames_by_level", build_mock)
    monkeypatch.setattr(training_datasets, "add_long_n_short_profit", profit_mock)
    monkeypatch.setattr(training_datasets, "_compute_safe_training_range", safe_range_mock)
    return build_mock, profit_mock, safe_range_mock


def _call(mt_ohlcv: pd.DataFrame, forecast_trigger_bars: int = 192) -> _TrainingFrames:
    # mt_ohlcv is a plain DataFrame here (not pandera-validated MultiTimeframe) since this test
    # isolates the caching logic from schema validation; training_datasets.py also resolves as
    # untyped (Any) from outside its own module — both pre-existing, not specific to this function.
    return training_datasets._cached_training_frames(
        mt_ohlcv,
        "4h",
        "1h",
        "15min",
        "5min",
        "15min",
        "trigger",
        forecast_trigger_bars,  # type: ignore[arg-type]
    )


def test_repeat_call_with_same_key_computes_prep_phase_once(monkeypatch: MonkeyPatch, mt_ohlcv: pd.DataFrame) -> None:
    build_mock, profit_mock, safe_range_mock = _stub_prep(monkeypatch)

    _call(mt_ohlcv)
    _call(mt_ohlcv)

    assert build_mock.call_count == 1
    assert profit_mock.call_count == 1
    assert safe_range_mock.call_count == 1


def test_different_forecast_trigger_bars_is_a_cache_miss(monkeypatch: MonkeyPatch, mt_ohlcv: pd.DataFrame) -> None:
    build_mock, *_ = _stub_prep(monkeypatch)

    _call(mt_ohlcv, forecast_trigger_bars=192)
    _call(mt_ohlcv, forecast_trigger_bars=96)

    assert build_mock.call_count == 2


def test_different_mt_ohlcv_object_is_a_cache_miss(monkeypatch: MonkeyPatch, mt_ohlcv: pd.DataFrame) -> None:
    build_mock, *_ = _stub_prep(monkeypatch)
    other_mt_ohlcv = mt_ohlcv.copy()

    _call(mt_ohlcv)
    _call(other_mt_ohlcv)

    assert build_mock.call_count == 2


def test_hit_returns_independent_copies_not_the_cached_object(monkeypatch: MonkeyPatch, mt_ohlcv: pd.DataFrame) -> None:
    _stub_prep(monkeypatch)

    _, _, _, first_dfs = _call(mt_ohlcv)
    first_dfs["structure"]["close"] = 999.0
    _, _, _, second_dfs = _call(mt_ohlcv)

    assert (second_dfs["structure"]["close"] == 1.0).all()


def test_returns_the_same_start_end_duration_as_the_underlying_computation(
    monkeypatch: MonkeyPatch, mt_ohlcv: pd.DataFrame
) -> None:
    _stub_prep(monkeypatch)

    train_safe_start, train_safe_end, duration_seconds, _ = _call(mt_ohlcv)

    assert train_safe_start == datetime(2024, 1, 1, tzinfo=UTC)
    assert train_safe_end == datetime(2024, 1, 2, tzinfo=UTC)
    assert duration_seconds == 3600

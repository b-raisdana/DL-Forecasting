"""Speed test and bottleneck analysis for `build_dataset()`."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from application.dataset_generation.extremum_features import build_branch_extremum
from application.dataset_generation.mfe_mae_om_labels import add_mfe_mae_om_labels
from application.model_implementations.tier1_000.datafeeder_input3_outcome1 import (
    _branch_features,
    build_dataset,
)
from application.model_implementations.tier1_000.model import BRANCH_TIMEFRAMES
from helper.data_preparation import single_timeframe
from infrastructure.ohlcv.ohlcv import read_multi_timeframe_ohlcv


@dataclass
class StageTiming:
    name: str
    seconds: float


@contextmanager
def _stage(name: str, timings: list[StageTiming]) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings.append(StageTiming(name=name, seconds=elapsed))


def _profile_build_dataset(symbol: str, date_range_str: str) -> None:
    timings: list[StageTiming] = []

    with _stage("read_multi_timeframe_ohlcv", timings):
        mt_ohlcv = read_multi_timeframe_ohlcv(date_range_str)

    with _stage("single_timeframe_extract", timings):
        base_ohlc = single_timeframe(mt_ohlcv, "5min")
        fifteen_min_ohlc = single_timeframe(mt_ohlcv, "15min")

    with _stage("add_mfe_mae_om_labels", timings):
        _labels = add_mfe_mae_om_labels(base_ohlc, fifteen_min_ohlc)

    with _stage("_branch_features", timings):
        features_by_tf = {
            tf_name: _branch_features(single_timeframe(mt_ohlcv, tf_name), tf_name) for tf_name in BRANCH_TIMEFRAMES
        }

    with _stage("build_branch_extremum", timings):
        _branch_extremum_by_tf = {
            tf_name: build_branch_extremum(features_by_tf[tf_name]) for tf_name in BRANCH_TIMEFRAMES
        }

    with _stage("build_dataset_full", timings):
        bundle = build_dataset(symbol, date_range_str)

    print(f"Symbol={symbol} range={date_range_str}")
    print(f"n_samples={bundle.n_samples}")
    print("\nPer-stage timings:")
    for t in timings:
        print(f"  {t.name:<35} {t.seconds:8.3f}s")
    total = sum(t.seconds for t in timings)
    print(f"  {'TOTAL':<35} {total:8.3f}s")
    return bundle


def main() -> None:
    symbol = "BTCUSDT"
    date_range_str = "22-02-10.00-00T25-04-01.00-00"
    print(f"Profiling build_dataset for {symbol} over {date_range_str}")
    bundle = _profile_build_dataset(symbol, date_range_str)
    print(f"\nResult: {bundle.n_samples} valid samples")


if __name__ == "__main__":
    main()

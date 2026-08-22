"""Speed test and bottleneck analysis for `build_dataset()`.

Runs `build_dataset()` across a representative symbol/range and prints per-stage timings.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import pandas as pd
from application.model_implementations.tier1_000.model import BRANCH_TIMEFRAMES, DatasetBundle
from config import app_config
from helper.data_preparation import single_timeframe
from helper.date_utils import date_range_to_string
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


def _profile_build_dataset(symbol: str, date_range_str: str) -> DatasetBundle:
    from application.dataset_generation.extremum_features import build_branch_extremum
    from application.dataset_generation.extremum_labels import add_extremum_labels
    from application.model_implementations.tier1_000.datafeeder_input3_outcome1 import (
        _branch_features,
        build_dataset,
    )

    timings: list[StageTiming] = []

    with _stage("read_multi_timeframe_ohlcv", timings):
        mt_ohlcv = read_multi_timeframe_ohlcv(date_range_str)

    with _stage("single_timeframe_extract", timings):
        base_ohlc = cast(pd.DataFrame, single_timeframe(mt_ohlcv, "5min"))
        fifteen_min_ohlc = cast(pd.DataFrame, single_timeframe(mt_ohlcv, "15min"))

    with _stage("branch_features", timings):
        features_by_tf = {
            tf_name: _branch_features(cast(pd.DataFrame, single_timeframe(mt_ohlcv, tf_name)), tf_name)
            for tf_name in BRANCH_TIMEFRAMES
        }

    with _stage("build_branch_extremum", timings):
        branch_extremum_by_tf = {
            tf_name: build_branch_extremum(features_by_tf[tf_name]) for tf_name in BRANCH_TIMEFRAMES
        }

    with _stage("add_extremum_labels", timings):
        add_extremum_labels(
            fifteen_min_ohlcv=fifteen_min_ohlc,
            five_min_ohlc=base_ohlc,
            branch_extremum_by_tf=branch_extremum_by_tf,
        )

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
    symbol = app_config.under_process_symbol
    # Use the full contiguous cached span for this symbol
    from presentation.dataset_generation.now_review_notebook import _full_cached_span

    start, end = _full_cached_span(symbol)
    date_range_str = date_range_to_string(start=start, end=end)
    print(f"Profiling build_dataset for {symbol} over {date_range_str}")
    bundle = _profile_build_dataset(symbol, date_range_str)
    print(f"\nResult: {bundle.n_samples} valid samples")


if __name__ == "__main__":
    main()

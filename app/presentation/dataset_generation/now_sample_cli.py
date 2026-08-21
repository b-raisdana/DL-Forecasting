"""Find missing OHLCV for a NOW-anchored training sample and force-fetch it.

For a given NOW timestamp, the dataset builder needs:
- history: enough base OHLCV to cover the widest branch window (1W x 128 weeks) plus indicator warmup
- future window: 240 minutes beyond NOW for extremum label lookahead

This script:
1. Builds the required date range around NOW.
2. Checks OHLCV and multi-timeframe cache gaps.
3. Fills the gaps via the existing fetch endpoint.
4. Builds one NOW sample to verify the cache now covers the period.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytz
from application.market_data.fetch_market_data import fill_ohlcv_gaps
from application.model_implementations.tier1_000.datafeeder_input3_outcome1 import build_dataset
from config import app_config
from domain.ohlcv.ohlcv import MULTI_TIMEFRAME_OHLCV_DATASET, OHLCV_DATASET
from helper.functions import date_range_to_string
from infrastructure.datastore_engine.disk_cache_gaps import find_cache_gaps


def _now_utc() -> datetime:
    return datetime.now(pytz.UTC)


def required_date_range_for_now(now: datetime) -> tuple[str, datetime, datetime]:
    """Compute the date_range_str needed to build a NOW sample.

    The 1W branch needs 128 weeks of lookback; ATR warmup for 1W uses 20 candles.
    We add a small buffer on both sides to ensure branch assembly has enough room.
    """
    lookback = timedelta(weeks=140)
    future_horizon = timedelta(minutes=300)
    start = now - lookback
    end = now + future_horizon
    return date_range_to_string(start=start, end=end), start, end


def report_gaps(symbol: str, date_range_str: str) -> dict[str, list[str]]:
    ohlcv_gaps = find_cache_gaps(OHLCV_DATASET, date_range_str)
    mt_gaps = find_cache_gaps(MULTI_TIMEFRAME_OHLCV_DATASET, date_range_str)
    return {"ohlcv": ohlcv_gaps, "multi_timeframe_ohlcv": mt_gaps}


def force_fetch_gaps(
    broker: str,
    trading_pair: str,
    date_range_str: str,
    market: str = "Spot",
) -> dict[str, list[str]]:
    """Fill OHLCV gaps, then rebuild multi-timeframe cache by reading it."""
    filled_ohlcv = fill_ohlcv_gaps(broker, trading_pair, date_range_str, market=market)
    mt_gaps = find_cache_gaps(MULTI_TIMEFRAME_OHLCV_DATASET, date_range_str)
    filled_mt: list[str] = []
    for gap in mt_gaps:
        from domain.ohlcv.ohlcv import get_multi_timeframe_ohlcv
        from helper.functions import date_range

        gap_start, gap_end = date_range(gap)
        if gap_end > _now_utc():
            continue
        get_multi_timeframe_ohlcv(gap)
        filled_mt.append(gap)
    return {"ohlcv": filled_ohlcv, "multi_timeframe_ohlcv": filled_mt}


def main() -> None:
    now = _now_utc()
    date_range_str, range_start, range_end = required_date_range_for_now(now)
    symbol = app_config.under_process_symbol
    broker = app_config.under_process_exchange.lower()
    market = app_config.under_process_market

    print(f"NOW (UTC): {now}")
    print(f"Required range: {range_start} -> {range_end}")
    print(f"Date range str: {date_range_str}")
    print(f"Symbol: {symbol}@{broker}/{market}")

    gaps = report_gaps(symbol, date_range_str)
    print(f"OHLCV gaps: {len(gaps['ohlcv'])}")
    for g in gaps["ohlcv"]:
        print(f"  {g}")
    print(f"Multi-timeframe OHLCV gaps: {len(gaps['multi_timeframe_ohlcv'])}")
    for g in gaps["multi_timeframe_ohlcv"]:
        print(f"  {g}")

    if gaps["ohlcv"] or gaps["multi_timeframe_ohlcv"]:
        print("\nForce-fetching missing data ...")
        filled = force_fetch_gaps(broker, symbol, date_range_str, market=market)
        print(f"Fetched OHLCV gaps: {filled['ohlcv']}")
        print(f"Rebuilt multi-timeframe gaps: {filled['multi_timeframe_ohlcv']}")
    else:
        print("\nNo gaps found — cache already covers the required range.")

    print("\nBuilding NOW sample ...")
    app_config.under_process_symbol = symbol
    app_config.under_process_exchange = broker.capitalize()
    app_config.under_process_market = market

    try:
        bundle = build_dataset(symbol, date_range_str)
        anchor = bundle.anchor_index[0]
        print(f"Sample anchor (NOW): {anchor}")
        print(f"  mfe={bundle.mfe[0, 0]:.6f} rer={bundle.rer[0, 0]:.6f}")
        action_idx = int(bundle.action[0].argmax())
        action_names = ["long", "short", "none"]
        print(f"  action={action_names[action_idx]}")
        print(f"  n_samples={bundle.n_samples}")
    except Exception as exc:
        print(f"Failed to build sample: {exc}")
        raise


if __name__ == "__main__":
    main()

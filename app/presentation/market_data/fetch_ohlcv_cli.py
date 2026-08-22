"""CLI entrypoint: fetch OHLCV candles for one trading pair from one broker and cache them to disk
(Parquet/ZSTD, under <data>/dataset_db/ohlcv/<market>/<trading_pair>/<broker>/ — see
`application.market_data.fetch_market_data.fetch_and_cache_ohlcv`/`fill_ohlcv_gaps`). A rerun over a
range already on disk is a cache hit — nothing is re-fetched from the broker.

With no `--date-range` and no `--days`, the default is a gap-fill backfill: every daily gap between
`app_config.ohlcv_oldest_fetch_date` and now is fetched newest-first, and "all up to date" is reported
once none remain (`fill_ohlcv_gaps`). `--list-gaps`/`--list-overlaps` report without fetching.

Examples:
    python -m presentation.market_data.fetch_ohlcv_cli --broker binance --trading-pair ETHUSDT
    python -m presentation.market_data.fetch_ohlcv_cli --broker kucoin --trading-pair BTCUSDT \\
        --date-range 24-01-01.00-00T24-01-31.23-59
    python -m presentation.market_data.fetch_ohlcv_cli --broker binance --trading-pair ETHUSDT --days 7
    python -m presentation.market_data.fetch_ohlcv_cli --broker binance --trading-pair ETHUSDT --list-gaps
"""

from __future__ import annotations

import argparse

from application.market_data.fetch_market_data import fetch_and_cache_ohlcv, fill_ohlcv_gaps
from config import app_config
from helper.date_utils import date_range_to_string, today_morning
from infrastructure.datastore_engine.disk_cache_gaps import find_cache_gaps, find_overlapping_cache_files
from infrastructure.market_data_fetch.ccxt_client import SUPPORTED_BROKERS
from infrastructure.ohlcv.ohlcv import OHLCV_DATASET


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and cache broker OHLCV candles for one trading pair.")
    parser.add_argument("--broker", default="kucoin", choices=sorted(SUPPORTED_BROKERS), help="broker to fetch from")
    parser.add_argument("--trading-pair", required=True, help="e.g. BTCUSDT")
    parser.add_argument(
        "--date-range",
        help="e.g. 24-01-01.00-00T24-01-31.23-59; if omitted (with --days also omitted), "
        "backfill every gap since app_config.ohlcv_oldest_fetch_date",
    )
    parser.add_argument("--days", type=float, default=None, help="fetch only the most recent N days")
    parser.add_argument("--market", default="Spot")
    parser.add_argument("--timeframe", default=None, help="base pandas timeframe, e.g. 1min; defaults to config")
    parser.add_argument(
        "--list-gaps",
        action="store_true",
        help="print missing daily ranges (over --date-range, or the full "
        "oldest-date-to-now span) and exit without fetching",
    )
    parser.add_argument(
        "--list-overlaps",
        action="store_true",
        help="print on-disk cache files overlapping the queried range and exit without fetching",
    )
    return parser.parse_args()


def _default_full_range() -> str:
    return date_range_to_string(start=app_config.ohlcv_oldest_fetch_datetime, end=today_morning())


def _print_ranges(empty_message: str, found_header: str, lines: list[str]) -> None:
    if not lines:
        print(f"[fetch_ohlcv_cli] {empty_message}")
        return
    print(f"[fetch_ohlcv_cli] {len(lines)} {found_header}:")
    for line in lines:
        print(f"  {line}")


def _run_list_gaps(args: argparse.Namespace, query_range: str) -> None:
    who = f"{args.trading_pair}@{args.broker} over {query_range}"
    gaps = find_cache_gaps(OHLCV_DATASET, query_range)
    _print_ranges(f"no gaps for {who}", f"gap(s) for {who}", gaps)


def _run_list_overlaps(args: argparse.Namespace, query_range: str) -> None:
    who = f"{args.trading_pair}@{args.broker} over {query_range}"
    overlaps = find_overlapping_cache_files(OHLCV_DATASET, query_range)
    lines = [f"ohlcv.{range_str}.{ext}" for range_str, ext in overlaps]
    _print_ranges(f"no overlapping cache files for {who}", f"overlapping cache file(s) for {who}", lines)


def _run_list_mode(args: argparse.Namespace) -> None:
    app_config.under_process_exchange = args.broker.capitalize()
    app_config.under_process_symbol = args.trading_pair
    app_config.under_process_market = args.market
    query_range = args.date_range or _default_full_range()

    if args.list_gaps:
        _run_list_gaps(args, query_range)
    if args.list_overlaps:
        _run_list_overlaps(args, query_range)


def _run_fetch_range(args: argparse.Namespace, date_range_str: str) -> None:
    ohlcv = fetch_and_cache_ohlcv(
        broker=args.broker,
        trading_pair=args.trading_pair,
        date_range_str=date_range_str,
        market=args.market,
        base_timeframe=args.timeframe,
    )
    print(
        f"[fetch_ohlcv_cli] cached {len(ohlcv)} candles for {args.trading_pair}@{args.broker} "
        f"({args.market}) over {date_range_str}"
    )


def _run_default_backfill(args: argparse.Namespace) -> None:
    full_range = _default_full_range()
    filled = fill_ohlcv_gaps(
        broker=args.broker,
        trading_pair=args.trading_pair,
        date_range_str=full_range,
        market=args.market,
        base_timeframe=args.timeframe,
    )
    if not filled:
        print(f"[fetch_ohlcv_cli] {args.trading_pair}@{args.broker} ({args.market}) all up to date over {full_range}")
    else:
        print(
            f"[fetch_ohlcv_cli] filled {len(filled)} gap(s) for {args.trading_pair}@{args.broker} "
            f"({args.market}): {', '.join(filled)}"
        )


def main(args: argparse.Namespace) -> None:
    if args.list_gaps or args.list_overlaps:
        _run_list_mode(args)
    elif args.date_range is not None:
        _run_fetch_range(args, args.date_range)
    elif args.days is not None:
        _run_fetch_range(args, date_range_to_string(days=args.days))
    else:
        _run_default_backfill(args)


if __name__ == "__main__":
    main(_parse_args())

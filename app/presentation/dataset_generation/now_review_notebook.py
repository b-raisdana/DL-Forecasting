import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.typing as pt
from application.model_implementations.tier1_000.datafeeder_input3_outcome1 import DatasetBundle, build_dataset
from application.model_implementations.tier1_000.model import (
    BRANCH_TIMEFRAMES,
    BRANCH_WINDOW_LENGTHS,
    CANDLE_FEATURE_COLUMNS,
)
from config import app_config
from domain.schemas.common.OHLCVA import MultiTimeframeOHLCVA
from helper.date_utils import date_range_to_string
from helper.pandera import pandera_validate
from plotly import graph_objects as go
from plotly.subplots import make_subplots

SYMBOLS = app_config.SYMBOLS
RANGE_DAYS = 14
_ACTION_LABELS = ["long", "short", "none"]
_AUX_FEATURE_NAMES = [f"{tf_name}_{column}" for tf_name in BRANCH_TIMEFRAMES for column in CANDLE_FEATURE_COLUMNS]
_CACHE_RE = re.compile(r"^(?P<kind>multi_timeframe_ohlcva?|ohlcva?)\.(?P<range>.+)\.(?:zip|feather|parquet)$")


@dataclass(frozen=True)
class CacheFile:
    path: Path
    data_frame_type: str
    date_range: str
    start: pd.Timestamp
    end: pd.Timestamp


def cached_multi_timeframe_files(symbol: str) -> list[CacheFile]:
    symbol_path = Path(app_config.path_of_data) / "dataset_db" / "multi_timeframe_ohlcv" / "Spot" / symbol / "Kucoin"
    if not symbol_path.exists():
        return []
    files: list[CacheFile] = []
    for path in symbol_path.iterdir():
        match = _CACHE_RE.match(path.name)
        if not match or not match.group("kind").startswith("multi_timeframe_ohlcv"):
            continue
        start, end = _parse_date_range(match.group("range"))
        files.append(CacheFile(path, match.group("kind"), match.group("range"), start, end))
    return sorted(files, key=lambda item: (item.start, item.end, item.path.name))


@pandera_validate
def cache_summary(symbols: list[str] = SYMBOLS) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        files = cached_multi_timeframe_files(symbol)
        if not files:
            rows.append({"symbol": symbol, "files": 0, "start": None, "end": None, "largest_days": None})
            continue
        largest = max(files, key=lambda item: item.end - item.start)
        rows.append(
            {
                "symbol": symbol,
                "files": len(files),
                "start": min(item.start for item in files),
                "end": max(item.end for item in files),
                "largest_days": round((largest.end - largest.start) / pd.Timedelta(days=1), 2),
            }
        )
    return pd.DataFrame(rows)


@pandera_validate
def load_cached_multi_timeframe_ohlcv(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame[MultiTimeframeOHLCVA]:
    """Stitches every locally cached daily file overlapping [start, end] together — the on-disk cache
    is one file per calendar day (see cache_summary()), so no single file ever covers a multi-day
    window on its own."""
    files = [item for item in cached_multi_timeframe_files(symbol) if item.end > start and item.start < end]
    if not files:
        raise FileNotFoundError(f"No local cached multi-timeframe OHLCV covers {symbol} {start} to {end}")
    df = pd.concat([_read_cached_frame(item) for item in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.loc[pd.IndexSlice[:, start:end], :]


def _full_cached_span(symbol: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    files = cached_multi_timeframe_files(symbol)
    if not files:
        raise FileNotFoundError(f"No local cached multi-timeframe OHLCV found for {symbol}")
    return min(item.start for item in files), max(item.end for item in files)


def random_cached_period(symbol: str, days: int = RANGE_DAYS, seed: int = 7) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = _full_cached_span(symbol)
    if end - start < pd.Timedelta(days=days):
        raise FileNotFoundError(f"No local cached {days}-day period found for {symbol}")
    rng = np.random.default_rng(abs(hash((symbol, seed))) % (2**32))
    max_offset_hours = int(((end - start) - pd.Timedelta(days=days)) / pd.Timedelta(hours=1))
    period_start = start + pd.Timedelta(hours=int(rng.integers(0, max(1, max_offset_hours + 1))))
    return period_start, period_start + pd.Timedelta(days=days)


def build_review_samples(symbol: str, sample_count: int = 5) -> list[dict[str, object]]:
    """Builds `sample_count` NOW samples straight from
    `datafeeder_input3_outcome1.build_dataset()`'s `DatasetBundle` — the literal per-branch feature
    windows + mfe/rer/action labels the Tier1_000 model trains on, not a separate ad hoc
    reconstruction of them. Uses the full locally cached span for `symbol` (the on-disk cache is
    fragmented into one file per calendar day, and the model's widest branch window (1W x 64) needs
    many months of lookback, far more than any single day) as `build_dataset()`'s `date_range_str` —
    stays an offline review of already-fetched data as long as that span is fully cached (no
    broker/exchange call — `build_dataset` only fetches on a cache miss).
    """
    start, end = _full_cached_span(symbol)
    date_range_str = date_range_to_string(start=start, end=end)
    bundle = build_dataset(symbol, date_range_str)
    if bundle.n_samples < sample_count:
        raise RuntimeError(f"Only {bundle.n_samples} samples built for {symbol} — need {sample_count}")
    selected = np.linspace(0, bundle.n_samples - 1, sample_count + 2, dtype=int)[1:-1]
    return [_extract_sample(symbol, bundle, int(idx)) for idx in selected]


def _extract_sample(symbol: str, bundle: DatasetBundle, idx: int) -> dict[str, object]:
    return {
        "symbol": symbol,
        "now": bundle.anchor_index[idx],
        "branch_windows": {
            tf_name: pd.DataFrame(
                bundle.branch_windows[tf_name][idx],
                columns=CANDLE_FEATURE_COLUMNS,
                index=np.arange(-(BRANCH_WINDOW_LENGTHS[tf_name] - 1), 1),
            )
            for tf_name in BRANCH_TIMEFRAMES
        },
        "auxiliary_features": pd.Series(bundle.auxiliary_features[idx], index=_AUX_FEATURE_NAMES),
        "mfe": float(bundle.mfe[idx, 0]),
        "rer": float(bundle.rer[idx, 0]),
        "action": dict(zip(_ACTION_LABELS, bundle.action[idx].tolist(), strict=True)),
    }


def plot_now_sample(sample: dict[str, object], title: str | None = None) -> go.Figure:
    """One row per branch timeframe, each plotting that branch's own CANDLE_FEATURE_COLUMNS
    (ATR-relative OHLC + volume, the columns model.py's branches actually consume) across the window,
    x=0 marking NOW. Labels (mfe/rer/action) — outcome_set=1's regression + classification targets —
    are summarized in the title rather than as price ladders, since mfe/rer are ATR-relative
    distances/ratios, not price levels this bundle carries.
    """
    branch_windows = sample["branch_windows"]
    assert isinstance(branch_windows, dict)
    fig = make_subplots(
        rows=len(BRANCH_TIMEFRAMES),
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"{tf_name} branch" for tf_name in BRANCH_TIMEFRAMES],
        vertical_spacing=0.04,
    )
    for row, tf_name in enumerate(BRANCH_TIMEFRAMES, start=1):
        frame = branch_windows[tf_name]
        for column in CANDLE_FEATURE_COLUMNS:
            fig.add_trace(
                go.Scatter(
                    x=frame.index, y=frame[column], mode="lines", name=column, legendgroup=column, showlegend=(row == 1)
                ),
                row=row,
                col=1,
            )
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=row, col=1)

    action = sample["action"]
    assert isinstance(action, dict)
    action_str = ", ".join(f"{name}={value:.0f}" for name, value in action.items())
    fig.update_layout(
        title=title
        or (
            f"{sample['symbol']} NOW {sample['now']} — mfe={sample['mfe']:.4f} rer={sample['rer']:.4f} "
            f"action=[{action_str}]"
        ),
        height=220 * len(BRANCH_TIMEFRAMES),
        legend={"orientation": "h"},
    )
    fig.update_xaxes(title_text="candles before NOW", row=len(BRANCH_TIMEFRAMES), col=1)
    return fig


def plot_random_period(symbol: str, start: pd.Timestamp, end: pd.Timestamp, chart_no: int) -> go.Figure:
    mt_ohlcv = load_cached_multi_timeframe_ohlcv(symbol, start, end)
    double = mt_ohlcv.loc[pd.IndexSlice["5min", :], :]
    fig = go.Figure()
    _add_candles(fig, double, f"{symbol} 5min", opacity=1.0)
    fig.update_layout(
        title=f"{symbol} cached period {chart_no}: {start} to {end}", height=520, xaxis_rangeslider_visible=False
    )
    return fig


@pandera_validate
def _add_candles(fig: go.Figure, df: pd.DataFrame, name: str, row: int | None = None, opacity: float = 1.0) -> None:
    trace = go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=name,
        opacity=opacity,
    )
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=1)


@pandera_validate
def _read_cached_frame(cache_file: CacheFile) -> pt.DataFrame[MultiTimeframeOHLCVA]:
    if cache_file.path.suffix == ".feather":
        df = pd.read_feather(cache_file.path)
    elif cache_file.path.suffix == ".zip":
        df = pd.read_csv(cache_file.path, compression="zip")
    else:
        df = pd.read_parquet(cache_file.path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index(["timeframe", "date"]).sort_index()
    return df


def _parse_date_range(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = value.split("T", 1)
    return _parse_cache_timestamp(start), _parse_cache_timestamp(end)


def _parse_cache_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(value, "%y-%m-%d.%H-%M"), tz="UTC")

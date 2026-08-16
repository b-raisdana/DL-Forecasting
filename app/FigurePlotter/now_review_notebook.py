import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ai_modelling.dataset_generator.classic_indicators import classic_indicator_columns
from ai_modelling.dataset_generator.relative_candle import relative_candle_columns
from ai_modelling.dataset_generator.training_datasets import (
    _cached_training_frames,
    _check_gap,
    _resolve_label_frame,
    normalize,
    slicing,
)
from ai_modelling.dataset_generator.volume_feature import volume_feature_columns
from Config import app_config
from helper.data_preparation import pattern_timeframe, trigger_timeframe
from plotly import graph_objects as go
from plotly.subplots import make_subplots

SYMBOLS = ["BTCUSDT", "ETHUSDT", "AIOUSDT", "TRXUSDT", "BNBUSDT"]
DEFAULT_STRUCTURE_TF = "4h"
DEFAULT_X_SHAPE = {
    "double": (255, 5),
    "trigger": (254, 5),
    "pattern": (253, 5),
    "structure": (127, 5),
}
DEFAULT_FORECAST_BARS = 192
RANGE_DAYS = 14
_CACHE_RE = re.compile(r"^(?P<kind>multi_timeframe_ohlcva?|ohlcva?)\.(?P<range>.+)\.(?:zip|feather)$")


@dataclass(frozen=True)
class CacheFile:
    path: Path
    data_frame_type: str
    date_range: str
    start: pd.Timestamp
    end: pd.Timestamp


def cached_multi_timeframe_files(symbol: str) -> list[CacheFile]:
    symbol_path = Path(app_config.path_of_data) / "Kucoin" / "Spot" / symbol
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


def load_cached_multi_timeframe_ohlcv(
    symbol: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    chosen = _select_cache_file(symbol, start, end)
    df = _read_cached_frame(chosen)
    if start is not None or end is not None:
        start = df.index.get_level_values("date").min() if start is None else start
        end = df.index.get_level_values("date").max() if end is None else end
        df = df.loc[pd.IndexSlice[:, start:end], :]
    return df.sort_index()


def _select_cache_file(symbol: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> CacheFile:
    files = cached_multi_timeframe_files(symbol)
    if not files:
        raise FileNotFoundError(f"No local cached multi-timeframe OHLCV found for {symbol}")
    if start is None or end is None:
        return max(files, key=lambda item: item.end - item.start)
    matches = [item for item in files if item.start <= start and item.end >= end]
    if not matches:
        raise FileNotFoundError(f"No single cached file covers {symbol} {start} to {end}")
    return min(matches, key=lambda item: item.end - item.start)


def random_cached_period(symbol: str, days: int = RANGE_DAYS, seed: int = 7) -> tuple[pd.Timestamp, pd.Timestamp]:
    files = [item for item in cached_multi_timeframe_files(symbol) if item.end - item.start >= pd.Timedelta(days=days)]
    if not files:
        raise FileNotFoundError(f"No local cached {days}-day period found for {symbol}")
    chosen = max(files, key=lambda item: item.end - item.start)
    rng = np.random.default_rng(abs(hash((symbol, seed))) % (2**32))
    max_offset_hours = int(((chosen.end - chosen.start) - pd.Timedelta(days=days)) / pd.Timedelta(hours=1))
    start = chosen.start + pd.Timedelta(hours=int(rng.integers(0, max(1, max_offset_hours + 1))))
    return start, start + pd.Timedelta(days=days)


def build_review_samples(
    symbol: str,
    sample_count: int = 5,
    structure_tf: str = DEFAULT_STRUCTURE_TF,
    forecast_bars: int = DEFAULT_FORECAST_BARS,
    label_tf: str | None = "5min",
    x_shape: dict[str, tuple[int, int]] = DEFAULT_X_SHAPE,
) -> list[dict[str, object]]:
    mt_ohlcv = load_cached_multi_timeframe_ohlcv(symbol)
    pattern_tf = pattern_timeframe(structure_tf)
    trigger_tf = trigger_timeframe(structure_tf)
    double_tf = pattern_timeframe(trigger_tf)
    resolved_label_tf, label_frame = _resolve_label_frame(structure_tf, pattern_tf, trigger_tf, double_tf, label_tf)
    train_safe_start, train_safe_end, _, dfs = _cached_training_frames(
        mt_ohlcv, structure_tf, pattern_tf, trigger_tf, double_tf, resolved_label_tf, label_frame, forecast_bars
    )
    candidates = dfs["double"].loc[pd.IndexSlice[train_safe_start:train_safe_end], :].index
    if len(candidates) < sample_count:
        raise RuntimeError(f"Only {len(candidates)} safe NOW candidates found for {symbol}")
    selected = candidates[np.linspace(0, len(candidates) - 1, sample_count + 2, dtype=int)[1:-1]]
    return [
        _build_sample_at_now(symbol, dfs, now, structure_tf, pattern_tf, trigger_tf, double_tf, x_shape, forecast_bars)
        for now in selected
    ]


def plot_now_sample(sample: dict[str, object], title: str | None = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05)
    _add_sample_traces(fig, sample, row=1, normalized=True)
    _add_sample_traces(fig, sample, row=2, normalized=False)
    _add_label_lines(fig, sample, row=1)
    _add_label_lines(fig, sample, row=2)
    fig.update_layout(
        title=title or f"{sample['symbol']} NOW {sample['now']}",
        height=850,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend={"orientation": "h"},
    )
    fig.update_yaxes(title_text="normalized", row=1, col=1)
    fig.update_yaxes(title_text="original", row=2, col=1)
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


def _build_sample_at_now(
    symbol: str,
    dfs: dict[str, pd.DataFrame],
    now: pd.Timestamp,
    structure_tf: str,
    pattern_tf: str,
    trigger_tf: str,
    double_tf: str,
    x_shape: dict[str, tuple[int, int]],
    forecast_bars: int,
) -> dict[str, object]:
    training_x_columns = (
        ["open", "high", "low", "close", "volume"]
        + classic_indicator_columns()
        + relative_candle_columns()
        + volume_feature_columns()
    )
    double_end = pd.Timestamp(now)
    trigger_end = double_end - x_shape["double"][0] * pd.to_timedelta(double_tf)
    pattern_end = trigger_end - x_shape["trigger"][0] * pd.to_timedelta(trigger_tf)
    structure_end = pattern_end - x_shape["pattern"][0] * pd.to_timedelta(pattern_tf)
    future = dfs["future"].loc[pd.IndexSlice[double_end:], :].iloc[:forecast_bars]
    double_slice, pattern_slice, structure_slice, trigger_slice = slicing(
        dfs, structure_end, pattern_end, trigger_end, double_end, training_x_columns, x_shape
    )
    gap_error = _check_gap(
        [
            (structure_tf, structure_slice, "structure"),
            (pattern_tf, pattern_slice, "pattern"),
            (trigger_tf, trigger_slice, "trigger"),
            (double_tf, double_slice, "double"),
        ],
        x_shape,
    )
    if gap_error is not None:
        raise RuntimeError(gap_error)
    sc_double, sc_pattern, sc_trigger, sc_structure, sc_future = normalize(
        structure_slice, pattern_slice, trigger_slice, double_slice, future
    )
    return {
        "symbol": symbol,
        "now": double_end,
        "original": {
            "structure": structure_slice,
            "pattern": pattern_slice,
            "trigger": trigger_slice,
            "double": double_slice,
            "future": future,
        },
        "normalized": {
            "structure": sc_structure,
            "pattern": sc_pattern,
            "trigger": sc_trigger,
            "double": sc_double,
            "future": sc_future,
        },
        "label": future.iloc[0],
    }


def _add_sample_traces(fig: go.Figure, sample: dict[str, object], row: int, normalized: bool) -> None:
    key = "normalized" if normalized else "original"
    frames = sample[key]
    assert isinstance(frames, dict)
    for level in ["structure", "pattern", "trigger", "double"]:
        _add_candles(fig, frames[level], f"{key} {level}", row=row, opacity=1.0)
    _add_candles(fig, frames["future"], f"{key} future labels", row=row, opacity=0.5)
    fig.add_vline(x=sample["now"], line_dash="dash", line_color="black", row=row, col=1)


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


def _add_label_lines(fig: go.Figure, sample: dict[str, object], row: int) -> None:
    label = sample["label"]
    now = sample["now"]
    assert isinstance(label, pd.Series)
    assert isinstance(now, pd.Timestamp)
    if float(label.get("long_signal", 0)) > 0:
        _add_ladder(fig, now, label["worst_long_open"], label["max_high"], label["long_sl_distance"], "long", row)
    if float(label.get("short_signal", 0)) > 0:
        _add_ladder(fig, now, label["worst_short_open"], label["min_low"], label["short_sl_distance"], "short", row)


def _add_ladder(
    fig: go.Figure, now: pd.Timestamp, entry: float, tp4: float, sl_distance: float, side: str, row: int
) -> None:
    color = "#1f9d55" if side == "long" else "#d64545"
    sl = entry - sl_distance if side == "long" else entry + sl_distance
    levels = {
        "SL": sl,
        "TP1": entry + (tp4 - entry) * 0.25,
        "TP2": entry + (tp4 - entry) * 0.5,
        "TP3": entry + (tp4 - entry) * 0.75,
        "TP4": tp4,
    }
    for name, y in levels.items():
        fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=f"{side} {name}", row=row, col=1)


def _read_cached_frame(cache_file: CacheFile) -> pd.DataFrame:
    if cache_file.path.suffix == ".feather":
        df = pd.read_feather(cache_file.path)
    else:
        df = pd.read_csv(cache_file.path, compression="zip")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index(["timeframe", "date"]).sort_index()
    return df


def _parse_date_range(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = value.split("T", 1)
    return _parse_cache_timestamp(start), _parse_cache_timestamp(end)


def _parse_cache_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(value, "%y-%m-%d.%H-%M"), tz="UTC")

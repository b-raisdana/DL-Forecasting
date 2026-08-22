import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from application.model_implementations.tier1_000.datafeeder_input3_outcome1 import (
    DatasetBundle,
    build_dataset,
)
from application.model_implementations.tier1_000.model import (
    BRANCH_TIMEFRAMES,
    CANDLE_FEATURE_COLUMNS,
)
from config import app_config
from helper.date_utils import date_range_to_string
from helper.pandera import pandera_validate
from plotly import graph_objects as go
from plotly.subplots import make_subplots

SYMBOLS = app_config.SYMBOLS
RANGE_DAYS = 14
_ACTION_LABELS = ["long", "short", "none"]
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


def _parse_date_range(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = value.split("T", 1)
    return _parse_cache_timestamp(start), _parse_cache_timestamp(end)


def _parse_cache_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(value, "%y-%m-%d.%H-%M"), tz="UTC")


@pandera_validate
def _read_cached_frame(cache_file: CacheFile) -> pd.DataFrame:
    if cache_file.path.suffix == ".parquet":
        df = pd.read_parquet(cache_file.path)
    elif cache_file.path.suffix == ".feather":
        df = pd.read_feather(cache_file.path)
    else:
        df = pd.read_csv(cache_file.path, compression="zip")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index(["timeframe", "date"]).sort_index()
    return df


@pandera_validate
def load_cached_multi_timeframe_ohlcv(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
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


def build_review_samples(symbol: str, sample_count: int = 5) -> list[dict[str, object]]:
    start, end = _full_cached_span(symbol)
    date_range_str = date_range_to_string(start=start, end=end)
    bundle = build_dataset(symbol, date_range_str)
    if bundle.n_samples < sample_count:
        raise RuntimeError(f"Only {bundle.n_samples} samples built for {symbol}")
    selected = np.linspace(0, bundle.n_samples - 1, sample_count + 2, dtype=int)[1:-1]
    return [_extract_sample(symbol, bundle, int(idx)) for idx in selected]


def _extract_sample(symbol: str, bundle: DatasetBundle, idx: int) -> dict[str, object]:
    anchor = bundle.anchor_index[idx]
    mt_ohlcv = load_cached_multi_timeframe_ohlcv(
        symbol,
        anchor - pd.Timedelta(minutes=120),
        anchor + pd.Timedelta(minutes=300),
    )
    base_5min = mt_ohlcv.loc[pd.IndexSlice["5min", :], :].sort_index()
    return {
        "symbol": symbol,
        "now": anchor,
        "base_5min": base_5min,
        "mfe": float(bundle.mfe[idx, 0]),
        "rer": float(bundle.rer[idx, 0]),
        "action": dict(zip(_ACTION_LABELS, bundle.action[idx].tolist(), strict=True)),
    }


def plot_labeled_future_window(sample: dict[str, object], title: str | None = None) -> go.Figure:
    base_5min = cast(pd.DataFrame, sample["base_5min"])
    anchor = cast(pd.Timestamp, sample["now"])

    future_end = anchor + pd.Timedelta(minutes=240)

    fig = go.Figure()

    mask_past = base_5min.index <= anchor
    mask_future = (base_5min.index > anchor) & (base_5min.index <= future_end)

    past = base_5min[mask_past]
    future = base_5min[mask_future]

    fig.add_trace(
        go.Candlestick(
            x=past.index,
            open=past["open"],
            high=past["high"],
            low=past["low"],
            close=past["close"],
            name="history",
            opacity=0.9,
        )
    )

    if not future.empty:
        fig.add_trace(
            go.Candlestick(
                x=future.index,
                open=future["open"],
                high=future["high"],
                low=future["low"],
                close=future["close"],
                name="future window",
                opacity=0.7,
            )
        )

        for i, (ts, row) in enumerate(future.iterrows()):
            fig.add_trace(
                go.Scatter(
                    x=[ts, ts],
                    y=[row["low"], row["high"]],
                    mode="lines",
                    line={"width": 1, "color": "rgba(255,0,0,0.15)"},
                    showlegend=(i == 0),
                    name="future range",
                )
            )

    action = cast(dict[str, float], sample["action"])
    action_str = ", ".join(f"{name}={value:.0f}" for name, value in action.items())

    fig.add_vline(
        x=anchor,
        line_dash="dash",
        line_color="black",
        annotation_text="NOW",
        annotation_position="top",
    )

    if not future.empty:
        label_x = future.index[-1]
        label_y = future["high"].max() * 1.002
        fig.add_annotation(
            x=label_x,
            y=label_y,
            text=f"action=[{action_str}]<br>mfe={sample['mfe']:.4f}<br>rer={sample['rer']:.4f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#636363",
            ax=-60,
            ay=-40,
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.9,
        )

    fig.update_layout(
        title=title or f"{sample['symbol']} NOW={anchor} — future window with labels",
        xaxis_title="time (UTC)",
        yaxis_title="price",
        height=520,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
    )
    return fig


def plot_labeled_branch_sample(sample: dict[str, object], title: str | None = None) -> go.Figure:
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

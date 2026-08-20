from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from archive_not_used_trash.application.dataset_generation.profit_loss.profit_loss_adder import add_long_n_short_profit
from config import app_config
from domain.schemas.common.MultiTimeframe import MultiTimeframe
from domain.technical_analysis.classic_indicators import classic_indicator_columns, scaleless_indicators
from archive_not_used_trash.domain.technical_analysis.classic_indicators import add_classic_indicators
from helper.data_preparation import single_timeframe
from archive_not_used_trash.helper.data_preparation import pattern_timeframe, trigger_timeframe
from helper.functions import date_range
from helper.importer import pt
from helper.logging.do_log import log_d

from application.dataset_generation.relative_candle import add_relative_candle_columns
from archive_not_used_trash.application.dataset_generation.relative_candle import relative_candle_columns
from application.dataset_generation.volume_feature import add_volume_feature_columns
from archive_not_used_trash.application.dataset_generation.volume_feature import volume_feature_columns

Shape = tuple[int, ...] | list[object] | dict[str, object] | None


def single_timeframe_n_indicators(mt_ohlcv: pt.DataFrame[MultiTimeframe], timeframe: str) -> pd.DataFrame:  # type: ignore[valid-type]
    ohlcv = single_timeframe(mt_ohlcv, timeframe)
    ohlcv = add_classic_indicators(ohlcv)  # type: ignore[no-untyped-call]
    ohlcv = add_relative_candle_columns(ohlcv)
    ohlcv = add_volume_feature_columns(ohlcv)
    return ohlcv


class _BatchSample(NamedTuple):
    x_slices: dict[str, pd.DataFrame]  # keys: double/trigger/pattern/structure -> sc_*_slice[training_x_columns]
    y_df: pd.DataFrame
    y_debug_df: pd.DataFrame
    is_actionable: bool


def _resolve_label_frame(
    structure_tf: str, pattern_tf: str, trigger_tf: str, double_tf: str, label_tf: str | None
) -> tuple[str, str]:
    label_frame_by_tf = {structure_tf: "structure", pattern_tf: "pattern", trigger_tf: "trigger", double_tf: "double"}
    resolved_label_tf = label_tf or trigger_tf
    if resolved_label_tf not in label_frame_by_tf:
        raise ValueError(f"label_tf={resolved_label_tf} must be one of {list(label_frame_by_tf.keys())}")
    return resolved_label_tf, label_frame_by_tf[resolved_label_tf]


def _build_frames_by_level(
    mt_ohlcv: pt.DataFrame[MultiTimeframe],  # type: ignore[valid-type]
    structure_tf: str,
    pattern_tf: str,
    trigger_tf: str,
    double_tf: str,
) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for df_name, timeframe in [
        ("structure", structure_tf),
        ("pattern", pattern_tf),
        ("trigger", trigger_tf),
        ("double", double_tf),
    ]:
        dfs[df_name] = single_timeframe_n_indicators(mt_ohlcv, timeframe)
    return dfs


# Producer loops (npz_batch.py/ram_batch.py/stream_loader.py) load mt_ohlcv once per quarter, then call
# train_data_of_mt_n_profit() up to ~100x against that same object with the same (structure_tf, label_tf,
# forecast_trigger_bars) — indicators and the rolling-window label computation in profit_loss_adder.py
# were being recomputed from scratch on every one of those calls. pd.DataFrame is unhashable
# (NDFrame.__hash__ = None), so the memo lives on the object itself via .attrs rather than as a dict key
# — self-bounded to mt_ohlcv's own lifetime. See the cache-or-generate skill.
_TRAINING_FRAME_CACHE_ATTR = "_dlf_training_frame_cache"


def _cached_training_frames(
    mt_ohlcv: pt.DataFrame[MultiTimeframe],  # type: ignore[valid-type]
    structure_tf: str,
    pattern_tf: str,
    trigger_tf: str,
    double_tf: str,
    label_tf: str,
    label_frame: str,
    forecast_trigger_bars: int,
) -> tuple[datetime, datetime, int, dict[str, pd.DataFrame]]:
    cache: dict[tuple[str, str, int], tuple[datetime, datetime, int, dict[str, pd.DataFrame]]] = (
        mt_ohlcv.attrs.setdefault(_TRAINING_FRAME_CACHE_ATTR, {})
    )
    cache_key = (structure_tf, label_tf, forecast_trigger_bars)
    if cache_key not in cache:
        dfs = _build_frames_by_level(mt_ohlcv, structure_tf, pattern_tf, trigger_tf, double_tf)
        dfs["future"] = add_long_n_short_profit(
            ohlc=dfs[label_frame],  # type: ignore[no-untyped-call]
            position_max_bars=forecast_trigger_bars,
            trigger_tf=label_tf,
        )
        cache[cache_key] = _compute_safe_training_range(dfs, structure_tf, label_tf, forecast_trigger_bars)
    train_safe_start, train_safe_end, duration_seconds, cached_dfs = cache[cache_key]
    return train_safe_start, train_safe_end, duration_seconds, {name: df.copy() for name, df in cached_dfs.items()}


def _compute_safe_training_range(
    dfs: dict[str, pd.DataFrame], structure_tf: str, label_tf: str, forecast_trigger_bars: int
) -> tuple[datetime, datetime, int, dict[str, pd.DataFrame]]:
    train_safe_end, train_safe_start, dfs = not_na_range(dfs)
    train_safe_start += pd.to_timedelta(structure_tf)
    train_safe_end -= 2 * forecast_trigger_bars * pd.to_timedelta(label_tf)
    duration_seconds = int((train_safe_end - train_safe_start) / timedelta(seconds=1))
    if duration_seconds <= 0:
        start, end = date_range(app_config.processing_date_range)
        raise RuntimeError(
            f"Extend date boundary +{-duration_seconds}s({duration_seconds / (60 * 60 * 24)}days, "
            f"start:{start}<{start + duration_seconds * timedelta(seconds=1)} or "
            f"end:{end}>{end - duration_seconds * timedelta(seconds=1)}) to make possible range of end dates positive!"
        )
    return train_safe_start, train_safe_end, duration_seconds, dfs


def _check_gap(slices_by_tf: list[tuple[str, pd.DataFrame, str]], x_shape: dict[str, tuple[int, int]]) -> str | None:
    for timeframe, slice_df, level in slices_by_tf:
        if (
            abs((slice_df.index.max() - slice_df.index.min()) / pd.to_timedelta(timeframe) - (x_shape[level][0] - 1))
            > app_config.max_x_gap
        ):
            return (
                f"Skipping: gap of > {app_config.max_x_gap} bars in {level}/{timeframe}-"
                f"{app_config.under_process_exchange}/{app_config.under_process_symbol}/{timeframe}:"
                f"{slice_df.index.min()}-{slice_df.index.max()}"
            )
    return None


def _shapes_ok_or_log(
    sc_slices: dict[str, pd.DataFrame],
    training_x_columns: list[str],
    x_shape: dict[str, tuple[int, int]],
    verbose: bool,
) -> bool:
    lengths_ok = {
        level: len(np.array(sc_slices[level][training_x_columns])) == x_shape[level][0]
        for level in ("double", "trigger", "pattern", "structure")
    }
    if all(lengths_ok.values()):
        return True
    if verbose:
        reasons = "".join(
            f"len(np.array(sc_{level}_slice[training_x_columns])) != x_shape['{level}'][0]"
            for level, ok in lengths_ok.items()
            if not ok
        )
        log_d(f"Skipped by:{reasons}")
    return False


def _build_one_sample(
    dfs: dict[str, pd.DataFrame],
    duration_seconds: int,
    double_tf: str,
    trigger_tf: str,
    pattern_tf: str,
    structure_tf: str,
    x_shape: dict[str, tuple[int, int]],
    train_safe_end: datetime,
    train_safe_start: datetime,
    forecast_trigger_bars: int,
    training_x_columns: list[str],
    training_y_columns: list[str],
    actionable_rate: float,
    actionable_batches: int,
    not_actionable_batches: int,
    verbose: bool,
) -> _BatchSample | None:
    double_end, trigger_end, pattern_end, structure_end = batch_ends(
        duration_seconds, double_tf, trigger_tf, pattern_tf, structure_tf, x_shape, train_safe_end, train_safe_start
    )
    future_slice = dfs["future"].loc[pd.IndexSlice[double_end:], :].iloc[:forecast_trigger_bars]
    if future_slice.shape[0] != forecast_trigger_bars:
        raise AssertionError(future_slice.shape[0] != forecast_trigger_bars)
    is_actionable = True
    if future_slice["long_signal"][0] == 0 and future_slice["short_signal"][0] == 0:
        is_actionable = False
        if actionable_batches == 0 or (not_actionable_batches / actionable_batches > (1 / actionable_rate - 1)):
            return None
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
        log_d(gap_error)
        return None
    (
        sc_double_slice,
        sc_pattern_slice,
        sc_trigger_slice,
        sc_structure_slice,
        sc_future,
    ) = normalize(structure_slice, pattern_slice, trigger_slice, double_slice, future_slice)
    sc_slices = {
        "double": sc_double_slice,
        "trigger": sc_trigger_slice,
        "pattern": sc_pattern_slice,
        "structure": sc_structure_slice,
    }
    if not _shapes_ok_or_log(sc_slices, training_x_columns, x_shape, verbose):
        return None
    return _BatchSample(
        x_slices={level: sc_slices[level][training_x_columns] for level in sc_slices},
        y_df=sc_future[training_y_columns],
        y_debug_df=sc_future,
        is_actionable=is_actionable,
    )


def _finalize_batch_arrays(
    x_dfs_by_level: dict[str, list[pd.DataFrame]],
    xs_lists: dict[str, list[npt.NDArray[np.float64]]],
    ys_list: list[npt.NDArray[np.float64]],
) -> tuple[dict[str, pd.DataFrame], dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    x_dfs: dict[str, pd.DataFrame] = {key: pd.concat(dfs_list) for key, dfs_list in x_dfs_by_level.items()}
    Xs: dict[str, npt.NDArray[np.float64]] = {key: np.array(values) for key, values in xs_lists.items()}
    ys: npt.NDArray[np.float64] = np.array(ys_list)
    return x_dfs, Xs, ys


def _assert_no_nan(Xs: dict[str, npt.NDArray[np.float64]], ys: npt.NDArray[np.float64]) -> None:
    for key in Xs:
        if np.isnan(Xs[key]).any():
            raise AssertionError(f"Found NA in Xs[{key}]")
    if np.isnan(ys).any():
        raise AssertionError("Found NA in ys")


# @profile_it
def train_data_of_mt_n_profit(
    structure_tf: str,
    mt_ohlcv: pt.DataFrame[MultiTimeframe],  # type: ignore[valid-type]
    x_shape: dict[str, tuple[int, int]],
    batch_size: int,
    dataset_batches: int = 100,
    forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1,
    actionable_rate: float = 0.2,  # try to generate 20% actionable, 80% not-actionable
    verbose: bool = True,
    label_tf: str | None = None,
    # Timeframe labels (max_high/min_low/long_signal/...) are generated on.
    # Defaults to trigger_tf (today's behavior: forecast_trigger_bars counts
    # trigger_tf bars). Pass double_tf (or any of structure/pattern/trigger/double
    # tf) to move label generation to that frame instead; forecast_trigger_bars
    # then counts bars of label_tf, so callers switching frames must also update it
    # (e.g. 48 bars for a 4h horizon on the 5-min double frame).
) -> tuple[
    dict[str, npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
    dict[str, pd.DataFrame],
    list[pd.DataFrame],
    str,
    list[pd.DataFrame],
]:
    training_x_columns = (
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        + classic_indicator_columns()  # type: ignore[no-untyped-call]
        + relative_candle_columns()
        + volume_feature_columns()
    )
    training_y_columns = [
        "long_signal",
        "short_signal",
        "min_low",
        "max_high",
        "long_profit",
        "short_profit",
        "long_risk",
        "short_risk",
        "long_drawdown",
        "short_drawdown",
        "long_drawdown",
        "short_drawdown",
    ]
    pattern_tf = pattern_timeframe(structure_tf)  # type: ignore[no-untyped-call]
    trigger_tf = trigger_timeframe(structure_tf)  # type: ignore[no-untyped-call]
    double_tf = pattern_timeframe(trigger_timeframe(structure_tf))  # type: ignore[no-untyped-call]
    label_tf, label_frame = _resolve_label_frame(structure_tf, pattern_tf, trigger_tf, double_tf, label_tf)
    train_safe_start, train_safe_end, duration_seconds, dfs = _cached_training_frames(
        mt_ohlcv, structure_tf, pattern_tf, trigger_tf, double_tf, label_tf, label_frame, forecast_trigger_bars
    )

    x_dfs_by_level: dict[str, list[pd.DataFrame]] = {"double": [], "trigger": [], "pattern": [], "structure": []}
    xs_lists: dict[str, list[npt.NDArray[np.float64]]] = {"double": [], "trigger": [], "pattern": [], "structure": []}
    y_dfs: list[pd.DataFrame] = []
    y_debug_dfs: list[pd.DataFrame] = []
    ys_list: list[npt.NDArray[np.float64]] = []
    remained_samples = batch_size * dataset_batches
    actionable_batches = 0
    not_actionable_batches = 0
    while remained_samples > 0:
        sample = _build_one_sample(
            dfs,
            duration_seconds,
            double_tf,
            trigger_tf,
            pattern_tf,
            structure_tf,
            x_shape,
            train_safe_end,
            train_safe_start,
            forecast_trigger_bars,
            training_x_columns,
            training_y_columns,
            actionable_rate,
            actionable_batches,
            not_actionable_batches,
            verbose,
        )
        if sample is None:
            continue
        for level in ("double", "trigger", "pattern", "structure"):
            x_dfs_by_level[level].append(sample.x_slices[level])
            xs_lists[level].append(np.array(x_dfs_by_level[level][-1]))
        y_dfs.append(sample.y_df)
        y_debug_dfs.append(sample.y_debug_df)
        ys_list.append(np.array(sample.y_df.iloc[0][["short_signal", "long_signal"]]))
        remained_samples -= 1
        if sample.is_actionable:
            actionable_batches += 1
        else:
            not_actionable_batches += 1
        if verbose and (remained_samples % 10) == 0 and remained_samples > 0:
            log_d(f"Remained Samples {remained_samples}/{batch_size}")

    x_dfs, Xs, ys = _finalize_batch_arrays(x_dfs_by_level, xs_lists, ys_list)
    shape_assertion(
        Xs=Xs,
        x_dfs=x_dfs,
        y_dfs=y_dfs,
        y_tester_dfs=y_debug_dfs,
        ys=ys,
        x_shape=x_shape,
        batch_size=batch_size,
        dataset_batched=dataset_batches,
        forecast_trigger_bars=forecast_trigger_bars,
    )
    _assert_no_nan(Xs, ys)
    return Xs, ys, x_dfs, y_dfs, trigger_tf, y_debug_dfs


def shape_assertion(
    Xs: dict[str, npt.NDArray[np.float64]],
    x_dfs: dict[str, pd.DataFrame],
    y_dfs: list[pd.DataFrame],
    y_tester_dfs: list[pd.DataFrame],
    ys: npt.NDArray[np.float64],
    x_shape: dict[str, tuple[int, int]],
    batch_size: int = 120,
    dataset_batched: int = 100,
    forecast_trigger_bars: int = 192,
    y_parameters: int = 2,
    y_df_parameters: int = 12,
) -> None:
    """
    x_shape = {'double': (255, 5), 'indicators': (129,), 'pattern': (253, 5), 'structure': (127, 5),
              'trigger': (254, 5)}
    """
    b_l = batch_size * dataset_batched
    # i_l = x_shape['indicators']
    x_shape_assertion(Xs, b_l, x_shape)
    if get_shape(ys) != (b_l, y_parameters):
        raise AssertionError(f"get_shape(ys) != (b_l, {y_parameters})")
    from deepdiff import DeepDiff

    if (
        DeepDiff(
            get_shape(x_dfs),
            {
                "double": (b_l * x_shape["double"][0], x_shape["double"][1]),
                "trigger": (b_l * x_shape["trigger"][0], x_shape["trigger"][1]),
                "pattern": (b_l * x_shape["pattern"][0], x_shape["pattern"][1]),
                "structure": (b_l * x_shape["structure"][0], x_shape["structure"][1]),
            },
        )
        != {}
    ):
        raise AssertionError("DeepDiff(get_shape(x_dfs), {")
    if get_shape(y_dfs) != [
        b_l,
        (
            forecast_trigger_bars,
            y_df_parameters,
        ),
    ]:
        raise AssertionError(f"get_shape(y_dfs) != [b_l, ({y_parameters},)]")
    if get_shape(y_tester_dfs) != [b_l, (forecast_trigger_bars, 44)]:
        raise AssertionError("get_shape(y_tester_dfs) != [b_l, (forecast_trigger_bars, 5)]")  # todo: this happens!


def x_shape_assertion(
    Xs: dict[str, npt.NDArray[np.float64]],
    batch_size: int,
    x_shape: dict[str, tuple[int, int]],
    num_of_indicators: int = (
        len(classic_indicator_columns())  # type: ignore[no-untyped-call]
        + len(relative_candle_columns())
        + len(volume_feature_columns())
    ),
) -> None:
    # i_l = x_shape['indicators'][0]
    b_l = batch_size
    if get_shape(Xs) != {
        "double": (b_l, x_shape["double"][0], 5 + num_of_indicators),
        # 'double-indicators': (b_l, i_l, num_of_indicators),
        "pattern": (b_l, x_shape["pattern"][0], 5 + num_of_indicators),
        # 'pattern-indicators': (b_l, i_l, num_of_indicators),
        "structure": (b_l, x_shape["structure"][0], 5 + num_of_indicators),
        # 'structure-indicators': (b_l, i_l, num_of_indicators),
        "trigger": (b_l, x_shape["trigger"][0], 5 + num_of_indicators),
        # 'trigger-indicators': (b_l, i_l, num_of_indicators)
    }:
        raise AssertionError("get_shape(Xs) != {")


def not_na_range(dfs: dict[str, pd.DataFrame]) -> tuple[datetime, datetime, dict[str, pd.DataFrame]]:
    future_end = dfs["future"].dropna(how="any", axis=0).index[-1]
    double_end = dfs["double"].dropna(how="any", axis=0).index[-1]
    train_safe_start = dfs["structure"].dropna(how="any", axis=0).index[1]
    train_safe_end = min(future_end, double_end)
    for df_name in dfs:
        dfs[df_name] = dfs[df_name].loc[pd.IndexSlice[train_safe_start:train_safe_end, :]].dropna(how="any", axis=0)
        if dfs[df_name].isna().any().any():
            raise AssertionError(f"Found NA in dfs[{df_name}]")
    return train_safe_end, train_safe_start, dfs


def get_shape(obj: object) -> Shape:
    if isinstance(obj, np.ndarray):
        return obj.shape
    elif isinstance(obj, pd.DataFrame):
        return obj.shape  # (rows, columns)
    elif isinstance(obj, pd.Series):
        return (obj.shape[0],)  # 1D shape
    elif isinstance(obj, (list, tuple)):
        return [len(obj)] + ([get_shape(obj[0])] if obj else [])
    elif isinstance(obj, dict):
        return {k: get_shape(v) for k, v in obj.items()}
    else:
        return None  # Base case for non-iterables


def scale_slice(
    slc: pd.DataFrame,
    price_shift: float,
    price_scale: float,
    volume_scale: float,
    mt_scale_adjuster: float = 1 / 20,
) -> pd.DataFrame:
    t = slc.copy()
    price_base_columns = (
        [
            "open",
            "high",
            "low",
            "close",
            "min_low",
            "max_high",
            "worst_long_open",
            "worst_short_open",
            "quantile_long_min_low",
            "quantile_short_max_high",
        ]
        + list(set(classic_indicator_columns()) - set(scaleless_indicators()))  # type: ignore[no-untyped-call]
    )
    atr_based_columns = [col for col in slc.columns if "_profit" in col or "_drawdown" in col]
    for column in slc.columns:
        if column in price_base_columns:
            t[column] = (t[column] + price_shift) * price_scale * mt_scale_adjuster
            continue
        if column in atr_based_columns:
            t[column] = t[column] * price_scale * mt_scale_adjuster
            continue

    if "volume" in slc.columns:
        t["volume"] = t["volume"] * volume_scale
        t["rsi"] = (t["rsi"] - 50) / 8
        t["mfi"] = (t["mfi"] - 50) / 8

    return t


def normalize(
    structure_slice: pd.DataFrame,
    pattern_slice: pd.DataFrame,
    trigger_slice: pd.DataFrame,
    double_slice: pd.DataFrame,
    future_slice: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    (
        price_scale,
        price_shift,
        volume_scale,
    ) = scaler_trainer(
        {
            "double": double_slice,
            "pattern": pattern_slice,
            "structure": structure_slice,
            "trigger": trigger_slice,
        },
        mean_atr=trigger_slice["atr"].mean(),
        close=double_slice.iloc[-1]["close"],
    )
    (sc_double_slice, sc_trigger_slice, sc_pattern_slice, sc_structure_slice, sc_future_slice) = (
        scale_slice(t, price_shift, price_scale, volume_scale)
        for t in [double_slice, trigger_slice, pattern_slice, structure_slice, future_slice]
    )
    return sc_double_slice, sc_pattern_slice, sc_trigger_slice, sc_structure_slice, sc_future_slice


def batch_ends(
    duration_seconds: int,
    double_tf: str,
    trigger_tf: str,
    pattern_tf: str,
    structure_tf: str,
    x_shape: dict[str, tuple[int, int]],
    train_safe_end: datetime,
    train_safe_start: datetime,
) -> tuple[datetime, datetime, datetime, datetime]:
    batch_length = int(
        (
            pd.to_timedelta(double_tf) * x_shape["double"][0]
            + pd.to_timedelta(trigger_tf) * x_shape["trigger"][0]
            + pd.to_timedelta(pattern_tf) * x_shape["pattern"][0]
            + pd.to_timedelta(structure_tf) * x_shape["structure"][0]
        ).total_seconds()
    )
    relative_double_end = np.random.randint(0, duration_seconds - batch_length)
    double_end: datetime = train_safe_end - relative_double_end * timedelta(seconds=1)
    trigger_end = double_end - x_shape["double"][0] * pd.to_timedelta(double_tf)
    pattern_end = trigger_end - x_shape["trigger"][0] * pd.to_timedelta(trigger_tf)
    structure_end = pattern_end - x_shape["pattern"][0] * pd.to_timedelta(pattern_tf)
    structure_start = structure_end - x_shape["structure"][0] * pd.to_timedelta(structure_tf)
    if structure_start < train_safe_start:
        raise AssertionError("structure_end is too soon!")
    return double_end, trigger_end, pattern_end, structure_end


def slicing(
    dfs: dict[str, pd.DataFrame],
    structure_end: datetime,
    pattern_end: datetime,
    trigger_end: datetime,
    double_end: datetime,
    training_x_columns: list[str],
    x_shape: dict[str, tuple[int, int]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    double_slice = dfs["double"].loc[pd.IndexSlice[:double_end], training_x_columns].iloc[-x_shape["double"][0] :]
    if len(double_slice) != x_shape["double"][0]:
        raise AssertionError("double dimension mismatch")
    if double_slice.isna().any().any():
        raise AssertionError("double_slice.isna().any().any()")

    trigger_slice = (
        dfs["trigger"].loc[pd.IndexSlice[:trigger_end], training_x_columns + ["atr"]].iloc[-x_shape["trigger"][0] :]
    )
    if len(trigger_slice) != x_shape["trigger"][0]:
        raise AssertionError("trigger dimension mismatch")
    if trigger_slice.isna().any().any():
        raise AssertionError("rigger_slice.isna().any().any()")

    pattern_slice = dfs["pattern"].loc[pd.IndexSlice[:pattern_end], training_x_columns].iloc[-x_shape["pattern"][0] :]
    if len(pattern_slice) != x_shape["pattern"][0]:
        raise AssertionError("pattern dimension mismatch")
    if pattern_slice.isna().any().any():
        raise AssertionError("pattern_slice.isna().any().any()")

    structure_slice = (
        dfs["structure"].loc[pd.IndexSlice[:structure_end], training_x_columns].iloc[-x_shape["structure"][0] :]
    )
    if len(structure_slice) != x_shape["structure"][0]:
        raise AssertionError("Structure dimension mismatch")
    if structure_slice.isna().any().any():
        raise AssertionError("structure_slice.isna().any().any()")

    return double_slice, pattern_slice, structure_slice, trigger_slice


def scaler_trainer(
    slices: dict[str, pd.DataFrame], mean_atr: float, close: float
) -> tuple[
    float,
    float,
    float,
]:
    price_scale = 1 / mean_atr
    price_shift = -close
    t_slice = pd.concat(slices)
    volume_scale = 1 / t_slice["volume"].mean()
    return (
        price_scale,
        price_shift,
        volume_scale,
    )


def ndarray_stats(input_array: npt.NDArray[np.float64], names: list[str]) -> pd.DataFrame:
    ys_mean = np.mean(input_array, axis=0)
    ys_std = np.std(input_array, axis=0)
    ys_min = np.min(input_array, axis=0)
    ys_max = np.max(input_array, axis=0)
    df_stats = pd.DataFrame({"min": ys_min, "max": ys_max, "mean": ys_mean, "std": ys_std}, index=names)
    return df_stats

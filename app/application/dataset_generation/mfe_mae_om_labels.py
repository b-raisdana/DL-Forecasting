"""mfe/mae/om/rer/action labels per docs/ML_Forecasting_System_Design/02-Data, Label & Feature
Engineering.md § targeting bid price / risk factors / TP-MAE-OM labels / model output targets.

New, self-contained implementation — not a modification of profit_loss_adder.py (that module backs
the separate cnn_lstm model's long_signal/short_signal targets and stays untouched). Fully vectorized
via numpy sliding windows instead of pandas' rolling().apply(lambda...) (Python-level per-window calls,
the pattern profit_loss_adder.py uses and docs/todos/02-training-data-labels.md flags as imprecise).
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from helper.importer import ta

HORIZON_BARS = 48  # 240 minutes / 5min
ATR_FLOOR_LENGTH = 255  # ATR(255, 15min) per § risk factors
FEE_RATE = 0.001  # F, § trading overhead
OM_GATE = 5.0  # § where can be a position?
RER_BOUND = 1 / (1 + OM_GATE - 1)  # (0, 1/4) at OM==5, per § model output targets

action_label_columns = ["action_long", "action_short", "action_none"]
regression_label_columns = ["mfe", "rer"]


def _atr_255_15min_floor(five_min_index: pd.DatetimeIndex, fifteen_min_ohlc: pd.DataFrame) -> npt.NDArray[np.float64]:
    """ATR(255, 15min), forward-filled onto five_min_index from the last *completed* 15min candle —
    merge_asof(direction='backward') so no anchor ever reads a still-forming or future 15min bar."""
    atr = ta.atr(
        high=fifteen_min_ohlc["high"],
        low=fifteen_min_ohlc["low"],
        close=fifteen_min_ohlc["close"],
        length=ATR_FLOOR_LENGTH,
    ).rename("atr_255_15min")
    anchors = pd.DataFrame(index=five_min_index.rename("date")).reset_index()
    atr_frame = atr.rename_axis("date").reset_index()
    merged = pd.merge_asof(anchors, atr_frame, on="date", direction="backward")
    return merged["atr_255_15min"].to_numpy()


def _direction_excursions(
    entry: npt.NDArray[np.float64],
    favorable_windows: npt.NDArray[np.float64],
    adverse_windows: npt.NDArray[np.float64],
    favorable_is_max: bool,
    atr_floor: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """MFE/MAE for one direction. favorable_windows/adverse_windows: (n_anchors, HORIZON_BARS).

    MFE = best favorable move from entry within the horizon.
    MAE = max(worst adverse move from entry up to (not after) the favorable-move's own bar, ATR floor)
    — "worst adverse excursion before TP4" per § risk factors; the favorable/adverse roles swap high/low
    by direction (favorable_is_max=True for Long: favorable=high, adverse=low; False for Short).
    """
    if favorable_is_max:
        best_bar = favorable_windows.argmax(axis=1)
        mfe = np.maximum(0.0, favorable_windows.max(axis=1) - entry)
    else:
        best_bar = favorable_windows.argmin(axis=1)
        mfe = np.maximum(0.0, entry - favorable_windows.min(axis=1))
    bar_positions = np.arange(favorable_windows.shape[1])[None, :]
    before_best = bar_positions <= best_bar[:, None]
    if favorable_is_max:
        # adverse = low; worst adverse before the best bar = the minimum low in that prefix
        adverse_extreme = np.where(before_best, adverse_windows, np.inf).min(axis=1)
        adverse_distance = np.maximum(0.0, entry - adverse_extreme)
    else:
        # adverse = high; worst adverse before the best bar = the maximum high in that prefix
        adverse_extreme = np.where(before_best, adverse_windows, -np.inf).max(axis=1)
        adverse_distance = np.maximum(0.0, adverse_extreme - entry)
    mae = np.maximum(adverse_distance, atr_floor)
    return mfe, mae


def add_mfe_mae_om_labels(five_min_ohlc: pd.DataFrame, fifteen_min_ohlc: pd.DataFrame) -> pd.DataFrame:
    """Per-anchor-candle Long/Short mfe/mae/om + the winning direction's action/mfe/rer.

    Entry price E = the best price reachable in the 5min candle immediately following the anchor
    (§ targeting bid price) — the horizon's own first bar, so no separate E computation is needed.
    Anchors within HORIZON_BARS of the end of five_min_ohlc have no complete horizon and are dropped
    (no lookahead past what's actually available).
    """
    high = five_min_ohlc["high"].to_numpy()
    low = five_min_ohlc["low"].to_numpy()
    n_anchors = len(high) - HORIZON_BARS
    if n_anchors <= 0:
        raise ValueError(f"five_min_ohlc has {len(high)} rows, needs > {HORIZON_BARS} for one full horizon")

    high_windows = np.lib.stride_tricks.sliding_window_view(high[1:], HORIZON_BARS)[:n_anchors]
    low_windows = np.lib.stride_tricks.sliding_window_view(low[1:], HORIZON_BARS)[:n_anchors]
    anchor_index = five_min_ohlc.index[:n_anchors]

    atr_floor = _atr_255_15min_floor(anchor_index, fifteen_min_ohlc)

    entry_long = low_windows[:, 0]
    entry_short = high_windows[:, 0]
    mfe_long, mae_long = _direction_excursions(entry_long, high_windows, low_windows, True, atr_floor)
    mfe_short, mae_short = _direction_excursions(entry_short, low_windows, high_windows, False, atr_floor)
    om_long = mfe_long / mae_long
    om_short = mfe_short / mae_short

    qualifies_long = om_long > OM_GATE
    qualifies_short = om_short > OM_GATE
    chosen_is_long = om_long >= om_short  # also the OM>5-both tie-break: higher OM wins
    action = np.select(
        [qualifies_long & ~qualifies_short, ~qualifies_long & qualifies_short, qualifies_long & qualifies_short],
        ["long", "short", np.where(chosen_is_long, "long", "short")],
        default="none",
    )

    mfe = np.where(chosen_is_long, mfe_long, mfe_short)
    mae = np.where(chosen_is_long, mae_long, mae_short)
    # rer's (0, 1/4) bound is only guaranteed under the OM>5 gate (§ model output targets) — clip so
    # non-actionable ("none") rows still carry a well-scaled regression target for the sigmoid/softplus heads.
    rer = np.clip(mae / (mfe - mae + np.finfo(np.float64).eps), 0.0, RER_BOUND)

    labels = pd.DataFrame(index=anchor_index)
    labels["mfe"] = mfe.astype(np.float32)
    labels["rer"] = rer.astype(np.float32)
    for cls in ("long", "short", "none"):
        labels[f"action_{cls}"] = (action == cls).astype(np.float32)
    return labels

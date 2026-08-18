# Appendix — MFE & RER Implementation

Actual pandas/numpy operations behind the `MFE`/`RER` labels specced in [02 § TP / MAE / OM labels](<02-Data, Label & Feature Engineering.md#tp--mae--om-labels>) and [§ model output targets](<02-Data, Label & Feature Engineering.md#model-output-targets>). Source: `app/application/dataset_generation/mfe_mae_om_labels.py` (`add_mfe_mae_om_labels`), tested in `tests/unit/dataset_generator/test_mfe_mae_om_labels.py`.

## Vectorization choice

Fully vectorized with numpy sliding windows, not pandas `rolling().apply(lambda ...)`. The older `profit_loss_adder.py` (separate model, untouched) uses that `rolling().apply` pattern (e.g. `ohlc["high"].rolling(w).apply(lambda x: x.argmax(), raw=True)`) — it calls back into Python per window, which is slow and was flagged as imprecise in `docs/todos/02-training-data-labels.md`. This module avoids both issues by building the whole future-window tensor at once with `numpy.lib.stride_tricks.sliding_window_view` and reducing it with array ops (`argmax`/`argmin`/`max`/`min` along an axis, `np.where`, `np.maximum`, `np.select`, `np.clip`) instead of a Python-level callback.

## Pipeline

1. **ATR floor, joined onto 5min anchors — `pd.merge_asof`**
   `ta.atr(high, low, close, length=255)` on 15min OHLC, then `pd.merge_asof(anchors, atr_frame, on="date", direction="backward")` — an as-of backward join so each 5min anchor picks up the last *completed* 15min ATR bar, never a still-forming or future one. Result becomes the per-anchor `atr_floor` array.

2. **Future windows — `np.lib.stride_tricks.sliding_window_view`**
   `high`/`low` arrays are turned into `(n_anchors, HORIZON_BARS=48)` window matrices via `sliding_window_view(high[1:], 48)[:n_anchors]` (and same for `low`) — one strided view, no copy, no Python loop. Row `i` is the 48 five-minute bars of anchor `i`'s 240-minute horizon; column 0 of each row is the entry bar, so `entry_long = low_windows[:, 0]`, `entry_short = high_windows[:, 0]`.

3. **Per-direction MFE/MAE — `_direction_excursions`, all axis-1 reductions**
   - `MFE`: `favorable_windows.max(axis=1) - entry` (Long) or `entry - favorable_windows.min(axis=1)` (Short), floored at 0 via `np.maximum(0.0, ...)`.
   - Best bar: `favorable_windows.argmax(axis=1)` / `argmin(axis=1)` — the column index of the MFE bar, per row.
   - Adverse prefix mask: `bar_positions <= best_bar[:, None]` (broadcast compare) marks every column up to and including the MFE bar.
   - Worst adverse *before* that bar: `np.where(mask, adverse_windows, ±inf).min/max(axis=1)` — masks out columns after the MFE bar with `+inf`/`-inf` so they never win the reduction, then takes the row-wise extreme over what remains.
   - `MAE = np.maximum(adverse_distance, atr_floor)` — element-wise max against the ATR floor array from step 1.

4. **OM, action, and tie-break — `np.select`**
   `om_long = mfe_long / mae_long` (and short, same form). `qualifies_* = om_* > 5`. `chosen_is_long = om_long >= om_short` doubles as both the single-direction pick and the both-qualify tie-break (higher OM wins). `np.select` maps the four `(qualifies_long, qualifies_short)` combinations to `"long"`/`"short"`/`"none"` in one vectorized call — no per-row Python branching.

5. **RER — element-wise formula + `np.clip`**
   `mfe`/`mae` for the *chosen* direction are picked with `np.where(chosen_is_long, mfe_long, mfe_short)` (and mae). Then:
   ```python
   rer = np.clip(mae / (mfe - mae + np.finfo(np.float64).eps), 0.0, RER_BOUND)
   ```
   `RER_BOUND = 1/(1 + OM_GATE - 1) = 0.25` (§ model output targets' `rer` bound at the `OM > 5` gate). The `eps` addend guards the `mfe == mae` edge case from a divide-by-zero; the outer `clip` re-imposes the `(0, 1/4)` bound for rows that don't clear the `OM > 5` gate (their raw ratio can exceed it), so `rer` stays a well-scaled regression target even for `action_none` rows.

## Output

One row per anchor (anchors within the last `HORIZON_BARS` rows are dropped — no complete horizon, no lookahead): `mfe`, `rer` (both `float32`), and one-hot `action_long`/`action_short`/`action_none`, indexed on the 5min anchor timestamp.

# current code — TP / SL / position-detection mechanisms

Scope: what is **actually implemented in code today**, not the design in
[planing.md](planing.md) / [training-data.md](training-data.md). The planning
docs describe a discrete TP1-4 + drawdown-curve label spec; that spec is
**not yet coded** (no `TP1`/`TP2`/`TP3`/`TP4` symbol exists anywhere in
`app/`). What exists instead is a continuous profit/risk/signal mechanism.
Gaps between plan and code are called out at the end.

There are two independent, unrelated mechanisms in the repo:

1. **ML training-label generator** — [profit_loss_adder.py](../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py),
   driven from [training_datasets.py](../app/ai_modelling/dataset_generator/training_datasets.py).
   This is the "hypothetical position opened near the anchor candle" mechanism —
   it answers "if a Long/Short were opened here, how good would it have been?"
   for every candle, to produce model targets. This is the primary mechanism
   the rest of this doc covers.
2. **Live/backtest bracket-order strategy** — [BasePatternStrategy.py](../app/Strategy/BasePatternStrategy.py),
   a `backtrader` strategy with its own, much simpler, single-level SL/TP.
   Covered briefly at the end for completeness — it is not related to (1) and
   does not use `profit_loss_adder.py`.

## anchor candle, in code terms

`training_datasets.py:train_data_of_mt_n_profit` calls `batch_ends(...)` to
pick `double_end` — the timestamp separating known history (`double`/`trigger`/
`pattern`/`structure` slices, the model's X) from the unseen future. `double_end`
*is* the anchor candle from the planning docs. The label window starts there:

```python
future_slice = dfs['future'].loc[pd.IndexSlice[double_end:], :].iloc[:forecast_trigger_bars]
```

`dfs['future']` is the output of `add_long_n_short_profit(ohlc=dfs['trigger'], ...)`
— every row of the trigger-timeframe OHLC gets long/short profit, risk, drawdown
and signal columns computed against its own forward window, so slicing at
`double_end` gives "the hypothetical Long/Short position opened right after the
anchor candle."

## best entry price near the anchor ("targeting bid price")

`max_profit_n_loss()` ([profit_loss_adder.py:17-78](../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L17-L78)):

- `worst_long_open` = rolling max of `high` over `action_delay` bars, shifted —
  the worst realistic fill price for a Long opened `action_delay` bars after
  the anchor (can't enter exactly at candle close, per `training-data.md`).
- `worst_short_open` = same idea, rolling min of `low`, for Short.

This is the implemented version of the "earliest possible entry ~5min after
NOW candle closes" rule — `action_delay` is the bar-count equivalent of that
delay, and the rolling max/high (long) / min/low (short) is the pessimistic
("worst-case") fill assumption used everywhere downstream as the true entry
price.

## TP / profit-target detection

**No discrete TP1/TP2/TP3/TP4 levels exist in code.** What's implemented is a
single continuous best-case target per position, plus internal quantile
scaffolding used only to compute drawdown (not exposed as separate TP labels):

- `max_profit_n_loss()` computes `max_high` / `min_low`: the best exit price
  reachable within the forward window (`rolling_window = position_max_bars -
  action_delay`), each shifted back `position_max_bars`, plus `max_high_distance`
  / `min_low_distance` (bar-count to that best point, via `argmax`/`argmin`).
- `quantile_maxes()` ([profit_loss_adder.py:81-141](../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L81-L141))
  slices the forward window into `quantiles` (default 50) sub-windows and
  computes `q{i}_max_high` / `q{i}_min_low` (+ their distances) for each. This
  is *not* TP1-4 — it's a fine-grained lookup table consumed only by
  `long_n_short_drawdown()` to find the worst adverse move *from entry*
  before the best-case point was reached (MAE — see next section, not a
  pullback from an interim peak), then dropped from the final frame by
  `drop_quantile_data()` before it's returned.
- `profit_n_loss()` turns the best-case target into `long_profit` /
  `short_profit` (absolute price distance to `max_high`/`min_low`) and
  `weighted_long_profit` / `weighted_short_profit` (same, minus a
  time-decayed risk-free-rate cost and `order_fee`, in ATR units).

So today's "TP" is effectively a single TP4-equivalent (max gainable profit in
the window) — the TP1 (breakeven)/TP2/TP3 (intermediate local-maxima before a
drawdown pullback) tiering described in `training-data.md` is spec only.

## what "drawdown" actually measures: MAE, not peak retracement

The `*_drawdown` columns are ambiguous by name. Two different things could be
meant:

1. **MAE (Maximum Adverse Excursion) — what's implemented**: distance from
   the entry price (`worst_long_open`/`worst_short_open`) down to the lowest
   point reached anywhere along the path to TP. "How far did price move
   against me, measured from where I got in?" Can be zero if price never
   dips below entry on the way to TP.
2. **Peak retracement — NOT implemented**: distance from an interim peak
   reached after some gain, back down to a later pullback low, even if price
   never goes below the entry price at all. Nothing in this pipeline computes
   this.

`long_n_short_drawdown()` computes (1): `max_high_quantile` picks the
quantile bucket matching how many bars it took to reach TP, then
`quantile_long_min_low` = the lowest low reached within a window of
approximately that same length — i.e. roughly "entry → TP," not "peak → TP."
`long_drawdown = (worst_long_open - quantile_long_min_low) / atr` is the MAE
in ATR units — a quantile-bucketed approximation, not an exact bar-by-bar
minimum.

The code and this doc keep the existing `*_drawdown` column/variable names
(no rename applied), but everywhere "drawdown" appears below, read it as
**MAE from entry**, not "retracement from a peak."

## SL detection

Two implementations exist, one active:

- **`stop_loss()`** ([profit_loss_adder.py:346-351](../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L346-L351))
  — the one actually wired into `add_long_n_short_profit()`. Sets
  `long_sl_distance` / `short_sl_distance` = `max(1, long_drawdown|short_drawdown)`,
  i.e. the SL distance in ATR units is just the MAE-to-the-best-case-point,
  floored at 1 ATR. No explicit price level is stored — only the distance,
  used as the denominator of the signal-strength calc below.
- **`zz_stop_loss()`** is old ignore it.

## strength of the anchor-suggested position

`profit_n_loss()` ([profit_loss_adder.py:221-319](../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L221-L319))
computes `long_signal` / `short_signal`:

```python
long_signal  = (1 - long_risk)  * (weighted_long_profit  / long_sl_distance)
short_signal = (1 - short_risk) * (weighted_short_profit / short_sl_distance)
```

where `long_risk = long_drawdown / weighted_long_profit` (capped/zeroed via
`max_risk`: any position with `weighted_*_profit <= 0` or `risk > max_risk` is
a "loser" — its risk is forced to 1 and its signal forced to 0). This is the
implemented "strength of anchor suggested position" — a single scalar per
side, per candle, combining reward, cost (fees + opportunity cost), and
risk (drawdown relative to reward), zeroed out for anything unprofitable
or over the max-risk threshold.

**"Best position for the anchor candle"** (which side, if either, to suggest)
is decided downstream in `train_data_of_mt_n_profit()`
([training_datasets.py:86-91](../app/ai_modelling/dataset_generator/training_datasets.py#L86-L91)):
a candle is `is_actionable` iff `long_signal != 0 or short_signal != 0` at the
anchor; the model's actual training target is
`ys = [short_signal, long_signal]` ([training_datasets.py:151](../app/ai_modelling/dataset_generator/training_datasets.py#L151))
— i.e. "best position" isn't collapsed into one label, both signals are kept
so the model learns relative Long-vs-Short strength directly, and No-Trade
falls out naturally when both are ~0. There's no separate discrete
Long/Short/No-Trade classification column in this pipeline yet — that's a
continuous two-headed regression, not the 3-way categorical action described
in `planing.md`.

## other position-related metrics (all in `profit_loss_adder.py`)

| column | producer | meaning |
| --- | --- | --- |
| `worst_long_open` / `worst_short_open` | `max_profit_n_loss` | pessimistic realistic fill price, `action_delay` bars after anchor |
| `max_high` / `min_low` | `max_profit_n_loss` | best-case exit price within the forward window |
| `max_high_distance` / `min_low_distance` | `max_profit_n_loss` | bars to that best-case point |
| `long_distance_time` / `short_distance_time` | `add_long_n_short_profit` | same distance, converted to a `timedelta` via `trigger_tf` |
| `quantile_long_min_low` / `quantile_short_max_high` | `long_n_short_drawdown` | lowest/highest price reached before the best-case point, looked up from the quantile table (used to compute MAE below) |
| `long_drawdown` / `short_drawdown` | `long_n_short_drawdown` | MAE (maximum adverse excursion) from entry to that point, in ATR units — see [MAE section](#what-drawdown-actually-measures-mae-not-peak-retracement) |
| `absolute_long_drawdown` / `absolute_short_drawdown` | `long_n_short_drawdown` | same MAE, in raw price units |
| `long_profit` / `short_profit` | `profit_n_loss` | raw price distance from worst-open to best-case exit |
| `weighted_long_profit` / `weighted_short_profit` | `profit_n_loss` | profit in ATR units, minus time-decayed risk-free cost and `order_fee` |
| `long_risk` / `short_risk` | `profit_n_loss` | drawdown / weighted-profit; forced to `1` (max) for unprofitable/over-`max_risk` positions |
| `long_sl_distance` / `short_sl_distance` | `stop_loss` | SL distance in ATR units, `max(1, drawdown)` |
| `long_signal` / `short_signal` | `profit_n_loss` | final position-strength score (see above); the actual model target |

All of the above are produced for every row of the trigger-timeframe frame by
`add_long_n_short_profit()` ([profit_loss_adder.py:361-433](../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L361-L433)),
which chains: `max_profit_n_loss` → `quantile_maxes` → `long_n_short_drawdown`
→ `stop_loss` → `profit_n_loss` → `drop_quantile_data`.

## secondary mechanism: live/backtest bracket orders (`BasePatternStrategy`)

Unrelated to the above — a much simpler, single-shot SL/TP used for actual
`backtrader` order placement in [BasePatternStrategy.add_signal()](../app/Strategy/BasePatternStrategy.py#L126-L180),
driven off detected base patterns (`internal_high`/`internal_low` of a
consolidation range), not off the anchor-candle ML labels:

- entry (`limit_price`) = pattern's breakout edge (`internal_high` for a Buy,
  `internal_low` for a Sell) offset by `atr * base_pattern_order_limit_price_margin_percentage`
  (`Config.py`, currently `0.05` = 5%).
- SL (`stop_loss`) = the *opposite* edge of the pattern (`internal_low` for a
  Buy, `internal_high` for a Sell) — i.e. SL = the far side of the
  consolidation range, one level, no ATR/quantile logic.
- TP (`take_profit`) = entry edge + `base_length * base_pattern_risk_reward_rate`,
  where `base_length = internal_high - internal_low` and
  `base_pattern_risk_reward_rate = 5` (`Config.py`) — a single fixed R-multiple
  target, not TP1-4.
- `is_trading_fee_reasonable()` rejects the signal if the TP-to-entry distance
  doesn't clear `trading_fee_safe_side_multiplier` (`3`) times the estimated
  round-trip commission cost.
- On fill, `notify_order()` handles the bracket: a filled SL cancels the TP
  leg and vice versa (via `get_order_group`), and a filled SL re-emits
  ("repeats") the original signal so the pattern keeps getting retried.

## gaps vs. the plan (`planing.md` / `training-data.md`)

Not implemented yet — still spec only:

- discrete TP1 (breakeven) / TP2 / TP3 (local-maxima-before-pullback) / TP4
  (global max) tiering; today there is only one best-case target (TP4-equivalent).
- the drawdown/pivot polyline ("acceleration curve", RDP-style segment
  splitting) and the `potential` metric built from it.
- the 3-way SL-hit / Timeout / TP-tier categorical outcome label.
- trading-overhead double-secure (fees added to SL risk *and* deducted from
  TP gain individually) — current code only subtracts a flat `order_fee` from
  profit, doesn't add it to the SL side.
- SL = `max(true negative movement, 1 ATR of next-higher timeframe)` — current
  `stop_loss()` uses same-timeframe MAE only, no higher-timeframe ATR term.

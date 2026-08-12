# TODO — training data / label preparation

Closing the gap between [training-data.md](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md) (the MAE/MFE/OM label spec) and what
`profit_loss_adder.py`/`training_datasets.py` actually compute today. See
[master-todo.md](master-todo.md) for how this topic fits the overall plan.

- [TODO — training data / label preparation](#todo--training-data--label-preparation)
  - [todo](#todo)
  - [appendix: current implementation status](#appendix-current-implementation-status)
    - [anchor candle, in code terms](#anchor-candle-in-code-terms)
    - [entry price ("targeting bid price")](#entry-price-targeting-bid-price)
    - [TP / profit-target detection](#tp--profit-target-detection)
    - [what "drawdown" actually measures: MAE, not peak retracement](#what-drawdown-actually-measures-mae-not-peak-retracement)
    - [SL detection](#sl-detection)
    - [strength of the anchor-suggested position](#strength-of-the-anchor-suggested-position)
    - [other position-related metrics](#other-position-related-metrics)
    - [secondary, unrelated mechanism: live/backtest bracket orders](#secondary-unrelated-mechanism-livebacktest-bracket-orders)

## todo

Ordered so each step is small, independently testable, and depends only on steps above it — safe to
hand to an agentic coding session one at a time. Written against the current MAE/MFE/OM spec (not the
earlier quantile-search draft this file's predecessor, `current-code.md`, was originally written
against). Steps marked **(decision)** change cross-cutting behavior (label shape/count, window size,
what `V` means) and are worth a one-line confirmation before implementing, since they ripple into
[model-architecture.md](model-architecture.md) and the model's input/output shapes; everything else is
a direct, self-contained code fix against the already-written spec.

1. **(decision, partially done) Fix window granularity/size.** All call sites pass `structure_tf='4h'`,
   which resolves `trigger_tf='15min'` and `double_tf='5min'` (`Config.py`'s `timeframe_shifter`) — but
   labels are generated on `dfs['trigger']` (15-min) with `forecast_trigger_bars` defaulting to
   `3*4*4*4*1 = 192` bars = **48 hours**, not the spec's 5-min NOW candle + 4-hour horizon (48 bars at
   5-min).
   - Done: `train_data_of_mt_n_profit` now takes an optional `label_tf` param (any of
     structure/pattern/trigger/double tf) selecting which frame labels are generated on; default `None`
     resolves to `trigger_tf`, so all existing call sites (which don't pass it) are byte-identical to
     before. `forecast_trigger_bars` now counts bars of `label_tf`, not always `trigger_tf`.
   - Still open: no call site passes `label_tf='5min'`/`double_tf` yet, and `forecast_trigger_bars` isn't
     switched to 48 anywhere — both are caller-side decisions once confirmed. `add_long_n_short_profit`'s
     `position_max_bars` default (`profit_loss_adder.py`) still says 768 bars in its stale docstring
     (step 11 cleanup). Build/test every step below against the corrected window once a call site flips.
2. **Rework entry price to the limit-order target (`E`)**: replace `worst_long_open`/`worst_short_open`'s
   worst-case rolling max/min-over-`action_delay` (a pessimistic market-order fill) with the best price
   reachable in the single 5-min candle immediately following NOW's close, per
   [training-data.md § targeting bid price](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#targeting-bid-price). Update
   `max_profit_n_loss()`. Upstream of `MFE`/`TP4`/`MAE`, so it must land before steps 3-4.
3. **Rename/reframe `MFE`/`TP4` onto the existing best-case columns.** `MFE` = `long_profit`/
   `short_profit` (recomputed against the fixed `E` from step 2); `TP4 = E ± MFE` = today's
   `max_high`/`min_low`. No new search algorithm needed — the spec dropped quantile-window TP search
   entirely (see the appendix note below); confirm this mapping with the characterization tests (see
   appendix) re-run against the step-2 entry price, then rename for clarity.
4. **Tighten `MAE`/SL precision and pin the ATR floor.** Current `long_n_short_drawdown()` finds the
   adverse move to the best-case point via a quantile-bucketed lookup (approximate); spec wants the
   exact worst adverse excursion before `TP4`. Replace the quantile-bucket lookup with an exact rolling
   min/max over the bars between entry and the `TP4` point (or keep the bucketed approach only if the
   precision loss is verified negligible via the characterization tests — measure, don't assume). Pin
   the ATR floor to an explicit `ATR(255, 15min)` term rather than relying on `dfs['trigger']`
   incidentally being 15-min (this incidental match breaks the moment step 1 moves labels to the 5-min
   frame).
5. **(decision) Add `Risk = MAE × (1 + F) × V`.** `F` = 0.001 fee rate is simple; `V` (position volume)
   has no equivalent in code today — confirm whether `V` should default to a fixed unit size (e.g. `1`)
   for label generation, or come from elsewhere. Add as an explicit new column, replacing the current
   flat `order_fee` subtraction from `weighted_long_profit`/`weighted_short_profit` (which matches
   neither the old double-secure formula nor the new risk-side-only one).
6. **Add `OM = MFE / MAE`.** Trivial once steps 3-4 land — a new column, no new logic beyond the
   division.
7. **Replace the direction-validity gate with `OM > 1`.** Swap `profit_n_loss()`'s current "loser"
   condition (`weighted_profit <= 0 or risk > max_risk`) for `OM <= 1` per
   [training-data.md § where can be a position?](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#where-can-be-a-position). Confirm
   whether `max_risk` is still needed for anything else before removing it, or whether `OM > 1` fully
   replaces its role.
8. **Single-label tie-break by `OM`.** When both directions have `OM > 1`, zero the signal of whichever
   has the lower `OM`, per the same spec section. Small, self-contained change once step 7 lands.
9. **Wire the new primary/auxiliary targets into `training_datasets.py`.** `training_y_columns` and the
   `ys` construction at
   [training_datasets.py:151](../../app/ai_modelling/dataset_generator/training_datasets.py#L151)
   currently hardcode `[short_signal, long_signal]`; replace with the spec's primary targets (`MAE`,
   `OM`) + auxiliary (`MFE`) for the winning direction, plus the entry-price target (step 2) and the
   Long/Short/None action head. Coordinate column naming with
   [model-architecture.md](model-architecture.md).
10. **Add a no-lookahead regression test.** Assert that perturbing FUTURE-slice data never changes a
    computed label at or before the anchor candle — the causal-by-construction claims below (anchor
    candle, entry price) are currently backed only by manual reasoning, not a test. Place under
    `app/tests/regression/` per the `pytest` skill, tagged `regression`; wire into whatever CI
    gate `xenon` runs, per [infrastructure.md](infrastructure.md).
11. **Cleanup pass.** Delete now-fully-dead code (`zz_stop_loss`, `singular_stop_loss`, `tops_mean` if
    nothing else calls them, the old flat-`order_fee`/`max_risk` weighted-profit path once steps 5/7
    land); trim `quantile_maxes()`'s 50-way scaffolding if step 4's exact computation no longer needs
    it; fix the stale `position_max_bars` docstring comment ("768 intervals = 16 hours" vs. the actual
    192) and any other docstrings this pass makes inaccurate.

**Not in scope for this file** (execution/live-trading concerns, not training data): the `TP1`-`TP3`
scale-out levels and their live-order-placement mechanics — spec explicitly marks these as non-ML-target
execution levels ([training-data.md § TP / MAE / OM labels](../ML_Forecasting_System_Design/02-Data, Label & Feature Engineering.md#tp--mae--om-labels)),
closer in spirit to `BasePatternStrategy` (see appendix below) than to the label generator this file
covers.

## appendix: current implementation status

Everything below is what's **actually implemented in code today** (verified against `app/` directly,
not assumed from the docs) — moved here from the retired `current-code.md` so this file leads with the
plan and keeps status as reference. Re-verified 2026-08-12.

There are two independent, unrelated mechanisms in the repo:

1. **ML training-label generator** — [profit_loss_adder.py](../../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py),
   driven from [training_datasets.py](../../app/ai_modelling/dataset_generator/training_datasets.py).
   Answers "if a Long/Short were opened here, how good would it have been?" for every candle, to
   produce model targets. The primary mechanism this file covers.
2. **Live/backtest bracket-order strategy** — [BasePatternStrategy.py](../../app/Strategy/BasePatternStrategy.py),
   a `backtrader` strategy with its own, much simpler, single-level SL/TP. Unrelated to (1); see the
   last subsection.

Characterization tests pinning today's actual output of `profit_loss_adder.py`'s five core functions
(`max_profit_n_loss`, `quantile_maxes`, `long_n_short_drawdown`, `stop_loss`, `profit_n_loss`) live in
[test_profit_loss_adder_characterization.py](../../app/tests/characterization/dataset_generator/profit_loss/test_profit_loss_adder_characterization.py)
— see the `pytest` skill. Re-run these after any todo step below changes this file's behavior:
an intentional change should fail exactly the affected test(s); re-capture expected values once confirmed
intentional.

### anchor candle, in code terms

`training_datasets.py:train_data_of_mt_n_profit` calls `batch_ends(...)` to pick `double_end` — the
timestamp separating known history (`double`/`trigger`/`pattern`/`structure` slices, the model's X)
from the unseen future. `double_end` is the anchor candle from the planning docs:

```python
future_slice = dfs['future'].loc[pd.IndexSlice[double_end:], :].iloc[:forecast_trigger_bars]
```

`dfs['future']` is the output of `add_long_n_short_profit(ohlc=dfs['trigger'], ...)` — every row of the
trigger-timeframe OHLC gets long/short profit, risk, drawdown and signal columns computed against its
own forward window, so slicing at `double_end` gives "the hypothetical Long/Short position opened right
after the anchor candle." All call sites pass `structure_tf='4h'`, which fixes `trigger_tf='15min'` and
`double_tf='5min'` — see todo step 1 for why this matters. `train_data_of_mt_n_profit` now accepts a
`label_tf` param to move which frame this runs on (defaults to `trigger_tf`, unused by any call site yet).

### entry price ("targeting bid price")

`max_profit_n_loss()` ([profit_loss_adder.py:17-78](../../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L17-L78)):

- `worst_long_open` = rolling max of `high` over `action_delay` bars (default 2), shifted — the worst
  realistic fill price for a Long opened `action_delay` bars after the anchor.
- `worst_short_open` = same idea, rolling min of `low`, for Short.

This is a worst-case **market-order** fill, pessimistic by design — not the spec's best-case
**limit-order** target (see todo step 2).

### TP / profit-target detection

No discrete TP1-4 levels exist in code. What's implemented is a single continuous best-case target per
position, plus internal quantile scaffolding used only to compute drawdown:

- `max_profit_n_loss()` computes `max_high`/`min_low`: the best exit price reachable within the forward
  window (`rolling_window = position_max_bars - action_delay`), plus `max_high_distance`/
  `min_low_distance` (bar-count to that point).
- `quantile_maxes()` ([profit_loss_adder.py:81-141](../../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L81-L141))
  slices the forward window into `quantiles` (default 50) sub-windows and computes `q{i}_max_high`/
  `q{i}_min_low` (+ distances). Consumed only by `long_n_short_drawdown()` to find the MAE, then dropped
  from the final frame by `drop_quantile_data()`.
- `profit_n_loss()` turns the best-case target into `long_profit`/`short_profit` (absolute distance to
  `max_high`/`min_low`) and `weighted_long_profit`/`weighted_short_profit` (same, minus a time-decayed
  risk-free-rate cost and flat `order_fee`, in ATR units).

Today's `max_high`/`min_low`/`long_profit`/`short_profit` map closely onto the current spec's
`TP4`/`MFE` (see todo step 3) — no quantile-window TP search is required by the current spec, that idea
was dropped. `quantile_maxes()`'s 50-way scaffolding predates that simplification.

### what "drawdown" actually measures: MAE, not peak retracement

The `*_drawdown` columns are ambiguous by name — two different things could be meant:

1. **MAE (Maximum Adverse Excursion) — what's implemented**: distance from the entry price down to the
   lowest point reached anywhere along the path to the best-case exit.
2. **Peak retracement — NOT implemented**: distance from an interim peak back down to a later pullback
   low. Nothing in this pipeline computes this.

`long_n_short_drawdown()` computes (1): `max_high_quantile` picks the quantile bucket matching how many
bars it took to reach the best-case exit, then `quantile_long_min_low` = the lowest low reached within a
window of approximately that same length — i.e. roughly "entry → best-case exit," a quantile-bucketed
approximation, not an exact bar-by-bar minimum (see todo step 4). Read every "drawdown" in this codebase
as **MAE from entry**, not "retracement from a peak."

### SL detection

Two implementations exist, one active:

- **`stop_loss()`** ([profit_loss_adder.py:346-351](../../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L346-L351))
  — the one wired into `add_long_n_short_profit()`. Sets `long_sl_distance`/`short_sl_distance` =
  `max(1, long_drawdown|short_drawdown)`, i.e. the SL distance in ATR units is the MAE floored at 1 ATR.
  No explicit price level is stored, only the distance. The ATR term is incidentally 15-min today (since
  `trigger_tf='15min'`), not an explicit higher-timeframe term — see todo step 4.
- **`zz_stop_loss()`** — dead code, not wired in.

### strength of the anchor-suggested position

`profit_n_loss()` ([profit_loss_adder.py:221-319](../../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L221-L319))
computes `long_signal`/`short_signal`:

```python
long_signal  = (1 - long_risk)  * (weighted_long_profit  / long_sl_distance)
short_signal = (1 - short_risk) * (weighted_short_profit / short_sl_distance)
```

where `long_risk = long_drawdown / weighted_long_profit` (capped/zeroed via `max_risk`: any position
with `weighted_*_profit <= 0` or `risk > max_risk` is a "loser" — risk forced to 1, signal forced to 0).
Downstream, `train_data_of_mt_n_profit()`
([training_datasets.py:86-91](../../app/ai_modelling/dataset_generator/training_datasets.py#L86-L91))
treats a candle as `is_actionable` iff `long_signal != 0 or short_signal != 0`; the model's actual
training target is `ys = [short_signal, long_signal]`
([training_datasets.py:151](../../app/ai_modelling/dataset_generator/training_datasets.py#L151)) — both
signals kept independently, no tie-break, no discrete Long/Short/None column. This whole mechanism
predates the spec's `MAE`/`OM`/`MFE` targets — see todo steps 5-9.

### other position-related metrics

All produced for every row of the trigger-timeframe frame by `add_long_n_short_profit()`
([profit_loss_adder.py:361-433](../../app/ai_modelling/dataset_generator/profit_loss/profit_loss_adder.py#L361-L433)),
which chains: `max_profit_n_loss` → `quantile_maxes` → `long_n_short_drawdown` → `stop_loss` →
`profit_n_loss` → `drop_quantile_data`.

| column                                               | producer                | meaning                                                              |
| ----------------------------------------------------- | ------------------------ | --------------------------------------------------------------------- |
| `worst_long_open` / `worst_short_open`               | `max_profit_n_loss`     | pessimistic realistic fill price, `action_delay` bars after anchor  |
| `max_high` / `min_low`                               | `max_profit_n_loss`     | best-case exit price within the forward window                      |
| `max_high_distance` / `min_low_distance`             | `max_profit_n_loss`     | bars to that best-case point                                        |
| `long_drawdown` / `short_drawdown`                   | `long_n_short_drawdown` | MAE from entry to that point, in ATR units                          |
| `absolute_long_drawdown` / `absolute_short_drawdown` | `long_n_short_drawdown` | same MAE, in raw price units                                        |
| `long_profit` / `short_profit`                       | `profit_n_loss`         | raw price distance from worst-open to best-case exit                |
| `weighted_long_profit` / `weighted_short_profit`     | `profit_n_loss`         | profit in ATR units, minus time-decayed risk-free cost and `order_fee` |
| `long_risk` / `short_risk`                           | `profit_n_loss`         | drawdown / weighted-profit; forced to `1` for unprofitable/over-risk |
| `long_sl_distance` / `short_sl_distance`             | `stop_loss`             | SL distance in ATR units, `max(1, drawdown)`                        |
| `long_signal` / `short_signal`                       | `profit_n_loss`         | final position-strength score; the actual model target today        |

### secondary, unrelated mechanism: live/backtest bracket orders

`BasePatternStrategy` — a much simpler, single-shot SL/TP used for actual `backtrader` order placement
in [BasePatternStrategy.add_signal()](../../app/Strategy/BasePatternStrategy.py#L126-L180), driven off
detected base patterns (`internal_high`/`internal_low` of a consolidation range), **not** off the
anchor-candle ML labels above:

- entry (`limit_price`) = pattern's breakout edge offset by
  `atr * base_pattern_order_limit_price_margin_percentage` (`Config.py`, `0.05` = 5%).
- SL (`stop_loss`) = the opposite edge of the pattern — one level, no ATR/quantile logic.
- TP (`take_profit`) = entry edge + `base_length * base_pattern_risk_reward_rate`
  (`base_pattern_risk_reward_rate = 5`) — a single fixed R-multiple target.
- `is_trading_fee_reasonable()` rejects the signal if TP-to-entry distance doesn't clear
  `trading_fee_safe_side_multiplier` (`3`) times estimated round-trip commission.
- On fill, `notify_order()` handles the bracket: a filled SL cancels the TP leg and vice versa, and a
  filled SL re-emits the original signal so the pattern keeps getting retried.

No documented path connects a trained model's prediction to this strategy placing an order — see
[model-architecture.md](model-architecture.md) for the deployment-layer gap this leaves open.

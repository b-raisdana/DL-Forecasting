# training data preparation

- [training data preparation](#training-data-preparation)
  - [where can be a position?](#where-can-be-a-position)
  - [trading overhead](#trading-overhead)
  - [targeting bid price](#targeting-bid-price)
  - [risk factors](#risk-factors)
  - [TP / MAE / OM labels](#tp--mae--om-labels)
  - [model output targets](#model-output-targets)
  - [glossary](#glossary)

## where can be a position?

- NOW candle timeframe: 5 minutes.
- for each NOW candle, use 4hours FUTURE knowledge (training-time only, not available at inference) to decide the candle's label: Long, Short, or None
- earliest possible entry: the 5-minute candle immediately following NOW (a position cannot open before NOW has closed; verify with 1-minute candles during backtesting)
- look forward boundary: 4H, exclusive of the boundary instant — e.g. NOW = 9:55-10:00 means the position must close before 14:00; the candle 13:55-14:00 is in-bounds, 14:00-14:05 is Timeout
- simulate opening a position at the targeted bid price (see targeting bid price) in each direction, applying trading overhead (see trading overhead): compute `MFE`, `SL`, `MAE`, `OM` per direction (see TP / MAE / OM labels)
- a direction is a valid label only if `OM > 1` (MFE exceeds MAE — some favorable edge over adverse risk); if neither direction clears it, the candle is labeled None
- tie-break when both directions clear `OM > 1`: pick whichever has the higher `OM`, zero the other direction's signal

## trading overhead

- trading fee rate `F` = 0.1% — no separate spread cost is modeled; a single targeted bid price is used for both Long and Short entries
- fees enter only on the risk side: `Risk = MAE × (1 + F) × V` (see [TP / MAE / OM labels](#tp--mae--om-labels)) — `MFE`/`OM` are raw price-derived, not fee-adjusted

## targeting bid price

- entry price `E` = the best price reachable within the 5-minute candle immediately following NOW's close
- limit-order target: the model commits open price + SL + TP together as one decision as soon as NOW closes; the best price in that next candle is the training label for the entry-price output
- at inference: if the predicted price isn't reached within that candle, the limit order never fills and no position opens

## risk factors

- SL = effective future adverse level = `max(worst adverse excursion before TP4, ATR floor distance)`, ATR floor = 1x `ATR(255, 15min)` (see glossary)
- `MAE = abs(E - SL)` — SL re-expressed as a distance from entry
- the ATR floor guards against a near-zero adverse move producing an unrealistically tight SL
- SL is a retrospective risk-sizing measure, not a live barrier: by construction it is never breached along the path to TP4. A live SL order is placed an epsilon farther out than this computed level.

## TP / MAE / OM labels

Per direction, computed from the FUTURE window (see [where can be a position?](#where-can-be-a-position)); favorable/adverse flips by direction — Long: favorable = higher price, adverse = lower price; Short: favorable = lower price, adverse = higher price.

- `MFE` = maximum favorable price excursion during the 4H horizon
- `TP4 = E ± MFE` — the MFE endpoint (`+` Long, `-` Short)
- `SL`, `MAE` = see [risk factors](#risk-factors)
- `Risk = MAE × (1 + F) × V` (`V` = position volume, `F` = fee rate — see [trading overhead](#trading-overhead))
- `OM` (Opportunity Multiple) = `MFE / MAE` — reward-to-risk ratio; drives the direction/None call in [where can be a position?](#where-can-be-a-position)

Execution levels (`TP1`-`TP4`) — discrete scale-out prices derived from `TP4`/`Risk`, for live order placement, not primary ML targets:

- `TP1_vol_ratio`, `TP1_risk_mult` — configurable params (fraction of `V` closed at TP1; risk multiplier sizing TP1's distance)
- `TP1_dist = Risk × TP1_risk_mult × TP1_vol_ratio`; `TP1 = E ± TP1_dist`
- `TP_step = abs(TP1 - TP4) / 3`; `TP2 = TP1 ± TP_step`; `TP3 = TP1 ± 2 × TP_step`; `TP4` = the MFE endpoint above

Rules:

- never use a future-derived value (`SL`, `MAE`, `MFE`, `TP4`, `OM`, or any `TP1`-`TP4` level) as an input feature — features carry only information available at entry
- always preserve raw `MAE`, `MFE`, `OM` even when only derived TP/Risk levels are consumed downstream
- each sample keeps its future horizon and trade direction fixed, so labels stay reproducible and path-dependent outcomes from different horizons/directions are never mixed
- supersedes the old three-way SL-hit/Timeout/TP outcome label: `TP4` is always the realized MFE endpoint (no TP-vs-Timeout ambiguity); SL-hit-vs-not still matters for live execution (see risk factors) but is no longer a separate ML label

## model output targets

What the model actually predicts, given the labels above (target/label definitions belong with the labeling spec; how the model represents/predicts them stays in [model-architecture-planning.md](model-architecture-planning.md#model-architecture--selection)):

- action head = Long / Short / None (see [where can be a position?](#where-can-be-a-position))
- primary regression targets = `MAE`, `OM` (see [TP / MAE / OM labels](#tp--mae--om-labels))
- auxiliary regression target = `MFE`
- confidence metric = open gap, no input features carry confidence information today — see [error-rating-and-evaluation.md § confidence & calibration metrics](error-rating-and-evaluation.md#confidence--calibration-metrics).

## glossary

- ATR = see [model-architecture-planning.md § glossary](model-architecture-planning.md#glossary)
- `E` / `V` / `F` = entry price / position volume / trading fee rate
- `MFE` (maximum favorable excursion) = best move _for_ the position from entry, over the horizon
- `MAE` (maximum adverse excursion) = worst move _against_ the position from entry to `SL` — not a pullback from an interim peak (that's "retracement")
- `OM` (Opportunity Multiple) = `MFE / MAE`, the reward-to-risk ratio
- `Risk` = fee-inflated `MAE` in position-volume terms, `MAE × (1 + F) × V`
- `TP1`-`TP4` = discrete execution scale-out levels between entry and the `MFE` endpoint (see [TP / MAE / OM labels](#tp--mae--om-labels))
- HISTORY / NOW / FUTURE = already-closed candles / the candle we're in / not-yet-started candles
- SL / TP = stop loss / take profit

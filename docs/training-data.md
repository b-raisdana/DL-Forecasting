# training data preparation

- [training data preparation](#training-data-preparation)
  - [where can be a position?](#where-can-be-a-position)
  - [trading overhead](#trading-overhead)
  - [naming candles](#naming-candles)
  - [targetting bid price](#targetting-bid-price)
  - [risk factors](#risk-factors)
  - [TP / MAE label](#tp--mae-label)

## where can be a position?

- for each NOW candle, use 4hours FUTURE knowledge (training-time only, not
  available at inference) to decide the candle's label: Long, Short, or None
- earliest possible entry: 5 minutes after the NOW candle closes (a position
  cannot open exactly at candle close; verify this with 1-minute candles
  during backtesting)
- lookforward boundary: 4H — no simulated deal may stay open for more than
  4H; if neither SL nor a qualifying TP is hit within 4H, the candle is
  labelled None
- simulate opening a position at the targeted bid price (see targetting bid
  price) in each direction and walk FUTURE candles forward, applying trading
  overhead (spread, fees, double secure):
  - per-candle check is simplified to a single side of each FUTURE candle:
    Low for Long, High for Short — e.g. if NOW = 9:00, the 2 later 5min
    candles are 9:05-9:10 and 9:10-9:15; only their Lows are checked against
    SL/TP for the Long simulation, only their Highs for the Short simulation
  - Long is a valid label if the walk-forward hits a qualifying TP before SL
  - Short is a valid label if the walk-forward hits a qualifying TP before SL
  - if neither direction hits a qualifying TP before SL within the 4H
    boundary, the candle is labelled None
- TP quality filter — a TP hit only qualifies as a valid label if its true
  gain is both:
  - at least 5x the trading fee
  - at least 3x the SL true risk
- tie-break when both Long and Short would independently qualify: pick
  whichever direction's qualifying TP is hit sooner (fewer candles) — the
  other direction's later success is counterfactual, since only one position
  can be open per candle
- same-candle ambiguity: if a single FUTURE candle's range touches both a
  direction's SL and TP, assume SL was touched first (pessimistic default)
  rather than guessing — avoids look-ahead bias

## trading overhead

- spread: 0.001%
- trading fee 0.1%
- double secure on each position: trading fees are added to the SL risk and
  deducted from the TP gain (SL/TP are distances from entry, so this
  applies the same way for both long and short positions)
  - SL true loss = SL distance + trading fees: fees are charged on top of
    the risked money when SL is hit
  - TP true gain = TP distance - trading fees: TP must be placed farther
    out so that the fee-adjusted gain still covers the SL risk

## naming candles

- HSITORY candles: the candles already passed and all are closed completly
- NOW: the candle we are in
- FUTURE: candle in the future (prediction) sery none of the has been started.

## targetting bid price

- best price between close of NOW and first-FUTURE range according to posionig type

## risk factors

- stop loss (SL) = `max(1 ATR, MAE-to-TP)` — matches the implemented
  `stop_loss()`; see [current-code.md](current-code.md#sl-detection)
- MAE (maximum adverse excursion) = worst move _against_ the position,
  measured from the entry price, along the path to TP — not a pullback from
  an interim peak (that's a different metric, "retracement", not computed
  here). See [MAE section](current-code.md#what-drawdown-actually-measures-mae-not-peak-retracement)
  for the exact definition. The code still names this `long_drawdown`/
  `short_drawdown` — read "drawdown" in this codebase as MAE-from-entry.

## TP / MAE label

Single TP per direction:

- TP = best-case exit price reachable within the 4H window (max `high` for
  Long, min `low` for Short) — the max-gainable-profit point, found via a
  rolling max/min over the lookahead window (hindsight, label-construction
  only, not a live feature).
- MAE = worst adverse move from entry before that point is reached, found by
  bucketing the lookahead window into quantiles and looking up which bucket
  the TP falls in — a cheap approximation, not a reversal-point polyline.
- SL = per risk factors above (MAE-derived, so SL is computed after TP).
- position strength = `(1 - risk) * (weighted_profit / sl_distance)`, zeroed
  for any position that's unprofitable or over the max-risk threshold — the
  value actually used as the model's training target.
- no-breakeven edge case: three-way outcome space, not forced TP/SL binary —
  (a) SL literally hit → SL/loss; (b) neither breakeven nor SL hit by
  timeout → distinct Timeout label (no stop was actually triggered); (c) TP
  reached → TP label.

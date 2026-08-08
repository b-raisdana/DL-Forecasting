# training data preparation

- [training data preparation](#training-data-preparation)
  - [where can be a position?](#where-can-be-a-position)
  - [trading overhead](#trading-overhead)
  - [naming candles](#naming-candles)
  - [targeting bid price](#targeting-bid-price)
  - [risk factors](#risk-factors)
  - [TP / MAE label](#tp--mae-label)

## where can be a position?

- NOW candle timeframe: 5 minutes.
- for each NOW candle, use 4hours FUTURE knowledge (training-time only, not available at inference) to decide the candle's label: Long, Short, or None
- earliest possible entry: the 5-minute candle immediately following NOW (a position cannot open before NOW has closed; verify with 1-minute candles during backtesting)
- look forward boundary: 4H, exclusive of the boundary instant — e.g. NOW = 9:55-10:00 means the position must close before 14:00; the candle 13:55-14:00 is in-bounds, 14:00-14:05 is Timeout
- simulate opening a position at the targeted bid price (see targeting bid price) in each direction, applying trading overhead (fees, double secure — see trading overhead): find TP as the best-case exit price reachable within the 4H window (see TP / MAE label) and derive SL from the MAE along the path to that TP (see risk factors)
- a direction is a valid label only if its TP clears the feasibility pre-filter (see TP / MAE label); if neither direction clears it, the candle is labeled None
- tie-break when both Long and Short clear the feasibility pre-filter: pick whichever direction's TP is reached in fewer bars, zero the other direction's signal

## trading overhead

- trading fee 0.1% — no separate spread cost is modeled; a single targeted bid price is used for both Long and Short entries
- double secure on each position: trading fees are added to the SL risk and deducted from the TP gain
  - SL true loss = SL distance + trading fees
  - TP true gain = TP distance - trading fees

## naming candles

- HISTORY candles: the candles already passed and all are closed completely
- NOW: the candle we are in
- FUTURE: candle in the future (prediction) series, none of which has started

## targeting bid price

- entry price = the best price reachable within the 5-minute candle immediately following NOW's close
- limit-order target: the model commits open price + SL + TP together as one decision as soon as NOW closes; the best price in that next candle is the training label for the entry-price output
- at inference: if the predicted price isn't reached within that candle, the limit order never fills and no position opens

## risk factors

- stop loss (SL) = `max(1 ATR(255, 15min), MAE-to-TP)`
- MAE (maximum adverse excursion) = worst move _against_ the position, measured from the entry price, along the path to TP — not a pullback from an interim peak, which is a different metric ("retracement").
- SL is a retrospective risk-sizing measure, not a live barrier: by construction it is never breached along the path to TP. A live SL order is placed an epsilon farther out than this computed level.
- the 1 ATR floor is the minimum SL distance, applied when MAE is smaller than 1 ATR.

## TP / MAE label

Feasibility pre-filter, per direction, before searching for TP:

- SL floor = 1x `ATR(255, 15min)` (see risk factors)
- feasible TP distance = `(ATR + trading_fee * close_price) * 3`
- does the gap between the targeted entry price (see targeting bid price) and the window's High (Long) / Low (Short) reach at least the feasible TP distance? If not, that direction is dropped.
- if neither direction's High/Low range clears it, label the candle None

TP search, per direction that survives the pre-filter:

- split the 4H timeout horizon into `n` (parameterized, default 10) equal quantile windows
- walk the quantiles in order; the first quantile whose local best-case exit (max `high` for Long / min `low` for Short) clears the feasible TP distance supplies TP
- if no quantile clears the bar, fall back to the window's global best-case exit price (max `high` for Long, min `low` for Short)
- MAE = worst adverse move from entry before that point is reached, found by bucketing the lookahead window into quantiles and looking up which bucket the TP falls in
- SL = per risk factors above (computed after TP)
- weighted profit = raw TP profit (in ATR units) minus order fee minus a time-decayed opportunity cost proportional to bars-to-TP
- position strength = `(1 - risk) * (weighted_profit / sl_distance)`, zeroed for any position that's unprofitable or over the max-risk threshold — the value used as the model's training target
- no-break-even edge case: three-way outcome — (a) SL literally hit (the live, epsilon-widened SL order — see risk factors) → SL/loss; (b) neither break-even nor SL hit by timeout → Timeout; (c) TP reached → TP label

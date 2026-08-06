# training data preparation

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
- TP2 quality filter — a TP hit only qualifies as a valid label if TP2 true
  gain is both:
  - at least 5x the trading fee
  - at least 3x the SL true risk
- TP close rate: 1->40%, 2->30%, 3->20%, 4->10%
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
  deducted from the TP gains (SL/TPn are distances from entry, so this
  applies the same way for both long and short positions)
  - SL true loss = SL distance + trading fees: fees are charged on top of
    the risked money when SL is hit
  - TPn true gain = TPn distance - trading fees: TP must be placed farther
    out so that the fee-adjusted gain still covers the SL risk

## naming candles

- HSITORY candles: the candles already passed and all are closed completly
- NOW: the candle we are in
- FUTURE: candle in the future (prediction) sery none of the has been started.

## targetting bid price

- best price between close of NOW and first-FUTURE range according to posionig type

## risk factors

- stop loss (SL): max of followings
  - maximum true negative movement
  - 1 ATR of 1higher timeframe (1H)
- win drawdown (WD): area under drawdown acceleration curve

## TP1-4 / drawdown labels

Within 4H timeout, labels built with hindsight (label-construction only, not a live feature):

- TP1 = break-even point (partial close → zero-loss + banked profit on remainder)
- TP4 = max gainable profit in 4H window (hindsight)
- TP2/TP3 = intermediate levels, local max-gainable-profit points before a max-drawdown pullback
- SL optimized per trade; TP1 defined relative to SL, so SL definition must come first
- TP2/TP3 selection: walk forward chronologically from TP1, take local maxima in
  time order (not size-sorted), each qualifying if followed by a drawdown
  pullback > threshold (e.g. fraction of ATR) before the next higher max. TP2
  = first qualifying max after TP1, TP3 = next after TP2, TP4 = global max
  (may coincide w/ TP3's successor). Fallback if <2 qualifying maxima:
  collapse/duplicate TP2/3 toward TP4 rather than leaving undefined (every
  example gets a complete label).
- no-breakeven edge case: three-way outcome space, not forced TP/SL binary —
  (a) SL literally hit → SL/loss; (b) neither breakeven nor SL hit by
  timeout → distinct Timeout label (no stop was actually triggered); (c)
  TP1+ reached → TP-tier labels.

## position potential

potential = total true gain at TP4 / SL true risk - DD-area

## DD-area

drawdown area = sum(distance of farest point of each candle from drawdown acceleration curve / ATR)

### drawdown/pivots acceleration curve

- drawdown: curve for calculating drawdowns
- pivots: curve for placing TPs
- polyline; segments have varying slope
- drawdown segments only increase in slope; pivot segments only decrease

**pivots path:** first candle's High → most significant Peaks → High of max-TP (Long)
first candle's Low → most significant Valleys → Low of max-TP (Short)
**drawdown path:** inverse of pivots

#### building the curve (drawdown; pivots = inverse)

Directional line-simplification (RDP-like), splitting only on adverse moves, not max distance either way.

1. Start with one segment: open → max-TP.
2. For each candle in a segment, measure distance to the segment line using the adverse side (low for Long, high for Short). Negative = pullback against trade direction, positive = favorable.
3. Split at the candle with the largest negative distance (worst pullback), if any:
   - segment → (prev point → candle) + (candle → next point)
   - candle becomes a new path point
4. Recurse into both new segments.
5. Stop splitting a segment once no candle in it has negative distance.

Result: path through reversal points from open to max-TP, per the endpoints/slope rules above.

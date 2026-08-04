# planing documentation

I awant to find best AI way to predict price movemtns to find best trading posiions to gain maximum profit with least risk.

## what am i lookng for?

### how to feed data?

- is my propsed data feed perfect?
- how to determine if any of supplied channels are not useful
- how to determine if i need to add new columns?
- what are alternative/candidates to test as new columns?

### how to normalize data?

- is the normalization suggested perfect?
- what are alternative/candidate normalization options to test?
- how to detrmine if modification of normalization improves my accuracy and perfomance?

### what is the best model implementation?

- what are alternative model candidate?
- what are the range of parameters feasible to test for parameter optimization?

## hardware

- NVIDIA Geforce RTX 4060 \*GB
- 64GB Ram
- 2 SSD HDD

## data feed

each canlde includes:

- OHLC prices / ATR
- candle height / pandas-ta.ATR(256)
- is candle a peak/valley? which timeframe
  - representing number (number of minutes) for highest timeframe in which this candle is a peak/valley
  - (+) for peak and (-) for valley
  - **resolved (causality):** no lookahead is used. A candle's confirmed peak/valley timeframe is capped by how much time has actually elapsed between it and the sequence's reference "now" candle — e.g. a candle from 4 months ago can be confirmed as a 4M peak (it was the max in the 4M before it and has stayed the max up to now), while a candle from 1 week ago can be at most a 1W peak/valley, since what happens after it isn't known yet. This is causal by construction.
    - OPEN QUESTION: when many overlapping training windows are generated across the ~1 year of history, is "now" for this confirmation always pinned to that specific window's own last candle (not to the actual present date the dataset was generated on)? If any window's peak/valley labels use information beyond that window's own "now", leakage is reintroduced. Worth double-checking directly in the feature-generation code.
  - see also the multi-horizon ATR-distance feature under "class imbalance" below, which extends this single "highest confirmed timeframe" number into several continuous per-horizon distances.
- timeframe in minutes
- size of the base sequence if the candle is in a base
- distance from now in minutes

### multi-timframe series

- each series 256 candles
- 15min, 1H, 4H, 1D timeframes
- on overlap: at-most 1 candle of higher tf allowed to overlap with lower tf series
- there is an assumtion about price mobments characterisitc: the patterns have near meaning in different time . if a pattern in 15min timeframe shows a long position the same pattern in 1H timeframe will show the same meaning.
- combination of pattern in different timeframes will give a detail view about price movement: the real truce behind a pattern in a 1D timeframe will be cleared by investigating the patterns of 4H, 1H and 15min timeframes of the same chart.
- how could i combine multiple models or distiribute data feed to best implement this nature with least effort and maximum efficiency?
- we are feeding data of more than 1 year to give maximum perpective to the model, but patterns we are looking for happen usaly in a day or a week. these patterns might be visible in last few candles of chart or might be in 2/3 candles of the chart or 50% of last candles? how to optimize where to foucous? can special attention mechanisms help?
- a pattern and be slowed-down or speeded-up with keeping it's nature, a compress-price or trend revesal might happen in different speed:
  - - what is the best of normalizing the spped?
  - - which model might transalte these different speed pattern the same better?
- where to focous? always on the now time(last candle) to determine go in long/short position or might it be helpful to detect it's too soon or too late to go in the position and expaculate the best time to open the position?

## validation / data splitting

- **resolved:** all 256-candle windows are built from contiguous, complete data — any time range with missing candles is filtered out and discarded entirely, so no window contains gaps.
- OPEN QUESTION: window completeness alone doesn't prevent leakage _between_ train and validation sets. Consecutive windows (e.g. one ending "now" and the next ending 15 minutes later) overlap heavily and are highly correlated. If windows are split into train/val randomly rather than by contiguous time blocks, near-duplicate information can leak across the split and inflate validation performance. Needs an explicit decision: chronological holdout (train on earlier dates, validate/test on strictly later dates) vs. some other scheme — and if chronological, how much gap/embargo to leave between the train and validation date ranges.

## required ouput

- current best action: Long / Short / No Trade
- how much are the targeted prices? TP1, 2, 3, 4
- how much is expected drawdown before TPs = SL
- how probable might be our forcasted TPs? meeting TPn before SL and before Timeout = 4H
- how probable is the drawdown breakout = SL hit?
- we are looking for escalping / same day positions to be closed in less than 4 hours.

### how TP1-4 / drawdown labels are built for training

Within the fixed 4H timeout, training labels are built with full knowledge of what happened next (valid only for constructing targets, not as a live/inference-time feature):

- **TP1** = the break-even point: the price level where, given SL, closing part of the position makes the trade zero-loss while banking some profit on the remaining share.
- **TP4** = the maximum gainable profit within the 4H window, known in hindsight.
- **TP2 / TP3** = intermediate levels between TP1 and TP4, chosen at local maximum-gainable-profit points that occur before a maximum-drawdown pullback.

OPEN QUESTIONS:

- Is SL fixed (e.g. a constant ATR multiple) before TP1 is computed, or is SL itself optimized per trade? TP1 is defined relative to SL, so SL's own definition needs to be pinned down first.
- What's the precise rule for picking TP2/TP3 when there are several candidate local maxima before drawdowns — e.g. the two largest, or the two most persistent (longest held before reversal)?
- The required output also asks for _probabilities_ of hitting each TP and of the drawdown, but the labels above are single deterministic outcomes per training example (what actually happened). Turning that into a probabilistic forecast is a separate modeling decision — e.g. quantile regression, a calibrated classifier per TP level, or an ensemble/Monte-Carlo spread over repeated runs — to be decided as part of "best model implementation."

### normalization

- (OHLC - close of now (latest candle) )/ devids by ATR of the same candle
- the current price (or close of last candle) equals to zero
- **resolved:** each candle is normalized by its _own_ rolling ATR (not by "ATR at now"). This is intentional and correct — standard volatility-normalization — since it expresses a move in "how many ATRs of that period's own volatility regime" rather than in raw price, which is exactly what makes moves from a calm period and a volatile period comparable.

## optimization

- chich optimization techniques to test?
- what KPIs to mintor for selecting best optimization?

## error calculation method

- how to measure the error rate?
- what are candidates to test?
- how to compare result and how to choose the best one?

### error metric vs. trading objective

- a low statistical loss (MSE/MAE/quantile loss on TP/drawdown predictions) does not guarantee profitability — it should be treated as a training-time signal, not the final selection criterion.
- final model/config selection should be based on backtested trading KPIs: win rate, profit factor, expectancy per trade, Sharpe/Sortino, max drawdown, Calmar ratio — computed by actually simulating trades from the model's TP/SL predictions.

OPEN QUESTION: which of these KPIs is primary (what you optimize/rank configs by), and which are just guardrails (e.g. "reject any config with max drawdown above X even if profit factor is best")?

## class imbalance

- test class-weighted loss vs. focal loss for any classification-style targets (e.g. peak/valley class, TP-hit-before-SL class) and compare.
- for each candle, provide (fractional) ATR-distance to the nearest peak/valley at each of several fixed horizons: 4H, 1D, 1W, 1M, 4M, 1Y. This turns the single categorical "highest confirmed timeframe" peak/valley feature into several continuous features, which sidesteps the imbalance problem for that feature.

OPEN QUESTION: does this multi-horizon ATR-distance feature replace the original single "highest timeframe" peak/valley feature entirely, or do both get fed in? And does the class-weight-vs-focal-loss test also apply to the TP-hit/drawdown side of the output (which can also be rare-event-like, e.g. TP4 being hit at all within 4H), or is this scoped only to the peak/valley input feature?

## experiment tracking (current priority)

- needed now, not deferred: with multiple normalization/model/optimization combinations planned, an ad hoc file-naming convention (as currently seen in /data, e.g. "- Copy (2).keras", ".bak", ".nan" suffixes) won't scale and makes it hard to know which run produced which result.
- decide on a lightweight tracking approach: at minimum a consistent naming/logging convention (config hash + date + key hyperparams); ideally a tool (MLflow, Weights & Biases, or a simple CSV/SQLite run log) recording config, dataset version, metrics (both loss and the trading KPIs above), and artifact path together.

OPEN QUESTION: local-only (e.g. MLflow with a local file backend) or does this need to be shared/remote given the single-machine hardware setup described above?

## deferred topics (not current concerns, kept as placeholders)

- **transaction costs / spread / slippage / execution latency**: fees and slippage can matter a lot for sub-4H scalping, but not addressed now. Revisit before any live/paper trading — backtest P&L without cost assumptions will overstate real performance.
- **risk / position sizing beyond TP targets**: stop-loss placement and position sizing are handled manually through existing procedures, not by the model. No AI work needed here for now.
- **market regime robustness / retraining cadence**: not addressed now. Revisit once a model is live for a while — crypto market character shifts (trend vs range, volatility regime) and an untouched model can decay silently.

## etc

- ATR = pandas-ta.ATR(256)
- base = sequence of 2

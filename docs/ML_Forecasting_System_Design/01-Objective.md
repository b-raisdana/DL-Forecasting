# Problem & Objective Engineering

## Trading prediction problem

- **Price prediction**: not used — no raw-price target.
- **Direction prediction**: derived, not modeled directly — Long/Short/None comes from a post-hoc rule (`OM > 1`, tie-break = higher `OM`) applied to regression outputs, not a classification head. See [02 § Label design](<02-Data, Label & Feature Engineering.md#label-design>).
- **Position prediction**: the direction call *is* the position decision (Long/Short/None).
- **Entry/exit prediction**: bundled, not separate — entry price + SL + full TP1–TP4 ladder are all committed as one decision at NOW-candle close (see [targeting bid price](<02-Data, Label & Feature Engineering.md#targeting-bid-price>)).
- **Multi-horizon prediction**: named in spec ("Multi-horizon labels") but unresolved — horizon is fixed at 240 minutes (a rolling window from NOW, not the next 4H candle); systematic horizon selection is an open gap ([00-ToC §1.2](<00-ToC & Coverage.md>)).
- **Classification vs regression vs ranking**: regression. Primary targets `MAE`, `OM` (continuous); auxiliary `MFE`. No classifier, no ranking.

## Objective definition

- **Prediction objective**: regress `MAE` and `OM` (+ auxiliary `MFE`), ATR-normalized, per direction.
- **Trading objective**: not in the training loss — expectancy/max-DD/Sortino are evaluation-only, computed in the backtest stage ([04](<04-Experimentation, Evaluation & Optimization.md>)).
- **Risk-adjusted objective**: same split — risk-adjustment happens at backtest evaluation, not at training time.
- **Profit vs accuracy**: reward/risk (`OM`) drives the design, not hit-rate/accuracy.
- **Reward/risk trade-off**: this *is* `OM = MFE / MAE`.
- **Multi-objective optimization**: de facto yes (dual regression target + auxiliary target), but no defined loss-weighting scheme yet — flagged missing ([05 A13](<05-Weakness Analysis.md>)).

## Prediction formulation

- **What exactly is predicted**: `MAE`, `OM` (primary), `MFE` (auxiliary); TP ladder, SL, and direction are all derived, not predicted.
- **Prediction horizon**: fixed 240 minutes.
- **Prediction frequency**: every closed 5-min NOW candle.
- **Single- vs multi-step**: single-step — one-shot 240-minute-ahead regression, no iterative rollout.
- **Point vs probability distribution**: point estimates only; no quantile/distributional output.
- **Absolute vs relative movement**: relative — ATR-normalized price distances, not absolute price levels.

## Problem decomposition

- **One model vs multiple models**: one model today. MoE / late-ensemble is a Tier-2 candidate in [04](<04-Experimentation, Evaluation & Optimization.md>), unresolved.
- **Direction + magnitude**: regression-then-rule, not joint heads — magnitude (`MAE`/`OM`/`MFE`) is regressed; direction is derived from `OM > 1`.
- **Entry + exit**: bundled into one decision at NOW close (entry + SL + TP1–TP4), not predicted separately.
- **Trend + reversal**: out of scope — no such decomposition exists.
- **Regime + prediction**: out of scope — no regime signal in the model (funding rate, changepoint/HMM regime detection are named-missing feature candidates, [05 A8/B6](<05-Weakness Analysis.md>)).
- **Multi-task learning**: not literal separate task heads — one head, multi-target regression (`MAE`, `OM`, `MFE`). Formal single- vs multi-task/complexity-control policy is unwritten ([00-ToC §1.4](<00-ToC & Coverage.md>)).

## Open gaps

Rationale behind these choices (why 240 minutes, why regression-then-rule over direct classification, why no loss-weighting) is undocumented — see [05-Weakness Analysis.md §A1–A4](<05-Weakness Analysis.md>) (importance 1–2).

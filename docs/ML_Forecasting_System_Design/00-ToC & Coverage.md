# ML_Forecasting_System_Design — Topic Coverage & Gaps

See [99-Weakness Analysis.md](99-Weakness Analysis.md) for importance-rated gaps and missing technique comparisons built on this doc.

## 1. Problem & Objective Engineering

The existing objective document already covers price/direction/position/entry-exit prediction, multi-horizon prediction, classification/regression/ranking, objectives, prediction formulation, and problem decomposition. fileciteturn3file2L96-L132

### 1.1 Forecasting target selection criteria **missing**

- How to decide whether the primary target should be:
  - price
  - return
  - direction
  - position
  - TP/SL-related quantities
  - probability/distribution
- Criteria for choosing a target based on signal quality, learnability, stability, and usefulness for downstream trading evaluation.

### 1.2 Target-horizon selection methodology **missing**

- How to choose prediction horizons systematically rather than manually.
- Relationship between:
  - prediction horizon
  - market timeframe
  - label noise
  - opportunity frequency
  - expected signal decay.
- How to compare candidate horizons fairly.

### 1.3 Objective alignment **missing**

- Formal mapping:

  `business/trading objective → model target → training objective → evaluation KPI`

- Detecting situations where optimization improves a model metric while degrading actual trading usefulness.

### 1.4 Problem complexity control **missing**

- Criteria for deciding whether a problem should remain:
  - single-target
  - multi-target
  - multi-task
  - hierarchical.
- Avoiding unnecessary prediction heads and complexity.

---

## 2. Data, Label & Feature Engineering

The existing data document covers Binance OHLCV, historical depth, completed-candle synchronization, missing-candle handling, data integrity, returns, normalization alternatives, the candle feature schema, future-information rules, labels/output targets, class imbalance, and dataset splitting. fileciteturn1file4L296-L342 fileciteturn3file8L304-L347

### 2.1 Data provenance and reproducibility **missing**

- Exact Binance dataset/version used for an experiment.
- Dataset snapshot identification.
- Reproducible reconstruction of the exact training dataset.
- Handling changes in locally cached data between experiments.

### 2.2 Data-quality severity policy **missing**

- Which data problems cause:
  - candle rejection
  - window rejection
  - dataset rejection
  - experiment rejection.
- Quantitative thresholds for acceptable corruption or missingness.

### 2.3 Label quality measurement **missing**

- Label consistency checks.
- Label-noise estimation.
- Ambiguous-label detection.
- Sensitivity of model performance to label thresholds.
- Measuring whether a proposed label contains sufficient predictive signal before expensive DL training.

### 2.4 Label alternative comparison **missing**

- A formal framework for comparing competing label definitions.
- Same-model/same-data evaluation of alternative labels.
- Detecting labels that are easier to predict but less useful for trading.

### 2.5 Feature interaction and redundancy **missing**

- Systematic detection of interactions between features.
- Distinguishing:
  - redundant features
  - complementary features
  - conditional features.
- Measuring whether combinations add signal beyond individual features.

### 2.6 Feature stability **missing**

- Stability of feature usefulness across:
  - assets
  - time periods
  - volatility regimes
  - market conditions.
- Detecting features whose historical usefulness is unstable.

### 2.7 Feature leakage beyond obvious future values **missing**

- Leakage through:
  - normalization windows
  - rolling calculations
  - resampling
  - multi-timeframe aggregation
  - cached derived data.
- Explicit point-in-time feature-generation tests.

### 2.8 Input-window selection **missing**

- Systematic comparison of:
  - sequence length
  - lookback duration
  - number of candles
  - different windows per timeframe.
- Relationship between context length, useful information, compute cost, and overfitting.

Per-tf window length is now flagged as an Optuna search dimension (uniform / independent-per-tf / tapering-schedule candidates), not fixed at the prior 256/tf default — see [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length". Still missing: the systematic comparison methodology itself (this section's actual ask) — the doc above names candidate schemes and the profiler-driven discipline to test them, not yet a completed comparison.

### 2.9 Multi-symbol generalization **missing**

- Formal testing methodology for:
  - BTC → ETH
  - BTC → other symbols
  - multiple training pairs → held-out pair.
- Measuring whether learned patterns are market-specific or transferable.

The current split is a 4-way scheme: other pairs for training, each training pair's own latest slice as Validation A (regime generalization), BTC/USDT as Validation B (cross-symbol generalization), and a recent BTC/USDT Final Test. fileciteturn3file0L31-L41 fileciteturn3file1L68-L77

---

## 3. Model & Architecture Engineering

The existing architecture document already covers candidate model families, input/feature embedding, local feature extraction, sequential modeling, multi-timeframe fusion, architecture combinations, output heads, hardware constraints, and implementation/design-checklist requirements. fileciteturn1file9L479-L505

### 3.1 Architecture-selection methodology

- Formal decision process for choosing:
  - CNN/TCN
  - LSTM/GRU
  - Transformer
  - SSM/Mamba
  - hybrid architectures.
- Matching architecture characteristics to observed market-pattern requirements.

The [Decision Framework](05-Prioritization Framework.md#decision-framework) tiers candidates for funding; the observed-characteristic → mechanism mapping this section asked for now lives in [03 § architecture-selection methodology](03-Model & Architecture Engineering.md#architecture-selection-methodology), feeding that framework's `domain_fit` factor.

### 3.2 Capacity selection

- How to determine appropriate:
  - depth
  - width
  - hidden dimension
  - number of heads
  - number of blocks.
- Relationship between model capacity, dataset size, and overfitting.

Capacity-ladder protocol (train/val-gap-driven, finalists-only) now in [03 § capacity sizing](03-Model & Architecture Engineering.md#capacity-sizing).

### 3.3 Architecture component independence

- How to determine whether an architectural component contributes independently.
- Interaction testing between components.
- Avoiding conclusions based on a component that only helps because of another component.

Per-stage zeroing/interaction protocol, reusing the [unified super-architecture skeleton](03-Model & Architecture Engineering.md#unified-super-architecture-skeleton), now in [03 § component-independence testing](03-Model & Architecture Engineering.md#component-independence-testing).

### 3.4 Combination-strategy decision rules

- When to prefer:
  - sequential composition
  - parallel branches
  - late ensemble
  - stacking
  - gating
  - mixture-of-experts.
- Quantifying whether additional complexity produces enough incremental value.

### 3.5 Architecture failure diagnosis

- Diagnosing whether poor performance comes from:
  - insufficient capacity
  - excessive capacity
  - inappropriate inductive bias
  - optimization difficulty
  - bad input representation
  - bad labels
  - insufficient context.

Seven-cause checklist with a cheapest-first testing order now in [03 § architecture failure diagnosis](03-Model & Architecture Engineering.md#architecture-failure-diagnosis).

### 3.6 Architecture robustness

- Stability of architecture rankings across:
  - random seeds
  - training periods
  - symbols
  - label variants.
- Detecting architecture choices that win only under one experimental condition.

Distinguished from the general ≥3-seeds-per-config discipline in [statistical validity of comparisons](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons) — that tests one split, this tests whether the _ranking_ survives a second axis of variation. Protocol in [03 § cross-seed and cross-condition robustness](03-Model & Architecture Engineering.md#cross-seed-and-cross-condition-robustness).

### 3.7 Parameter-count vs effective-capacity analysis

- Measuring actual computational/representational cost rather than using parameter count alone.
- Comparing attention complexity, sequence length, activation memory, and throughput.

Required per-candidate comparison (activation memory, measured throughput at matched param count, wall-clock examples/sec) now in [03 § param-count vs effective-capacity analysis](03-Model & Architecture Engineering.md#param-count-vs-effective-capacity-analysis), extending the worked example under [hardware constraints](03-Model & Architecture Engineering.md#hardware-constraints).

### 3.8 Architecture simplification

- When a simpler model should be preferred despite slightly lower statistical performance.
- Minimum-complexity rule for accepting architectural improvements.

Minimum-complexity rule (prefer the simpler candidate unless the more complex one clears the paired-test CI _and_ isn't attributable to one dominant component) now in [03 § simplification rule](03-Model & Architecture Engineering.md#simplification-rule), generalizing the combination-strategy doc's existing single-backend-wins default.

---

## 4. Training Engineering

The existing material covers losses, class-weight/focal alternatives, regularization, training dynamics, mixed precision/checkpointing, and the architecture-specific training interface. fileciteturn3file8L331-L347 fileciteturn3file7L273-L277

Training-strategy selection, batch-size strategy, epoch/budget selection, loss-weight selection, training stability, sampling strategy, and augmentation are now covered in [04 § Training Engineering](04-Experimentation, Evaluation & Optimization.md#training-engineering).

### 4.4 Initialization and reproducibility **missing**

- Weight initialization alternatives.
- Random seed control.
- Deterministic vs nondeterministic GPU behavior.
- Measuring seed sensitivity.

---

## 5. Experimentation, Evaluation & Optimization Engineering

This is already the project's central comparison/optimization document. It covers alternative design, comparison criteria, baselines, controlled experiments, feasibility, statistical metrics, trading KPIs, experiment design, ablation, statistical validity, parameter optimization, hyperparameter optimization, search strategies, and cross-architecture fairness. fileciteturn2file1L43-L67 fileciteturn2file3L123-L138

### 5.1 Experiment hierarchy

- Define a consistent hierarchy such as:
  - screening
  - candidate validation
  - optimization
  - finalist confirmation.
- Prevent expensive full training from being used for every low-value hypothesis.

### 5.2 Experiment-budget allocation **missing**

- How much compute to allocate to:
  - feature screening
  - label experiments
  - architecture search
  - hyperparameter search
  - finalist confirmation.
- Dynamic reallocation of budget based on evidence.

### 5.3 Experiment stopping criteria

- When an alternative has enough evidence to be:
  - accepted
  - rejected
  - deferred.
- Early stopping of experiments that cannot plausibly become competitive.

### 5.4 Fair comparison across fundamentally different approaches

- Equalizing:
  - data
  - preprocessing
  - training budget
  - GPU budget
  - search budget
  - seeds
  - evaluation conditions.
- Preventing an architecture from winning simply because it received more optimization effort.

The current architecture comparison already establishes a shared split, shared GPU-hour budget, architecture-agnostic pruning, and a minimum grace period. fileciteturn3file9L358-L368

### 5.5 Multi-stage metric hierarchy

- Explicit distinction between:
  - training metrics
  - development diagnostics
  - trading KPIs
  - final selection criteria.
- Rules for resolving conflicts between metrics.

The project already establishes that statistical loss is not the trading objective and that final selection should use backtested trading KPIs. fileciteturn2file1L65-L80

### 5.6 Error taxonomy **missing**

- Standardized classification of failures:
  - wrong direction
  - wrong magnitude
  - wrong confidence
  - bad TP/SL estimate
  - regime-specific failure
  - missed opportunity
  - excessive false signals.
- Mapping each failure type to possible engineering interventions.

### 5.7 Error-driven optimization **missing**

- Formal loop:
  - measure failure
  - identify likely cause
  - generate targeted alternatives
  - test
  - retain/reject.
- Preventing random experimentation without a hypothesis.

### 5.8 Search-result validation

- Confirming that the best Optuna/search result remains competitive after:
  - additional seeds
  - longer training
  - independent rerun.
- Detecting optimizer overfitting.

### 5.9 Optimization overfitting

- Detecting overfitting to:
  - validation symbol
  - validation period
  - repeated experiments
  - KPI selection.
- Rules for when a validation set becomes contaminated through repeated decisions.

### 5.10 Feasibility frontier **missing**

- Explicit Pareto-style comparison between:
  - profitability
  - drawdown
  - generalization
  - compute
  - latency
  - memory
  - model complexity.
- Define when a small performance gain is not worth additional complexity/cost.

### 5.11 Final-selection protocol

- Exact procedure for locking:
  - architecture
  - features
  - labels
  - normalization
  - hyperparameters
  - thresholds
    before touching the final holdout.
- Rules for what happens if final-holdout performance is materially worse.

The final holdout is already specified as untouched until all tuning decisions are locked. fileciteturn3file1L68-L77

---

## 6. Deployment, Monitoring & Continuous Learning

This entire area is currently **out of scope**, so there are no additional in-scope sub-topics to add.

Production deployment, production monitoring/alerting, and online/continuous-learning pipelines are explicitly excluded. fileciteturn1file0L92-L95

---

## 7. Decision/System Architecture

This area is currently **out of scope**.

The current project stops at forecasting/model evaluation/backtesting and does not design the complete automated trading decision/execution system. End-to-end trading-system architecture and an automated decision layer built on model output are explicitly excluded. fileciteturn1file0L97-L99

The following remain outside the current scope:

- exchange/broker execution
- order-management architecture
- live position management
- production risk-control infrastructure
- complete automated signal-to-order pipeline.

---

## Excluded Topics

### 1. Security

Not part of the current ML research scope:

- ML security
- adversarial attacks
- data poisoning
- model extraction
- cybersecurity
- prompt injection.

These concern protecting the model/system from malicious behavior rather than improving market forecasting quality. fileciteturn1file0L17-L23

### 2. MLOps / Infrastructure

Excluded:

- Kubernetes
- cloud infrastructure
- CI/CD
- distributed serving
- infrastructure orchestration
- production infrastructure engineering.

The project can use the existing local development/training environment; production-scale infrastructure engineering is not needed to answer the current forecasting research questions. fileciteturn1file0L25-L31

### 3. Reinforcement Learning

Excluded:

- Q-learning
- DQN
- policy gradients
- actor-critic
- PPO
- offline RL
- multi-agent RL.

The current approach is supervised/self-supervised ML/DL forecasting from historical market data, not learning a trading policy through environment interaction and rewards. fileciteturn1file0L33-L40

### 4. Production Decision Layer

Excluded:

- live trading execution
- broker/exchange execution architecture
- operational position management
- production risk-control infrastructure.

The model may predict Long/Short/None and related quantities for research/backtesting, but building the production mechanism that converts those predictions into live orders is outside this project. fileciteturn1file0L42-L46

### 5. System-Level Architecture

Excluded:

- large-scale distributed AI architecture
- production system architecture
- microservices architecture
- infrastructure architecture.

The project is concerned with the forecasting/modeling research pipeline, not production software architecture. fileciteturn1file0L48-L52

### 6. Meta-Optimization / AutoML

Excluded as a project discipline:

- fully automated feature engineering
- neural architecture search as the primary methodology
- fully automated model selection
- fully automated experiment generation.

Manual/hypothesis-driven candidate selection plus controlled automated hyperparameter optimization is sufficient. The project already uses architecture choice as a controlled categorical search variable rather than unrestricted AutoML. fileciteturn1file0L54-L58

### 7. Final Production Validation

Excluded:

- production acceptance testing
- canary deployment
- blue/green deployment
- production A/B testing
- long-term production certification.

Historical out-of-sample and backtesting validation remain in scope; production certification is not. fileciteturn1file0L60-L65

### 8. LLM / General AI

Excluded for now:

- LLMs
- tokenization
- prompt engineering
- RAG
- agents
- language-model fine-tuning
- multimodal AI
- image/audio/video models
- general text processing.

The current problem is numerical time-series forecasting from financial-market data, so these techniques do not belong in the present research scope. fileciteturn1file0L67-L78

### 9. Irrelevant Data Modalities

Excluded:

- images
- audio
- video
- generic sensor data
- generic text datasets.

The input domain is market time series: OHLCV, derived prices, volume, technical indicators, temporal features, and market-structure features. fileciteturn1file0L80-L85

### 10. Alternative Market-Data Acquisition

Excluded:

- generic web scraping
- unrelated external data acquisition
- multiple unrelated market-data providers.

The project assumes locally cached Binance market data. fileciteturn1file0L87-L90

---

## Deferred / Revisit Later

These are not permanently excluded; they are deliberately postponed because they become important at a later project stage.

### Transaction Costs, Spread, Slippage & Latency

Deferred because the current focus is forecasting research rather than live/sub-4H execution economics.

Before paper/live trading, backtesting must incorporate realistic:

- fees
- spread
- slippage
- execution latency.

A cost-free backtest can materially overstate real performance. fileciteturn1file0L101-L103

### Risk & Position Sizing Beyond TP Targets

Currently handled by the existing manual procedure rather than learned by the model.

Revisit if the project later moves toward:

- model-generated position sizing
- dynamic exposure
- portfolio-level risk optimization.

fileciteturn1file0L101-L103

### Market-Regime Robustness & Retraining Cadence

Not currently developed as a continuous-learning problem.

Revisit after sufficient live/paper experience to determine:

- whether model performance decays
- which regimes cause degradation
- how quickly market dynamics change
- when retraining should occur.

Crypto markets can shift between trend/range/volatility regimes, so a model that remains unchanged indefinitely may silently degrade. fileciteturn1file0L101-L104

---

## Coverage Summary

| Area                                                         | Current status                                                                                                                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Problem & Objective Engineering                              | Core definition exists; target-selection and objective-alignment methodology need expansion                                                                                                             |
| Data, Label & Feature Engineering                            | Substantial coverage; data provenance, feature stability, leakage depth, and systematic target/input selection need expansion                                                                           |
| Model & Architecture Engineering                             | Strong coverage, incl. architecture-selection mapping, capacity sizing, component-independence testing, failure diagnosis, cross-condition robustness, effective-capacity analysis, simplification rule |
| Training Engineering                                         | Core mechanisms exist; training-strategy, stability, initialization, sampling, and loss-weight methodology need expansion                                                                               |
| Experimentation, Evaluation & Optimization Engineering       | Strongest-covered area; experiment hierarchy, budget allocation, stopping rules, error taxonomy, and optimization-overfitting need expansion                                                            |
| Deployment, Monitoring & Continuous Learning                 | Explicitly excluded                                                                                                                                                                                     |
| Decision/System Architecture                                 | Explicitly excluded                                                                                                                                                                                     |
| Security / MLOps / RL / LLM / AutoML / Production validation | Explicitly excluded                                                                                                                                                                                     |
| Transaction costs / risk sizing / regime retraining          | Deferred                                                                                                                                                                                                |

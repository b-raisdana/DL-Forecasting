# ML_Forecasting_System_Design — Topic Coverage

Index of what each design doc covers, and where in that doc.

## 1. Problem & Objective Engineering

[01](01-Objective.md) covers price/direction/position/entry-exit prediction, multi-horizon prediction, classification/regression/ranking, objectives, prediction formulation, and problem decomposition. fileciteturn3file2L96-L132

---

## 2. Data, Label & Feature Engineering

[02](<02-Data, Label & Feature Engineering.md>) covers Binance OHLCV, historical depth, completed-candle synchronization, missing-candle handling, data integrity, returns, normalization alternatives, the candle feature schema, future-information rules, labels/output targets, class imbalance, and dataset splitting. fileciteturn1file4L296-L342 fileciteturn3file8L304-L347

### 2.8 Input-window selection

Per-tf window length is now an Optuna search dimension (uniform / independent-per-tf / tapering-schedule candidates) — see [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length".

### 2.9 Multi-symbol generalization

Current 4-way split (other pairs for training, each pair's own latest slice as Validation A / regime generalization, BTC/USDT as Validation B / cross-symbol generalization, a recent BTC/USDT Final Test): [02 § validation & train/test splitting](<02-Data, Label & Feature Engineering.md#validation--traintest-splitting>).

---

## 3. Model & Architecture Engineering

[03](<03-Model & Architecture Engineering.md>) covers candidate model families, input/feature embedding, local feature extraction, sequential modeling, multi-timeframe fusion, architecture combinations, output heads, hardware constraints, and implementation/design-checklist requirements. fileciteturn1file9L479-L505

### 3.1 Architecture-selection methodology

The [Decision Framework](05-Prioritization Framework.md#decision-framework) tiers candidates for funding; the observed-characteristic → mechanism mapping lives in [03 § architecture-selection methodology](03-Model & Architecture Engineering.md#architecture-selection-methodology), feeding that framework's `domain_fit` factor.

### 3.2 Capacity selection

Capacity-ladder protocol (train/val-gap-driven, finalists-only): [03 § capacity sizing](03-Model & Architecture Engineering.md#capacity-sizing).

### 3.3 Architecture component independence

Per-stage zeroing/interaction protocol, reusing the [unified super-architecture skeleton](03-Model & Architecture Engineering.md#unified-super-architecture-skeleton): [03 § component-independence testing](03-Model & Architecture Engineering.md#component-independence-testing).

### 3.4 Combination-strategy decision rules

[05 § combination strategy](05-Prioritization Framework.md#combination-strategy) tiers sequential/parallel/ensemble/stacking/gating/MoE candidates.

### 3.5 Architecture failure diagnosis

Seven-cause checklist with a cheapest-first testing order: [03 § architecture failure diagnosis](03-Model & Architecture Engineering.md#architecture-failure-diagnosis).

### 3.6 Architecture robustness

Distinguished from the general ≥3-seeds-per-config discipline in [statistical validity of comparisons](<04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons>) — that tests one split, this tests whether the _ranking_ survives a second axis of variation. Protocol: [03 § cross-seed and cross-condition robustness](03-Model & Architecture Engineering.md#cross-seed-and-cross-condition-robustness).

### 3.7 Parameter-count vs effective-capacity analysis

Required per-candidate comparison (activation memory, measured throughput at matched param count, wall-clock examples/sec): [03 § param-count vs effective-capacity analysis](03-Model & Architecture Engineering.md#param-count-vs-effective-capacity-analysis), extending the worked example under [hardware constraints](03-Model & Architecture Engineering.md#hardware-constraints).

### 3.8 Architecture simplification

Minimum-complexity rule (prefer the simpler candidate unless the more complex one clears the paired-test CI _and_ isn't attributable to one dominant component): [03 § simplification rule](03-Model & Architecture Engineering.md#simplification-rule).

---

## 4. Training Engineering

[04](<04-Experimentation, Evaluation & Optimization.md>) covers losses, class-weight/focal alternatives, regularization, training dynamics, mixed precision/checkpointing, the architecture-specific training interface, training-strategy selection, batch-size strategy, epoch/budget selection, loss-weight selection, training stability, sampling strategy, and augmentation — see [04 § Training Engineering](<04-Experimentation, Evaluation & Optimization.md#training-engineering>). fileciteturn3file8L331-L347 fileciteturn3file7L273-L277

---

## 5. Experimentation, Evaluation & Optimization Engineering

[04](<04-Experimentation, Evaluation & Optimization.md>) is the project's central comparison/optimization document: alternative design, comparison criteria, baselines, controlled experiments, feasibility, statistical metrics, trading KPIs, experiment design, ablation, statistical validity, parameter optimization, hyperparameter optimization, search strategies, cross-architecture fairness. fileciteturn2file1L43-L67 fileciteturn2file3L123-L138

### 5.4 Fair comparison across fundamentally different approaches

Shared split, shared GPU-hour budget, architecture-agnostic pruning, minimum grace period: [04 § cross-architecture fairness](<04-Experimentation, Evaluation & Optimization.md#cross-architecture-fairness>).

### 5.5 Multi-stage metric hierarchy

Statistical loss is a training-time signal only, never the selection criterion; final selection always uses [04 § backtested trading KPIs](<04-Experimentation, Evaluation & Optimization.md#backtested-trading-kpis-final-selection>).

### 5.11 Final-selection protocol

Locking order and the "materially worse than Validation B" rule (investigate, don't re-tune — that needs a fresh holdout): [04 § model-selection pipeline](<04-Experimentation, Evaluation & Optimization.md#model-selection-pipeline>) step 4.

---

## 6. ML-Ops

[06](06-ML-Ops.md) covers training-run resource monitoring — GPU compute/VRAM, RAM, CPU, disk I/O KPIs and thresholds for this project's single-GPU local hardware, plus the diagnostic decision tree for isolating which resource is actually the bottleneck. Concrete implementation: `ResourceSampler` in `app/ai_modelling/tier1_000/train.py`. Scoped to this project's own local hardware, not production/serving MLOps — see [99-Exclusion.md § MLOps / Infrastructure](99-Exclusion.md#mlops--infrastructure).

---

Topic searched here and not found? Check [99-Weakness Analysis.md](99-Weakness Analysis.md) (gaps) and [99-Exclusion.md](99-Exclusion.md) (exclusions).

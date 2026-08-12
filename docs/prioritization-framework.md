# Prioritization Framework — Tiering Candidate Techniques

Scoring method for sorting candidate techniques (architecture blocks, GBM variants, normalization schemes, anything this project evaluates as a named alternative) into fund-now / test-later / parked tiers, so that judgment stays consistent across the many candidate lists in [model-architecture-planning.md](model-architecture-planning.md) and its companion docs rather than being decided fresh, informally, per bullet.

- [Prioritization Framework — Tiering Candidate Techniques](#prioritization-framework--tiering-candidate-techniques)
  - [why](#why)
  - [tiers](#tiers)
  - [scoring factors](#scoring-factors)
  - [tool-identity test: when a X/Y grouping stays one row](#tool-identity-test-when-a-xy-grouping-stays-one-row)
  - [combination formula](#combination-formula)
  - [mandatory-floor exception](#mandatory-floor-exception)
  - [how to apply](#how-to-apply)
  - [tiered candidates by layer](#tiered-candidates-by-layer)
    - [normalization strategy](#normalization-strategy)
    - [model architecture \& selection](#model-architecture--selection)
      - [input / feature embedding](#input--feature-embedding)
      - [local feature extraction](#local-feature-extraction)
      - [sequential encoding](#sequential-encoding)
      - [attention / dependency](#attention--dependency)
      - [global representation](#global-representation)
      - [current Stage-1 candidate set](#current-stage-1-candidate-set)
      - [activation mechanisms](#activation-mechanisms)
      - [combination strategy](#combination-strategy)
      - [fusion mechanism](#fusion-mechanism)
    - [multi-timeframe fusion](#multi-timeframe-fusion)
    - [auxiliary tabular models (GBM-family)](#auxiliary-tabular-models-gbm-family)

## why

The design-space lists across this project's planning docs (embedding, local extraction, sequential encoding, attention, fusion, GBM-family alternatives, normalization schemes, etc.) are long and growing faster than the single-GPU budget can test everything. Those docs already make tiering judgments informally in prose — e.g. ModernTCN "worth promoting straight into Stage-1 profiling" vs. KAN-based blocks "parked... pending evidence" (see [current Stage-1 candidate set](model-architecture-planning.md#current-stage-1-candidate-set)). This doc makes that judgment repeatable: a fixed factor list and a formula, so a new candidate gets the same treatment a prior one got, instead of re-litigating "how excited should we be about this" per bullet.

## tiers

- **Tier 1 — fund now:** active Stage-1 candidate, or a near-term addition to whatever categorical search / screening step is relevant.
- **Tier 2 — secondary:** test post-primary-search, as a refinement/alt within an already-covered role, or once finalist/budget headroom allows.
- **Tier 3 — parked:** logged for awareness only; revisit only on new evidence (a cheaper-proxy result, a new benchmark, an unblocked dependency) — not on a schedule.

## scoring factors

Score each 0–2: 0 = absent/weak, 1 = moderate, 2 = strong. **Score in context, not in the abstract** — "evidence" and "dominance" mean _evidence/dominance that this candidate wins on this project's actual multivariate, multi-tf, non-stationary setup_, not general field fame. A technique that's a textbook standard elsewhere but structurally doesn't fit here (classic univariate ARIMA/GARCH against a genuinely multivariate schema, for instance) scores low on those factors despite the broad reputation — see the [current Stage-1 candidate set](#current-stage-1-candidate-set) table for that exact case scored against a superficially similar but better-fitting alternative.

**pull factors (raise tier):**

1. **research/reference-document support** — peer-reviewed papers or published benchmarks (the kind already cited inline across this project's docs, e.g. TSMixer's own ablations, TabArena 2025) showing the candidate beats relevant alternatives _on a comparable problem_ — not just "it's been used somewhere."
2. **field dominance / standard-of-practice** — is it the current default choice in the relevant application area (time-series forecasting, NLP-derived sequence modeling, tabular ML), independent of project-specific evidence? De-facto standards carry lower integration risk.
3. **modernity / replacement trajectory** — is it a newer technique explicitly designed to supersede something already in the candidate set (ModernTCN→TCN, MLA→GQA)? Adopting it should retire complexity, not just add a parallel option.
4. **resource fit (hardware/compute budget)** — narrowly: does it run within the single-GPU VRAM/RAM/wall-clock ceiling that governs every candidate (see [hardware constraints](model-architecture-candidate-sets.md#hardware-constraints))? This is a feasibility check, not a value judgment — it doesn't ask whether the technique is _worth_ its cost, only whether the cost is affordable at all.
5. **domain/problem fit** — two things bundled deliberately: (a) _structural_ fit — does the technique's required input shape (multivariate channels, multi-tf branches, ~256-candle sequences, "needs a graph," "assumes univariate") match what this project actually has, and (b) _characteristic_ fit — does its benefit target something this data actually exhibits (non-stationary, noisy OHLCV, genuinely multivariate schema) rather than a generic capability bump never tested against this kind of noise. A technique that structurally can't consume this input at all (e.g. a graph-structured model with no graph in the data) scores 0 here regardless of how well-regarded it is elsewhere.
6. **marginal impact vs. cost** — a _value_ judgment layered on top of the two feasibility checks above: given it fits (resource_fit) and applies (domain_fit), does adopting it directly close a gap already flagged elsewhere (O(n²)/VRAM ceiling, confidence-metric gap, class imbalance) cheaply, or is it a large lift for a small/speculative gain? Two candidates can both score well on resource_fit/domain_fit and still diverge sharply here — a technique can be cheap and applicable but bring only a marginal, already-covered benefit (e.g. Mish activation), or bring the exact fix to a bottleneck this doc already names (e.g. Perceiver-style latent-bottleneck attention against the flagged multi-tf VRAM cost).

**modifiers (lower tier, never raise it):**

1. **risk / reversibility** — an incremental, low-risk swap into an already-covered role (e.g. xLSTM within the LSTM floor) vs. a structurally new, unproven mechanism that needs its own validation cycle.
2. **tooling/library maturity** — drop-in within libraries already in scope (LightGBM/XGBoost/Optuna/Keras) vs. a custom implementation from a paper.
3. **dependency / gating status** — hard gate, not a score: blocked behind another unresolved decision (e.g. the fusion-mechanism choice only matters once "combination strategy" ≠ single-backend-wins, see [combination strategy](model-architecture-planning.md#combination-strategy)) demotes the tier, per the rule below, until the blocker resolves.

## tool-identity test: when a X/Y grouping stays one row

Several candidates below are named `X/Y` (LSTM/GRU, TCN/ModernTCN, GQA/MQA, ...). That notation means two different things depending on the pair: a merged row hides whichever of the two items is actually weaker, and a reader can't tell "these are drop-in-swappable" from "these are lumped for convenience" just by looking at the slash.

Apply one mechanical test: **would implementing both, in the library this project actually uses (Keras/TF for architecture blocks, LightGBM/XGBoost/statsmodels/etc for GBM-family), mean instantiating two different classes, or calling the same class/function with a different argument?**

- **Different classes/tools → split into separate rows**, each scored independently against its neighbors. Example: `tf.keras.layers.LSTM` vs `tf.keras.layers.GRU` are two distinct layer classes with different gating internals — LSTM/GRU splits.
- **Same class/function, different argument → stays one row.** Example: TCN vs ModernTCN are both a stacked `Conv1D` block; ModernTCN is that same block called with a larger kernel size and grouped/depthwise convolution — a parameter choice, not a different tool. GQA vs MQA are the same multi-query-attention mechanism parameterized by KV-group count (MQA = GQA with `num_kv_groups=1`) — also stays merged.

This test is orthogonal to how similar the mechanisms _feel_: GRU is arguably more similar to LSTM internally than ModernTCN is to plain TCN, but GRU still gets its own row because it's a separate class you'd import and swap in, while ModernTCN is a config change on the block already in the row above it. When in doubt, ask whether a hyperparameter-search tool (Optuna categorical dimension) would need a new class/import to add the second option, or just a new value for an existing parameter — the former splits, the latter doesn't.

## combination formula

```text
raw      = evidence + dominance + modernity + resource_fit + domain_fit + impact_vs_cost   (0–12)
adjusted = raw − (1 if speculative/unproven mechanism else 0)     [risk modifier]
               − (1 if no mature drop-in library/impl else 0)     [tooling modifier]
```

- **Tier 1** if `adjusted ≥ 8` and not gated
- **Tier 2** if `4 ≤ adjusted < 8`, **or** `adjusted ≥ 8` but gated — the gate is a _ceiling_, not a floor: it demotes an otherwise-Tier-1 score, it never promotes a low score
- **Tier 3** if `adjusted < 4`, regardless of gating

Recalibrate the 8/4 cutoffs, not the mandatory-floor exception below, if the formula ever disagrees with a clear case.

## mandatory-floor exception

Treat a "mandatory floor" (naive/persistence baseline, GBM-on-flattened floor — see [current Stage-1 candidate set](model-architecture-planning.md#current-stage-1-candidate-set)) as always-Tier-1, orthogonal to this formula. Those score low on most pull factors (nothing novel or dominant about a persistence baseline) but aren't optional: they exist to prove a learned candidate beats doing nothing, not to compete on the same axes as everything else here. Both worked examples below (naive baseline, GBM-on-flattened-as-floor) score under the natural Tier-1 cutoff on the formula alone — the exception is what actually promotes them, not the arithmetic.

## how to apply

When adding a new candidate anywhere in the planning docs (a new architecture block, a new GBM variant, a new normalization scheme):

1. Score the six pull factors and three modifiers per the definitions above, **against the other candidates already scored in the same section** below — not in isolation. A technique's "dominance" or "modernity" score should reflect its rank relative to its actual peers in that layer, not a global absolute; see the [scoring factors](#scoring-factors) note on scoring in context.
2. Run the combination formula.
3. Check the dependency/gating modifier — a hard gate demotes an otherwise-Tier-1 score to Tier 2; it never raises a Tier-3 score.
4. Add the candidate's row to the relevant table below (or create a new section if it's a new layer/topic), so the tiering judgment stays visible and comparable to its neighbors instead of being re-derived later.

## tiered candidates by layer

Section order and headings mirror [model-architecture-planning.md](model-architecture-planning.md)'s own structure, so a given layer's tiering sits next to the equivalent stage there. Only _open_ alternatives are scored — items a doc section already marks as resolved (e.g. ATR-relative normalization as primary, decision-anchor point, higher-tf in-progress-candle handling) aren't live candidates and are skipped. Scores are illustrative starting points from the current doc text, not final measurements — recalibrate any row once real MI/backtest/profiling evidence exists, per this project's own "measured evidence only" discipline (see [error metric ≠ trading objective](error-rating-and-evaluation.md#core-principle-error-metric--trading-objective)).

### normalization strategy

| candidate                                       | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier  |
| ----------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ----- |
| hybrid: ATR-norm price + raw log-return channel | 1        | 1         | 1         | 2            | 2          | 2           | 9   | 0    | 0       | 9        | **1** |
| log-return norm (scale-free)                    | 1        | 2         | 0         | 2            | 1          | 1           | 7   | 0    | 0       | 7        | **2** |
| rolling z-score                                 | 1        | 2         | 0         | 2            | 1          | 1           | 7   | 0    | 0       | 7        | **2** |
| min-max per window                              | 0        | 1         | 0         | 2            | 0          | 0           | 3   | 0    | 0       | 3        | **3** |

The hybrid scheme wins on domain fit (position + velocity, tailored to this project) despite log-return/z-score being more textbook-dominant in general finance — a direct illustration of the "score in context" note above. `no normalization` and `min-max as primary` are already rejected in the source doc, not re-scored here.

### model architecture & selection

#### input / feature embedding

| candidate                       | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier          |
| ------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ------------- |
| linear/MLP projection (default) | 2        | 2         | 0         | 2            | 2          | 2           | 10  | 0    | 0       | 10       | **1**         |
| PatchTST-style patch embedding  | 2        | 1         | 2         | 2            | 1          | 2           | 10  | 0    | −1      | 9        | **1**         |
| per-tf learned tf-id embedding  | 1        | 1         | 1         | 2            | 2          | 1           | 8   | 0    | 0       | 8        | **2** (gated) |

`per-tf tf-id embedding` only matters if the flat/shared-encoder architecture branch is chosen over the per-tf-branch design that's currently the working assumption (see [multi-timeframe fusion](model-architecture-planning.md#multi-timeframe-fusion)) — that dependency gate demotes an otherwise-qualifying score to Tier 2.

#### local feature extraction

| candidate                                                                                                                                                                                      | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ----- |
| TCN / ModernTCN (large-kernel/grouped-conv is a `Conv1D` param choice on the same block, not a separate tool — see [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row)) | 2        | 1         | 2         | 2            | 2          | 2           | 11  | 0    | 0       | 11       | **1** |
| plain/vanilla 1D conv (current code)                                                                                                                                                           | 2        | 2         | 0         | 2            | 2          | 2           | 10  | 0    | 0       | 10       | **1** |
| InceptionTime (multi-kernel-size Inception modules)                                                                                                                                            | 2        | 2         | 1         | 1            | 1          | 2           | 9   | 0    | 0       | 9        | **1** |
| FCN (fully-convolutional, global-pooled stack)                                                                                                                                                 | 2        | 1         | 0         | 2            | 1          | 2           | 8   | 0    | 0       | 8        | **1** |
| conv stem → Transformer (downsampling lever)                                                                                                                                                   | 1        | 1         | 1         | 2            | 1          | 1           | 7   | 0    | 0       | 7        | **2** |
| ResNet (residual conv blocks, TSC baseline)                                                                                                                                                    | 2        | 1         | 0         | 2            | 1          | 1           | 7   | 0    | 0       | 7        | **2** |
| TimesNet-style 1D→2D reshape                                                                                                                                                                   | 1        | 0         | 2         | 1            | 1          | 0           | 5   | −1   | −1      | 3        | **3** |
| SCINet                                                                                                                                                                                         | 1        | 0         | 1         | 1            | 1          | 1           | 5   | −1   | −1      | 3        | **3** |

TCN and ModernTCN stay one row: both are the same stacked-`Conv1D` block, distinguished only by kernel-size/dilation/grouping parameters, exactly the drop-in-parameter case the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row) says to keep merged — consistent with the doc calling ModernTCN "a direct, low-risk upgrade path," not a replacement that invalidates the floor.

`residual-CNN TSC baselines` (ResNet/FCN/InceptionTime) are scored as three separate rows because ResNet, FCN, and InceptionTime are three separately-implemented network topologies (residual blocks / plain fully-conv stack / multi-branch multi-kernel-size Inception modules — distinct classes in any TSC library, e.g. `aeon`/`tsai`), not parameter choices on one function, and scoring them independently surfaces a real difference an averaged score would hide: InceptionTime (the newest, best-benchmarked of the three, but costlier per module) and FCN (the cheapest, still competitive) both clear Tier 1 on their own merits, while plain ResNet — the oldest and weakest of the three on `impact/cost` since InceptionTime generally supersedes it in TSC benchmarks — lands Tier 2. TimesNet and SCINet land Tier 3 on the same pattern: real modernity, but the doc itself flags both as untested/plausible-not-confirmed for this pipeline specifically (risk modifier applies to both).

#### sequential encoding

| candidate                     | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier          |
| ----------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ------------- |
| LSTM (sanity floor)           | 2        | 2         | 0         | 2            | 2          | 2           | 10  | 0    | 0       | 10       | **1** (floor) |
| Mamba (selective state-space) | 2        | 1         | 2         | 2            | 1          | 2           | 10  | −1   | −1      | 8        | **1**         |
| GRU (alt within LSTM floor)   | 1        | 1         | 0         | 2            | 2          | 1           | 7   | 0    | 0       | 7        | **2**         |
| ConvLSTM                      | 1        | 1         | 0         | 1            | 1          | 1           | 5   | 0    | 0       | 5        | **2**         |
| S4 (fixed/HiPPO state-space)  | 1        | 1         | 1         | 2            | 1          | 1           | 7   | −1   | −1      | 5        | **2**         |
| xLSTM                         | 1        | 0         | 2         | 1            | 1          | 1           | 6   | −1   | −1      | 4        | **2**         |
| Hyena (implicit long conv)    | 1        | 0         | 2         | 1            | 1          | 1           | 6   | −1   | −1      | 4        | **2**         |

`LSTM` and `GRU` score as separate rows, and `Mamba` and `S4` score as separate rows, per the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row): `tf.keras.layers.LSTM` and `tf.keras.layers.GRU` are separate classes (not a parameter choice on one layer), and Mamba (input-selective scan, `mamba_ssm`) and S4 (fixed HiPPO-initialized state matrices, the `state-spaces/s4` repo) are separately-implemented mechanisms, not versions of one function. Scoring them independently surfaces real information an averaged score would hide: GRU joins xLSTM/ConvLSTM as a Tier-2 alt tested _within_ the LSTM floor role rather than sharing the floor's Tier-1 status outright — LSTM stays the floor because it's what the existing code already implements (`cnn_lstm_model.py`), so testing it costs nothing extra, while GRU is a genuinely new run. Likewise Mamba clears Tier 1 on modernity/impact but S4 — understood as Mamba's superseded predecessor rather than a co-equal option — lands Tier 2. ConvLSTM still outranks xLSTM/Hyena on `adjusted` despite a lower `raw` score — its mechanism is older and less flashy, but mature and low-risk, while xLSTM/Hyena's modernity edge gets clawed back by the risk/tooling modifiers. That's the intended shape of the formula: novelty alone doesn't win, it has to survive the discount for being unproven. `Hyena (implicit long conv)` is a single candidate, not a grouped pair — "implicit long convolution" is just Hyena's mechanism name.

#### attention / dependency

| candidate                                       | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier  |
| ----------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ----- |
| GQA/MQA                                         | 2        | 2         | 1         | 2            | 1          | 2           | 10  | 0    | 0       | 10       | **1** |
| cross-tf attention over pooled reps             | 1        | 1         | 1         | 2            | 2          | 2           | 9   | 0    | 0       | 9        | **1** |
| Longformer-style sliding-window + global tokens | 2        | 1         | 0         | 2            | 1          | 2           | 8   | 0    | 0       | 8        | **1** |
| standard self-attention (full, no mitigation)   | 2        | 2         | 0         | 0            | 2          | 1           | 7   | 0    | 0       | 7        | **2** |
| iTransformer                                    | 2        | 1         | 2         | 1            | 2          | 1           | 9   | −1   | −1      | 7        | **2** |
| Informer (ProbSparse)                           | 2        | 1         | 1         | 2            | 1          | 2           | 9   | −1   | −1      | 7        | **2** |
| MLA                                             | 1        | 0         | 2         | 2            | 1          | 2           | 8   | −1   | −1      | 6        | **2** |
| NSA                                             | 1        | 0         | 2         | 2            | 2          | 1           | 8   | −1   | −1      | 6        | **2** |
| BigBird-style + random attention                | 1        | 1         | 1         | 1            | 1          | 1           | 6   | 0    | −1      | 5        | **2** |
| Differential Attention                          | 1        | 0         | 2         | 1            | 2          | 1           | 7   | −1   | −1      | 5        | **2** |
| Autoformer                                      | 2        | 1         | 1         | 1            | 1          | 1           | 7   | −1   | −1      | 5        | **2** |
| FEDformer                                       | 2        | 0         | 1         | 1            | 1          | 1           | 6   | −1   | −1      | 4        | **2** |
| Performer (FAVOR+ kernel attention)             | 0        | 1         | 1         | 2            | 0          | 0           | 4   | −1   | −1      | 2        | **3** |
| Linformer (low-rank seq-length projection)      | 0        | 1         | 0         | 2            | 0          | 0           | 3   | −1   | −1      | 1        | **3** |

`GQA/MQA` stays one row per the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row): MQA is GQA with its KV-group count set to 1, the same attention call with one config value changed, not a separate mechanism. `Longformer` and `BigBird-style + random attention` score as separate rows because they're separate published models with separate reference implementations (`LongformerModel`/`BigBirdModel`), not the same function with a different argument: BigBird adds a third, block-sparse "random attention" component on top of Longformer's window+global pattern, which is enough extra implementation complexity (no straightforward dense-mask trick, needs a real block-sparse kernel to be efficient) to cost it a tooling point that plain Longformer doesn't pay — Longformer clears Tier 1, BigBird lands Tier 2. `Performer` and `Linformer` score as separate rows the same way — FAVOR+ random-feature kernels (Performer) and a fixed low-rank sequence-length projection (Linformer) are different approximation mechanisms, not parameter variants of one attention layer — both still land Tier 3, matching the doc's own "weakest... on modeling quality... bottom-of-priority fallback only" characterization; Performer edges out Linformer only on the modernity of its approximation, not on any evidence either belongs above the mitigations that already outrank both. `standard self-attention` — the un-mitigated default — sits below GQA/MQA and the sliding-window fallback precisely because of the VRAM cost this doc already flags as the binding constraint (`resource_fit = 0`); the mitigations exist specifically to outrank the mechanism they're mitigating. FlashAttention and mixed-precision AMP aren't scored here — they're always-on infrastructure applied under whichever mechanism wins, not competing candidates (see [hardware constraints](model-architecture-candidate-sets.md#hardware-constraints)).

#### global representation

| candidate                                        | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier  |
| ------------------------------------------------ | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ----- |
| pooling (mean/max/attn-pool/last-token, default) | 2        | 2         | 0         | 2            | 2          | 2           | 10  | 0    | 0       | 10       | **1** |
| N-HiTS (hierarchical interpolation, multi-rate)  | 2        | 1         | 2         | 2            | 1          | 2           | 10  | 0    | 0       | 10       | **1** |
| N-BEATS (stacked residual basis blocks)          | 2        | 1         | 0         | 2            | 1          | 1           | 7   | 0    | 0       | 7        | **2** |

`pooling (mean/max/attn-pool/last-token, default)` stays one row per the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row) — this project's `global_repr` stage builds it as one function taking a `kind` argument (mean/max/attention/last-token are branches inside `global_pool(x, kind=...)`, not separate imported classes), the same pattern as GQA/MQA above. `N-BEATS` and `N-HiTS` score as separate rows because they're separate model classes with different internals (plain stacked residual MLP blocks vs. hierarchical multi-rate interpolation), not a parameter choice on one block. N-HiTS pulls ahead because the doc's own text singles it out specifically for long-horizon efficiency against the flagged VRAM ceiling — a real reason an averaged score would obscure; N-BEATS alone, without that specific efficiency angle, lands Tier 2.

Prediction heads (action / MAE-OM regression / confidence) aren't scored — they're required output slots defined by the label design in [training-data.md](training-data.md#model-output-targets), not competing techniques to tier.

#### current Stage-1 candidate set

| candidate                                             | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier          |
| ----------------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ------------- |
| TCN / ModernTCN                                       | 2        | 1         | 2         | 2            | 2          | 2           | 11  | 0    | 0       | 11       | **1**         |
| LSTM (sanity floor)                                   | 2        | 2         | 0         | 2            | 2          | 2           | 10  | 0    | 0       | 10       | **1** (floor) |
| CNN-LSTM(-attention) (implemented)                    | 2        | 1         | 0         | 2            | 2          | 2           | 9   | 0    | 0       | 9        | **1**         |
| Transformer w/ per-tf embed + cross-tf attn           | 2        | 2         | 1         | 1            | 2          | 1           | 9   | 0    | 0       | 9        | **1**         |
| TSMixer                                               | 2        | 1         | 1         | 2            | 1          | 2           | 9   | 0    | 0       | 9        | **1**         |
| DLinear                                               | 2        | 1         | 0         | 2            | 1          | 2           | 8   | 0    | 0       | 8        | **1**         |
| Mamba (selective state-space)                         | 2        | 1         | 2         | 2            | 1          | 2           | 10  | −1   | −1      | 8        | **1**         |
| Perceiver (latent-bottleneck cross-attn)              | 1        | 1         | 2         | 2            | 1          | 2           | 9   | 0    | −1      | 8        | **1**         |
| naive/persistence baseline                            | 0        | 2         | 0         | 2            | 0          | 0           | 4   | 0    | 0       | 4        | **1** (floor) |
| hybrid CNN→Transformer                                | 1        | 1         | 1         | 2            | 1          | 1           | 7   | 0    | 0       | 7        | **2**         |
| GRU (alt within LSTM floor)                           | 1        | 1         | 0         | 2            | 2          | 1           | 7   | 0    | 0       | 7        | **2**         |
| TFT (Temporal Fusion Transformer)                     | 2        | 2         | 0         | 1            | 1          | 1           | 7   | 0    | 0       | 7        | **2**         |
| S4 (fixed/HiPPO state-space)                          | 1        | 1         | 1         | 2            | 1          | 1           | 7   | −1   | −1      | 5        | **2**         |
| 1-NN w/ DTW distance                                  | 2        | 1         | 0         | 1            | 1          | 0           | 5   | 0    | 0       | 5        | **2** (gated) |
| 4 separate per-tf models + late ensemble (as primary) | 1        | 1         | 0         | 0            | 1          | 0           | 3   | 0    | 0       | 3        | **3**         |
| GBM on flattened features (as primary architecture)   | 0        | 1         | 0         | 2            | 0          | 0           | 3   | 0    | 0       | 3        | **3**         |
| pure MLP on flattened features                        | 0        | 0         | 0         | 2            | 0          | 0           | 2   | 0    | 0       | 2        | **3**         |
| ARIMA / SARIMA                                        | 0        | 1         | 0         | 1            | 0          | 0           | 2   | 0    | 0       | 2        | **3**         |
| exponential smoothing / ETS                           | 0        | 1         | 0         | 1            | 0          | 0           | 2   | 0    | 0       | 2        | **3**         |
| GARCH (volatility, not point-forecast)                | 0        | 1         | 0         | 1            | 0          | 0           | 2   | 0    | 0       | 2        | **3**         |
| TimeKAN (frequency-decomposition KAN backbone)        | 0        | 0         | 2         | 1            | 1          | 0           | 4   | −1   | −1      | 2        | **3**         |
| KANMixer (TSMixer-style block, KAN edges)             | 0        | 0         | 2         | 1            | 1          | 0           | 4   | −1   | −1      | 2        | **3**         |
| GNN over tf/symbol nodes                              | 0        | 0         | 1         | 1            | 1          | 0           | 3   | −1   | −1      | 1        | **3**         |

This table re-lists candidates already scored in the "sequential encoding," "global representation," and "attention/dependency" sections above in their Stage-1-set role, inheriting those scores rather than re-deriving them. A few groupings are worth spelling out explicitly, per the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row):

- **`LSTM` / `GRU`** → LSTM keeps the floor's Tier 1 (it's what the current code already runs, so testing it is free); GRU joins as a Tier-2 alt within that floor role, same treatment as xLSTM/ConvLSTM.
- **`state-space: Mamba` / `S4`** → Mamba clears Tier 1 on modernity/impact; S4, read as Mamba's superseded predecessor rather than a co-equal, lands Tier 2.
- **`all-MLP mixer: TSMixer` / `DLinear`** → both clear Tier 1 on their own scores, but for different reasons: TSMixer on the doc's own cited ablation evidence, DLinear on being the cheapest possible "does the DL machinery even earn its complexity" floor.
- **`TFT` / `Perceiver`** → two separate architectures, scored independently. Perceiver's mechanism is the same one scored **raw 11** in the [multi-timeframe fusion](#multi-timeframe-fusion) table — but that's a different, narrower role (fusing pooled per-tf reps, where `domain_fit = 2` because it's purpose-fit for exactly that join), not the same score copied over. Scored here as a whole-backend choice instead, `domain_fit` drops to 1 (a generalist backend, not purpose-built for this project's multi-tf shape the way it is for the fusion slot specifically) giving `raw 9`/adjusted 8 — same technique, same "same mechanism different role different tier" pattern the `GBM-on-flattened` note below spells out, still enough to clear Tier 1. TFT, despite higher field `dominance` as the more established forecasting-specific architecture, doesn't target a bottleneck this doc has flagged and lands Tier 2.
- **`classic univariate stats: ARIMA/SARIMA/ETS/GARCH`** → kept the same `raw 2`/Tier 3 score across all of them rather than inventing false differentiation: the rejection reason (single-series, univariate-by-design, doesn't extend to this project's multivariate multi-tf schema without reinvention) applies identically regardless of which classical model it is. ARIMA and SARIMA are merged into one row since SARIMA is ARIMA's own seasonal extension (`statsmodels.tsa.SARIMAX` is a strict superset call, not a separate mechanism) — the one classic-stats pair that actually does pass the tool-identity test's "same function, different argument" bar. ETS and GARCH get their own rows since they're separately-implemented model families (`ExponentialSmoothing` vs. `arch_model`); GARCH is also flagged separately since it's a volatility model, not a point-forecaster — its natural role here, if any, is a risk/position-sizing feature input, not a Stage-1 backbone.
- **`KAN-based blocks: TimeKAN` / `KANMixer`** → kept identical scores deliberately: both are 2025-era, unproven at this project's scale, and parked for the same reason: TimeKAN and KANMixer are separate architectures (frequency-decomposition backbone vs. TSMixer-style block, both using KAN spline edges), but neither has enough project-specific signal yet to differentiate beyond "both new, both parked."

Two calibration points worth flagging explicitly:

- **`naive/persistence baseline` scores 4/12 raw** — below the natural Tier-1 cutoff — **but is Tier 1 anyway**, via the [mandatory-floor exception](#mandatory-floor-exception): it's not competing on merit, it's the thing everything else has to beat.
- **`classic univariate stats` (raw 2 each) vs. `1-NN w/ DTW` (raw 5)** is the "score in context" rule in action: classic stats models are more field-dominant in the abstract (evidence/dominance would score higher on a generic finance-forecasting task) but score `domain_fit = 0` here because the doc is explicit they don't extend to this project's actual multivariate, multi-tf schema without being reinvented — a structural mismatch, not a budget/priority problem, hence Tier 3 rather than a deferred Tier 2. DTW-1NN, despite being a narrower/older technique, gets `domain_fit = 1` because the doc ties it to a fallback mechanism (DTW preprocessing) already live elsewhere in the design, and lands Tier 2 (gated on that fallback ever being built) rather than Tier 3.

#### activation mechanisms

| candidate                        | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier  |
| -------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ----- |
| GELU                             | 2        | 2         | 1         | 2            | 1          | 1           | 9   | 0    | 0       | 9        | **1** |
| GLU-family gating (GEGLU/SwiGLU) | 2        | 2         | 2         | 1            | 1          | 1           | 9   | 0    | 0       | 9        | **1** |
| ReLU                             | 2        | 2         | 0         | 2            | 1          | 1           | 8   | 0    | 0       | 8        | **1** |
| SiLU/Swish                       | 2        | 1         | 1         | 2            | 1          | 1           | 8   | 0    | 0       | 8        | **1** |
| Mish                             | 1        | 0         | 0         | 1            | 0          | 0           | 2   | 0    | 0       | 2        | **3** |

All Tier-1 rows here still fall under the doc's own scope rule: activation choice is "a cheap post-hoc refinement" tested _within_ whichever backend/profile wins the primary search, not folded into it — this table ranks priority _among activations_, not their priority against the architecture-level candidates above.

#### combination strategy

| candidate                                        | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier          |
| ------------------------------------------------ | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ------------- |
| single-backend-wins (current default)            | 2        | 2         | 0         | 2            | 2          | 2           | 10  | 0    | 0       | 10       | **1**         |
| single hybrid backend (= hybrid CNN→Transformer) | 1        | 1         | 1         | 2            | 1          | 1           | 7   | 0    | 0       | 7        | **2**         |
| knowledge distillation (multi-teacher → student) | 2        | 1         | 0         | 0            | 1          | 1           | 5   | 0    | 0       | 5        | **2** (gated) |
| MoE gating                                       | 1        | 1         | 2         | 1            | 1          | 1           | 7   | −1   | −1      | 5        | **2** (gated) |
| late ensembling of independent backbones         | 1        | 1         | 0         | 0            | 1          | 1           | 4   | 0    | 0       | 4        | **2** (gated) |
| EffiCANet-style conv+attn fusion block           | 1        | 0         | 1         | 1            | 1          | 0           | 4   | −1   | −1      | 2        | **3**         |
| differentiable/block-level NAS (DARTS-style)     | 1        | 0         | 1         | 0            | 1          | 0           | 3   | −1   | −1      | 1        | **3**         |

Every row except `single-backend-wins` is dependency-gated: the doc's own default assumption is single-backend-wins until one of these is _measured_ to beat it (see [combination strategy](model-architecture-planning.md#combination-strategy) → "status: unresolved"), so none of the alternatives can be Tier 1 yet regardless of score.

#### fusion mechanism

| candidate                          | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier          |
| ---------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ------------- |
| concatenation + MLP projection     | 2        | 2         | 0         | 2            | 1          | 1           | 8   | 0    | 0       | 8        | **2** (gated) |
| cross-attention fusion             | 2        | 1         | 1         | 0            | 2          | 1           | 7   | 0    | 0       | 7        | **2**         |
| gated fusion (GLU-style)           | 1        | 1         | 1         | 1            | 1          | 1           | 6   | 0    | 0       | 6        | **2**         |
| weighted sum / learned scalar gate | 1        | 1         | 0         | 2            | 1          | 1           | 6   | 0    | 0       | 6        | **2**         |

`concatenation + MLP` is the clearest illustration of the gate-as-ceiling rule in this whole doc: its raw/adjusted score (8) clears the Tier-1 bar outright, but the gate — fusion only matters once a combination strategy other than single-backend-wins is adopted — demotes it to Tier 2 anyway. Same gate applies to all rows here.

### multi-timeframe fusion

Only the still-open sub-choices from [multi-timeframe fusion](model-architecture-planning.md#multi-timeframe-fusion) are scored; the per-tf-encoders overall approach, ATR-relative scale-invariance, decision-anchor point, and completed-candles-only rule are already resolved there, not live candidates.

| candidate                                                        | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier  |
| ---------------------------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ----- |
| Perceiver-style latent-bottleneck cross-attn                     | 2        | 1         | 2         | 2            | 2          | 2           | 11  | 0    | −1      | 10       | **1** |
| long-window: attention/state-space over pooling (the "standard") | 1        | 2         | 1         | 1            | 2          | 1           | 8   | 0    | 0       | 8        | **1** |
| higher-tf-as-query cross-attn shape (default)                    | 1        | 1         | 0         | 2            | 2          | 1           | 7   | 0    | 0       | 7        | **2** |
| bidirectional cross-tf attention                                 | 1        | 1         | 1         | 1            | 2          | 1           | 7   | 0    | 0       | 7        | **2** |
| fixed recency weighting (cheap baseline)                         | 1        | 1         | 0         | 2            | 0          | 0           | 4   | 0    | 0       | 4        | **2** |
| explicit DTW preprocessing (fallback/diagnostic)                 | 1        | 1         | 0         | 0            | 1          | 0           | 3   | 0    | 0       | 3        | **3** |

Perceiver-style latent-bottleneck cross-attention outranks even the plain higher-tf-as-query default here — it's not just an alternative attention shape, it's the one that specifically targets the longest branches (15min/1H) where the doc already flags quadratic cost as worst, so `resource_fit` and `impact/cost` both max out. `long-window: attention/state-space over pooling` is not an `X/Y` grouping in the split-able sense — it names a multi-tf _strategy_ (don't downsample the long window, run a full backend over it) that's backend-agnostic; which backend (attention vs. state-space) is a separate, already-answered choice scored in its own [sequential encoding](#sequential-encoding)/[attention](#attention--dependency) tables above, not a second implementation hiding inside this row.

### auxiliary tabular models (GBM-family)

Screening/meta-labeling/class-imbalance-proxy _roles_ for GBMs (see [auxiliary tabular models](model-architecture-planning.md#auxiliary-tabular-models-gbm-family)) aren't scored — they're already-settled uses, not competing techniques. The library/model choices _within_ those roles are:

| candidate                                                       | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | raw | risk | tooling | adjusted | tier          |
| --------------------------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | --- | ---- | ------- | -------- | ------------- |
| LightGBM (native quantile/pinball objective)                    | 2        | 2         | 1         | 2            | 2          | 2           | 11  | 0    | 0       | 11       | **1**         |
| XGBoost (native quantile/pinball objective)                     | 2        | 2         | 0         | 2            | 2          | 2           | 10  | 0    | 0       | 10       | **1**         |
| CatBoost                                                        | 2        | 2         | 1         | 2            | 1          | 2           | 10  | 0    | 0       | 10       | **1**         |
| TabPFN v2/2.5                                                   | 2        | 1         | 2         | 2            | 1          | 2           | 10  | −1   | 0       | 9        | **1**         |
| NGBoost                                                         | 2        | 1         | 1         | 1            | 1          | 2           | 8   | 0    | 0       | 8        | **1**         |
| GBM-on-flattened as floor/diagnostic (not primary architecture) | 1        | 1         | 0         | 2            | 1          | 2           | 7   | 0    | 0       | 7        | **1** (floor) |
| TabICL                                                          | 1        | 0         | 2         | 1            | 1          | 1           | 6   | −1   | −1      | 4        | **2**         |
| quantile regression forests / multivariate extensions           | 1        | 0         | 1         | 1            | 1          | 1           | 5   | −1   | −1      | 3        | **3**         |
| retrieval-augmented in-context tabular learning (TabR-style)    | 0        | 0         | 2         | 1            | 1          | 0           | 4   | −1   | −1      | 2        | **3**         |
| hybrid GBM + TabPFN ensemble                                    | 1        | 0         | 1         | 0            | 1          | 0           | 3   | −1   | −1      | 1        | **3**         |
| FT-Transformer                                                  | 1        | 0         | 1         | 1            | 0          | 0           | 3   | −1   | −1      | 1        | **3**         |
| hybrid GBM + LLM ensemble                                       | 0        | 0         | 2         | 0            | 0          | 0           | 2   | −1   | −1      | 0        | **3**         |
| ResNet-tabular                                                  | 1        | 0         | 0         | 1            | 0          | 0           | 2   | −1   | −1      | 0        | **3**         |

Four groupings are worth spelling out explicitly, per the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row):

- **`LightGBM` / `XGBoost` (native pinball objective)** → scored as separate rows because LightGBM and XGBoost are separate library classes (`lightgbm.LGBMRegressor` vs. `xgboost.XGBRegressor`), not one function with a different argument. Both still land Tier 1: they're mature, interchangeable-in-practice GBM tools; LightGBM's native categorical-feature handling gives it a slight modernity edge rather than a real priority gap between them.
- **`TabPFN v2/2.5` / `TabICL`** → `TabPFN v2/2.5` stays one row (version numbers of the same `TabPFNClassifier`/`TabPFNRegressor` package, not separate tools — the "stays merged" side of the test). `TabICL` is a genuinely separate, less mature implementation and gets its own row, landing Tier 2 on weaker evidence and no drop-in library yet.
- **`hybrid GBM + TabPFN ensemble` / `hybrid GBM + LLM ensemble`** → scored as separate rows because "TabPFN" and "LLM" name different tools entirely, and they aren't equally speculative: GBM+TabPFN inherits TabPFN's own Tier-1 standalone score, while GBM+LLM scores at the very bottom of this table — an LLM has no natural structural fit to this numeric OHLCV/candle schema and its inference cost is well outside the single-GPU budget, worse than merely "unproven."
- **`FT-Transformer` / `ResNet-tabular`** → scored as separate rows; `TabR` isn't scored a third time here since it already has its own row above (`retrieval-augmented in-context tabular learning (TabR-style)`).

`GBM-on-flattened` appears twice in this doc under two different roles, and the roles score differently: as a **primary sequence-architecture candidate** it scores Tier 3 (raw 3, see [current Stage-1 candidate set](#current-stage-1-candidate-set) — rejected, discards sequence structure). As a **floor/diagnostic** measuring how much signal is sequence-dependent at all, the exact same technique is Tier 1 via the [mandatory-floor exception](#mandatory-floor-exception). Same mechanism, different role, different tier — the role a candidate is being evaluated _for_ is part of what's being scored, not just the technique in isolation.

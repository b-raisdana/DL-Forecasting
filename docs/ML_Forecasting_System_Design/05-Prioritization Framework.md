# Prioritization Framework

## Tiered candidates by layer

Section order and headings mirror [model-architecture-planning.md](03-Model & Architecture Engineering.md)'s own structure, so a given layer's tiering sits next to the equivalent stage there. Only _open_ alternatives are scored — items a doc section already marks as resolved (e.g. ATR-relative normalization as primary, decision-anchor point, higher-tf in-progress-candle handling) aren't live candidates and are skipped. Scores are illustrative starting points from the current doc text, not final measurements — recalibrate any row once real MI/backtest/profiling evidence exists, per this project's own "measured evidence only" discipline (see [error metric ≠ trading objective](04-Experimentation, Evaluation & Optimization.md#core-principle-error-metric--trading-objective)).

### Normalization strategy

| candidate                                       | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier  |
| ----------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ----- |
| hybrid: ATR-norm price + raw log-return channel | 1        | 1         | 1         | 2            | 2          | 2           | 0    | 0       | 9        | **1** |
| log-return norm (scale-free)                    | 1        | 2         | 0         | 2            | 1          | 1           | 0    | 0       | 7        | **2** |
| rolling z-score                                 | 1        | 2         | 0         | 2            | 1          | 1           | 0    | 0       | 7        | **2** |
| min-max per window                              | 0        | 1         | 0         | 2            | 0          | 0           | 0    | 0       | 3        | **3** |

The hybrid scheme wins on domain fit (position + velocity, tailored to this project) despite log-return/z-score being more textbook-dominant in general finance — a direct illustration of the "score in context" note above. `no normalization` and `min-max as primary` are already rejected in the source doc, not re-scored here.

### model architecture & selection

#### input / feature embedding

| candidate                       | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier          |
| ------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ------------- |
| linear/MLP projection (default) | 2        | 2         | 0         | 2            | 2          | 2           | 0    | 0       | 10       | **1**         |
| PatchTST-style patch embedding  | 2        | 1         | 2         | 2            | 1          | 2           | 0    | −1      | 9        | **1**         |
| per-tf learned tf-id embedding  | 1        | 1         | 1         | 2            | 2          | 1           | 0    | 0       | 8        | **2** (gated) |

`per-tf tf-id embedding` only matters if the flat/shared-encoder architecture branch is chosen over the per-tf-branch design that's the working assumption (see [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion)) — that dependency gate demotes an otherwise-qualifying score to Tier 2.

#### local feature extraction

| candidate                                                                                                                                                                                      | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ----- |
| TCN / ModernTCN (large-kernel/grouped-conv is a `Conv1D` param choice on the same block, not a separate tool — see [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row)) | 2        | 1         | 2         | 2            | 2          | 2           | 0    | 0       | 11       | **1** |
| plain/vanilla 1D conv (default)                                                                                                                                                                | 2        | 2         | 0         | 2            | 2          | 2           | 0    | 0       | 10       | **1** |
| InceptionTime (multi-kernel-size Inception modules)                                                                                                                                            | 2        | 2         | 1         | 1            | 1          | 2           | 0    | 0       | 9        | **1** |
| FCN (fully-convolutional, global-pooled stack)                                                                                                                                                 | 2        | 1         | 0         | 2            | 1          | 2           | 0    | 0       | 8        | **1** |
| conv stem → Transformer (downsampling lever)                                                                                                                                                   | 1        | 1         | 1         | 2            | 1          | 1           | 0    | 0       | 7        | **2** |
| ResNet (residual conv blocks, TSC baseline)                                                                                                                                                    | 2        | 1         | 0         | 2            | 1          | 1           | 0    | 0       | 7        | **2** |
| TimesNet-style 1D→2D reshape                                                                                                                                                                   | 1        | 0         | 2         | 1            | 1          | 0           | −1   | −1      | 3        | **3** |
| SCINet                                                                                                                                                                                         | 1        | 0         | 1         | 1            | 1          | 1           | −1   | −1      | 3        | **3** |

Matches the source doc's own framing of ModernTCN as "a direct, low-risk upgrade path," not a replacement that invalidates the floor.

`residual-CNN TSC baselines` (ResNet/FCN/InceptionTime) score as three separate rows (distinct topologies, not parameter variants — see [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row)). InceptionTime (best-benchmarked, costlier) and FCN (cheapest, still competitive) clear Tier 1; ResNet, superseded by InceptionTime in TSC benchmarks, lands Tier 2. TimesNet/SCINet land Tier 3: real modernity, but unconfirmed for this pipeline (risk modifier applies to both).

#### local feature extraction — placement

Which conv block to use is scored above; this table scores _where_ it runs (see [local feature extraction — placement](03-Model & Architecture Engineering.md#local-feature-extraction--placement)). `pre-sequential only` is already resolved elsewhere and isn't re-scored here.

| candidate                                     | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier  |
| --------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ----- |
| post-attention only (`local_extraction_post`) | 1        | 0         | 1         | 2            | 1          | 1           | 0    | 0       | 6        | **2** |
| both (sandwich, pre + post)                   | 0        | 0         | 1         | 1            | 1          | 0           | −1   | 0       | 2        | **3** |

`post-attention only`'s moderate `evidence`/`modernity` comes from an adjacent-domain precedent (Conformer-style conv-after-attention in ASR), not direct evidence here, hence `dominance = 0`; `resource_fit = 2` since it adds negligible compute on top of the already-scored conv block. Tier 2: cheap to test but not evidenced enough to jump the Tier-1 queue. `both (sandwich)` carries `risk = −1` for confounding two unproven levers in one trial (see [statistical validity of comparisons](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons)) — Tier 3 until either placement shows a standalone win.

#### sequential encoding

| candidate                     | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier          |
| ----------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ------------- |
| LSTM (sanity floor)           | 2        | 2         | 0         | 2            | 2          | 2           | 0    | 0       | 10       | **1** (floor) |
| Mamba (selective state-space) | 2        | 1         | 2         | 2            | 1          | 2           | −1   | −1      | 8        | **1**         |
| GRU (alt within LSTM floor)   | 1        | 1         | 0         | 2            | 2          | 1           | 0    | 0       | 7        | **2**         |
| ConvLSTM                      | 1        | 1         | 0         | 1            | 1          | 1           | 0    | 0       | 5        | **2**         |
| S4 (fixed/HiPPO state-space)  | 1        | 1         | 1         | 2            | 1          | 1           | −1   | −1      | 5        | **2**         |
| xLSTM                         | 1        | 0         | 2         | 1            | 1          | 1           | −1   | −1      | 4        | **2**         |
| Hyena (implicit long conv)    | 1        | 0         | 2         | 1            | 1          | 1           | −1   | −1      | 4        | **2**         |

GRU joins xLSTM/ConvLSTM as a Tier-2 alt tested _within_ the LSTM floor role, rather than sharing the floor's Tier-1 status (LSTM/GRU and Mamba/S4 both split per the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row)). S4, read as Mamba's superseded predecessor, lands Tier 2. ConvLSTM outranks xLSTM/Hyena on `adjusted` despite scoring lower on the six pull factors before modifiers: its mechanism is older but mature and low-risk, while xLSTM/Hyena's modernity edge gets clawed back by the risk/tooling modifiers — novelty alone doesn't win, it has to survive the unproven-mechanism discount. `Hyena (implicit long conv)` is a single candidate, not a grouped pair — "implicit long convolution" is just its mechanism name.

#### attention / dependency

| candidate                                       | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier  |
| ----------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ----- |
| GQA/MQA                                         | 2        | 2         | 1         | 2            | 1          | 2           | 0    | 0       | 10       | **1** |
| cross-tf attention over pooled reps             | 1        | 1         | 1         | 2            | 2          | 2           | 0    | 0       | 9        | **1** |
| Longformer-style sliding-window + global tokens | 2        | 1         | 0         | 2            | 1          | 2           | 0    | 0       | 8        | **1** |
| standard self-attention (full, no mitigation)   | 2        | 2         | 0         | 0            | 2          | 1           | 0    | 0       | 7        | **2** |
| iTransformer                                    | 2        | 1         | 2         | 1            | 2          | 1           | −1   | −1      | 7        | **2** |
| Informer (ProbSparse)                           | 2        | 1         | 1         | 2            | 1          | 2           | −1   | −1      | 7        | **2** |
| MLA                                             | 1        | 0         | 2         | 2            | 1          | 2           | −1   | −1      | 6        | **2** |
| NSA                                             | 1        | 0         | 2         | 2            | 2          | 1           | −1   | −1      | 6        | **2** |
| BigBird-style + random attention                | 1        | 1         | 1         | 1            | 1          | 1           | 0    | −1      | 5        | **2** |
| Differential Attention                          | 1        | 0         | 2         | 1            | 2          | 1           | −1   | −1      | 5        | **2** |
| Autoformer                                      | 2        | 1         | 1         | 1            | 1          | 1           | −1   | −1      | 5        | **2** |
| FEDformer                                       | 2        | 0         | 1         | 1            | 1          | 1           | −1   | −1      | 4        | **2** |
| Performer (FAVOR+ kernel attention)             | 0        | 1         | 1         | 2            | 0          | 0           | −1   | −1      | 2        | **3** |
| Linformer (low-rank seq-length projection)      | 0        | 1         | 0         | 2            | 0          | 0           | −1   | −1      | 1        | **3** |

`GQA/MQA` stays one row (MQA = GQA with `num_kv_groups=1`, see [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row)). `Longformer`/`BigBird-style + random attention` split: BigBird's extra block-sparse "random attention" component needs a real block-sparse kernel (no dense-mask shortcut), costing it a tooling point Longformer doesn't pay — Longformer clears Tier 1, BigBird lands Tier 2. `Performer`/`Linformer` split the same way (different approximation mechanisms — FAVOR+ kernels vs. fixed low-rank projection); both land Tier 3, matching the source doc's "bottom-of-priority fallback only" framing, with Performer edging ahead only on modernity. `standard self-attention` sits below GQA/MQA and the sliding-window fallback on the flagged VRAM cost (`resource_fit = 0`) — the mitigations exist specifically to outrank the mechanism they mitigate. FlashAttention/mixed-precision AMP aren't scored: always-on infrastructure under whichever mechanism wins, not competing candidates (see [hardware constraints](03-Model & Architecture Engineering.md#hardware-constraints)).

#### global representation

| candidate                                        | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier  |
| ------------------------------------------------ | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ----- |
| pooling (mean/max/attn-pool/last-token, default) | 2        | 2         | 0         | 2            | 2          | 2           | 0    | 0       | 10       | **1** |
| N-HiTS (hierarchical interpolation, multi-rate)  | 2        | 1         | 2         | 2            | 1          | 2           | 0    | 0       | 10       | **1** |
| N-BEATS (stacked residual basis blocks)          | 2        | 1         | 0         | 2            | 1          | 1           | 0    | 0       | 7        | **2** |

`pooling` stays one row: this project's `global_repr` stage implements mean/max/attention/last-token as branches of one `global_pool(x, kind=...)` function, not separate classes (same pattern as GQA/MQA above). `N-BEATS`/`N-HiTS` split (separate model classes — stacked residual MLP vs. hierarchical multi-rate interpolation); N-HiTS pulls ahead because the source doc singles it out for long-horizon efficiency against the flagged VRAM ceiling, N-BEATS lands Tier 2 without that angle.

Prediction heads (action / MAE-OM regression / confidence) aren't scored — they're required output slots defined by the label design in [training-data.md](02-Data, Label & Feature Engineering.md#model-output-targets), not competing techniques to tier.

#### current Stage-1 candidate set

| candidate                                             | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier          |
| ----------------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ------------- |
| TCN / ModernTCN                                       | 2        | 1         | 2         | 2            | 2          | 2           | 0    | 0       | 11       | **1**         |
| LSTM (sanity floor)                                   | 2        | 2         | 0         | 2            | 2          | 2           | 0    | 0       | 10       | **1** (floor) |
| CNN-LSTM(-attention)                                  | 2        | 1         | 0         | 2            | 2          | 2           | 0    | 0       | 9        | **1**         |
| Transformer w/ per-tf embed + cross-tf attn           | 2        | 2         | 1         | 1            | 2          | 1           | 0    | 0       | 9        | **1**         |
| TSMixer                                               | 2        | 1         | 1         | 2            | 1          | 2           | 0    | 0       | 9        | **1**         |
| DLinear                                               | 2        | 1         | 0         | 2            | 1          | 2           | 0    | 0       | 8        | **1**         |
| Mamba (selective state-space)                         | 2        | 1         | 2         | 2            | 1          | 2           | −1   | −1      | 8        | **1**         |
| Perceiver (latent-bottleneck cross-attn)              | 1        | 1         | 2         | 2            | 1          | 2           | 0    | −1      | 8        | **1**         |
| naive/persistence baseline                            | 0        | 2         | 0         | 2            | 0          | 0           | 0    | 0       | 4        | **1** (floor) |
| hybrid CNN→Transformer                                | 1        | 1         | 1         | 2            | 1          | 1           | 0    | 0       | 7        | **2**         |
| GRU (alt within LSTM floor)                           | 1        | 1         | 0         | 2            | 2          | 1           | 0    | 0       | 7        | **2**         |
| TFT (Temporal Fusion Transformer)                     | 2        | 2         | 0         | 1            | 1          | 1           | 0    | 0       | 7        | **2**         |
| S4 (fixed/HiPPO state-space)                          | 1        | 1         | 1         | 2            | 1          | 1           | −1   | −1      | 5        | **2**         |
| 1-NN w/ DTW distance                                  | 2        | 1         | 0         | 1            | 1          | 0           | 0    | 0       | 5        | **2** (gated) |
| 4 separate per-tf models + late ensemble (as primary) | 1        | 1         | 0         | 0            | 1          | 0           | 0    | 0       | 3        | **3**         |
| GBM on flattened features (as primary architecture)   | 0        | 1         | 0         | 2            | 0          | 0           | 0    | 0       | 3        | **3**         |
| pure MLP on flattened features                        | 0        | 0         | 0         | 2            | 0          | 0           | 0    | 0       | 2        | **3**         |
| ARIMA / SARIMA                                        | 0        | 1         | 0         | 1            | 0          | 0           | 0    | 0       | 2        | **3**         |
| exponential smoothing / ETS                           | 0        | 1         | 0         | 1            | 0          | 0           | 0    | 0       | 2        | **3**         |
| GARCH (volatility, not point-forecast)                | 0        | 1         | 0         | 1            | 0          | 0           | 0    | 0       | 2        | **3**         |
| TimeKAN (frequency-decomposition KAN backbone)        | 0        | 0         | 2         | 1            | 1          | 0           | −1   | −1      | 2        | **3**         |
| KANMixer (TSMixer-style block, KAN edges)             | 0        | 0         | 2         | 1            | 1          | 0           | −1   | −1      | 2        | **3**         |
| GNN over tf/symbol nodes                              | 0        | 0         | 1         | 1            | 1          | 0           | −1   | −1      | 1        | **3**         |

This table re-lists candidates already scored in the "sequential encoding," "global representation," and "attention/dependency" sections above in their Stage-1-set role, inheriting those scores rather than re-deriving them (LSTM/GRU and Mamba/S4 splits are covered in [sequential encoding](#sequential-encoding) above). A few further groupings are worth spelling out explicitly, per the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row):

- **`all-MLP mixer: TSMixer` / `DLinear`** → both clear Tier 1, but for different reasons: TSMixer on the doc's own cited ablation evidence, DLinear on being the cheapest possible "does the DL machinery even earn its complexity" floor.
- **`TFT` / `Perceiver`** → scored independently. Perceiver scores **adjusted 10** in the [multi-timeframe fusion](#multi-timeframe-fusion) table for a narrower role (`domain_fit = 2`, purpose-fit for fusing pooled per-tf reps); here, as a whole-backend choice, `domain_fit` drops to 1 (generalist, not purpose-built for this shape), giving `adjusted 8` — same technique, same mechanism-different-role-different-tier pattern as the `GBM-on-flattened` note below, still Tier 1. TFT, despite higher field `dominance`, doesn't target a flagged bottleneck and lands Tier 2.
- **`classic univariate stats: ARIMA/SARIMA/ETS/GARCH`** → same `adjusted 2`/Tier 3 for all: the rejection reason (univariate-by-design, doesn't extend to this project's multivariate multi-tf schema) applies identically. ARIMA/SARIMA merge into one row (`statsmodels.tsa.SARIMAX` is ARIMA's own seasonal extension, not a separate mechanism — the one pair that passes the tool-identity test's "same function, different argument" bar). ETS/GARCH keep separate rows (separate model families); GARCH is also a volatility model, not a point-forecaster — its natural role, if any, is a risk/position-sizing feature, not a Stage-1 backbone.
- **`KAN-based blocks: TimeKAN` / `KANMixer`** → identical scores deliberately: both 2025-era, unproven at this project's scale, separate architectures (frequency-decomposition backbone vs. TSMixer-style block, both KAN spline edges) with no project-specific signal yet to differentiate beyond "both new, both parked."

Two calibration points worth flagging explicitly:

- **`naive/persistence baseline` scores adjusted 4** — below the natural Tier-1 cutoff — **but is Tier 1 anyway**, via the [mandatory-floor exception](#mandatory-floor-exception): it's not competing on merit, it's the thing everything else has to beat.
- **`classic univariate stats` (adjusted 2 each) vs. `1-NN w/ DTW` (adjusted 5)** is the "score in context" rule in action: classic stats models are more field-dominant in the abstract (evidence/dominance would score higher on a generic finance-forecasting task) but score `domain_fit = 0` here because the doc is explicit they don't extend to this project's actual multivariate, multi-tf schema without being reinvented — a structural mismatch, not a budget/priority problem, hence Tier 3 rather than a deferred Tier 2. DTW-1NN, despite being a narrower/older technique, gets `domain_fit = 1` because the doc ties it to a fallback mechanism (DTW preprocessing) already live elsewhere in the design, and lands Tier 2 (gated on that fallback ever being built) rather than Tier 3.

#### activation mechanisms

| candidate                        | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier  |
| -------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ----- |
| GELU                             | 2        | 2         | 1         | 2            | 1          | 1           | 0    | 0       | 9        | **1** |
| GLU-family gating (GEGLU/SwiGLU) | 2        | 2         | 2         | 1            | 1          | 1           | 0    | 0       | 9        | **1** |
| ReLU                             | 2        | 2         | 0         | 2            | 1          | 1           | 0    | 0       | 8        | **1** |
| SiLU/Swish                       | 2        | 1         | 1         | 2            | 1          | 1           | 0    | 0       | 8        | **1** |
| Mish                             | 1        | 0         | 0         | 1            | 0          | 0           | 0    | 0       | 2        | **3** |

All Tier-1 rows here still fall under the doc's own scope rule: activation choice is "a cheap post-hoc refinement" tested _within_ whichever backend/profile wins the primary search, not folded into it — this table ranks priority _among activations_, not their priority against the architecture-level candidates above.

#### combination strategy

| candidate                                        | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier          |
| ------------------------------------------------ | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ------------- |
| single-backend-wins (default)                    | 2        | 2         | 0         | 2            | 2          | 2           | 0    | 0       | 10       | **1**         |
| single hybrid backend (= hybrid CNN→Transformer) | 1        | 1         | 1         | 2            | 1          | 1           | 0    | 0       | 7        | **2**         |
| knowledge distillation (multi-teacher → student) | 2        | 1         | 0         | 0            | 1          | 1           | 0    | 0       | 5        | **2** (gated) |
| MoE gating                                       | 1        | 1         | 2         | 1            | 1          | 1           | −1   | −1      | 5        | **2** (gated) |
| late ensembling of independent backbones         | 1        | 1         | 0         | 0            | 1          | 1           | 0    | 0       | 4        | **2** (gated) |
| EffiCANet-style conv+attn fusion block           | 1        | 0         | 1         | 1            | 1          | 0           | −1   | −1      | 2        | **3**         |
| differentiable/block-level NAS (DARTS-style)     | 1        | 0         | 1         | 0            | 1          | 0           | −1   | −1      | 1        | **3**         |

Every row except `single-backend-wins` is dependency-gated: the doc's own default assumption is single-backend-wins until one of these is _measured_ to beat it (see [combination strategy](03-Model & Architecture Engineering.md#combination-strategy) → "status: unresolved"), so none of the alternatives can be Tier 1 yet regardless of score.

#### fusion mechanism

| candidate                          | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier          |
| ---------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ------------- |
| concatenation + MLP projection     | 2        | 2         | 0         | 2            | 1          | 1           | 0    | 0       | 8        | **2** (gated) |
| cross-attention fusion             | 2        | 1         | 1         | 0            | 2          | 1           | 0    | 0       | 7        | **2**         |
| gated fusion (GLU-style)           | 1        | 1         | 1         | 1            | 1          | 1           | 0    | 0       | 6        | **2**         |
| weighted sum / learned scalar gate | 1        | 1         | 0         | 2            | 1          | 1           | 0    | 0       | 6        | **2**         |

`concatenation + MLP` is the clearest illustration of the gate-as-ceiling rule in this whole doc: its adjusted score (8) clears the Tier-1 bar outright, but the gate — fusion only matters once a combination strategy other than single-backend-wins is adopted — demotes it to Tier 2 anyway. Same gate applies to all rows here.

### multi-timeframe fusion

Only the still-open sub-choices from [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion) are scored; the per-tf-encoders overall approach, ATR-relative scale-invariance, decision-anchor point, and completed-candles-only rule are already resolved there, not live candidates.

| candidate                                                        | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier  |
| ---------------------------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ----- |
| Perceiver-style latent-bottleneck cross-attn                     | 2        | 1         | 2         | 2            | 2          | 2           | 0    | −1      | 10       | **1** |
| long-window: attention/state-space over pooling (the "standard") | 1        | 2         | 1         | 1            | 2          | 1           | 0    | 0       | 8        | **1** |
| higher-tf-as-query cross-attn shape (default)                    | 1        | 1         | 0         | 2            | 2          | 1           | 0    | 0       | 7        | **2** |
| bidirectional cross-tf attention                                 | 1        | 1         | 1         | 1            | 2          | 1           | 0    | 0       | 7        | **2** |
| fixed recency weighting (cheap baseline)                         | 1        | 1         | 0         | 2            | 0          | 0           | 0    | 0       | 4        | **2** |
| explicit DTW preprocessing (fallback/diagnostic)                 | 1        | 1         | 0         | 0            | 1          | 0           | 0    | 0       | 3        | **3** |

Perceiver-style latent-bottleneck cross-attention outranks even the plain higher-tf-as-query default here — it's not just an alternative attention shape, it's the one that specifically targets the longest branches (15min/1H) where the doc already flags quadratic cost as worst, so `resource_fit` and `impact/cost` both max out. `long-window: attention/state-space over pooling` is not an `X/Y` grouping in the split-able sense — it names a multi-tf _strategy_ (don't downsample the long window, run a full backend over it) that's backend-agnostic; which backend (attention vs. state-space) is a separate, already-answered choice scored in its own [sequential encoding](#sequential-encoding)/[attention](#attention--dependency) tables above, not a second implementation hiding inside this row.

### auxiliary tabular models (GBM-family)

Screening/meta-labeling/class-imbalance-proxy _roles_ for GBMs (see [auxiliary tabular models](03-Model & Architecture Engineering.md#auxiliary-tabular-models-gbm-family)) aren't scored — they're already-settled uses, not competing techniques. The library/model choices _within_ those roles are:

| candidate                                                       | evidence | dominance | modernity | resource_fit | domain_fit | impact/cost | risk | tooling | adjusted | tier          |
| --------------------------------------------------------------- | -------- | --------- | --------- | ------------ | ---------- | ----------- | ---- | ------- | -------- | ------------- |
| LightGBM (native quantile/pinball objective)                    | 2        | 2         | 1         | 2            | 2          | 2           | 0    | 0       | 11       | **1**         |
| XGBoost (native quantile/pinball objective)                     | 2        | 2         | 0         | 2            | 2          | 2           | 0    | 0       | 10       | **1**         |
| CatBoost                                                        | 2        | 2         | 1         | 2            | 1          | 2           | 0    | 0       | 10       | **1**         |
| TabPFN v2/2.5                                                   | 2        | 1         | 2         | 2            | 1          | 2           | −1   | 0       | 9        | **1**         |
| NGBoost                                                         | 2        | 1         | 1         | 1            | 1          | 2           | 0    | 0       | 8        | **1**         |
| GBM-on-flattened as floor/diagnostic (not primary architecture) | 1        | 1         | 0         | 2            | 1          | 2           | 0    | 0       | 7        | **1** (floor) |
| TabICL                                                          | 1        | 0         | 2         | 1            | 1          | 1           | −1   | −1      | 4        | **2**         |
| quantile regression forests / multivariate extensions           | 1        | 0         | 1         | 1            | 1          | 1           | −1   | −1      | 3        | **3**         |
| retrieval-augmented in-context tabular learning (TabR-style)    | 0        | 0         | 2         | 1            | 1          | 0           | −1   | −1      | 2        | **3**         |
| hybrid GBM + TabPFN ensemble                                    | 1        | 0         | 1         | 0            | 1          | 0           | −1   | −1      | 1        | **3**         |
| FT-Transformer                                                  | 1        | 0         | 1         | 1            | 0          | 0           | −1   | −1      | 1        | **3**         |
| hybrid GBM + LLM ensemble                                       | 0        | 0         | 2         | 0            | 0          | 0           | −1   | −1      | 0        | **3**         |
| ResNet-tabular                                                  | 1        | 0         | 0         | 1            | 0          | 0           | −1   | −1      | 0        | **3**         |

Four groupings are worth spelling out explicitly, per the [tool-identity test](#tool-identity-test-when-a-xy-grouping-stays-one-row):

- **`LightGBM` / `XGBoost` (native pinball objective)** → separate library classes (`lightgbm.LGBMRegressor` vs. `xgboost.XGBRegressor`), split into separate rows. Both land Tier 1 as mature, interchangeable-in-practice GBM tools; LightGBM's native categorical-feature handling gives it a slight modernity edge, not a real priority gap.
- **`TabPFN v2/2.5` / `TabICL`** → `TabPFN v2/2.5` stays one row (version numbers of the same `TabPFNClassifier`/`TabPFNRegressor` package). `TabICL`, a genuinely separate and less mature implementation, gets its own row and lands Tier 2 on weaker evidence and no drop-in library yet.
- **`hybrid GBM + TabPFN ensemble` / `hybrid GBM + LLM ensemble`** → split (different tools, unequally speculative): GBM+TabPFN inherits TabPFN's Tier-1 standalone score, while GBM+LLM scores at the bottom — an LLM has no structural fit to this numeric OHLCV/candle schema and its inference cost is outside the single-GPU budget.
- **`FT-Transformer` / `ResNet-tabular`** → split; `TabR` isn't scored a third time since it already has its own row above (`retrieval-augmented in-context tabular learning (TabR-style)`).

`GBM-on-flattened` appears twice in this doc under two different roles, and the roles score differently: as a **primary sequence-architecture candidate** it scores Tier 3 (adjusted 3, see [current Stage-1 candidate set](#current-stage-1-candidate-set) — rejected, discards sequence structure). As a **floor/diagnostic** measuring how much signal is sequence-dependent at all, the exact same technique is Tier 1 via the [mandatory-floor exception](#mandatory-floor-exception). Same mechanism, different role, different tier — the role a candidate is being evaluated _for_ is part of what's being scored, not just the technique in isolation.

### Time-Series Foundation Models (TSFMs)

Excluded from [AI Trading System — Planning Notes](03-Model & Architecture Engineering.md#model-architecture--selection) — that doc covers custom/from-scratch architecture design only; this file covers the pretrained-checkpoint alternative track.

Chronos, TimesFM, Moirai, Lag-Llama, PatchTST-based pretrained checkpoints

These are a different category entirely: not language models repurposed, but transformer/patching architectures pretrained from scratch on large corpora of numeric time series across many domains, then fine-tuned or used zero-shot on a new series. This is architecturally much closer to what's already in your candidate pool (decoder-only transformer over patched sequences) than to LLM-reprogramming — the "foundation model" part is about the pretraining corpus size/diversity, not about language.

full pretraining from scratch is out of scope (that's what makes them "foundation" models — large corpora, large compute, not a single-GPU exercise). But downloading a pretrained checkpoint and fine-tuning locally on your BTC/USDT + cross-pair data is plausible within your 8GB budget for the smaller Chronos/TimesFM variants, and zero-shot inference (no training at all) is cheap enough to run as a baseline comparison point. Two open questions your doc's own methodology already answers how to handle:

- (a) these models are pretrained mostly on non-financial series (retail demand, weather, web traffic, etc.) — whether that pretraining transfers to crypto price dynamics is exactly the kind of thing your MI/GBM screening and backtested-KPI discipline should settle, not assumption;
- (b) they typically output point/quantile forecasts of the raw series, not your TP/SL/action/MAE-OM label structure (see [training-data.md § model output targets](02-Data, Label & Feature Engineering.md#model-output-targets)) — you'd need a task-specific head on top regardless.

## Feasibility & Trade-offs

**Open gap — feasibility frontier** ([99-Weakness Analysis.md § A18](99-Weakness%20Analysis.md), Imp 3): no formal Pareto-style framework yet for jointly trading off profitability/drawdown/compute/latency/complexity, or for deciding when a small performance gain isn't worth added cost. The individual ingredients already exist: prediction quality/profitability/risk from [backtested trading KPIs](04-Experimentation, Evaluation & Optimization.md#backtested-trading-kpis-final-selection); training time/GPU/memory from [hardware constraints](03-Model & Architecture Engineering.md#hardware-constraints) and the `profile_trial_cost()` profiler; complexity from the [simplification rule](03-Model & Architecture Engineering.md#simplification-rule). Inference latency is out of scope — this project doesn't cover production/live serving (see [99-Exclusion.md](99-Exclusion.md)). What's missing is the joint methodology combining these, not the individual metrics.

## Decision Framework

### why

The design-space lists across this project's planning docs (embedding, local extraction, sequential encoding, attention, fusion, GBM-family alternatives, normalization schemes, etc.) are long and growing faster than the single-GPU budget can test everything. Those docs already make tiering judgments informally in prose — e.g. ModernTCN "worth promoting straight into Stage-1 profiling" vs. KAN-based blocks "parked... pending evidence" (see [current Stage-1 candidate set](#current-stage-1-candidate-set)). This doc makes that judgment repeatable: a fixed factor list and a formula, so a new candidate gets the same treatment a prior one got, instead of re-litigating "how excited should we be about this" per bullet.

### tiers

- **Tier 1 — fund now:** active Stage-1 candidate, or a near-term addition to whatever categorical search / screening step is relevant.
- **Tier 2 — secondary:** test post-primary-search, as a refinement/alt within an already-covered role, or once finalist/budget headroom allows.
- **Tier 3 — parked:** logged for awareness only; revisit only on new evidence (a cheaper-proxy result, a new benchmark, an unblocked dependency) — not on a schedule.

### scoring factors

Score each 0–2: 0 = absent/weak, 1 = moderate, 2 = strong. **Score in context, not in the abstract** — "evidence" and "dominance" mean _evidence/dominance that this candidate wins on this project's actual multivariate, multi-tf, non-stationary setup_, not general field fame. A technique that's a textbook standard elsewhere but structurally doesn't fit here (classic univariate ARIMA/GARCH against a genuinely multivariate schema, for instance) scores low on those factors despite the broad reputation — see the [current Stage-1 candidate set](#current-stage-1-candidate-set) table for that exact case scored against a superficially similar but better-fitting alternative.

**pull factors (raise tier):**

1. **research/reference-document support** — peer-reviewed papers or published benchmarks (the kind already cited inline across this project's docs, e.g. TSMixer's own ablations, TabArena 2025) showing the candidate beats relevant alternatives _on a comparable problem_ — not just "it's been used somewhere."
2. **field dominance / standard-of-practice** — is it the current default choice in the relevant application area (time-series forecasting, NLP-derived sequence modeling, tabular ML), independent of project-specific evidence? De-facto standards carry lower integration risk.
3. **modernity / replacement trajectory** — is it a newer technique explicitly designed to supersede something already in the candidate set (ModernTCN→TCN, MLA→GQA)? Adopting it should retire complexity, not just add a parallel option.
4. **resource fit (hardware/compute budget)** — narrowly: does it run within the single-GPU VRAM/RAM/wall-clock ceiling that governs every candidate (see [hardware constraints](03-Model & Architecture Engineering.md#hardware-constraints))? This is a feasibility check, not a value judgment — it doesn't ask whether the technique is _worth_ its cost, only whether the cost is affordable at all.
5. **domain/problem fit** — two things bundled deliberately: (a) _structural_ fit — does the technique's required input shape (multivariate channels, multi-tf branches, per-tf window-length sequences — placeholder default 256/tf, now itself a search dimension, see [multi-timeframe fusion](03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length" — "needs a graph," "assumes univariate") match what this project actually has, and (b) _characteristic_ fit — does its benefit target something this data actually exhibits (non-stationary, noisy OHLCV, genuinely multivariate schema) rather than a generic capability bump never tested against this kind of noise. A technique that structurally can't consume this input at all (e.g. a graph-structured model with no graph in the data) scores 0 here regardless of how well-regarded it is elsewhere.
6. **marginal impact vs. cost** — a _value_ judgment layered on top of the two feasibility checks above: given it fits (resource_fit) and applies (domain_fit), does adopting it directly close a gap already flagged elsewhere (O(n²)/VRAM ceiling, confidence-metric gap, class imbalance) cheaply, or is it a large lift for a small/speculative gain? Two candidates can both score well on resource_fit/domain_fit and still diverge sharply here — a technique can be cheap and applicable but bring only a marginal, already-covered benefit (e.g. Mish activation), or bring the exact fix to a bottleneck this doc already names (e.g. Perceiver-style latent-bottleneck attention against the flagged multi-tf VRAM cost).

**modifiers (lower tier, never raise it):**

1. **risk / reversibility** — an incremental, low-risk swap into an already-covered role (e.g. xLSTM within the LSTM floor) vs. a structurally new, unproven mechanism that needs its own validation cycle.
2. **tooling/library maturity** — drop-in within libraries already in scope (LightGBM/XGBoost/Optuna/Keras) vs. a custom implementation from a paper.
3. **dependency / gating status** — hard gate, not a score: blocked behind another unresolved decision (e.g. the fusion-mechanism choice only matters once "combination strategy" ≠ single-backend-wins, see [combination strategy](03-Model & Architecture Engineering.md#combination-strategy)) demotes the tier, per the rule below, until the blocker resolves.

### tool-identity test: when a X/Y grouping stays one row

Several candidates below are named `X/Y` (LSTM/GRU, TCN/ModernTCN, GQA/MQA, ...). That notation means two different things depending on the pair: a merged row hides whichever of the two items is actually weaker, and a reader can't tell "these are drop-in-swappable" from "these are lumped for convenience" just by looking at the slash.

Apply one mechanical test: **would implementing both, in the library this project actually uses (Keras/TF for architecture blocks, LightGBM/XGBoost/statsmodels/etc for GBM-family), mean instantiating two different classes, or calling the same class/function with a different argument?**

- **Different classes/tools → split into separate rows**, each scored independently against its neighbors. Example: `tf.keras.layers.LSTM` vs `tf.keras.layers.GRU` are two distinct layer classes with different gating internals — LSTM/GRU splits.
- **Same class/function, different argument → stays one row.** Example: TCN vs ModernTCN are both a stacked `Conv1D` block; ModernTCN is that same block called with a larger kernel size and grouped/depthwise convolution — a parameter choice, not a different tool. GQA vs MQA are the same multi-query-attention mechanism parameterized by KV-group count (MQA = GQA with `num_kv_groups=1`) — also stays merged.

This test is orthogonal to how similar the mechanisms _feel_: GRU is arguably more similar to LSTM internally than ModernTCN is to plain TCN, but GRU still gets its own row because it's a separate class you'd import and swap in, while ModernTCN is a config change on the block already in the row above it. When in doubt, ask whether a hyperparameter-search tool (Optuna categorical dimension) would need a new class/import to add the second option, or just a new value for an existing parameter — the former splits, the latter doesn't.

### combination formula

```text
adjusted = (evidence + dominance + modernity + resource_fit + domain_fit + impact_vs_cost)   (0–12)
           − (1 if speculative/unproven mechanism else 0)     [risk modifier]
           − (1 if no mature drop-in library/impl else 0)     [tooling modifier]
```

- **Tier 1** if `adjusted ≥ 8` and not gated
- **Tier 2** if `4 ≤ adjusted < 8`, **or** `adjusted ≥ 8` but gated — the gate is a _ceiling_, not a floor: it demotes an otherwise-Tier-1 score, it never promotes a low score
- **Tier 3** if `adjusted < 4`, regardless of gating

Recalibrate the 8/4 cutoffs, not the mandatory-floor exception below, if the formula ever disagrees with a clear case.

### mandatory-floor exception

Treat a "mandatory floor" (naive/persistence baseline, GBM-on-flattened floor — see [current Stage-1 candidate set](#current-stage-1-candidate-set)) as always-Tier-1, orthogonal to this formula. Those score low on most pull factors (nothing novel or dominant about a persistence baseline) but aren't optional: they exist to prove a learned candidate beats doing nothing, not to compete on the same axes as everything else here. Both worked examples below (naive baseline, GBM-on-flattened-as-floor) score under the natural Tier-1 cutoff on the formula alone — the exception is what actually promotes them, not the arithmetic.

### how to apply

When adding a new candidate anywhere in the planning docs (a new architecture block, a new GBM variant, a new normalization scheme):

1. Score the six pull factors and three modifiers per the definitions above, **against the other candidates already scored in the same section** below — not in isolation. A technique's "dominance" or "modernity" score should reflect its rank relative to its actual peers in that layer, not a global absolute; see the [scoring factors](#scoring-factors) note on scoring in context.
2. Run the combination formula.
3. Check the dependency/gating modifier — a hard gate demotes an otherwise-Tier-1 score to Tier 2; it never raises a Tier-3 score.
4. Add the candidate's row to the relevant table below (or create a new section if it's a new layer/topic), so the tiering judgment stays visible and comparable to its neighbors instead of being re-derived later.

## Iterative Optimization

**Open gap — error-driven optimization loop** ([99-Weakness Analysis.md § A16](99-Weakness%20Analysis.md), Imp 2): "establish baseline → identify bottleneck → generate alternatives → experiment → measure → select improvement → repeat" is the project's implicit workflow, but there's no standardized error taxonomy (wrong direction / wrong magnitude / wrong confidence / bad TP-SL estimate / regime-specific failure / missed opportunity / excessive false signals) mapping each failure type to a targeted intervention, and no formal loop enforcing "measure failure → identify likely cause → generate targeted alternatives → test → retain/reject" over ad hoc experimentation. The building blocks exist piecemeal — [model-selection pipeline](04-Experimentation, Evaluation & Optimization.md#model-selection-pipeline) for the baseline/experiment/measure mechanics, [architecture failure diagnosis](03-Model & Architecture Engineering.md#architecture-failure-diagnosis) for one failure-diagnosis case — but not assembled into the general loop this section names.

# AI Trading System — Planning Notes

Goal: best AI method to predict price moves → optimal trade positions, max profit / min risk.

- [AI Trading System — Planning Notes](#ai-trading-system--planning-notes)
  - [key questions](#key-questions)
    - [data feed design](#data-feed-design)
    - [normalization strategy](#normalization-strategy)
    - [model architecture \& selection](#model-architecture--selection)
      - [top level architecture](#top-level-architecture)
        - [prioritization framework — tiering candidate techniques](#prioritization-framework--tiering-candidate-techniques)
        - [input / feature embedding](#input--feature-embedding)
        - [local feature extraction](#local-feature-extraction)
        - [sequential encoding](#sequential-encoding)
        - [attention / dependency](#attention--dependency)
        - [multi-tf fusion](#multi-tf-fusion)
        - [global representation](#global-representation)
        - [prediction heads](#prediction-heads)
        - [current Stage-1 candidate set](#current-stage-1-candidate-set)
      - [activation mechanisms — testing strategy](#activation-mechanisms--testing-strategy)
      - [combination strategy](#combination-strategy)
      - [fusion mechanism](#fusion-mechanism)
  - [multi-timeframe fusion](#multi-timeframe-fusion)
  - [validation \& train/test splitting](#validation--traintest-splitting)
  - [optimization strategy](#optimization-strategy)
  - [class imbalance handling](#class-imbalance-handling)
  - [auxiliary tabular models (GBM-family)](#auxiliary-tabular-models-gbm-family)
    - [modern GBM-family alternatives](#modern-gbm-family-alternatives)
    - [uncertainty-native GBM variants — confidence-metric gap](#uncertainty-native-gbm-variants--confidence-metric-gap)
  - [experiment tracking (current priority)](#experiment-tracking-current-priority)
  - [companion docs (broken out for size)](#companion-docs-broken-out-for-size)
  - [excluded topics (broken out into separate files)](#excluded-topics-broken-out-into-separate-files)
  - [deferred topics (not current concerns, placeholders)](#deferred-topics-not-current-concerns-placeholders)
  - [glossary](#glossary)

## key questions

### data feed design

Candle-level feature schema, the feature-set completeness-testing workflow (ablation, MI/GBM candidate screening, candidate feature pool), and screening methodology are broken out to [input features & embedding](input-features.md) for size — see [companion docs](#companion-docs-broken-out-for-size).

### normalization strategy

- for all price based inputs use rolling /ATR scheme.
- **alternative schemes to test:**
  - log-return norm (scale-free)
  - rolling z-score
  - min-max per window (cheap, likely worse — loses cross-window vol comparability)
  - hybrid — ATR-norm price + separate raw log-return channel (position + velocity)
    Alt:
  - no normalization — rejected, non-stationary
  - min-max as primary — rejected, loses vol-regime comparability; kept only as test candidate
- **testing protocol** for a normalization change: same discipline as the [seed-count workflow](error-rating-and-evaluation.md#statistical-validity-of-comparisons) — ≥3 seeds/scheme, compare backtested-KPI distributions (not train loss), same train/validate split scheme (see "validation & train/test splitting").
  Alt: single-run train-loss comparison (rejected — exactly the noise risk flagged).

### model architecture & selection

#### top level architecture

Two views of the same design space live here: a **pipeline-stage view** (what building block fills each stage of the network) and the **current Stage-1 candidate set** (the concrete whole-architecture options actually fed into the Optuna categorical search today, per "optimization strategy" below / [cross-architecture fairness](model-architecture-candidate-sets.md#cross-architecture-fairness)). The stage view is the fuller design space — useful for reasoning about new techniques and for a possible future block-level NAS (see "combination strategy" → differentiable/block-level NAS) — but Stage-1 currently picks one whole architecture from the candidate set, not an arbitrary per-stage mix, until/unless that NAS alternative is adopted.

##### prioritization framework — tiering candidate techniques

Full scoring factors, tiers, and combination formula for sorting candidates below into fund-now / test-later / parked broken out to [prioritization framework](prioritization-framework.md) — see [companion docs](#companion-docs-broken-out-for-size). Use it when adding a new candidate anywhere in this doc, so tiering stays consistent rather than vibes-based per bullet — it's the general form of the judgment already made informally throughout the bullets below (e.g. ModernTCN "worth promoting straight into Stage-1 profiling" vs. KAN blocks "parked... pending evidence").

##### input / feature embedding

Full detail broken out to [input features & embedding](input-features.md#input--feature-embedding) — see [companion docs](#companion-docs-broken-out-for-size). Embedding options in play:

- linear/MLP projection of the per-candle feature vector → `d_model` (current default, shared first step across all Stage-1 candidates)
- per-tf learned tf-id embedding (flat/shared-encoder archs only; implicit via branch identity otherwise)
- PatchTST-style patch embedding (cheap lever on the attention VRAM cost flagged under [hardware constraints](model-architecture-candidate-sets.md#hardware-constraints))

##### local feature extraction

- **plain/vanilla 1D conv** (non-causal, non-dilated, same-padding stacked `Conv1D`) — the simplest possible learned local-extraction baseline, and worth naming explicitly since it's easy to jump straight to TCN/dilation and never test whether that added complexity earns its keep. This is also what the existing (pre-planning-doc) code already implements — `cnn_lstm_block()` in [cnn_lstm_attention_model.py](../app/ai_modelling/cnn_lstm_attention/cnn_lstm_attention_model.py) and [cnn_lstm_model.py](../app/ai_modelling/cnn_lstm/cnn_lstm_model.py) both stack same-padding `Conv1D` layers ahead of LSTM — see "current Stage-1 candidate set" below for that architecture as a whole.
- dilated causal conv (TCN) — multi-scale local pattern extraction, cheap.
- **residual-CNN time-series-classification baselines** — three separately-implemented network topologies from the UCR/UEA time-series-classification benchmark literature, distinct lineage from the NLP-derived TCN/ModernTCN line above; different classes in any TSC library (e.g. `aeon`/`tsai`), not one function with a parameter choice — see [prioritization framework](prioritization-framework.md#local-feature-extraction) for why each is scored independently rather than as one candidate:
  - **ResNet** — stacked residual conv blocks. The oldest and weakest of the three here (InceptionTime generally supersedes it in TSC benchmarks).
  - **FCN** — plain fully-convolutional, global-average-pooled stack, no residual connections. Simplest and cheapest of the three, still competitive.
  - **InceptionTime** — multi-kernel-size Inception-style modules in parallel per block. Newest and best-benchmarked of the three, at higher per-module cost.

  All three: cheap, non-causal, no dilation-schedule tuning needed — worth testing as an independent floor for the plain-CNN/TCN/ModernTCN family, not assumed inferior just because the lineage is older.
- **ModernTCN** — large-kernel, grouped/depthwise convolution in place of TCN's small-kernel stacked-dilation approach, explicitly designed to capture cross-time *and* cross-variable dependency in one pass. A direct, low-risk upgrade to the plain TCN block above — same role in the pipeline, same cost class, better-tested internals — worth promoting straight into Stage-1 profiling rather than parking as a deferred alternative. Particularly relevant here since the feature schema is genuinely multivariate (relative-HLC, volume/ATR, gap, multiple top-distance channels per candle — see [candle feature schema](input-features.md#candle-feature-schema)); plain TCN's per-channel/shared-filter convolution treats cross-variable interaction only incidentally, ModernTCN's grouped-conv design treats it deliberately.
- conv stem ahead of a Transformer (the hybrid CNN→Transformer candidate) — same cost-reduction goal as patching above, via learned downsampling instead of fixed patches.
- **TimesNet-style 1D→2D reshape** — reshapes the series into a 2D grid keyed by detected/candidate periodicities, then applies 2D (inception-style) conv blocks to capture multi-periodicity directly. Relevant if session/cyclical features (hour/day sin-cos, session-open flags — see [candidate feature pool](input-features.md#candidate-feature-pool)) are meaningfully periodic; untested assumption, not yet screened.
- **SCINet** — recursive downsample→convolve→interact structure that extracts multi-resolution features hierarchically from a single input series. Conceptually close to what this doc's "multi-timeframe fusion" section already does by hand across resampled branches (5min/15min/1H/4H/1D/1W); SCINet does something structurally similar *within* one branch, so it reads as a plausible swap for the conv stage inside each per-tf encoder rather than a whole-pipeline replacement.
  Alt: no local-extraction stage, feed embedded scalars straight to sequential/attention stage — cheaper, ablation baseline.

##### sequential encoding

- LSTM recurrence — sanity-check floor, sequential context without attention's O(n²) cost. This is what the existing code already runs (`cnn_lstm_model.py`), so it's the floor, not just an option.
  - **GRU** — `tf.keras.layers.GRU`, a separate layer class from `LSTM` (merged forget/input gate, no separate cell state, ~25% fewer parameters), not a parameter of the same layer — see [prioritization framework](prioritization-framework.md#tool-identity-test-when-a-xy-grouping-stays-one-row) for why this and xLSTM/ConvLSTM below get the same "alt within the floor role" treatment despite one being a different class and the others being config choices. Tested as an alt within this same floor role, not a new pipeline stage.
  - **xLSTM** — modernized LSTM (exponential gating, matrix memory), recently used as the backbone of at least one zero-shot time-series foundation model. Not a mechanism swap in the strict sense (still recurrent) — the relevant question is "is a modernized recurrent block worth it," tested as an alt within this same floor role rather than a new pipeline stage.
  - **ConvLSTM** — convolutional gates inside the recurrent cell itself, a genuinely different mechanism from stacking a separate conv stage ahead of a plain LSTM (which is what the existing CNN-LSTM(-attention) code already does — see "current Stage-1 candidate set" below). Worth testing as an alt within this floor role, not a separate pipeline stage.
- state-space — linear-time long-context alternative to attention. Two separately-implemented mechanisms, not one function with a parameter (see [prioritization framework](prioritization-framework.md#sequential-encoding)):
  - **Mamba** — input-selective scan (`mamba_ssm`), the current standard-bearer for this family; directly targets the flagged O(n²)/VRAM ceiling.
  - **S4** — fixed, HiPPO-initialized state matrices, computed via convolution; largely superseded by Mamba in the literature, kept as a lower-priority alt rather than a co-equal option.
  - **Hyena (implicit long convolution)** — parameterizes an implicit convolution whose effective kernel spans the whole sequence, aiming for attention-like long-range coverage at sub-quadratic cost. Sits conceptually between the TCN dilation line below and this SSM line rather than beside either — treat as a variant to test within the SSM branch, not a separate architecture line.
- TCN dilation stack — same block as "Local Feature Extraction" above; listed here too because dilation depth is what gives it long-range context, not just a local receptive field.
  Alt: no dedicated sequential-encoding stage, rely entirely on attention for all-range dependency — the pure-Transformer candidate already covers this.

##### attention / dependency

- standard self-attention over the concatenated multi-tf sequence — most expressive, most expensive; the O(n²) cost [hardware constraints](model-architecture-candidate-sets.md#hardware-constraints) already flags as VRAM-dominant.
- cross-tf attention over per-tf pooled representations (not raw concat seq) — cheaper, see "multi-tf fusion" stage below.
- **Informer (ProbSparse attention)** — sparsifies attention to sub-O(n²) cost, a direct mitigation for the flagged VRAM bottleneck.
- **Autoformer (series decomposition + autocorrelation)** — replaces dot-product attention with trend/seasonal decomposition and autocorrelation-based dependency discovery.
- **FEDformer (frequency-domain attention)** — attends in the frequency domain; another angle on the same O(n²) mitigation as Informer/Autoformer.
- **iTransformer (inverted attention)** — attends across variates (feature channels) instead of across time. Interesting given the candle schema's many engineered feature channels per step — worth testing as a channel-dependency finder rather than a time-dependency finder.
- **GQA (Grouped-Query Attention) / MQA (Multi-Query Attention)** — multiple query heads share fewer key/value projections, cutting KV memory footprint directly. Standard, well-tested, cheap to add as a categorical option on any Transformer/hybrid/TFT branch above — targets the same O(n²)/VRAM cost this doc's hardware notes already flag, but the memory dimension of it specifically rather than the compute dimension Informer/Autoformer/FEDformer target.
- **MLA (Multi-head Latent Attention)** — newer than GQA (introduced with DeepSeek-V2): compresses keys/values into a low-rank latent space before caching, going further than GQA on memory reduction while reportedly preserving more representational power than sharing heads outright. Worth testing head-to-head against GQA specifically, since the flagged bottleneck is activations/KV-cache, not parameter count — a more targeted comparison than either vs. plain attention.
- **Native Sparse Attention (NSA)** — combines compressed/coarse attention, selective fine-grained attention, and sliding-window attention as parallel trainable branches, rather than a fixed post-hoc sparsity pattern. Structurally close to this doc's own multi-tf design (coarse/fine/local view ≈ higher-tf/lower-tf/local-window reasoning, see "multi-timeframe fusion" below) — a stronger, more current candidate than older sparse schemes for the same O(n²) mitigation Informer/Autoformer/FEDformer already target.
- **Longformer-style sliding-window + global tokens** — simpler, older, well-understood: local attention window plus a small number of globally-attending tokens. Cheaper fallback if NSA proves too complex to get working within the single-GPU budget.
  - **BigBird-style + random attention** — adds a third, block-sparse "random attention" component on top of Longformer's window+global pattern. A separate reference implementation (`BigBirdModel`), not a config flag on Longformer's — see [prioritization framework](prioritization-framework.md#attention--dependency) for why it's scored as its own row (a real block-sparse kernel needed to be efficient, not a dense-mask trick) and lands one tier below plain Longformer rather than sharing its score.
- **linear/kernel attention** — approximates the attention matrix for linear-in-sequence-length cost; generally the weakest of the sub-quadratic options on modeling quality in published comparisons, but cheapest to implement — bottom-of-priority fallback only. Two separately-implemented approximation mechanisms, not variants of one function:
  - **Performer** — FAVOR+ random-feature kernel approximation of softmax attention.
  - **Linformer** — fixed low-rank projection of the sequence-length dimension.
- **Differential Attention** — computes two separate softmax attention maps and subtracts one, cancelling common-mode attention noise and sharpening focus on the signal that differs between them. Pitched specifically as a signal-to-noise improvement, directly relevant given OHLCV/candlestick signal is inherently noisy — a genuinely case-specific quality candidate, not a copy-the-LLM-trend pick.
  Alt:
  - no attention stage, conv/recurrence/state-space only — cheapest, ablation baseline given attention's flagged cost.
  - FlashAttention — not a competing mechanism to choose among above; an exact-attention implementation that fuses the softmax computation to avoid materializing the full attention matrix. Same category as mixed-precision AMP under [hardware constraints](model-architecture-candidate-sets.md#hardware-constraints): a near-free win, on by default under whichever variant above is chosen, not a separate arm of this search.

##### multi-tf fusion

Full strategy list lives in the standalone "multi-timeframe fusion" section below (per-tf encoders + fusion block vs. flat full-attention vs. hierarchical/wavelet vs. late-ensemble) — this stage is where that resolved/candidate choice plugs into the overall pipeline. The concat/weighted-sum/cross-attention/gated menu under "fusion mechanism" below applies here too, not only to backend-type combination.

##### global representation

- pooling over the final sequence/fused representation → fixed-size vector (mean / max / attention-pooling / last-token).
- **N-BEATS / N-HiTS** — two separate model classes (e.g. `neuralforecast`'s `NBEATS`/`NHITS`), not a parameter choice on one block; an alternative path through this stage that skips the sequential-encoding and attention stages entirely, producing hierarchical basis-function forecasts directly from pure feedforward stacked residual blocks. **N-HiTS** specifically adds hierarchical interpolation/multi-rate sampling targeting long-horizon efficiency and is cheap on VRAM — worth a Stage-1-adjacent candidate slot given the hardware ceiling. **N-BEATS**, without that specific efficiency angle, is a secondary refinement rather than a co-equal candidate — see [prioritization framework](prioritization-framework.md#global-representation) for the scoring.
  Alt: no explicit global-representation stage, feed the full sequence directly to per-position heads — viable for head designs needing full sequence context; deferred, adds head complexity.

##### prediction heads

- action head (Long/Short/None), MAE/OM regression (auxiliary MFE) or quantile heads, confidence head — targets defined in [training-data.md](training-data.md#model-output-targets).
- loss/metric choice per head — see [error rating & model evaluation](error-rating-and-evaluation.md#per-head-statistical-metrics-dev-diagnostics); not duplicated here.

##### current Stage-1 candidate set

Full hyperparameter profiles (S1/S2/S3 = depth-heavy/width-heavy/context-heavy per architecture), search-space-bounds methodology, and the cross-architecture-fairness protocol live in [Stage-1 candidate sets](model-architecture-candidate-sets.md) — see [companion docs](#companion-docs-broken-out-for-size). Architecture options currently in the set:

- **CNN-LSTM(-attention)** — the architecture already implemented pre-planning-doc, not a new proposal: plain (non-causal, non-dilated) `Conv1D` stack → LSTM stack → (attention variant only) self-attention → pooling → dense heads, per branch. See [cnn_lstm_model.py](../app/ai_modelling/cnn_lstm/cnn_lstm_model.py) (no attention) and [cnn_lstm_attention_model.py](../app/ai_modelling/cnn_lstm_attention/cnn_lstm_attention_model.py) (with attention). Named explicitly here so the existing baseline is measured against the newer candidates below in the same Optuna study, not left as an untracked assumption of "obviously superseded."
- Transformer w/ per-tf embedding + cross-tf attention
- TCN — dilated convs for multi-scale, good single-GPU baseline; **ModernTCN** (large-kernel/grouped-conv variant, see "local feature extraction" above) is a direct, low-risk upgrade path for this line — worth profiling alongside/in place of plain TCN rather than as a separate candidate slot.
- hybrid CNN→Transformer
- state-space: **Mamba** — cheap long-seq alt to attention. (**S4**, its predecessor, is a lower-priority alt within this same slot — see [sequential encoding](#sequential-encoding) above; the two are separately-implemented mechanisms, not one candidate.)
- **LSTM** — sanity-check floor. (**GRU** is a Tier-2 alt within this floor role, same as xLSTM/ConvLSTM — see [sequential encoding](#sequential-encoding) above; it's a separate Keras layer class, not a parameter of `LSTM`.)
- **naive/persistence baseline** (e.g. "no change" / carry-forward the last signal) — not a learned model at all, the mandatory floor beneath even the LSTM/GBM floors: proves any learned candidate beats doing nothing before crediting it with real skill. Cheap enough it should just always be computed alongside backtested KPIs, not treated as a Stage-1 categorical option.
- **all-MLP mixer: TSMixer and DLinear** — two separate model implementations (structured time-mixing/feature-mixing MLP layers vs. a single linear layer per decomposed component), not variants of one function; both drop conv/attention/recurrence entirely (distinct from the already-rejected "pure MLP on flattened features" alt below, which discards sequence structure by flattening it away — TSMixer/DLinear keep the time axis intact and mix along it explicitly). A cheap second floor parallel to the LSTM floor above, but architecturally further from the conv/attention assumptions baked into the other candidates, which makes it a more informative floor: the TSMixer paper's own ablations found CNNs underperforming it on the more non-stationary of two benchmarks despite costing more compute, and there's a broader, well-known result in this literature of simple linear models (DLinear) beating heavier recurrent/attention forecasters on standard benchmarks — relevant given how non-stationary crypto price data is. Same "cheap, worth adding" priority as ModernTCN above.
- **TFT (Temporal Fusion Transformer)** and **Perceiver** — named in "combination strategy" below; not yet profiled with S1/S2/S3 hyperparams. Two separate architectures, scored independently: Perceiver's latent-bottleneck mechanism is already independently scored highest in this doc's [multi-timeframe fusion](prioritization-framework.md#multi-timeframe-fusion) table for directly targeting the flagged O(n²)/VRAM ceiling, while TFT is the more field-dominant but less bottleneck-targeted of the two — see [prioritization framework](prioritization-framework.md#current-stage-1-candidate-set) for the scoring.

Alt (kept as non-candidates, rationale in the candidate-sets doc): pure MLP on flattened features, GBM on flattened features, 4 separate per-tf models + late ensembling, GNN over tf/symbol nodes, **KAN-based time-series blocks: TimeKAN (frequency-decomposition backbone) and KANMixer (TSMixer-style block), both using learnable spline-based edge functions in place of fixed nonlinearities** — two separate architectures, both genuinely new (2025-era) and unproven at scale; same parked tier as GNN, logged pending independent MI/backtest evidence rather than funded a Stage-1 slot.

- **classic univariate statistical models** — rejected as primary candidates, not measured: these are single-series univariate-by-design and don't naturally extend to this doc's multi-tf, multivariate feature schema (relative-HLC/volume/top-distance channels per candle, see [candle feature schema](input-features.md#candle-feature-schema)) without discarding most of it back down to a single price series — would need reinventing as a per-feature ensemble to be a fair comparison. This rejection reason applies identically to all of them, so they're grouped by implementation family rather than scored separately: **ARIMA/SARIMA** (SARIMA is ARIMA's own seasonal extension, `statsmodels.tsa.SARIMAX` — a parameter choice, not a separate mechanism), **exponential smoothing/ETS**, and **GARCH** (a separately-implemented volatility model, not a point-forecaster — its plausible role, if any, is a risk/position-sizing feature input, not a Stage-1 backbone). The naive/persistence and all-MLP-mixer (DLinear) floors above already serve the "is the DL machinery earning its complexity" sanity-check role these would otherwise fill.
- **1-nearest-neighbor w/ DTW distance** — classic, parameter-light time-series-classification baseline (near-mandatory floor in academic TSC benchmarks). Ties to the DTW-preprocessing fallback already named under [multi-timeframe fusion](#multi-timeframe-fusion) → "pattern speed-invariance," but as a full pattern-matching classifier rather than a preprocessing step — worth adding as a floor if that DTW fallback ever gets built anyway, deferred until then rather than a separately-funded Stage-1 slot.

#### activation mechanisms — testing strategy

- candidates: ReLU (cheap baseline), GELU (Transformer-standard, smoother gradient), SiLU/Swish (used in Mamba/S4-style gating and modern conv nets), Mish (occasional marginal gains, costlier), GLU-family gating (GEGLU/SwiGLU — gates the feedforward block itself, common in modern Transformer variants) — applies to whichever Stage-1 architecture wins, not a separate architecture axis.
- **test scope:** swap within a fixed architecture/profile (the Stage-1 categorical-search winner), not folded into the primary search — activation choice is a cheap post-hoc refinement.
- **method:** same seed-count/backtested-KPI discipline as other hparam tests (≥3 seeds, paired stat test across matched folds, per [statistical validity of comparisons](error-rating-and-evaluation.md#statistical-validity-of-comparisons)) — not a train-loss-only comparison.
  Alt:
  - activation as an Optuna categorical dimension inside the main Stage-1 study — rejected as primary, expands the search space for a knob with low expected impact; viable secondary refinement once backend + combination strategy are decided.
  - fix GELU everywhere, untested — rejected, cheap enough to test given how often activation choice measurably matters in practice.

#### combination strategy

This axis is about combining across backend **types** (Transformer/TCN/SSM/LSTM/etc — the Stage-1 categorical search above), distinct from the multi-**timeframe** combination axis ("multi-timeframe fusion" below, which combines across tfs within one chosen backend). The two axes are orthogonal; each needs its own resolved strategy, though the fusion-mechanism menu below is shared between them.

- **single-backend-wins (no combination)** — run the Stage 1 categorical search (Transformer vs TCN vs SSM vs LSTM vs GRU, vs TFT vs Perceiver once added — see "current Stage-1 candidate set" above), take the winner, stop. Cheapest option; this is effectively the current doc's implicit baseline. Rejected as final answer only if a combo empirically beats it — needs to be measured, not assumed.
- **late ensembling of independently-trained single-backend models** — train each backend type separately to convergence, combine only at inference (average / weighted vote / stacked meta-learner on top of frozen predictions). Distinct from architectural fusion: no shared gradient path, no joint training. Cheaper to implement, more expensive at inference time (N full forward passes); the direct backend-type analog of the "4 separate per-tf models + late ensemble" alt already listed under "multi-timeframe fusion" — could reuse that machinery.
- **mixture-of-experts (MoE) gating** — a learned gate routes each example (or token/candle) to one or a sparse subset of backend "experts," rather than always running and fusing all of them. Different failure mode than architectural fusion: conditional compute (cheaper inference) vs. always-on fusion (more expensive, more information). Worth flagging as a candidate given the VRAM ceiling.
- **knowledge distillation from multiple single-backend teachers → one compact student** — train each backend separately (diagnostic/upper-bound), then distill their combined behavior into a single deployable model. Buys some of the diversity benefit without paying multi-encoder training/inference cost long-term; costs more up front (train N+1 models instead of 1).
- **differentiable/block-level NAS (DARTS-style)** — instead of hand-naming which backend combos to test, let a search algorithm learn which blocks (conv/attn/recurrent/state-space — the pipeline-stage menu above) to compose and in what arrangement. Subsumes the conv-position/layer-order questions into one search rather than a manually enumerated grid. Higher implementation cost, probably deferred given the single-GPU budget — same reasoning already applied to rejecting "very wide ranges relying only on Hyperband."
- **single hybrid backend with block-level composability, not true parallel dual-encoders** — this is actually what the existing "hybrid CNN→Transformer" candidate (and the pipeline-stage view above) already is: one encoder stack with swappable internal blocks, versus multi-backend fusion's two-plus full independent encoder stacks fused at a later stage. Worth stating explicitly since it's easy to conflate the two — the hybrid candidate is cheap and already in the Stage-1 set; true multi-encoder fusion is a separate, pricier decision covered by the strategies above.
  - **EffiCANet-style conv+attention fusion** — combines local conv feature extraction with attention over the conv output *inside one block*, rather than this doc's current sequential CNN→Transformer staging. A concrete, marginal-value variant of this same "single hybrid backend" idea; lower priority than the ModernTCN/TSMixer additions under "current Stage-1 candidate set" since it refines an already-covered candidate rather than adding a new capability.

**status:** unresolved, not yet measured. Default assumption = single-backend-wins (cheapest, current doc's implicit baseline); any other strategy adopted only on measured evidence it beats that baseline on backtested KPIs, per [core principle: error metric ≠ trading objective](error-rating-and-evaluation.md#core-principle-error-metric--trading-objective).

#### fusion mechanism

How to combine multiple representations, once a combination strategy other than single-backend-wins actually requires fusing them (the architectural-fusion / hybrid multi-encoder case above, and the "multi-tf fusion" pipeline stage):

- concatenation + MLP projection — cheapest, no learned interaction beyond the MLP.
- weighted sum / learned scalar gate per source — cheap, assumes representations already share a compatible space.
- cross-attention fusion — one source's representation attends over another's; most expressive, most expensive.
- gated fusion (GLU-style gate deciding per-dimension how much of each source to keep) — a middle ground between weighted-sum and cross-attention.
  Alt: no explicit fusion mechanism needed for single-backend-wins or late-ensembling — late-ensembling fuses only at the final prediction (average/weighted-vote/stacked meta-learner), not at representation level; see "combination strategy" above.

This same menu applies to both axes: fusing across backend **types** (this section) and fusing across **timeframes** ("multi-timeframe fusion" below, "multi-tf combination approach" bullet) — same concat/weighted-sum/cross-attention/gated choice either way.

## multi-timeframe fusion

- each series 256 candles
- from anchor backwards: first 6 of tf-ordered-list (5min–1W) — the actual input series; 1M/4M/1Y are peak/valley-confirmation-only, per [candle feature schema](input-features.md#candle-feature-schema)
- overlap: ≤1 higher-tf candle may overlap a lower-tf series.
- domain assumption: pattern meaning is scale-invariant across tf (15min compress-price-pattern ≈ 1H compress-price-pattern); combining tfs clarifies the "real truth" behind any one tf's pattern.
- **multi-tf combination approach:** per-tf encoders (small TCN/Transformer per series) → concat/pool → shared cross-tf fusion block (small Transformer over pooled reps, or concat+MLP as cheaper baseline). Lower effort than full cross-attention over the concatenated sequence; natural first arch to profile before the pricier full-attention option. Per the timeframe-in-minutes resolution in [candle feature schema](input-features.md#candle-feature-schema), this per-tf-branch design drops that field entirely (branch identity already tells the encoder the tf); it's only added back if the arch choice switches to the flat/shared-encoder option below.
  - **cross-tf attention shape, once a Transformer-based fusion block is chosen:** higher-tf-as-query/lower-tf-as-key-value (coarse context refining fine detail) is one specific shape, not the only one worth testing. **Bidirectional cross-attention** (both directions, letting lower-tf representations also get refined by higher-tf context, not just the reverse) and **Perceiver-style latent-bottleneck cross-attention** (attend into a small fixed-size learned latent array instead of the raw longer lower-tf sequence) are both viable alternatives — the latter specifically targets the longest branches (15min, 1H) where quadratic cost is worst, and ties directly to the Perceiver candidate already in "current Stage-1 candidate set".
  Alt:
  - 4 separate models + late ensemble — rejected as primary, loses cross-tf interaction; cheap baseline only
  - flat Transformer full self-attn over concat seq, no per-tf stage — most expensive, candidate only if profiling allows
  - hierarchical/wavelet decomposition — deferred, more complex, no evidence needed
- **long-window focus:** attention/state-space over fixed pooling is the standard approach, since 1yr+ of data is fed but target patterns can live anywhere from the last few candles to half the sequence, and the relevant window shifts case-to-case. Include as an arch candidate; compare vs recency-weighted-pooling baseline to confirm it earns its cost.
  Alt:
  - fixed recency weighting w/o learned attention — rejected as sole approach, can't adapt; kept as cheap baseline
  - manual windowing/hand-picked N — rejected, reintroduces the problem attention solves
- **pattern speed-invariance** (same pattern over 3 vs 30 candles) — a time-warping problem:
  - TCN multi-dilation captures multi-scale shape w/o explicit warp
  - attention has no fixed receptive field either — test directly
  - Explicit DTW preprocessing kept as fallback/diagnostic if the architectural approach fails empirically (test: does model score known same-pattern-diff-speed examples similarly?)
    Alt:
  - DTW preprocessing as default — rejected, heavier engineering, hurts real-time variable-length inference; diagnostic/fallback only
  - volatility-based logical-candle resampling — deferred, hard to reconcile w/ existing windowing/multi-tf design
- **pattern scale-invariance** (same pattern, different price magnitude): largely already handled by the resolved ATR-relative normalization (scale expressed relative to volatility by construction). Open part is architectural: conv layers naturally somewhat scale-robust; attention has no strong scale bias — evaluate empirically. Treat as validated by construction; architecture comparison is the remaining lever, not a new normalization step.
  Alt: separate explicit "scale normalization" beyond ATR-relative (rejected — double-normalizes, unnecessary).
- **decision-anchor point** keep the primary Long/Short/No-Trade decision anchored at the anchor candle for now (simplest, matches current label design). Treat entry-timing (too-soon/too-late detection) as a secondary/future output — changes label design non-trivially, wait for anchor-based baseline first.
  Alt:
  - build entry-timing into v1 — rejected for now, adds complexity before baseline validated
  - separate downstream "timing" model post-Long/Short — viable future step, deferred
- **higher-tf "in progress" candles — decision:** use only completed candles — lowest tf 15min; 1H candles = 256×15min prior so most-recent is fully closed; same for higher tfs. Sidesteps partial-candle state and cross-tf boundary-alignment leakage by construction.

## validation & train/test splitting

- windows built only from contiguous complete data; any gap-containing range discarded entirely.

- **split scheme — resolved (simplified):** train on all other trading pairs, validate on BTC/USDT. This is a cross-symbol (leave-one-symbol-out) split, not a temporal one — since train and validation are entirely different assets, there's no same-symbol window overlap to leak across, so the walk-forward-vs-embargo machinery isn't needed. Use the full BTC/USDT history as the validation set.
  Alt:
  - walk-forward / random-split-with-embargo within a single symbol's own history — previous approach, dropped as unnecessary complexity now that validation is cross-symbol
  - rotating leave-one-symbol-out across all pairs rather than always BTC/USDT — viable generalization check, deferred; BTC/USDT fixed as the validation symbol since it's the primary target market
- **final holdout — resolved:** reserve the most-recent contiguous block of BTC/USDT (≥ several weeks, enough trade outcomes at 4H scale); never touches training-pair selection or any tuning decision; used exactly once, after arch/hparams/normalization/threshold are locked in from the BTC/USDT validation split, for final reported KPIs. Materially worse holdout result than validation = overfitting-to-tuning signal → investigate, don't re-tune against it (would require a fresh holdout).
  Alt: no separate final holdout, reporting the BTC/USDT validation KPIs directly as final (rejected — still risks overfitting through repeated validation-set tuning).

## optimization strategy

- optimization = **one search** across (1) arch/model-combo choice + (2) each arch's hparams, not two disjoint phases.
- architecture = single categorical param inside same Optuna study as hparams (conditional sub-params per arch), not exhaustive — bad archs pruned early instead of full independent sweeps each. Impl: `app/ai_modelling/parameter_optimizser/optuna_optimizer.py`.
- Optuna TPE (sample-efficient, single-GPU budget) + Hyperband pruning.
- GA/NSGA-II for optional 2nd refinement stage.
- Pareto front across competing KPIs (e.g. Sortino vs max-DD).
- per-trial time measured not assumed.
  - runs real training steps per arch, measures wall-clock+peak-VRAM;
  - `estimate_total_budget()`/`max_trials_for_budget()` → projected total + trial-count cap before full study.
- `OptunaPruningCallback` reports val_loss/epoch, prunes Hyperband-unpromising or NaN/Inf trials.
- best-run selection KPI: see [error rating & model evaluation](error-rating-and-evaluation.md#backtested-trading-kpis-final-selection) — primary=expectancy, guardrail=max-DD, secondary=Sortino, once the backtest module is built; until then `val_loss` remains the training-time proxy (`compute_fitness()`), explicitly interim not final.

## class imbalance handling

- test class-weighted vs focal loss for classification-style targets (peak/valley class, TP-hit-before-SL class), compare.
- multi-horizon ATR-distance to nearest peak/valley at fixed horizons (tf-ordered-list 4H-to-1Y) — turns single categorical "highest confirmed tf" feature into continuous features, sidesteps imbalance for that feature.

- **multi-horizon vs categorical peak/valley feature** feed both. Continuous ATR-distance features = primary (solve imbalance); keep categorical "highest confirmed tf" too as a cheap compact discrete summary — may capture something continuous version doesn't, esp. at low data volume. Confirm via ablation, don't assume; drop categorical only if ablation shows zero marginal contribution.
  Alt:
  - continuous-only, drop categorical now — rejected/deferred, no ablation evidence yet it's safe
  - categorical-only, skip multi-horizon — rejected, reintroduces the imbalance problem it solves
  - replace w/o ever testing — rejected, riskier than testing first, no evidence
- **class-weight-vs-focal test scope** applies to both peak/valley and TP-hit/drawdown targets, as two separate experiments (not one shared decision):
  - peak/valley target if kept as aux output
  - ~~TP-hit/SL-hit/Timeout target~~ — removed from spec (see [training-data.md § TP / MAE / OM labels](training-data.md#tp--mae--om-labels)); `MAE`/`OM` are now continuous regression targets, not a categorical outcome, so this class-weight/focal question no longer applies here
  - Tune class weights/focal-gamma per target, not globally.
    Alt:
  - scope to peak/valley only — rejected, leaves TP/SL imbalance, likely worse, unaddressed
  - one blanket loss choice untested per-target — rejected, no evidence of transfer, gamma likely needs per-target tuning
  - resampling alternatives:
    - SMOTE-style — awkward for sequential windows, likely rejected
    - class-balanced batch sampling — deferred, complementary
    - inference-time cost-sensitive thresholding — relates to No-Trade threshold answer; complementary lever, not replacement
- **prevalence measurement — next action, not yet run:** actual prevalence (% candles peak/valley per horizon, % trades reaching each TP vs SL-hit vs Timeout) isn't known — measure empirically once the labeling pipeline exists, via a data-profiling script, before finalizing the class-weight/focal choice above.
  Alt: assume prevalence from market-structure intuition, no measurement (rejected — exactly what this step avoids).
- **cheap iteration proxy for the class-weight/focal choice** — a GBM on flattened features is a cheap place to iterate which weighting scheme/gamma looks promising before committing to a full DL retrain cycle, same cheap-proxy-before-expensive-run pattern already used for feature screening; see "auxiliary tabular models (GBM-family)".

## auxiliary tabular models (GBM-family)

Scoped roles for GBMs and related tabular models in this pipeline, distinct from the primary sequence-model architecture search above. GBMs operate on a flat, point-in-time feature vector per example — no notion of "this candle came before that one" beyond whatever's hand-engineered into lagged features — so they're deliberately scoped to screening/meta-labeling/floor/proxy roles, never the primary architecture search covered under "model architecture & selection".

- **candidate-feature screening tiebreaker** — already resolved, see "candidate-feature screening — method": MI screen first (cheap, no GPU), small LightGBM/XGBoost only if MI is ambiguous, full DL run reserved for candidates passing both. Correct scoped use because the question at that stage is "does this one feature carry signal at all," not "does the sequence pattern matter."
- **meta-labeling classifier** (López de Prado framework) — the canonical GBM use case for this kind of problem. Primary model (this doc's DL architecture) proposes a side (Long/Short) and size; a secondary GBM classifier answers a narrower question, "given this signal fired, should I actually take it," trained on a much smaller, better-balanced label set than the primary model's full action space — e.g. a binary "is `OM` above threshold" derived from [training-data.md § TP / MAE / OM labels](training-data.md#tp--mae--om-labels), since meta-labeling is a flat point-in-time binary classification problem, not a sequence problem.
- **flattened-feature floor baseline** — parallel to, but a genuinely different question than, the LSTM/GRU "sanity-check floor" role (see "sequential encoding" / "current Stage-1 candidate set"). LSTM/GRU still respects sequence order; a GBM on flattened features doesn't see order at all. A GBM floor answers "how much of the signal is even sequence-dependent, versus just present in a flattened feature snapshot" — if GBM-on-flattened gets close to the sequence models, that's informative (the sequence-modeling machinery may be buying less than expected); if it's far behind, that validates the sequence-architecture investment.
- **class-imbalance / prevalence experiment proxy** — cheap place to iterate the class-weighted-vs-focal-loss decision (which weighting scheme, which gamma) before committing to a full DL retrain cycle; see "class imbalance handling".
- **where GBMs are a poor fit:** anything that needs the actual multi-timeframe sequence, cross-tf attention, or pattern-shape/speed-invariance reasoning from "multi-timeframe fusion". GBMs have no notion of sequence order beyond hand-engineered lagged features — exactly why they're scoped to screening/meta-labeling/floor/proxy roles, not the primary architecture search.
  Alt:
  - GBM as a serious primary-architecture candidate — rejected, discards sequence structure the same way the "pure MLP on flattened features" alt does under "current Stage-1 candidate set"

### modern GBM-family alternatives

- **CatBoost** — same GBDT family, different engineering choices (ordered boosting to reduce prediction shift, native categorical handling without manual encoding). Recent large benchmarks (McElfresh et al., NeurIPS 2023, 176 datasets; TabArena 2025) put CatBoost ahead of XGBoost and LightGBM on mean rank among GBDTs, particularly on high-cardinality categoricals, mixed-type features, and moderate-to-high noise datasets. Worth adding as a third option anywhere this doc currently plans LightGBM/XGBoost (feature screening, meta-labeling) — near-identical API, cheap to swap in.
- **Random Forest** — bagging rather than boosting: decorrelated trees averaged instead of sequentially fit to residuals. Not a GBDT variant, a genuinely different tree-ensemble family — more robust to overfitting/noisy labels out of the box, less tuning-sensitive, though typically a notch behind well-tuned GBDTs on mean benchmark accuracy. Cheap floor to add alongside LightGBM/XGBoost/CatBoost in the same screening/meta-labeling roles, given how noisy the crypto label set likely is (see "class imbalance handling" → prevalence measurement, not yet run).
- **tabular foundation models (TabPFN v2/2.5, TabICL)** — transformer-based in-context learners pretrained on large volumes of synthetic tabular data, requiring no dataset-specific hyperparameter tuning at inference time. Recent benchmarks report TabPFN with default hyperparameters on average outperforming tuned GBDTs, and a broader 300-dataset evaluation found tabular foundation models now match or surpass GBDTs on many tasks. Genuinely cheap to try in the feature-screening or meta-labeling slot given "no tuning needed" fits the single-GPU budget — but treat this the same way this doc treats LLM-based forecasting under "excluded topics": promising benchmarks, unproven specifically on financial/noisy-label data, so it belongs in the same screening/meta-labeling role as a candidate to test against the GBM baseline via backtested-KPI discipline, not an assumed upgrade.
- **general tabular deep learning (FT-Transformer, TabR, ResNet-tabular)** — the gap between these and GBDTs has been narrowing in recent literature, though GBDTs remain the practical default for most tabular tasks given faster training and easier tuning. Lower priority here specifically, given the much larger, more directly relevant DL architecture search already underway for the sequence-modeling half of the problem.
  Alt: adopt CatBoost/TabPFN by benchmark reputation alone, skip local evaluation — rejected, same reasoning as everywhere else in this doc: measured backtested-KPI evidence only, per [core principle: error metric ≠ trading objective](error-rating-and-evaluation.md#core-principle-error-metric--trading-objective).

### uncertainty-native GBM variants — confidence-metric gap

Candidate techniques for the confidence-metric gap — see [error rating & model evaluation § confidence & calibration metrics](error-rating-and-evaluation.md#confidence--calibration-metrics) for the problem statement and what's measured; this section owns *which technique* produces it. This family produces calibrated uncertainty as a mechanism of the model itself, not by engineering extra confidence-specific input features.

- **quantile GBM** (native quantile/pinball objective in LightGBM/XGBoost) — try first: reuses the pinball-loss work already planned for price-level heads (see [per-head statistical metrics](error-rating-and-evaluation.md#per-head-statistical-metrics-dev-diagnostics)), and gives prediction intervals (e.g. 10th/50th/90th percentile TP) rather than a single point number. Cheapest option in this group, reuses infrastructure already planned.
- **NGBoost** — a genuinely different mechanism than standard GBM: instead of gradient-boosting toward a point estimate, it boosts the parameters of an entire output distribution using natural gradients, so every prediction carries a full predictive distribution rather than a point forecast plus a bolted-on score. Base-learner-agnostic (works with tree learners) — a small step from the existing GBM screening code, not a rebuild. Clean fit for meta-labeling (calibrated P(TP-hit) instead of a bare 0/1) or for the TP/MAE/OM heads directly.
- **quantile regression forests / newer multivariate extensions** (e.g. Tomographic Quantile Forests) — same idea, extended to jointly-calibrated uncertainty across correlated outputs, relevant if TP and SL should have correlated (not independently-estimated) uncertainty bands rather than separately-quantiled heads that might contradict each other.
  Alt:
  - hybrid GBM + TabPFN/LLM-boosted ensembles — recent work explicitly boosts GBDTs with TabPFN or LLM components as additional weak learners rather than treating tree-based and foundation-model approaches as competitors; best-of-both efficiency + foundation-model signal, but a more involved implementation than swapping libraries — ranked below the uncertainty-native options for near-term priority.
  - retrieval-augmented in-context tabular learning (TabR-style: at inference, look up similar historical rows via nearest-neighbor search in a learned embedding space, condition the prediction on them) — conceptually close to what the zigzag/nearest-top-distance features already hand-engineer (see "candle feature schema"), but research-stage, not yet a mature drop-in library the way NGBoost/quantile-LightGBM are; deferred.

**priority, given this doc's cheap-proxy-before-expensive-run discipline:** quantile GBM first (cheapest, reuses planned pinball-loss work, native in libraries already in scope) → NGBoost second (bigger uncertainty-quantification payoff, still a straightforward library addition) → hybrid/retrieval approaches deferred, same "parked, not funded a slot" tier as the LLM-reprogramming and GNN entries, pending evidence from the cheaper options first.

## experiment tracking (current priority)

- needed now: ad hoc file-naming (`- Copy (2).keras`, `.bak`, `.nan` in /data) won't scale, can't trace which run→which result.
- decide lightweight tracking: min = consistent naming/logging convention (config hash+date+key hparams); ideally a tool (MLflow/W&B/CSV-SQLite) logging config+dataset-version+metrics(loss+trading KPIs)+artifact path together.
- local-only (e.g. MLflow w/ local file backend).
  Alt:
  - W&B/cloud-hosted — rejected for now, conflicts w/ local-only; revisit if collaboration/remote-dashboard becomes a real need
  - bare CSV/SQLite log, no dedicated tool — viable fallback if MLflow local-server overhead isn't worth it
  - no formal tracking — rejected, explicitly doesn't scale, see above

## companion docs (broken out for size)

These hold detail kept out of this doc purely for navigability — still fully in scope, not excluded (contrast with "excluded topics" below, which is genuinely out of scope). This doc keeps short option-lists with pointers into these at the relevant sections.

- [input features & embedding](input-features.md) — candle feature schema, feature-set completeness-testing workflow, input/feature-embedding stage detail.
- [Stage-1 candidate sets](model-architecture-candidate-sets.md) — hardware constraints and VRAM/RAM budget split, the per-candidate design checklist, pseudocode-level architecture detail for both single-backend candidates and combined/super-model strategies, full S1/S2/S3 hyperparameter profiles per architecture, search-space bounds, cross-architecture-fairness protocol.
- [prioritization framework](prioritization-framework.md) — the tiering scoring method (factors, combination formula, mandatory-floor exception) plus a full fund-now/test-later/parked breakdown of every candidate named across this doc, organized by the same stage/layer headings.
- [error rating & model evaluation](error-rating-and-evaluation.md) — per-head statistical metrics, confidence/calibration metrics, backtested trading KPIs, seed-count/statistical-validity workflow, model-selection pipeline.

## excluded topics (broken out into separate files)

Approaches out of scope for this doc, tracked in their own file instead of inline here. Add future exclusions as new bullets + files, same pattern.

- [Time-Series Foundation Models (TSFMs)](timeseries-foundation-models-architecture-planning.md) — pretrained/fine-tuned checkpoints (Chronos, TimesFM, Moirai, Lag-Llama, PatchTST); this doc's "model architecture & selection" section covers custom/from-scratch architecture design only.

## deferred topics (not current concerns, placeholders)

- **transaction costs/spread/slippage/latency**: matters for sub-4H scalping, not addressed now. Revisit before live/paper trading — cost-free backtest overstates real perf.
- **risk/position sizing beyond TP targets**: handled manually via existing procedure, not by model. No AI work needed now.
- **market regime robustness/retraining cadence**: not addressed now. Revisit once live a while — crypto regime shifts (trend/range/vol), untouched model can decay silently.

## glossary

- ATR = pandas-ta.ATR(256)
- anchor candle = last candle of a 256-candle window; the "as of" point for a prediction (training or live)
- tf-ordered-list = 5min, 15min, 1H, 4H, 1D, 1W, 1M, 4M, 1Y
- tf = timeframe
- natural price distance = signed distance from a top: + = price higher than the top, − = lower (not adjusted for peak vs. valley)
- normal price distance = natural price distance with sign flipped for valleys, so + always means "away from the top" for both peaks and valleys
- volume strength of tops = SUM(volume) / ATR(volume) of the 2-tf-lower candles (e.g. 4H top → 15min) within ±256 top-tf candles, restricted to candles whose [L,H] overlaps the top's price range (peak-high/valley-low ± 2-tf-lower ATR(256))
- Stage-1 = the current architecture-search phase; picks one whole model architecture from the candidate set (see "model architecture & selection")
- S1/S2/S3 = hyperparameter-profile labels per Stage-1 architecture candidate: depth-heavy / width-heavy / context-heavy (see [Stage-1 candidate sets](model-architecture-candidate-sets.md))
- GBM = gradient boosting machine (LightGBM/XGBoost/CatBoost family) — see "auxiliary tabular models (GBM-family)"

# Model & Architecture Engineering

## Data feed design

Candle-level feature schema, the feature-set completeness-testing workflow (ablation, MI/GBM candidate screening, candidate feature pool), and screening methodology are broken out to [input features & embedding](02-Data, Label & Feature Engineering.md#candle-feature-schema) for size.

## Model architecture & selection

### top level architecture

Two views of the same design space live here: a **pipeline-stage view** (what building block fills each stage of the network) and the **current Stage-1 candidate set** (the concrete whole-architecture options actually fed into the Optuna categorical search today, per "optimization strategy" below / [cross-architecture fairness](04-Experimentation, Evaluation & Optimization.md#cross-architecture-fairness)). The stage view is the fuller design space — useful for reasoning about new techniques and for a possible future block-level NAS (see "combination strategy" → differentiable/block-level NAS) — but Stage-1 currently picks one whole architecture from the candidate set, not an arbitrary per-stage mix, until/unless that NAS alternative is adopted.

#### prioritization framework — tiering candidate techniques

Full scoring factors, tiers, and combination formula for sorting candidates below into fund-now / test-later / parked broken out to [prioritization framework](04-Experimentation, Evaluation & Optimization.md#decision-framework). Use it when adding a new candidate anywhere in this doc, so tiering stays consistent rather than vibes-based per bullet — it's the general form of the judgment already made informally throughout the bullets below (e.g. ModernTCN "worth promoting straight into Stage-1 profiling" vs. KAN blocks "parked... pending evidence").

#### architecture diagnosis, capacity and robustness

That framework decides which candidates get funded/tested. It doesn't cover what to do with a candidate once it's trained: matching mechanism to observed data characteristics before scoring it, sizing depth/width against dataset size, testing whether a multi-stage candidate's components each pull weight, diagnosing why a trained candidate underperforms, checking a ranking survives beyond the split it was picked on, comparing candidates on measured cost rather than param count alone, and when a more-complex winner isn't worth adopting. Broken out to [Architecture diagnosis, capacity & robustness](#architecture-diagnosis-capacity-and-robustness) below for size.

#### input / feature embedding

Full detail broken out to [input features & embedding](#input--feature-embedding). Embedding options in play:

- linear/MLP projection of the per-candle feature vector → `d_model` (current default, shared first step across all Stage-1 candidates)
- per-tf learned tf-id embedding (flat/shared-encoder archs only; implicit via branch identity otherwise)
- PatchTST-style patch embedding (cheap lever on the attention VRAM cost flagged under [hardware constraints](#hardware-constraints))

- linear/MLP projection of the per-candle feature vector → `d_model` — shared first step across all Stage-1 candidates.
- per-tf embedding — learned tf-id embedding for flat/shared-encoder archs; implicit via branch identity for per-tf-branch archs (see the timeframe-in-minutes resolution under [candle feature schema](02-Data, Label & Feature Engineering.md#candle-feature-schema) above).
- **PatchTST-style patch embedding** — groups contiguous candles into patches before projecting, shortening the effective sequence length fed to attention. Directly relevant given the VRAM-cost note under [hardware constraints](#hardware-constraints) → "max feasible model size" (full attention over the concatenated multi-tf sequence is the dominant cost, not param count) — patching is a cheap lever on that cost. This is the patching _mechanism_ trained from scratch as part of the Stage-1 candidate, distinct from the pretrained PatchTST-based checkpoints covered (and excluded) under [TSFMs](04-Experimentation, Evaluation & Optimization.md#time-series-foundation-models-tsfms).
  Alt: raw per-candle projection, no patching — simpler, longer effective sequence into attention; current default.

#### local feature extraction

- **plain/vanilla 1D conv** (non-causal, non-dilated, same-padding stacked `Conv1D`) — the simplest possible learned local-extraction baseline, and worth naming explicitly since it's easy to jump straight to TCN/dilation and never test whether that added complexity earns its keep. This is also what the existing (pre-planning-doc) code already implements — `cnn_lstm_block()` in [cnn_lstm_attention_model.py](../../app/ai_modelling/cnn_lstm_attention/cnn_lstm_attention_model.py) and [cnn_lstm_model.py](../../app/ai_modelling/cnn_lstm/cnn_lstm_model.py) both stack same-padding `Conv1D` layers ahead of LSTM — see "current Stage-1 candidate set" below for that architecture as a whole.
- dilated causal conv (TCN) — multi-scale local pattern extraction, cheap.
- **residual-CNN time-series-classification baselines** — three separately-implemented network topologies from the UCR/UEA time-series-classification benchmark literature, distinct lineage from the NLP-derived TCN/ModernTCN line above; different classes in any TSC library (e.g. `aeon`/`tsai`), not one function with a parameter choice — see [prioritization framework](04-Experimentation, Evaluation & Optimization.md#local-feature-extraction) for why each is scored independently rather than as one candidate:
  - **ResNet** — stacked residual conv blocks. The oldest and weakest of the three here (InceptionTime generally supersedes it in TSC benchmarks).
  - **FCN** — plain fully-convolutional, global-average-pooled stack, no residual connections. Simplest and cheapest of the three, still competitive.
  - **InceptionTime** — multi-kernel-size Inception-style modules in parallel per block. Newest and best-benchmarked of the three, at higher per-module cost.

  All three: cheap, non-causal, no dilation-schedule tuning needed — worth testing as an independent floor for the plain-CNN/TCN/ModernTCN family, not assumed inferior just because the lineage is older.

- **ModernTCN** — large-kernel, grouped/depthwise convolution in place of TCN's small-kernel stacked-dilation approach, explicitly designed to capture cross-time _and_ cross-variable dependency in one pass. A direct, low-risk upgrade to the plain TCN block above — same role in the pipeline, same cost class, better-tested internals — worth promoting straight into Stage-1 profiling rather than parking as a deferred alternative. Particularly relevant here since the feature schema is genuinely multivariate (relative-HLC, volume/ATR, gap, multiple top-distance channels per candle — see [candle feature schema](02-Data, Label & Feature Engineering.md#candle-feature-schema)); plain TCN's per-channel/shared-filter convolution treats cross-variable interaction only incidentally, ModernTCN's grouped-conv design treats it deliberately.
- conv stem ahead of a Transformer (the hybrid CNN→Transformer candidate) — same cost-reduction goal as patching above, via learned downsampling instead of fixed patches.
- **TimesNet-style 1D→2D reshape** — reshapes the series into a 2D grid keyed by detected/candidate periodicities, then applies 2D (inception-style) conv blocks to capture multi-periodicity directly. Relevant if session/cyclical features (hour/day sin-cos, session-open flags — see [candidate feature pool](02-Data, Label & Feature Engineering.md#candidate-feature-pool)) are meaningfully periodic; untested assumption, not yet screened.
- **SCINet** — recursive downsample→convolve→interact structure that extracts multi-resolution features hierarchically from a single input series. Conceptually close to what this doc's "multi-timeframe fusion" section already does by hand across resampled branches (5min/15min/1H/4H/1D/1W); SCINet does something structurally similar _within_ one branch, so it reads as a plausible swap for the conv stage inside each per-tf encoder rather than a whole-pipeline replacement.
  Alt: no local-extraction stage, feed embedded scalars straight to sequential/attention stage — cheaper, ablation baseline.

#### sequential encoding

- LSTM recurrence — sanity-check floor, sequential context without attention's O(n²) cost. This is what the existing code already runs (`cnn_lstm_model.py`), so it's the floor, not just an option.
  - **GRU** — `tf.keras.layers.GRU`, a separate layer class from `LSTM` (merged forget/input gate, no separate cell state, ~25% fewer parameters), not a parameter of the same layer — see [prioritization framework](04-Experimentation, Evaluation & Optimization.md#tool-identity-test-when-a-xy-grouping-stays-one-row) for why this and xLSTM/ConvLSTM below get the same "alt within the floor role" treatment despite one being a different class and the others being config choices. Tested as an alt within this same floor role, not a new pipeline stage.
  - **xLSTM** — modernized LSTM (exponential gating, matrix memory), recently used as the backbone of at least one zero-shot time-series foundation model. Not a mechanism swap in the strict sense (still recurrent) — the relevant question is "is a modernized recurrent block worth it," tested as an alt within this same floor role rather than a new pipeline stage.
  - **ConvLSTM** — convolutional gates inside the recurrent cell itself, a genuinely different mechanism from stacking a separate conv stage ahead of a plain LSTM (which is what the existing CNN-LSTM(-attention) code already does — see "current Stage-1 candidate set" below). Worth testing as an alt within this floor role, not a separate pipeline stage.
- state-space — linear-time long-context alternative to attention. Two separately-implemented mechanisms, not one function with a parameter (see [prioritization framework](04-Experimentation, Evaluation & Optimization.md#sequential-encoding)):
  - **Mamba** — input-selective scan (`mamba_ssm`), the current standard-bearer for this family; directly targets the flagged O(n²)/VRAM ceiling.
  - **S4** — fixed, HiPPO-initialized state matrices, computed via convolution; largely superseded by Mamba in the literature, kept as a lower-priority alt rather than a co-equal option.
  - **Hyena (implicit long convolution)** — parameterizes an implicit convolution whose effective kernel spans the whole sequence, aiming for attention-like long-range coverage at sub-quadratic cost. Sits conceptually between the TCN dilation line below and this SSM line rather than beside either — treat as a variant to test within the SSM branch, not a separate architecture line.
- TCN dilation stack — same block as "Local Feature Extraction" above; listed here too because dilation depth is what gives it long-range context, not just a local receptive field.
  Alt: no dedicated sequential-encoding stage, rely entirely on attention for all-range dependency — the pure-Transformer candidate already covers this.

#### attention / dependency

- standard self-attention over the concatenated multi-tf sequence — most expressive, most expensive; the O(n²) cost [hardware constraints](#hardware-constraints) already flags as VRAM-dominant.
- cross-tf attention over per-tf pooled representations (not raw concat seq) — cheaper, see "multi-tf fusion" stage below.
- **Informer (ProbSparse attention)** — sparsifies attention to sub-O(n²) cost, a direct mitigation for the flagged VRAM bottleneck.
- **Autoformer (series decomposition + autocorrelation)** — replaces dot-product attention with trend/seasonal decomposition and autocorrelation-based dependency discovery.
- **FEDformer (frequency-domain attention)** — attends in the frequency domain; another angle on the same O(n²) mitigation as Informer/Autoformer.
- **iTransformer (inverted attention)** — attends across variates (feature channels) instead of across time. Interesting given the candle schema's many engineered feature channels per step — worth testing as a channel-dependency finder rather than a time-dependency finder.
- **GQA (Grouped-Query Attention) / MQA (Multi-Query Attention)** — multiple query heads share fewer key/value projections, cutting KV memory footprint directly. Standard, well-tested, cheap to add as a categorical option on any Transformer/hybrid/TFT branch above — targets the same O(n²)/VRAM cost this doc's hardware notes already flag, but the memory dimension of it specifically rather than the compute dimension Informer/Autoformer/FEDformer target.
- **MLA (Multi-head Latent Attention)** — newer than GQA (introduced with DeepSeek-V2): compresses keys/values into a low-rank latent space before caching, going further than GQA on memory reduction while reportedly preserving more representational power than sharing heads outright. Worth testing head-to-head against GQA specifically, since the flagged bottleneck is activations/KV-cache, not parameter count — a more targeted comparison than either vs. plain attention.
- **Native Sparse Attention (NSA)** — combines compressed/coarse attention, selective fine-grained attention, and sliding-window attention as parallel trainable branches, rather than a fixed post-hoc sparsity pattern. Structurally close to this doc's own multi-tf design (coarse/fine/local view ≈ higher-tf/lower-tf/local-window reasoning, see "multi-timeframe fusion" below) — a stronger, more current candidate than older sparse schemes for the same O(n²) mitigation Informer/Autoformer/FEDformer already target.
- **Longformer-style sliding-window + global tokens** — simpler, older, well-understood: local attention window plus a small number of globally-attending tokens. Cheaper fallback if NSA proves too complex to get working within the single-GPU budget.
  - **BigBird-style + random attention** — adds a third, block-sparse "random attention" component on top of Longformer's window+global pattern. A separate reference implementation (`BigBirdModel`), not a config flag on Longformer's — see [prioritization framework](04-Experimentation, Evaluation & Optimization.md#attention--dependency) for why it's scored as its own row (a real block-sparse kernel needed to be efficient, not a dense-mask trick) and lands one tier below plain Longformer rather than sharing its score.
- **linear/kernel attention** — approximates the attention matrix for linear-in-sequence-length cost; generally the weakest of the sub-quadratic options on modeling quality in published comparisons, but cheapest to implement — bottom-of-priority fallback only. Two separately-implemented approximation mechanisms, not variants of one function:
  - **Performer** — FAVOR+ random-feature kernel approximation of softmax attention.
  - **Linformer** — fixed low-rank projection of the sequence-length dimension.
- **Differential Attention** — computes two separate softmax attention maps and subtracts one, cancelling common-mode attention noise and sharpening focus on the signal that differs between them. Pitched specifically as a signal-to-noise improvement, directly relevant given OHLCV/candlestick signal is inherently noisy — a genuinely case-specific quality candidate, not a copy-the-LLM-trend pick.
  Alt:
  - no attention stage, conv/recurrence/state-space only — cheapest, ablation baseline given attention's flagged cost.
  - FlashAttention — not a competing mechanism to choose among above; an exact-attention implementation that fuses the softmax computation to avoid materializing the full attention matrix. Same category as mixed-precision AMP under [hardware constraints](#hardware-constraints): a near-free win, on by default under whichever variant above is chosen, not a separate arm of this search.

#### multi-tf fusion

Full strategy list lives in the standalone "multi-timeframe fusion" section below (per-tf encoders + fusion block vs. flat full-attention vs. hierarchical/wavelet vs. late-ensemble) — this stage is where that resolved/candidate choice plugs into the overall pipeline. The concat/weighted-sum/cross-attention/gated menu under "fusion mechanism" below applies here too, not only to backend-type combination.

#### global representation

- pooling over the final sequence/fused representation → fixed-size vector (mean / max / attention-pooling / last-token).
- **N-BEATS / N-HiTS** — two separate model classes (e.g. `neuralforecast`'s `NBEATS`/`NHITS`), not a parameter choice on one block; an alternative path through this stage that skips the sequential-encoding and attention stages entirely, producing hierarchical basis-function forecasts directly from pure feedforward stacked residual blocks. **N-HiTS** specifically adds hierarchical interpolation/multi-rate sampling targeting long-horizon efficiency and is cheap on VRAM — worth a Stage-1-adjacent candidate slot given the hardware ceiling. **N-BEATS**, without that specific efficiency angle, is a secondary refinement rather than a co-equal candidate — see [prioritization framework](04-Experimentation, Evaluation & Optimization.md#global-representation) for the scoring.
  Alt: no explicit global-representation stage, feed the full sequence directly to per-position heads — viable for head designs needing full sequence context; deferred, adds head complexity.

#### prediction heads

- action head (Long/Short/None), MAE/OM regression (auxiliary MFE) or quantile heads, confidence head — targets defined in [training-data.md](02-Data, Label & Feature Engineering.md#model-output-targets).
- **point-estimate (baseline) vs probabilistic `MFE`/`MAE` heads (alternative):** baseline regresses `MAE`/`OM`/`MFE` as single point values; the alternative outputs distribution parameters (mean/std/skew/kurtosis, added incrementally) via a distributional NLL loss (Gaussian → skew-normal → skew-t as moments are added) instead of MAE/MSE/pinball. Full framing, testing order, and the derived TP/SL-probability payoff live in [training-data.md § model output targets](02-Data, Label & Feature Engineering.md#model-output-targets) — this section only owns the head/architecture shape.
- loss/metric choice per head — see [error rating & model evaluation](04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics); not duplicated here.

#### current Stage-1 candidate set

Full hyperparameter profiles (S1/S2/S3 = depth-heavy/width-heavy/context-heavy per architecture), search-space-bounds methodology, and the cross-architecture-fairness protocol live in [Stage-1 candidate sets](#stage-1-candidate-sets). Architecture options currently in the set:

- **CNN-LSTM(-attention)** — the architecture already implemented pre-planning-doc, not a new proposal: plain (non-causal, non-dilated) `Conv1D` stack → LSTM stack → (attention variant only) self-attention → pooling → dense heads, per branch. See [cnn_lstm_model.py](../../app/ai_modelling/cnn_lstm/cnn_lstm_model.py) (no attention) and [cnn_lstm_attention_model.py](../../app/ai_modelling/cnn_lstm_attention/cnn_lstm_attention_model.py) (with attention). Named explicitly here so the existing baseline is measured against the newer candidates below in the same Optuna study, not left as an untracked assumption of "obviously superseded."
- Transformer w/ per-tf embedding + cross-tf attention
- TCN — dilated convs for multi-scale, good single-GPU baseline; **ModernTCN** (large-kernel/grouped-conv variant, see "local feature extraction" above) is a direct, low-risk upgrade path for this line — worth profiling alongside/in place of plain TCN rather than as a separate candidate slot.
- hybrid CNN→Transformer
- state-space: **Mamba** — cheap long-seq alt to attention. (**S4**, its predecessor, is a lower-priority alt within this same slot — see [sequential encoding](#sequential-encoding) above; the two are separately-implemented mechanisms, not one candidate.)
- **LSTM** — sanity-check floor. (**GRU** is a Tier-2 alt within this floor role, same as xLSTM/ConvLSTM — see [sequential encoding](#sequential-encoding) above; it's a separate Keras layer class, not a parameter of `LSTM`.)
- **naive/persistence baseline** (e.g. "no change" / carry-forward the last signal) — not a learned model at all, the mandatory floor beneath even the LSTM/GBM floors: proves any learned candidate beats doing nothing before crediting it with real skill. Cheap enough it should just always be computed alongside backtested KPIs, not treated as a Stage-1 categorical option.
- **all-MLP mixer: TSMixer and DLinear** — two separate model implementations (structured time-mixing/feature-mixing MLP layers vs. a single linear layer per decomposed component), not variants of one function; both drop conv/attention/recurrence entirely (distinct from the already-rejected "pure MLP on flattened features" alt below, which discards sequence structure by flattening it away — TSMixer/DLinear keep the time axis intact and mix along it explicitly). A cheap second floor parallel to the LSTM floor above, but architecturally further from the conv/attention assumptions baked into the other candidates, which makes it a more informative floor: the TSMixer paper's own ablations found CNNs underperforming it on the more non-stationary of two benchmarks despite costing more compute, and there's a broader, well-known result in this literature of simple linear models (DLinear) beating heavier recurrent/attention forecasters on standard benchmarks — relevant given how non-stationary crypto price data is. Same "cheap, worth adding" priority as ModernTCN above.
- **TFT (Temporal Fusion Transformer)** and **Perceiver** — named in "combination strategy" below; not yet profiled with S1/S2/S3 hyperparams. Two separate architectures, scored independently: Perceiver's latent-bottleneck mechanism is already independently scored highest in this doc's [multi-timeframe fusion](04-Experimentation, Evaluation & Optimization.md#multi-timeframe-fusion) table for directly targeting the flagged O(n²)/VRAM ceiling, while TFT is the more field-dominant but less bottleneck-targeted of the two — see [prioritization framework](04-Experimentation, Evaluation & Optimization.md#current-stage-1-candidate-set) for the scoring.

Alt (kept as non-candidates, rationale in the candidate-sets doc): pure MLP on flattened features, GBM on flattened features, 4 separate per-tf models + late ensembling, GNN over tf/symbol nodes, **KAN-based time-series blocks: TimeKAN (frequency-decomposition backbone) and KANMixer (TSMixer-style block), both using learnable spline-based edge functions in place of fixed nonlinearities** — two separate architectures, both genuinely new (2025-era) and unproven at scale; same parked tier as GNN, logged pending independent MI/backtest evidence rather than funded a Stage-1 slot.

- **classic univariate statistical models** — rejected as primary candidates, not measured: these are single-series univariate-by-design and don't naturally extend to this doc's multi-tf, multivariate feature schema (relative-HLC/volume/top-distance channels per candle, see [candle feature schema](02-Data, Label & Feature Engineering.md#candle-feature-schema)) without discarding most of it back down to a single price series — would need reinventing as a per-feature ensemble to be a fair comparison. This rejection reason applies identically to all of them, so they're grouped by implementation family rather than scored separately: **ARIMA/SARIMA** (SARIMA is ARIMA's own seasonal extension, `statsmodels.tsa.SARIMAX` — a parameter choice, not a separate mechanism), **exponential smoothing/ETS**, and **GARCH** (a separately-implemented volatility model, not a point-forecaster — its plausible role, if any, is a risk/position-sizing feature input, not a Stage-1 backbone). The naive/persistence and all-MLP-mixer (DLinear) floors above already serve the "is the DL machinery earning its complexity" sanity-check role these would otherwise fill.
- **1-nearest-neighbor w/ DTW distance** — classic, parameter-light time-series-classification baseline (near-mandatory floor in academic TSC benchmarks). Ties to the DTW-preprocessing fallback already named under [multi-timeframe fusion](#multi-timeframe-fusion) → "pattern speed-invariance," but as a full pattern-matching classifier rather than a preprocessing step — worth adding as a floor if that DTW fallback ever gets built anyway, deferred until then rather than a separately-funded Stage-1 slot.

#### activation mechanisms — testing strategy

- candidates: ReLU (cheap baseline), GELU (Transformer-standard, smoother gradient), SiLU/Swish (used in Mamba/S4-style gating and modern conv nets), Mish (occasional marginal gains, costlier), GLU-family gating (GEGLU/SwiGLU — gates the feedforward block itself, common in modern Transformer variants) — applies to whichever Stage-1 architecture wins, not a separate architecture axis.
- **test scope:** swap within a fixed architecture/profile (the Stage-1 categorical-search winner), not folded into the primary search — activation choice is a cheap post-hoc refinement.
- **method:** same seed-count/backtested-KPI discipline as other hparam tests (≥3 seeds, paired stat test across matched folds, per [statistical validity of comparisons](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons)) — not a train-loss-only comparison.
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
  - **EffiCANet-style conv+attention fusion** — combines local conv feature extraction with attention over the conv output _inside one block_, rather than this doc's current sequential CNN→Transformer staging. A concrete, marginal-value variant of this same "single hybrid backend" idea; lower priority than the ModernTCN/TSMixer additions under "current Stage-1 candidate set" since it refines an already-covered candidate rather than adding a new capability.

**status:** unresolved, not yet measured. Default assumption = single-backend-wins (cheapest, current doc's implicit baseline); any other strategy adopted only on measured evidence it beats that baseline on backtested KPIs, per [core principle: error metric ≠ trading objective](04-Experimentation, Evaluation & Optimization.md#core-principle-error-metric--trading-objective).

#### fusion mechanism

How to combine multiple representations, once a combination strategy other than single-backend-wins actually requires fusing them (the architectural-fusion / hybrid multi-encoder case above, and the "multi-tf fusion" pipeline stage):

- concatenation + MLP projection — cheapest, no learned interaction beyond the MLP.
- weighted sum / learned scalar gate per source — cheap, assumes representations already share a compatible space.
- cross-attention fusion — one source's representation attends over another's; most expressive, most expensive.
- gated fusion (GLU-style gate deciding per-dimension how much of each source to keep) — a middle ground between weighted-sum and cross-attention.
  Alt: no explicit fusion mechanism needed for single-backend-wins or late-ensembling — late-ensembling fuses only at the final prediction (average/weighted-vote/stacked meta-learner), not at representation level; see "combination strategy" above.

This same menu applies to both axes: fusing across backend **types** (this section) and fusing across **timeframes** ("multi-timeframe fusion" below, "multi-tf combination approach" bullet) — same concat/weighted-sum/cross-attention/gated choice either way.

## Multi-timeframe fusion

- each series 256 candles
- from anchor backwards: first 6 of tf-ordered-list (5min–1W) — the actual input series; 1M/4M/1Y are peak/valley-confirmation-only, per [candle feature schema](02-Data, Label & Feature Engineering.md#candle-feature-schema)
- overlap: ≤1 higher-tf candle may overlap a lower-tf series.
- domain assumption: pattern meaning is scale-invariant across tf (15min compress-price-pattern ≈ 1H compress-price-pattern); combining tfs clarifies the "real truth" behind any one tf's pattern.
- **multi-tf combination approach:** per-tf encoders (small TCN/Transformer per series) → concat/pool → shared cross-tf fusion block (small Transformer over pooled reps, or concat+MLP as cheaper baseline). Lower effort than full cross-attention over the concatenated sequence; natural first arch to profile before the pricier full-attention option. Per the timeframe-in-minutes resolution in [candle feature schema](02-Data, Label & Feature Engineering.md#candle-feature-schema), this per-tf-branch design drops that field entirely (branch identity already tells the encoder the tf); it's only added back if the arch choice switches to the flat/shared-encoder option below.
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

## Auxiliary tabular models (GBM-family)

Scoped roles for GBMs and related tabular models in this pipeline, distinct from the primary sequence-model architecture search above. GBMs operate on a flat, point-in-time feature vector per example — no notion of "this candle came before that one" beyond whatever's hand-engineered into lagged features — so they're deliberately scoped to screening/meta-labeling/floor/proxy roles, never the primary architecture search covered under "model architecture & selection".

- **candidate-feature screening tiebreaker** — already resolved, see "candidate-feature screening — method": MI screen first (cheap, no GPU), small LightGBM/XGBoost only if MI is ambiguous, full DL run reserved for candidates passing both. Correct scoped use because the question at that stage is "does this one feature carry signal at all," not "does the sequence pattern matter."
- **meta-labeling classifier** (López de Prado framework) — the canonical GBM use case for this kind of problem. Primary model (this doc's DL architecture) proposes a side (Long/Short) and size; a secondary GBM classifier answers a narrower question, "given this signal fired, should I actually take it," trained on a much smaller, better-balanced label set than the primary model's full action space — e.g. a binary "is `OM` above threshold" derived from [training-data.md § TP / MAE / OM labels](02-Data, Label & Feature Engineering.md#tp--mae--om-labels), since meta-labeling is a flat point-in-time binary classification problem, not a sequence problem.
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
  Alt: adopt CatBoost/TabPFN by benchmark reputation alone, skip local evaluation — rejected, same reasoning as everywhere else in this doc: measured backtested-KPI evidence only, per [core principle: error metric ≠ trading objective](04-Experimentation, Evaluation & Optimization.md#core-principle-error-metric--trading-objective).

### uncertainty-native GBM variants — confidence-metric gap

Candidate techniques for the confidence-metric gap — see [error rating & model evaluation § confidence & calibration metrics](04-Experimentation, Evaluation & Optimization.md#confidence--calibration-metrics) for the problem statement and what's measured; this section owns _which technique_ produces it. This family produces calibrated uncertainty as a mechanism of the model itself, not by engineering extra confidence-specific input features.

- **quantile GBM** (native quantile/pinball objective in LightGBM/XGBoost) — try first: reuses the pinball-loss work already planned for price-level heads (see [per-head statistical metrics](04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics)), and gives prediction intervals (e.g. 10th/50th/90th percentile TP) rather than a single point number. Cheapest option in this group, reuses infrastructure already planned.
- **NGBoost** — a genuinely different mechanism than standard GBM: instead of gradient-boosting toward a point estimate, it boosts the parameters of an entire output distribution using natural gradients, so every prediction carries a full predictive distribution rather than a point forecast plus a bolted-on score. Base-learner-agnostic (works with tree learners) — a small step from the existing GBM screening code, not a rebuild. Clean fit for meta-labeling (calibrated P(TP-hit) instead of a bare 0/1) or for the TP/MAE/OM heads directly.
- **quantile regression forests / newer multivariate extensions** (e.g. Tomographic Quantile Forests) — same idea, extended to jointly-calibrated uncertainty across correlated outputs, relevant if TP and SL should have correlated (not independently-estimated) uncertainty bands rather than separately-quantiled heads that might contradict each other.
  Alt:
  - hybrid GBM + TabPFN/LLM-boosted ensembles — recent work explicitly boosts GBDTs with TabPFN or LLM components as additional weak learners rather than treating tree-based and foundation-model approaches as competitors; best-of-both efficiency + foundation-model signal, but a more involved implementation than swapping libraries — ranked below the uncertainty-native options for near-term priority.
  - retrieval-augmented in-context tabular learning (TabR-style: at inference, look up similar historical rows via nearest-neighbor search in a learned embedding space, condition the prediction on them) — conceptually close to what the zigzag/nearest-top-distance features already hand-engineer (see "candle feature schema"), but research-stage, not yet a mature drop-in library the way NGBoost/quantile-LightGBM are; deferred.

**priority, given this doc's cheap-proxy-before-expensive-run discipline:** quantile GBM first (cheapest, reuses planned pinball-loss work, native in libraries already in scope) → NGBoost second (bigger uncertainty-quantification payoff, still a straightforward library addition) → hybrid/retrieval approaches deferred, same "parked, not funded a slot" tier as the LLM-reprogramming and GNN entries, pending evidence from the cheaper options first.

**relationship to the primary DL head:** the same mean/std/skew/kurtosis-parametric idea (NGBoost's mechanism) applies directly to the primary sequence model's `MFE`/`MAE` heads too, not only this GBM-auxiliary role — see [training-data.md § model output targets](02-Data, Label & Feature Engineering.md#model-output-targets) for that (larger) alternative. If adopted there, TP/SL probability/risk estimates derive from the primary model's own fitted distribution, which may make a separate GBM-based confidence mechanism unnecessary.

## Stage-1 candidate sets

Full architectural detail for candidate models: the hardware/memory budget every candidate design must fit, the design checklist each candidate write-up must satisfy, pseudocode-level block design for each Stage-1 single-backend candidate, the same detail for **combined/super models** (multi-backend fusion, ensembling, MoE, distillation), full S1/S2/S3 hyperparameter profiles, the search-space-bounds methodology, and the cross-architecture-fairness protocol. The main doc's [top level architecture](#top-level-architecture) → "current Stage-1 candidate set" keeps just the option list and links here for detail.

### hardware constraints

This doc is where every candidate/combined-model design has to actually fit, so the budget lives next to what it constrains rather than in the higher-level planning doc.

- RTX 4060 Laptop GPU, 8GB VRAM (8188 MiB per `nvidia-smi`), 64GB RAM, 2 SSD/HDD.

- **max feasible model size**
  - don't hand-calculate — `profile_trial_cost()`/`estimate_total_budget()`/`max_trials_for_budget()`
  - measure real wall-clock+VRAM on this exact card.
    - Rough prior only: 4tf×256candles×few scalars is a modest seq length; 8GB should fit small/med Transformer/TCN at batch 16–64 w/ mixed precision; VRAM more likely bound by hidden-dim/full-attention-over-concat-seq than seq length itself. If full cross-attention doesn't fit, caps toward cheaper fusion alternatives (per-tf encoders + light fusion) — see [multi-timeframe fusion](#multi-timeframe-fusion).
  - Alt:
    - gradient checkpointing — fallback if needed, slows training
    - mixed precision AMP — near-free win, enable by default, not as fallback
    - FlashAttention — same "near-free win, on by default" category as mixed-precision AMP above; fuses the softmax computation so the full attention matrix is never materialized, applies under whichever attention variant is chosen (see [attention / dependency](#attention--dependency)), not an approximation or a separate arm of the search
    - GQA/MQA or MLA on the attention stage — targeted KV-cache/activation reduction, the more surgical lever if profiling shows attention-over-concat-seq specifically (not param count) is the binding VRAM constraint; see [attention / dependency](#attention--dependency) for the head-to-head comparison
    - gradient accumulation — fallback if batch-size-bound
    - cloud/rented GPU — rejected, conflicts w/ local-only decision; revisit only if hard-bottlenecked
    - model parallelism — n/a, single GPU

- **every candidate/combined design in this doc must state a total-parameter estimate against this budget** — see [design checklist](#per-candidate-requirements) below. Worked illustrative example (not a measured number, order-of-magnitude only): Transformer S2 (d_model=384, num_heads=8, num_encoder_layers=2, d_ff=1024) ≈ 4·d_model² (Q/K/V/O) + 2·d_model·d_ff (FFN) per layer ≈ 1.37M params/layer × 2 layers ≈ 2.7M params. Even w/ Adam's 2 extra moment buffers + gradients (~4× params, fp32) that's ~40MB — negligible against an 8GB card. This confirms the existing "batch size, not param count, is the primary lever" framing above: **every S1/S2/S3 profile in [architecture candidates](#architecture-candidates) is order ~0.1M–5M params, nowhere near the binding constraint** — activation memory (attention over the concatenated multi-tf sequence, batch size) is what actually caps size, not parameter count. Flag here, not re-derived per candidate below.

### vram/ram budget split

Rough prior, not yet profiler-confirmed — confirm via `profile_trial_cost()` alongside the hyperparam-bounds profiling pass, same as the section above. Nothing else in the repo sizes VRAM/RAM or specifies a pre-loading pipeline; the closest related content is `infrastructure.md`'s [Repository design pattern](../infrastructure.md#repository-design-pattern) (disk-level cache format/ownership for candle/indicator artifacts) and [pandas-ta](../infrastructure.md#pandas-ta)/[TensorFlow](../infrastructure.md#tensorflow) library choices — linked below, not duplicated.

- **VRAM (8GB card, 8188 MiB)**: ~5% CUDA/driver context reserve, ~40% model params+gradients+optimizer states, ~35% activations (batch fwd/bwd — dominated by attention over the concatenated multi-tf sequence for Transformer-like archs, not by param count — see the worked example above), ~10% input-batch staging (pinned-memory transfer buffer), ~10% fragmentation headroom.
- **RAM (64GB)**: ~10% OS/Python/dataloader-worker overhead, ~65% in-memory feature/candle cache (64GB is generous relative to the 8GB card, so the full or near-full multi-tf dataset can likely stay resident), ~15% pinned staging buffers for CPU→GPU transfer, ~10% headroom.
- **pre-loading / prefetch pipeline** — how the RAM-resident cache actually reaches the GPU without stalling it:
  - in-memory feature cache (the RAM 65% slice above) is populated once per training run from the on-disk artifact cache that `infrastructure.md`'s repository pattern already owns (parquet/npz, cached CSV/zip) — this doc only covers what happens after that disk read, not the disk cache format/ownership itself.
  - `tf.data.Dataset` pipeline (per [TensorFlow](../infrastructure.md#tensorflow) as the chosen framework — see [pseudocode convention](#pseudocode-convention)) reads windows from the RAM-resident cache, batches, and calls `.prefetch(tf.data.AUTOTUNE)` so the next batch is staged into the pinned transfer buffer (RAM's 15% slice) while the GPU is still computing the current step — overlaps CPU-side windowing/collation with GPU compute instead of serializing them.
  - `.cache()` on the `tf.data` pipeline is redundant with the already-resident in-memory feature cache above if the full dataset fits in the 65% RAM slice — only add it if profiling shows re-windowing cost (not the disk read) is itself a bottleneck.
  - Alt:
    - no explicit split, let the allocator decide dynamically — rejected as a sizing guide; describes runtime behavior but doesn't help set search-space bounds up front
    - cache nothing, stream from disk every epoch — rejected, wastes the 64GB RAM headroom and adds I/O bottleneck given SSD/HDD mix
    - memory-map the on-disk cache instead of loading fully into RAM — viable if the dataset ever exceeds the RAM budget; not needed yet given the 64GB headroom vs. current data volume

### design checklist

Start of "the rest of this doc" — a checklist to run against every candidate/combined-model design below (and any new one added later) so designs stay consistent, complete, and don't silently skip a concern. Two parts: what each write-up must _state_, and what layers of design it must _pass through_ to get there.

#### per-candidate requirements

Every architecture write-up in [architecture candidates](#architecture-candidates) or [combination strategies](#combination-strategies-combinedsuper-models) must specify:

- **input/output contract** — input shape per tf branch (seq_len × n_features, per [candle feature schema](02-Data, Label & Feature Engineering.md#candle-feature-schema)), output heads and their shapes (per [training-data.md § model output targets](02-Data, Label & Feature Engineering.md#model-output-targets)).
- **stage-by-stage block list** — which [pipeline stage](#top-level-architecture) each layer belongs to (embedding/local-extraction/sequential/attention/fusion/global-repr/heads), expressed as a `stage_config` per [unified super-architecture skeleton](#unified-super-architecture-skeleton) — including which stages are zeroed (skipped) for this candidate.
- **hyperparam profile table** — S1/S2/S3 (depth-heavy/width-heavy/context-heavy), per [architecture candidates](#architecture-candidates).
- **param-count order-of-magnitude estimate** — against the [hardware constraints](#hardware-constraints) budget; flag if it's not obviously negligible like the worked example there.
- **pseudocode** — for the block(s) that make this candidate distinct from a plain pass-through of the skeleton, per [pseudocode convention](#pseudocode-convention). Reuse the skeleton's stage functions rather than re-deriving; only write out what's new.
- **rejected alternatives** — `Alt:` list, same convention as the rest of this doc/`model-architecture-planning.md`, so "why not X" is answered inline instead of re-litigated later.

#### design layers to pass

Concerns to verify, roughly in dependency order — a candidate isn't "designed" until all of these are addressed (even if the answer is "inherits from the skeleton, nothing candidate-specific here"):

1. **data/interface layer** — input shape and feature schema match [candle feature schema](02-Data, Label & Feature Engineering.md#candle-feature-schema); output heads match [training-data.md § model output targets](02-Data, Label & Feature Engineering.md#model-output-targets).
2. **representation layer** — the stage-by-stage block choice itself (embedding → local-extraction → sequential → attention → fusion → global-repr), per [top level architecture](#top-level-architecture).
3. **capacity/sizing layer** — param count vs. [hardware constraints](#hardware-constraints); activation memory (batch × seq × d_model) vs. the [vram/ram budget split](#vramram-budget-split), since that's the actual binding constraint, not param count.
4. **regularization layer** — dropout, weight decay, normalization placement (pre-/post-norm) — held constant across S1/S2/S3 per the existing convention (see [architecture candidates](#architecture-candidates) intro), stated explicitly per candidate anyway so it isn't silently assumed.
5. **training-dynamics layer** — gradient flow (residual connections around any new block), mixed-precision (AMP) compatibility, checkpointing compatibility — per [hardware constraints](#hardware-constraints) → Alt list.
6. **evaluation-interface layer** — loss/metric hookup per head, per [error rating & model evaluation](04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics) — not re-derived here, just confirmed wired correctly for this candidate's head shapes.
7. **combination layer** (only for [combined/super models](#combination-strategies-combinedsuper-models)) — fusion mechanism (per [fusion mechanism](#fusion-mechanism)) + stage-slot placement config, since a combined model is this same checklist applied once per constituent backend plus one more pass for how they're fused.

### pseudocode convention

Every candidate below is described with **Keras (`tf.keras`) functional-API-style pseudocode** — shape-annotated layer composition (`y = Layer(...)(x)  # shape: (...)`), not full runnable code.

Rationale, briefly, since this is worth remembering rather than re-deciding later: [infrastructure.md](../infrastructure.md#tensorflow) already locks TensorFlow as this repo's DL framework (`tensorflow[and-cuda]`, Docker base `tensorflow:25.01-tf2-py3`) for the existing CNN-LSTM-attention models. Keras is TensorFlow's own high-level API, so pseudocode written this way is close to directly transcribable into real code later, not a translation exercise from a different framework's idioms. It's also simpler to read as architecture-shape sudo-code than raw PyTorch `nn.Module`/`forward()` boilerplate (explicit `__init__`/`forward` split, manual shape bookkeeping) — Keras's functional style (`y = Layer(config)(x)`) reads closer to the stage-pipeline diagrams already used in `model-architecture-planning.md`.

Convention used throughout:

- `# shape: (B, T, F)` comments track tensor shape through the pipeline (batch, sequence, feature-dim).
- hyperparameter names match the S1/S2/S3 tables below exactly (`d_model`, `num_heads`, `hidden_channels`, ...) so a profile row can be read directly into the pseudocode's function signature.
- a stage function returning its input unchanged (`kind == 0`) represents that stage being **zeroed** — see [unified super-architecture skeleton](#unified-super-architecture-skeleton).

### unified super-architecture skeleton

For the "which combination of stages, in which placement (before/after, start/middle/end)" question: rather than designing a separate architecture per placement variant, this doc designs **one maximally-complex skeleton** containing every pipeline stage as a slot, and tests placement by **zeroing** (disabling) or **numbering** (selecting which block type occupies) each slot — not by drawing a new diagram per combination. This is also literally what a "combined/super model" is in this doc: the skeleton itself, parameterized by `stage_config`.

- the skeleton's stage **order is fixed** and maps directly onto start/middle/end: embedding = start, local-extraction → sequential → attention → fusion = middle, global-representation → heads = end. This fixed order already matches the one resolved case in `model-architecture-planning.md` (conv-then-transformer in the hybrid CNN→Transformer candidate, per [local feature extraction](#local-feature-extraction)) — so "placement" here means _which slots are active_, not reordering the skeleton itself.
- `stage_config[stage] = 0` → that stage is `tf.identity` (skipped entirely) — the "zeroing" test, e.g. does attention earn its cost over conv-only.
- `stage_config[stage] = <block name>` → the "numbering" test — which block implementation occupies that fixed slot, e.g. `attention: "self_attn"` vs `"informer"` vs `"itransformer"` (per [attention / dependency](#attention--dependency)).
- reordering the skeleton itself (e.g. attention _before_ local-extraction instead of after) is a genuinely different, larger search — that's what [differentiable/block-level NAS](#combination-strategy) would search over; out of scope for this fixed-skeleton default, deferred same as NAS is deferred in the main doc.

```python
def build_super_architecture(stage_config: dict, profile: str, tf_list: list[str]):
    # stage_config values: 0 = identity (stage disabled/zeroed).
    #                      non-zero = which block type occupies this stage's slot.
    per_tf_branches = []
    for tf_name in tf_list:                                        # start: one branch per input timeframe
        x = Input(shape=(seq_len, n_features), name=f"{tf_name}_in")     # shape: (B, T, F)

        x = embed(x, kind=stage_config["embedding"], profile=profile)          # shape: (B, T, d_model)
        x = local_extract(x, kind=stage_config["local_extraction"], profile=profile)  # 0 = identity
        x = sequential_encode(x, kind=stage_config["sequential"], profile=profile)    # 0 = identity
        x = attend(x, kind=stage_config["attention"], profile=profile)                # 0 = identity

        per_tf_branches.append(x)

    fused = fuse(per_tf_branches, kind=stage_config["fusion"])            # middle/end boundary — shape: (B, T', d_model) or (B, d_model)
    pooled = global_repr(fused, kind=stage_config["global_repr"])         # shape: (B, d_model)
    outputs = {name: head(pooled, kind=cfg) for name, cfg in stage_config["heads"].items()}  # end
    return Model(inputs=[b.input for b in per_tf_branches], outputs=outputs)
```

Each `stage_config` dict below is a complete "placement" test: which slots are zeroed, which block occupies the rest. No separate architecture diagram needed per placement — only a different `stage_config`.

### architecture candidates

hyperparam sets below are illustrative starting candidates for `profile_trial_cost()` to evaluate, not fixed a priori — see [search-space bounds](04-Experimentation, Evaluation & Optimization.md#hyperparam-search-space-bounds). Profiles S1/S2/S3 are roughly parameter/compute-matched _within_ each architecture — same capacity weight, different capability focus — not a size ladder: **S1 = depth-heavy, S2 = width-heavy, S3 = context-heavy** (max attention span/receptive field/state capacity, arch-dependent). dropout is a separate regularization knob, held constant across profiles. Sizing follows the [vram/ram budget split](#vramram-budget-split) above; batch size, not param count, is the primary lever for using available VRAM, per [max feasible model size](#hardware-constraints).

Per the [design checklist](#per-candidate-requirements), each candidate below states its `stage_config` (relative to the skeleton above) before its hyperparam profile, plus a short pseudocode block only for what's distinct from a plain skeleton pass-through.

- **CNN-LSTM(-attention)** — the pre-existing baseline, not a new proposal; see [current Stage-1 candidate set](#current-stage-1-candidate-set). Included here mainly as a worked illustration of the skeleton itself: the two variants that already exist in code — [cnn_lstm_model.py](../../app/ai_modelling/cnn_lstm/cnn_lstm_model.py) and [cnn_lstm_attention_model.py](../../app/ai_modelling/cnn_lstm_attention/cnn_lstm_attention_model.py) — differ from each other by exactly one `stage_config` value (`attention: 0` vs `attention: "self_attn"`), which is the zeroing mechanism working as designed, not a coincidence.
  - `stage_config`: `{embedding: 0, local_extraction: "plain_cnn", sequential: "rnn", attention: 0 | "self_attn", fusion: "concat_mlp", global_repr: "pool"}` — `embedding: 0` because the existing code feeds raw per-candle features straight into the conv stack, no separate linear-projection stage.
  - distinguishing block (plain, non-causal, non-dilated conv — the base option the TCN/ModernTCN candidates below build on top of):

    ```python
    def local_extract_plain_cnn(x, cnn_count, base_filters, base_kernel_size, dropout):
        for i in range(cnn_count):
            x = Conv1D(base_filters * (i + 1), base_kernel_size + i, padding="same", activation="relu")(x)  # shape: (B, T, filters_i)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
        return x
    ```

  - per-branch tail (both variants): `sequential_encode_rnn()` (LSTM stack, `return_sequences=True`) → optionally `attend_self_attn()` if `attention != 0` → `BatchNormalization` → `GlobalAveragePooling1D` → `Dense(64)` → `Dense(128)`; branches `fuse(kind="concat_mlp")` → `Dense(256)` + `LeakyReLU` → per-head `Dense`.
  - S1/S2/S3 profiles not yet assigned — backfill via `profile_trial_cost()` same as the others before including it in the Optuna study, not hand-picked.
    Alt:
  - **residual-CNN time-series-classification baselines** as the `local_extraction` block instead of the plain stacked-conv above — same slot, three separate implementations, not one: **ResNet** (residual skip connections), **FCN** (global-pooled fully-conv stack, no residual), **InceptionTime** (multi-kernel-size Inception modules); see [local feature extraction](#local-feature-extraction) and the scoring in [prioritization framework](04-Experimentation, Evaluation & Optimization.md#local-feature-extraction). None yet reduced to a `stage_config`-ready block here — flagged as a follow-up, not designed in this pass.
  - **ConvLSTM** in place of `sequential: "rnn"` — convolutional gates inside the recurrent cell itself, a different mechanism than conv-then-LSTM stacking; see [sequential encoding](#sequential-encoding). Same follow-up status as the residual-CNN alt above.

- **Transformer w/ per-tf embedding + cross-tf attention**
  - `stage_config`: `{embedding: "linear", local_extraction: 0, sequential: 0, attention: "self_attn", fusion: "cross_attn", global_repr: "pool"}`
  - distinguishing block:

    ```python
    def attend_self_attn(x, d_model, num_heads, num_encoder_layers, d_ff, dropout):
        for _ in range(num_encoder_layers):
            attn_out = MultiHeadAttention(num_heads, d_model // num_heads)(x, x)   # shape: (B, T, d_model)
            x = LayerNorm()(x + Dropout(dropout)(attn_out))
            ff = Dense(d_ff, activation="gelu")(x)
            ff = Dense(d_model)(ff)
            x = LayerNorm()(x + Dropout(dropout)(ff))
        return x
    ```

  - d_model: S1:160, S2:384, S3:256
  - num_heads: S1:4, S2:8, S3:16
  - num_encoder_layers: S1:8, S2:2, S3:4
  - d_ff (feedforward dim): S1:640, S2:1024, S3:768
  - seq_len per tf (capped 256/tf): 256 (all profiles)
  - dropout: 0.1 (all profiles)

- **TCN** — cheaper, dilated convs for multi-scale, good single-GPU baseline
  - `stage_config`: `{embedding: "linear", local_extraction: "tcn_dilated", sequential: 0, attention: 0, fusion: "concat_mlp", global_repr: "pool"}`
  - distinguishing block:

    ```python
    def local_extract_tcn(x, hidden_channels, kernel_size, num_dilated_levels, dropout):
        for level in range(num_dilated_levels):
            dilation = 2 ** level
            residual = x
            y = Conv1D(hidden_channels, kernel_size, dilation_rate=dilation,
                       padding="causal", activation="relu")(x)              # shape: (B, T, hidden_channels)
            y = Dropout(dropout)(y)
            x = residual + y if residual.shape[-1] == hidden_channels else Conv1D(hidden_channels, 1)(residual) + y
        return x
    ```

  - hidden_channels: S1:40, S2:96, S3:56
  - kernel_size: S1:3, S2:3, S3:9
  - num_dilated_levels: S1:10, S2:3, S3:6
  - dropout: 0.1 (all profiles)

- **hybrid CNN→Transformer**
  - `stage_config`: `{embedding: "linear", local_extraction: "conv_stem", sequential: 0, attention: "self_attn", fusion: "concat_mlp", global_repr: "pool"}` — conv-then-attention is the skeleton's fixed order, so this candidate is literally `local_extract_tcn()` (as a short conv stem, `num_conv_layers` instead of `num_dilated_levels`, no dilation growth, optional stride-2 pooling to shorten `T` before attention — the VRAM-cost lever noted under [hardware constraints](#hardware-constraints)) feeding directly into `attend_self_attn()` above; no new block needed, just both non-zero.
  - conv_channels: S1:48, S2:112, S3:64
  - conv_kernel_size: S1:3, S2:3, S3:9
  - num_conv_layers: S1:4, S2:2, S3:2
  - transformer d_model: S1:160, S2:320, S3:224
  - transformer num_heads: S1:4, S2:8, S3:8
  - transformer num_layers: S1:4, S2:2, S3:3
- **state-space: Mamba** — cheap long-seq alt to attention. (**S4** is a lower-priority alt within this same `sequential: "ssm"` slot — swap `MambaBlock` below for an `S4Block`-equivalent; a separate implementation, not a parameter of `MambaBlock`, per [sequential encoding](#sequential-encoding) and the scoring in [prioritization framework](04-Experimentation, Evaluation & Optimization.md#sequential-encoding).)
  - `stage_config`: `{embedding: "linear", local_extraction: 0, sequential: "ssm", attention: 0, fusion: "concat_mlp", global_repr: "pool"}`
  - distinguishing block:

    ```python
    def sequential_encode_ssm(x, d_model, d_state, num_layers, conv_kernel):
        for _ in range(num_layers):
            x = MambaBlock(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel)(x)  # shape preserved: (B, T, d_model)
        return x
    ```

    `MambaBlock` isn't a native `tf.keras` layer — needs a third-party/custom selective-scan implementation; flagged here as a build dependency, not assumed available.

  - d_model: S1:128, S2:320, S3:192
  - d_state: S1:16, S2:16, S3:64
  - num_layers: S1:8, S2:2, S3:4
  - conv_kernel (local conv width): 4 (all profiles)

- **LSTM** — sanity-check floor (S3 focus = bidirectional context, not receptive field — RNN context is sequential, not attention-based)
  - `stage_config`: `{embedding: "linear", local_extraction: 0, sequential: "rnn", attention: 0, fusion: "concat_mlp", global_repr: "last_token"}`
  - distinguishing block:

    ```python
    def sequential_encode_rnn(x, hidden_size, num_layers, bidirectional, dropout, cell_type="lstm"):
        cell_cls = LSTM if cell_type == "lstm" else GRU  # GRU is a separate tf.keras layer class, not an LSTM param — see prioritization-framework.md's tool-identity test
        for _ in range(num_layers):
            rnn = cell_cls(hidden_size, return_sequences=True, dropout=dropout)
            x = Bidirectional(rnn)(x) if bidirectional else rnn(x)   # shape: (B, T, hidden_size [*2 if bidirectional])
        return x
    ```

  - hidden_size: S1:96, S2:320, S3:176
  - num_layers: S1:4, S2:1, S3:2
  - bidirectional: S1:false, S2:false, S3:true
  - dropout: 0.1 (all profiles)
  - **GRU** (`cell_type="gru"` above) — Tier-2 alt tested within this same floor role, not a separate candidate slot; see [sequential encoding](#sequential-encoding) and the scoring in [prioritization framework](04-Experimentation, Evaluation & Optimization.md#sequential-encoding).
    Alt:
  - **naive/persistence baseline** — not a `stage_config` at all, no learned stages; "no change"/carry-forward the last signal. The floor beneath this floor — computed alongside backtested KPIs for every run, not a Stage-1 categorical option; see [current Stage-1 candidate set](#current-stage-1-candidate-set).
  - pure MLP on flattened features — rejected as serious candidate, discards seq structure; trivial baseline only
  - GBM (LightGBM, XGBoost, CatBoost — three separate library classes) on flattened features — kept as a cheap non-sequence floor, distinct from the LSTM floor above (which still respects sequence order); see [auxiliary tabular models (GBM-family)](#auxiliary-tabular-models-gbm-family) for what this comparison is meant to answer
  - Random Forest on flattened features — same scoped role as GBM above, bagging instead of boosting; see [modern GBM-family alternatives](#modern-gbm-family-alternatives)
  - 1-nearest-neighbor w/ DTW distance — classic parameter-light TSC floor; see [current Stage-1 candidate set](#current-stage-1-candidate-set) Alt list
  - 4 separate per-tf models + late ensembling — kept as cheap baseline, see [multi-timeframe fusion](#multi-timeframe-fusion) and [combination strategies](#combination-strategies-combinedsuper-models) below
  - GNN over tf/symbol nodes — deferred, no evidence needed yet
  - **TFT (Temporal Fusion Transformer)** and **Perceiver** — named as additional Stage-1 categorical options in the main doc's [combination strategy](#combination-strategy) section; not yet added with S1/S2/S3 hyperparam profiles here — pending the same profiling treatment as the candidates above. Two separate stage-level compositions, each expressible via the skeleton (TFT ≈ per-feature gating + LSTM sequential stage + interpretable multi-head attention stage; Perceiver ≈ cross-attention fusion stage over a fixed-size latent array instead of concat/cross-tf attention) rather than needing a new skeleton — see the scoring in [prioritization framework](04-Experimentation, Evaluation & Optimization.md#current-stage-1-candidate-set) for why they're evaluated independently rather than as one row.

**shared fusion/global-repr/head building blocks** (referenced by `stage_config` above, not repeated per candidate):

```python
def fuse(branches, kind, d_model):
    if kind == "concat_mlp":
        x = Concatenate()(branches)                          # shape: (B, T, d_model * n_tf)
        return Dense(d_model, activation="gelu")(x)           # shape: (B, T, d_model)
    if kind == "cross_attn":
        # each per-tf branch attends over the concat of all others — see fusion mechanism menu
        return CrossAttentionFusion(d_model)(branches)
    if kind == "gated":
        return GatedFusion(d_model)(branches)                 # GLU-style per-dim gate, per fusion mechanism menu
    raise ValueError(kind)

def global_repr(x, kind):
    if kind == "pool":
        return GlobalAveragePooling1D()(x)                    # shape: (B, d_model)
    if kind == "last_token":
        return x[:, -1, :]                                    # shape: (B, d_model)
    raise ValueError(kind)

def head(x, kind):
    # action / MAE-OM regression / confidence heads — see training-data.md § model output targets
    return Dense(kind["units"], activation=kind.get("activation"))(x)
```

See [fusion mechanism](#fusion-mechanism) for the rationale/tradeoffs behind each fusion kind — not re-derived here, this section only pins the pseudocode.

### combination strategies (combined/super models)

Pseudocode for the "combined/super model" strategies named in the main doc's [combination strategy](#combination-strategy) section — status there is **unresolved, not yet measured, default = single-backend-wins**; this section only adds the implementation-level design (per the [design checklist](#design-checklist)) for when/if one of these is tested, not a re-argument of the rationale or priority (see the main doc for that).

- **late ensembling** — N independently-trained `build_super_architecture()` instances, no shared gradient path:

  ```python
  def late_ensemble(stage_configs: list[dict], profile: str, weights=None):
      models = [build_super_architecture(cfg, profile, tf_list) for cfg in stage_configs]  # trained separately, frozen
      def predict(x):
          preds = [m(x) for m in models]
          return weighted_average(preds, weights)   # or a stacked meta-learner on top of frozen preds
      return predict
  ```

- **mixture-of-experts (MoE) gating** — a learned gate routes per-example, conditional compute instead of always-on fusion:

  ```python
  def build_moe(expert_configs: list[dict], profile: str, tf_list: list[str]):
      experts = [build_super_architecture(cfg, profile, tf_list) for cfg in expert_configs]
      gate = Dense(len(expert_configs), activation="softmax")             # shape: (B, num_experts)
      def forward(x):
          gate_weights = gate(GlobalAveragePooling1D()(embed(x, "linear", profile)))
          expert_outs = tf.stack([e(x) for e in experts], axis=1)          # shape: (B, num_experts, ...)
          return tf.reduce_sum(gate_weights[..., None] * expert_outs, axis=1)
      return forward
  ```

- **knowledge distillation** — N teachers → one compact deployable student:

  ```python
  def distill(teacher_configs: list[dict], student_config: dict, profile: str, tf_list: list[str], distill_weight: float):
      teachers = [build_super_architecture(cfg, profile, tf_list) for cfg in teacher_configs]  # trained to convergence, frozen
      student = build_super_architecture(student_config, profile, tf_list)
      def loss(x, y_true):
          soft_targets = weighted_average([t(x) for t in teachers])
          return task_loss(student(x), y_true) + distill_weight * kd_loss(student(x), soft_targets)
      return student, loss
  ```

- **differentiable/block-level NAS (DARTS-style)** — no separate pseudocode here: this is exactly the [unified super-architecture skeleton](#unified-super-architecture-skeleton)'s `stage_config`, but with each slot's choice made a differentiable softmax mixture over block options during search, discretized to a hard `stage_config` after convergence, instead of hand-picked per the presets in [architecture candidates](#architecture-candidates). Deferred per the main doc; the skeleton above is already what a future NAS pass would search over, so adopting it later needs no new architecture design, only a search loop over the existing `stage_config` space.
- **single hybrid backend with block-level composability** — already covered: this is the hybrid CNN→Transformer candidate in [architecture candidates](#architecture-candidates), i.e. one `stage_config` with two non-zero middle slots, not a separate multi-encoder fusion case. Listed in the main doc only to flag the naming ambiguity — no separate design needed here.

## Architecture diagnosis, capacity and robustness

Companion to [prioritization framework](04-Experimentation, Evaluation & Optimization.md#decision-framework): that framework decides which candidates get compute (tiering, before training). This section covers what happens after a candidate is trained — matching mechanism to data characteristics before scoring, sizing capacity against dataset size, testing component independence, diagnosing underperformance, checking ranking robustness beyond one split, comparing candidates on measured cost rather than param count, and a rule for when a more-complex winner isn't worth adopting. Addresses [05 A11](05-Weakness Analysis.md).

### architecture-selection methodology

The [prioritization framework](04-Experimentation, Evaluation & Optimization.md#decision-framework)'s `domain_fit` factor already asks "does this candidate's mechanism target something this data actually exhibits" — this makes that mapping explicit and required before scoring, rather than left implicit in prose the way it currently is (e.g. ModernTCN's grouped-conv reasoning under [local feature extraction](#local-feature-extraction)).

**characteristic → mechanism mapping**, filled in per candidate before scoring:

| observed data characteristic (from [02](02-Data, Label & Feature Engineering.md)) | mechanism that targets it | current candidates |
| --- | --- | --- |
| long-range dependency, unknown window (target pattern anywhere in the fed sequence) | attention / state-space, not fixed pooling | Transformer, Mamba/S4, long-window-focus attention |
| multi-scale local pattern | dilated / multi-kernel conv | TCN, ModernTCN, InceptionTime |
| genuinely multivariate feature channels (relative-HLC, volume, top-distance, ATR) | cross-variable-aware mechanism, not per-channel/shared-filter conv | ModernTCN (grouped conv), iTransformer (inverted attention) |
| noisy OHLCV signal | mechanisms explicitly targeting signal/noise separation | Differential Attention |
| pattern speed/scale invariance | multi-dilation, no fixed receptive field | TCN, attention (see [multi-timeframe fusion](#multi-timeframe-fusion) → pattern speed-invariance) |

- A candidate with no row here isn't disqualified, but its `domain_fit` score should reflect that it's a generic capability bump rather than a targeted fit (see [scoring factors](04-Experimentation, Evaluation & Optimization.md#scoring-factors) factor 5).
  Alt: score novelty/benchmark reputation only, skip the explicit mapping — rejected; this is what happens informally per-bullet today and is what let this gap exist in the first place. The mapping forces a falsifiable hypothesis ("this should help because X") that [architecture failure diagnosis](#architecture-failure-diagnosis) below can then confirm or reject empirically instead of guessing post-hoc.

### capacity sizing

No formal depth/width-vs-dataset-size rule exists beyond "S1/S2/S3 are illustrative starting points, `profile_trial_cost()` measures real cost" ([architecture candidates](#architecture-candidates) intro) and the param-count-is-negligible finding under [hardware constraints](#hardware-constraints). That finding answers "does it fit the GPU," not "is it the right size for the data" — capacity can be VRAM-affordable and still over/underfit the ~1yr, cross-symbol training set.

- **capacity ladder, not just profile shape**: S1/S2/S3 vary depth/width/context shape at roughly matched capacity (see [architecture candidates](#architecture-candidates) intro). Add a capacity ladder within the winning shape — e.g. 0.5×/1×/2× width at fixed depth — to find where val-KPI plateaus or degrades. Finalists only, from the same budget reserve as [statistical validity](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons)'s multi-seed re-run — not run during the main search.
- **train/val gap as the empirical signal**: at each rung, track train-loss vs. val-KPI gap. Gap flat or shrinking as capacity grows → not yet over capacity. Gap growing while val-KPI stalls or worsens → that rung is excess capacity; the rung below is the ceiling for this dataset size.
- **directional prior, not a hard cap**: classic params-vs-N guidance (params ≲ O(N)) doesn't transfer cleanly to modern regularized DL, which routinely exceeds N params without overfitting — treat as a sanity flag (worth a second look if params/N is unusually large vs. the S1–S3 range already profiled), not a rejection rule.
  Alt: fixed capacity by literature convention (copy a published config) — rejected, no evidence that config matches this dataset's actual size or noise level.

### component-independence testing

Guards against crediting a combined candidate's win to "the architecture" when really one stage did the work and the rest are along for the ride (00-ToC §3.3).

- **the mechanism already exists, it just needs a protocol**: the [unified super-architecture skeleton](#unified-super-architecture-skeleton)'s `stage_config` zeroing is the tool. For any candidate with 2+ non-zero middle stages (the hybrid CNN→Transformer, any future combined/super model), zero each non-zero stage individually and compare **backtested KPI**, not train-loss, against the full candidate — reuse the [statistical validity](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons) ≥3-seed paired-test discipline per zeroing config, not one run each.
- **interaction check, when a component is only motivated by another already being present** (e.g. attention-after-conv-stem in the hybrid candidate, motivated by the conv stem shortening the effective sequence first — see [local feature extraction](#local-feature-extraction) "conv stem ahead of a Transformer"): test the suspected components in isolation too (attention alone, without the conv stem — not only single-zeroing from the full config). Targeted to stage pairs the [architecture-selection methodology](#architecture-selection-methodology) mapping explicitly motivated jointly, not full factorial over every stage combination — unaffordable within the single-GPU budget.
- **output**: a per-stage marginal-contribution table (one row per zeroing config, backtested-KPI delta vs. the full candidate ± CI) attached to any combined-candidate write-up before it's credited as beating the single-backend floor.
  Alt: judge the combined candidate's overall KPI alone, skip per-stage attribution — rejected, this is exactly how a dominant single stage gets mistaken for a validated combination.

### architecture failure diagnosis

When a candidate underperforms — loses to a simpler floor (naive, LSTM, GBM-on-flattened) or underperforms its literature reputation — work this checklist before discarding the candidate or the hypothesis behind it. Architecture-specific instance of 00-ToC §5.7's general "measure → identify cause → targeted retest" loop.

1. **insufficient capacity** — train and val loss both high, both plateaued early. → one rung up the [capacity sizing](#capacity-sizing) ladder, retest.
2. **excessive capacity** — train loss low, val loss high or diverging, gap grows with capacity. → one rung down, or confirm a smaller model matches current performance (feeds the [simplification rule](#simplification-rule)).
3. **inappropriate inductive bias** — capacity looks appropriate (train/val gap reasonable) but a structurally simpler floor still wins. → the mechanism may not match this data; revisit the candidate's row (or lack of one) in [architecture-selection methodology](#architecture-selection-methodology) rather than re-tuning hyperparameters. Most expensive item to accept — implies the candidate is structurally wrong, not just mistuned — so rule out 4–7 first.
4. **optimization difficulty** — loss curve unstable, non-monotonic, or NaN/Inf despite reasonable capacity/bias. → gradient flow (residual connections around new blocks, per [design layers to pass](#design-layers-to-pass) step 5), LR/warmup, mixed-precision numerics; ties to 00-ToC §4.6 training stability.
5. **bad input representation** — swap embedding/normalization (per [input / feature embedding](#input--feature-embedding)) and re-check ranking. If a different architecture wins under one embedding but not another, the failure may be representation-level, not architecture-level.
6. **bad labels** — cross-check against label-quality/noise measurement (00-ToC §2.3); a label-noise ceiling looks identical to "no architecture can fit this" from inside the architecture comparison alone.
7. **insufficient context** — shorten/lengthen the input window (00-ToC §2.8); performance may be context-length-bound, not architecture-bound.

Cheapest-first order: 5/6/7 (representation/labels/context) are cheap swaps reusing the existing architecture; 1/2 (capacity) is a hyperparameter resweep; 3 (inductive bias) — the actual "this architecture is wrong for this problem" conclusion — comes last, only once 4–7 are ruled out.

### cross-seed and cross-condition robustness

Distinct from [statistical validity of comparisons](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons)'s "≥3 seeds, paired test excluding zero": that establishes whether config A beats config B on one split. This asks whether the **ranking** across architectures holds beyond the split it was measured on.

- **protocol**: after [model-selection pipeline](04-Experimentation, Evaluation & Optimization.md#model-selection-pipeline) step 2 (finalists re-run across ≥3 seeds), re-run the top 2–3 finalists' seed comparison across one more axis of variation beyond seed — a second train-period slice (reuses the rolling-slice check named in [05 A17](05-Weakness Analysis.md), once built) or a second training-symbol subset — and confirm the **ranking**, not just each config's absolute KPI, holds.
- A ranking that flips under the second condition is not a resolved winner yet, even if each run individually cleared the paired-test bar on its own slice — treat as "insufficient evidence to pick a winner," not "pick whichever wins more slices" (that reintroduces the multiple-comparison problem [05 A14](05-Weakness Analysis.md) already flags).
- **budget**: 2–3 finalists × 1 extra axis, from the same "finalists post-search only" reserve already carved out under [statistical validity](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons) — not a second full search.
  Alt: trust the single-split ranking once it clears the paired-test bar — rejected as sole criterion, conflates "beats its peer on this slice" with "reliably beats its peer."

### param-count vs effective-capacity analysis

Extends the one-off worked example under [hardware constraints](#hardware-constraints) (Transformer S2 param math, ~2.7M params, negligible against 8GB) into a required per-candidate comparison, since that example's own conclusion is that param count isn't the binding constraint — activation memory is.

Required alongside the [design checklist](#per-candidate-requirements)'s existing "param-count order-of-magnitude estimate," all obtainable from the same `profile_trial_cost()` pass already run for [hyperparam search-space bounds](04-Experimentation, Evaluation & Optimization.md#hyperparam-search-space-bounds):

- **activation memory** — batch × seq × d_model, the term the hardware-constraints worked example already flags as dominant.
- **attention/sequence complexity, measured not assumed** — sub-quadratic candidates (Mamba/S4, Informer, linear attention) should report measured throughput at matched param count on this card, not just cite their asymptotic complexity class; an O(n) mechanism with a large constant factor can lose to O(n²) at this project's sequence length (≤256/tf × 6 tf branches).
- **wall-clock throughput** (examples/sec at the VRAM-fitting batch size) — the actual per-trial cost `estimate_total_budget()` needs.

Purpose: stop a param-count-cheap candidate from being credited with "efficiency" purely on asymptotic reputation if its measured profile on this hardware/sequence-length doesn't actually beat a param-count-larger candidate — asymptotic complexity is a prior for what to test, not a substitute for the profiler's number.
Alt: param count alone as the capacity/cost proxy — rejected, the hardware-constraints worked example already shows it's off by orders of magnitude from the actual binding constraint on this hardware.

### simplification rule

Formalizes what's already this doc's implicit default for backend combination ([combination strategy](#combination-strategy): "status: unresolved... default assumption = single-backend-wins... adopted only on measured evidence") into a general rule covering every capacity/component/combination decision, not just that one axis.

- **rule**: given two candidates whose backtested-KPI distributions are not distinguishable by the [statistical validity](04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons) paired test (CI includes zero), prefer the simpler one — fewer non-zero `stage_config` slots, lower param count, cheaper [combination strategy](#combination-strategy) tier (single-backend-wins < late-ensembling < MoE/distillation, ascending cost).
- **complexity must clear the noise floor, not just beat the mean** — a more complex candidate is accepted over a simpler one only if (a) it wins outside the paired-test CI, **and** (b) [component-independence testing](#component-independence-testing) shows the win isn't attributable to one dominant stage a simpler candidate could also carry (in which case, adopt that one stage into the simpler candidate instead of the whole combination).
- Applies at every layer this doc scores candidates: activation choice, capacity rung, attention mechanism, backend-combination strategy.
  Alt: prefer whichever candidate has the single highest point-estimate KPI, complexity untiebroken — rejected, this is the same backtest-overfitting risk [05 A14](05-Weakness Analysis.md) already flags: with enough candidates tested, some more-complex one wins by noise alone.

## Glossary

Shared terms (ATR, anchor candle, tf-ordered-list, tf, Stage-1, S1/S2/S3, GBM) live in [02-Data, Label & Feature Engineering.md § glossary](02-Data, Label & Feature Engineering.md#glossary).

- `stage_config` — per-candidate dict selecting, per pipeline stage, either `0` (zeroed/skipped) or a block-type name; the full parameterization of the [unified super-architecture skeleton](#unified-super-architecture-skeleton).
- super-architecture / combined model — this doc's single fixed-order skeleton with pluggable stage slots; "combined/super model" = any `stage_config` with more than one non-zero middle stage, or any strategy in [combination strategies](#combination-strategies-combinedsuper-models) composing multiple full skeleton instances.
- zeroing / numbering (placement) — the two ways `stage_config` values vary: zeroing tests whether a stage is needed at all (`0` vs. non-zero); numbering tests which block occupies an already-active slot.

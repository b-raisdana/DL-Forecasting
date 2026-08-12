# Model Architecture Candidate Sets

Full architectural detail for candidate models: the hardware/memory budget every candidate design must fit, the design checklist each candidate write-up must satisfy, pseudocode-level block design for each Stage-1 single-backend candidate, the same detail for **combined/super models** (multi-backend fusion, ensembling, MoE, distillation), full S1/S2/S3 hyperparameter profiles, the search-space-bounds methodology, and the cross-architecture-fairness protocol. The main doc's [top level architecture](model-architecture-planning.md#top-level-architecture) → "current Stage-1 candidate set" keeps just the option list and links here for detail.

- [Model Architecture Candidate Sets](#model-architecture-candidate-sets)
  - [hardware constraints](#hardware-constraints)
  - [vram/ram budget split](#vramram-budget-split)
  - [design checklist](#design-checklist)
    - [per-candidate requirements](#per-candidate-requirements)
    - [design layers to pass](#design-layers-to-pass)
  - [pseudocode convention](#pseudocode-convention)
  - [unified super-architecture skeleton](#unified-super-architecture-skeleton)
  - [architecture candidates](#architecture-candidates)
  - [combination strategies (combined/super models)](#combination-strategies-combinedsuper-models)
  - [hyperparam search-space bounds](#hyperparam-search-space-bounds)
  - [cross-architecture fairness](#cross-architecture-fairness)
  - [glossary](#glossary)

## hardware constraints

This doc is where every candidate/combined-model design has to actually fit, so the budget lives next to what it constrains rather than in the higher-level planning doc.

- RTX 4060 Laptop GPU, 8GB VRAM (8188 MiB per `nvidia-smi`), 64GB RAM, 2 SSD/HDD.

- **max feasible model size**
  - don't hand-calculate — `profile_trial_cost()`/`estimate_total_budget()`/`max_trials_for_budget()`
  - measure real wall-clock+VRAM on this exact card.
    - Rough prior only: 4tf×256candles×few scalars is a modest seq length; 8GB should fit small/med Transformer/TCN at batch 16–64 w/ mixed precision; VRAM more likely bound by hidden-dim/full-attention-over-concat-seq than seq length itself. If full cross-attention doesn't fit, caps toward cheaper fusion alternatives (per-tf encoders + light fusion) — see [multi-timeframe fusion](model-architecture-planning.md#multi-timeframe-fusion).
  - Alt:
    - gradient checkpointing — fallback if needed, slows training
    - mixed precision AMP — near-free win, enable by default, not as fallback
    - FlashAttention — same "near-free win, on by default" category as mixed-precision AMP above; fuses the softmax computation so the full attention matrix is never materialized, applies under whichever attention variant is chosen (see [attention / dependency](model-architecture-planning.md#attention--dependency)), not an approximation or a separate arm of the search
    - GQA/MQA or MLA on the attention stage — targeted KV-cache/activation reduction, the more surgical lever if profiling shows attention-over-concat-seq specifically (not param count) is the binding VRAM constraint; see [attention / dependency](model-architecture-planning.md#attention--dependency) for the head-to-head comparison
    - gradient accumulation — fallback if batch-size-bound
    - cloud/rented GPU — rejected, conflicts w/ local-only decision; revisit only if hard-bottlenecked
    - model parallelism — n/a, single GPU

- **every candidate/combined design in this doc must state a total-parameter estimate against this budget** — see [design checklist](#per-candidate-requirements) below. Worked illustrative example (not a measured number, order-of-magnitude only): Transformer S2 (d_model=384, num_heads=8, num_encoder_layers=2, d_ff=1024) ≈ 4·d_model² (Q/K/V/O) + 2·d_model·d_ff (FFN) per layer ≈ 1.37M params/layer × 2 layers ≈ 2.7M params. Even w/ Adam's 2 extra moment buffers + gradients (~4× params, fp32) that's ~40MB — negligible against an 8GB card. This confirms the existing "batch size, not param count, is the primary lever" framing above: **every S1/S2/S3 profile in [architecture candidates](#architecture-candidates) is order ~0.1M–5M params, nowhere near the binding constraint** — activation memory (attention over the concatenated multi-tf sequence, batch size) is what actually caps size, not parameter count. Flag here, not re-derived per candidate below.

## vram/ram budget split

Rough prior, not yet profiler-confirmed — confirm via `profile_trial_cost()` alongside the hyperparam-bounds profiling pass, same as the section above. Nothing else in the repo sizes VRAM/RAM or specifies a pre-loading pipeline; the closest related content is `infrastructure.md`'s [Repository design pattern](infrastructure.md#repository-design-pattern) (disk-level cache format/ownership for candle/indicator artifacts) and [pandas-ta](infrastructure.md#pandas-ta)/[TensorFlow](infrastructure.md#tensorflow) library choices — linked below, not duplicated.

- **VRAM (8GB card, 8188 MiB)**: ~5% CUDA/driver context reserve, ~40% model params+gradients+optimizer states, ~35% activations (batch fwd/bwd — dominated by attention over the concatenated multi-tf sequence for Transformer-like archs, not by param count — see the worked example above), ~10% input-batch staging (pinned-memory transfer buffer), ~10% fragmentation headroom.
- **RAM (64GB)**: ~10% OS/Python/dataloader-worker overhead, ~65% in-memory feature/candle cache (64GB is generous relative to the 8GB card, so the full or near-full multi-tf dataset can likely stay resident), ~15% pinned staging buffers for CPU→GPU transfer, ~10% headroom.
- **pre-loading / prefetch pipeline** — how the RAM-resident cache actually reaches the GPU without stalling it:
  - in-memory feature cache (the RAM 65% slice above) is populated once per training run from the on-disk artifact cache that `infrastructure.md`'s repository pattern already owns (parquet/npz, cached CSV/zip) — this doc only covers what happens after that disk read, not the disk cache format/ownership itself.
  - `tf.data.Dataset` pipeline (per [TensorFlow](infrastructure.md#tensorflow) as the chosen framework — see [pseudocode convention](#pseudocode-convention)) reads windows from the RAM-resident cache, batches, and calls `.prefetch(tf.data.AUTOTUNE)` so the next batch is staged into the pinned transfer buffer (RAM's 15% slice) while the GPU is still computing the current step — overlaps CPU-side windowing/collation with GPU compute instead of serializing them.
  - `.cache()` on the `tf.data` pipeline is redundant with the already-resident in-memory feature cache above if the full dataset fits in the 65% RAM slice — only add it if profiling shows re-windowing cost (not the disk read) is itself a bottleneck.
  - Alt:
    - no explicit split, let the allocator decide dynamically — rejected as a sizing guide; describes runtime behavior but doesn't help set search-space bounds up front
    - cache nothing, stream from disk every epoch — rejected, wastes the 64GB RAM headroom and adds I/O bottleneck given SSD/HDD mix
    - memory-map the on-disk cache instead of loading fully into RAM — viable if the dataset ever exceeds the RAM budget; not needed yet given the 64GB headroom vs. current data volume

## design checklist

Start of "the rest of this doc" — a checklist to run against every candidate/combined-model design below (and any new one added later) so designs stay consistent, complete, and don't silently skip a concern. Two parts: what each write-up must *state*, and what layers of design it must *pass through* to get there.

### per-candidate requirements

Every architecture write-up in [architecture candidates](#architecture-candidates) or [combination strategies](#combination-strategies-combinedsuper-models) must specify:

- **input/output contract** — input shape per tf branch (seq_len × n_features, per [candle feature schema](input-features.md#candle-feature-schema)), output heads and their shapes (per [training-data.md § model output targets](training-data.md#model-output-targets)).
- **stage-by-stage block list** — which [pipeline stage](model-architecture-planning.md#top-level-architecture) each layer belongs to (embedding/local-extraction/sequential/attention/fusion/global-repr/heads), expressed as a `stage_config` per [unified super-architecture skeleton](#unified-super-architecture-skeleton) — including which stages are zeroed (skipped) for this candidate.
- **hyperparam profile table** — S1/S2/S3 (depth-heavy/width-heavy/context-heavy), per [architecture candidates](#architecture-candidates).
- **param-count order-of-magnitude estimate** — against the [hardware constraints](#hardware-constraints) budget; flag if it's not obviously negligible like the worked example there.
- **pseudocode** — for the block(s) that make this candidate distinct from a plain pass-through of the skeleton, per [pseudocode convention](#pseudocode-convention). Reuse the skeleton's stage functions rather than re-deriving; only write out what's new.
- **rejected alternatives** — `Alt:` list, same convention as the rest of this doc/`model-architecture-planning.md`, so "why not X" is answered inline instead of re-litigated later.

### design layers to pass

Concerns to verify, roughly in dependency order — a candidate isn't "designed" until all of these are addressed (even if the answer is "inherits from the skeleton, nothing candidate-specific here"):

1. **data/interface layer** — input shape and feature schema match [candle feature schema](input-features.md#candle-feature-schema); output heads match [training-data.md § model output targets](training-data.md#model-output-targets).
2. **representation layer** — the stage-by-stage block choice itself (embedding → local-extraction → sequential → attention → fusion → global-repr), per [top level architecture](model-architecture-planning.md#top-level-architecture).
3. **capacity/sizing layer** — param count vs. [hardware constraints](#hardware-constraints); activation memory (batch × seq × d_model) vs. the [vram/ram budget split](#vramram-budget-split), since that's the actual binding constraint, not param count.
4. **regularization layer** — dropout, weight decay, normalization placement (pre-/post-norm) — held constant across S1/S2/S3 per the existing convention (see [architecture candidates](#architecture-candidates) intro), stated explicitly per candidate anyway so it isn't silently assumed.
5. **training-dynamics layer** — gradient flow (residual connections around any new block), mixed-precision (AMP) compatibility, checkpointing compatibility — per [hardware constraints](#hardware-constraints) → Alt list.
6. **evaluation-interface layer** — loss/metric hookup per head, per [error rating & model evaluation](error-rating-and-evaluation.md#per-head-statistical-metrics-dev-diagnostics) — not re-derived here, just confirmed wired correctly for this candidate's head shapes.
7. **combination layer** (only for [combined/super models](#combination-strategies-combinedsuper-models)) — fusion mechanism (per [fusion mechanism](model-architecture-planning.md#fusion-mechanism)) + stage-slot placement config, since a combined model is this same checklist applied once per constituent backend plus one more pass for how they're fused.

## pseudocode convention

Every candidate below is described with **Keras (`tf.keras`) functional-API-style pseudocode** — shape-annotated layer composition (`y = Layer(...)(x)  # shape: (...)`), not full runnable code.

Rationale, briefly, since this is worth remembering rather than re-deciding later: [infrastructure.md](infrastructure.md#tensorflow) already locks TensorFlow as this repo's DL framework (`tensorflow[and-cuda]`, Docker base `tensorflow:25.01-tf2-py3`) for the existing CNN-LSTM-attention models. Keras is TensorFlow's own high-level API, so pseudocode written this way is close to directly transcribable into real code later, not a translation exercise from a different framework's idioms. It's also simpler to read as architecture-shape sudo-code than raw PyTorch `nn.Module`/`forward()` boilerplate (explicit `__init__`/`forward` split, manual shape bookkeeping) — Keras's functional style (`y = Layer(config)(x)`) reads closer to the stage-pipeline diagrams already used in `model-architecture-planning.md`.

Convention used throughout:

- `# shape: (B, T, F)` comments track tensor shape through the pipeline (batch, sequence, feature-dim).
- hyperparameter names match the S1/S2/S3 tables below exactly (`d_model`, `num_heads`, `hidden_channels`, ...) so a profile row can be read directly into the pseudocode's function signature.
- a stage function returning its input unchanged (`kind == 0`) represents that stage being **zeroed** — see [unified super-architecture skeleton](#unified-super-architecture-skeleton).

## unified super-architecture skeleton

For the "which combination of stages, in which placement (before/after, start/middle/end)" question: rather than designing a separate architecture per placement variant, this doc designs **one maximally-complex skeleton** containing every pipeline stage as a slot, and tests placement by **zeroing** (disabling) or **numbering** (selecting which block type occupies) each slot — not by drawing a new diagram per combination. This is also literally what a "combined/super model" is in this doc: the skeleton itself, parameterized by `stage_config`.

- the skeleton's stage **order is fixed** and maps directly onto start/middle/end: embedding = start, local-extraction → sequential → attention → fusion = middle, global-representation → heads = end. This fixed order already matches the one resolved case in `model-architecture-planning.md` (conv-then-transformer in the hybrid CNN→Transformer candidate, per [local feature extraction](model-architecture-planning.md#local-feature-extraction)) — so "placement" here means *which slots are active*, not reordering the skeleton itself.
- `stage_config[stage] = 0` → that stage is `tf.identity` (skipped entirely) — the "zeroing" test, e.g. does attention earn its cost over conv-only.
- `stage_config[stage] = <block name>` → the "numbering" test — which block implementation occupies that fixed slot, e.g. `attention: "self_attn"` vs `"informer"` vs `"itransformer"` (per [attention / dependency](model-architecture-planning.md#attention--dependency)).
- reordering the skeleton itself (e.g. attention *before* local-extraction instead of after) is a genuinely different, larger search — that's what [differentiable/block-level NAS](model-architecture-planning.md#combination-strategy) would search over; out of scope for this fixed-skeleton default, deferred same as NAS is deferred in the main doc.

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

## architecture candidates

hyperparam sets below are illustrative starting candidates for `profile_trial_cost()` to evaluate, not fixed a priori — see [search-space bounds](#hyperparam-search-space-bounds). Profiles S1/S2/S3 are roughly parameter/compute-matched _within_ each architecture — same capacity weight, different capability focus — not a size ladder: **S1 = depth-heavy, S2 = width-heavy, S3 = context-heavy** (max attention span/receptive field/state capacity, arch-dependent). dropout is a separate regularization knob, held constant across profiles. Sizing follows the [vram/ram budget split](#vramram-budget-split) above; batch size, not param count, is the primary lever for using available VRAM, per [max feasible model size](#hardware-constraints).

Per the [design checklist](#per-candidate-requirements), each candidate below states its `stage_config` (relative to the skeleton above) before its hyperparam profile, plus a short pseudocode block only for what's distinct from a plain skeleton pass-through.

- **CNN-LSTM(-attention)** — the pre-existing baseline, not a new proposal; see [current Stage-1 candidate set](model-architecture-planning.md#current-stage-1-candidate-set). Included here mainly as a worked illustration of the skeleton itself: the two variants that already exist in code — [cnn_lstm_model.py](../app/ai_modelling/cnn_lstm/cnn_lstm_model.py) and [cnn_lstm_attention_model.py](../app/ai_modelling/cnn_lstm_attention/cnn_lstm_attention_model.py) — differ from each other by exactly one `stage_config` value (`attention: 0` vs `attention: "self_attn"`), which is the zeroing mechanism working as designed, not a coincidence.
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
  - **residual-CNN time-series-classification baselines** as the `local_extraction` block instead of the plain stacked-conv above — same slot, three separate implementations, not one: **ResNet** (residual skip connections), **FCN** (global-pooled fully-conv stack, no residual), **InceptionTime** (multi-kernel-size Inception modules); see [local feature extraction](model-architecture-planning.md#local-feature-extraction) and the scoring in [prioritization framework](prioritization-framework.md#local-feature-extraction). None yet reduced to a `stage_config`-ready block here — flagged as a follow-up, not designed in this pass.
  - **ConvLSTM** in place of `sequential: "rnn"` — convolutional gates inside the recurrent cell itself, a different mechanism than conv-then-LSTM stacking; see [sequential encoding](model-architecture-planning.md#sequential-encoding). Same follow-up status as the residual-CNN alt above.
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
- **state-space: Mamba** — cheap long-seq alt to attention. (**S4** is a lower-priority alt within this same `sequential: "ssm"` slot — swap `MambaBlock` below for an `S4Block`-equivalent; a separate implementation, not a parameter of `MambaBlock`, per [sequential encoding](model-architecture-planning.md#sequential-encoding) and the scoring in [prioritization framework](prioritization-framework.md#sequential-encoding).)
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
  - **GRU** (`cell_type="gru"` above) — Tier-2 alt tested within this same floor role, not a separate candidate slot; see [sequential encoding](model-architecture-planning.md#sequential-encoding) and the scoring in [prioritization framework](prioritization-framework.md#sequential-encoding).
  Alt:
  - **naive/persistence baseline** — not a `stage_config` at all, no learned stages; "no change"/carry-forward the last signal. The floor beneath this floor — computed alongside backtested KPIs for every run, not a Stage-1 categorical option; see [current Stage-1 candidate set](model-architecture-planning.md#current-stage-1-candidate-set).
  - pure MLP on flattened features — rejected as serious candidate, discards seq structure; trivial baseline only
  - GBM (LightGBM, XGBoost, CatBoost — three separate library classes) on flattened features — kept as a cheap non-sequence floor, distinct from the LSTM floor above (which still respects sequence order); see [auxiliary tabular models (GBM-family)](model-architecture-planning.md#auxiliary-tabular-models-gbm-family) for what this comparison is meant to answer
  - Random Forest on flattened features — same scoped role as GBM above, bagging instead of boosting; see [modern GBM-family alternatives](model-architecture-planning.md#modern-gbm-family-alternatives)
  - 1-nearest-neighbor w/ DTW distance — classic parameter-light TSC floor; see [current Stage-1 candidate set](model-architecture-planning.md#current-stage-1-candidate-set) Alt list
  - 4 separate per-tf models + late ensembling — kept as cheap baseline, see [multi-timeframe fusion](model-architecture-planning.md#multi-timeframe-fusion) and [combination strategies](#combination-strategies-combinedsuper-models) below
  - GNN over tf/symbol nodes — deferred, no evidence needed yet
  - **TFT (Temporal Fusion Transformer)** and **Perceiver** — named as additional Stage-1 categorical options in the main doc's [combination strategy](model-architecture-planning.md#combination-strategy) section; not yet added with S1/S2/S3 hyperparam profiles here — pending the same profiling treatment as the candidates above. Two separate stage-level compositions, each expressible via the skeleton (TFT ≈ per-feature gating + LSTM sequential stage + interpretable multi-head attention stage; Perceiver ≈ cross-attention fusion stage over a fixed-size latent array instead of concat/cross-tf attention) rather than needing a new skeleton — see the scoring in [prioritization framework](prioritization-framework.md#current-stage-1-candidate-set) for why they're evaluated independently rather than as one row.

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

See [fusion mechanism](model-architecture-planning.md#fusion-mechanism) for the rationale/tradeoffs behind each fusion kind — not re-derived here, this section only pins the pseudocode.

## combination strategies (combined/super models)

Pseudocode for the "combined/super model" strategies named in the main doc's [combination strategy](model-architecture-planning.md#combination-strategy) section — status there is **unresolved, not yet measured, default = single-backend-wins**; this section only adds the implementation-level design (per the [design checklist](#design-checklist)) for when/if one of these is tested, not a re-argument of the rationale or priority (see the main doc for that).

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

## hyperparam search-space bounds

not fixed a priori — `profile_trial_cost()` measures real wall-clock/VRAM per arch+hparam combo on this card; `max_trials_for_budget()` derives trial cap. Search-space priors: seq len capped 256/tf; batch size s.t. largest arch/seq combo fits VRAM at batch≥8; hidden-dim/depth kept modest vs ~1yr data (small vs NLP-scale). Concrete bounds from profiler's first pass, not hand-picked.
Alt:

- fixed ranges from DL-literature defaults w/o profiling — rejected, wrong hardware/dataset scale
- very wide ranges relying only on Hyperband — rejected as primary, wastes trials in OOM regions

## cross-architecture fairness

- architecture = categorical param in one Optuna study (not N sweeps) → fairness enforced at study level:
  - (1) same train-pairs/BTC-USDT split every trial (see [validation & train/test splitting](model-architecture-planning.md#validation--traintest-splitting));
  - (2) one shared GPU-hour budget via `estimate_total_budget()`, not per-arch;
  - (3) Hyperband pruning arch-agnostic;
  - (4) min grace-period epochs before pruning (protects slow-converging archs); post-study sanity-check trial counts per arch, top-up budget if one is starved.
- Alt:
  - separate sweeps w/ equalized budgets — rejected, old approach, wastes compute
  - fixed wall-clock per arch — rejected, same waste, time-boxed
  - compare only best trial per arch — rejected, too seed-sensitive

## glossary

Stage-1, S1/S2/S3 — see [model-architecture-planning.md § glossary](model-architecture-planning.md#glossary).

- `stage_config` — per-candidate dict selecting, per pipeline stage, either `0` (zeroed/skipped) or a block-type name; the full parameterization of the [unified super-architecture skeleton](#unified-super-architecture-skeleton).
- super-architecture / combined model — this doc's single fixed-order skeleton with pluggable stage slots; "combined/super model" = any `stage_config` with more than one non-zero middle stage, or any strategy in [combination strategies](#combination-strategies-combinedsuper-models) composing multiple full skeleton instances.
- zeroing / numbering (placement) — the two ways `stage_config` values vary: zeroing tests whether a stage is needed at all (`0` vs. non-zero); numbering tests which block occupies an already-active slot.

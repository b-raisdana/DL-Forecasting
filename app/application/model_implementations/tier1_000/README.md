# Tier-1_000 Model Implementation

Spec: `docs/ML_Forecasting_System_Design/designsets/Tier-1_000.par_branch_mtcn_lstm_perc_gqa_mlp_lgbm(handmade).base.jsonc`
Input variation: `input-data-feature/Tier-1_000.atr_rel_ohlc_log_sma_v_extm_rel_6tf(handmade).input.jsonc` → variation 3
Outcome set: `outcome-label-target-head/Tier-1_000.action_mfe_rer(handmade).outcome.jsonc` → set 1

## Input Features

Each of the 6 branches (5min, 15min, 1h, 4h, 1D, 1W) produces a `(window_len, 15)` tensor of `CANDLE_FEATURE_COLUMNS` (`model.py:53`). Window lengths per variation 3: 256, 256, 128, 128, 64, 64 (`model.py:45`).

### relative_OHLC (5 columns)
- `relative_normal_close`: `(close - anchor_close) / ATR(256)` — anchor is the LAST 5min candle before the label timestamp (`datafeeder_input3_outcome1.py:229`)
- `rel_high_close`: `(high - close) / ATR` (`relative_candle.py:31`)
- `rel_close_low`: `(close - low) / ATR` (`relative_candle.py:32`)
- `gap`: `(open - prev_close) / ATR` (`relative_candle.py:33`)
- `rel_candle_height`: `(high - low) / ATR` (`relative_candle.py:34`)

ATR is `pandas_ta.atr(length=256)` except 1W uses length=20 to avoid warmup crash (`datafeeder_input3_outcome1.py:79`).

### V (1 column)
- `log_volume_sma_ratio`: `log((volume + eps) / (SMA(volume, 256) + eps))` (`volume_feature.py:36`)

### higher_extremum_distance (8 columns)
Computed in `extremum_features.py` via `compute_higher_extremum_distance()`.

Step A — `compute_true_extremum()` (`domain/price_action/CausalExtremum.py:126`): O(n) monotonic-stack pass finds each candle's full-hindsight peak/valley reach in minutes, then `floor_to_tf_ladder()` snaps to the nearest TF ladder rung (5min/15min/1h/4h/1D/1W/1M/4M/1Y). `extremum_sign` is +1 for peaks, -1 for valleys (peak wins ties).

Step B — causal cap: `observed = min(true_reach, floor_to_tf_ladder(age_minutes))` where `age_minutes` is elapsed minutes from the candle to the anchor. This is the closed-form derivation in `domain/price_action/CausalExtremum.py:27`.

Step D — for each window position, `_nearest_and_last()` finds the price-nearest and time-last eligible extremum from the plus2TF/plus3TF source branch's pool (events whose reach already exceeds the target TF threshold). Raw price diff is divided by the source branch's ATR (aligned causally via `align_source_atr()`); raw elapsed minutes get `log1p(elapsed / target_tf_minutes)` (`application/dataset_generation/extremum_features.py:196-200`).

Source branches: plus2TF/plus3TF targets not in the 6 cached timeframes (1M/4M/1Y) fall back to the 1W branch's own extremum (`datafeeder_input3_outcome1.py:200`).

### extremum_weight (1 column)
`sign × log1p(observed_reach / tf_minutes) × min(1, age_minutes / observed_reach)`, zero if not an extremum (`application/dataset_generation/extremum_features.py:103`). `observed_reach` uses the same causal cap as Step B.

### auxiliary_features
LAST candle of every branch, all 15 features flattened: shape `(n_samples, 90)` (`datafeeder_input3_outcome1.py:283`).

## Outcome Features

Set 1 produces three targets (`mfe_mae_om_labels.py`):

- `action`: 3-class one-hot (long/short/none). A direction qualifies if its `om = mfe / mae > OM_GATE (5.0)`. If both qualify, the higher OM wins. Neither qualifies → none (`mfe_mae_om_labels.py:107-114`).
- `mfe`: Maximum favorable excursion from entry (best high for long, best low for short) within `HORIZON_BARS = 48` (240 min) (`mfe_mae_om_labels.py:62-65`). Entry price is the first bar of the horizon (`mfe_mae_om_labels.py:100-101`).
- `rer`: `mae / (mfe - mae)`, clipped to `(0, 1/4)` (`mfe_mae_om_labels.py:120`). `mae` is the worst adverse excursion before the best bar, floored at `ATR(255, 15min)` (`mfe_mae_om_labels.py:76`).

Loss: categorical crossentropy for action; Gaussian NLL (`gaussian_nll_loss` in `model.py:397`) for mfe/rer mean+std pairs trained jointly.

## Architecture Backbone

### Six parallel branches
`TimeframeBranchEncoder` per timeframe (`model.py:147`): `Dense(channels)` → `ModernTCNBlock × depth` → `LSTM × layers` (all `return_sequences=True`).

### ModernTCN
`ModernTCNBlock` (`model.py:124`): pre-norm → causal depthwise conv (kernel=7, left-padded) → dropout → pre-norm → ConvFFN (4x expansion, GELU) → residual. Depth=8, channels=192.

### LSTM
4 unidirectional layers, hidden=384, `return_sequences=True` (`model.py:167`).

### Perceiver fusion
`PerceiverFusion` (`model.py:220`): learnable timeframe identity embeddings (shape `(1,1,hidden_units)`) broadcast-added to each branch's sequence; all 6 sequences concatenated; `Perceiver_latent_tokens=96` learnable latents cross-attend to the concatenated sequence via `MultiHeadAttention` (heads=8, key_dim=768/8) for 8 layers, each with pre-norm residual + FFN. No self-attention among latents — that role is filled by downstream GQA.

### GQA
`GQAEncoder` (`model.py:289`): 8 layers of hand-rolled GroupedQueryAttention (heads=8, kv_heads=2, d_model=768=`Perceiver_latent_dim`) + FFN, pre-norm residual. Hand-rolled because Keras `MultiHeadAttention` lacks `kv_heads < heads` (`model.py:182`).

### Pooling
`last_token` by default (`model.py:321`): takes the final latent token.

### Prediction head
`PredictionHead` (`model.py:331`): MLP trunk (depth=4, width=512, GELU, dropout=0.1) on `Concatenate([pooled, auxiliary_features])`. Three output heads:
- `action`: Dense(3, softmax, float32)
- `mfe_params`: Dense(2, softplus, float32) → [mean, std]
- `rer_params`: Dense(2, [sigmoid, softplus], float32) → [mean, std]

### Training
AdamW (lr=3e-4, weight_decay=1e-4, clipnorm=1.0), cosine scheduler, mixed_float16, batch_size=8 (reduced from spec's 128 due to 8GB VRAM OOM profiling — `model.py:93`).

## Resource Usage (Measured)

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 5560 MB VRAM.

| Metric | Value |
|--------|-------|
| Model parameters | 143.8M |
| batch_size=8 GPU peak (forward+backward) | 4521.6 MB (81.3% of VRAM) |
| batch_size=8 stability | OOM on 2nd gradient step — BFC allocator fragmentation near cap |
| batch_size=4 GPU peak | 4727.9 MB |
| batch_size=4 stability | 5+ steps stable |
| Steady-state step time (batch=4, after XLA warmup) | ~6–10s |
| Throughput (batch=4) | ~0.3–0.7 samples/sec |

The spec's own transient_activation_memory estimate identified the Perceiver's cross-attention over the concatenated 1536-token multi-tf sequence as the dominant scalable term. The measured peak confirms this: activation memory, not parameter count, binds VRAM on this hardware.

## Hyperparameter Optimization Judgment

Optimization is required before broad search.

- batch_size=8 is **unstable** on this hardware (OOM on step 2). This blocks normal training. The highest-leverage fix is gradient checkpointing on the Perceiver cross-attention layers, trading ~20-30% compute for ~500-800 MB activation savings and restoring batch_size=8 stability.
- Perceiver_latent_tokens (96) and Perceiver_cross_attention_layers (8) are the next best knobs. Reducing either cuts the dominant activation term directly. A token reduction to 48 or layer reduction to 4 should fit comfortably under 5560 MB at batch_size=8.
- Backbone widths (ModernTCN_channels=192, LSTM_hidden_units=384) are the main parameter drivers. If activation memory remains tight after Perceiver changes, reducing these toward base (128/256) saves ~30-40% parameters with modest accuracy trade-off.
- GQA heads/layers (8/8) and MLP width/depth (512/4) are lower priority — GQA operates on already-compressed 96 tokens, so its footprint is small; MLP width is held constant across candidates by design.
- Learning rate, weight decay, scheduler, and optimizer are spec-resolved at reasonable defaults. They cannot be judged without actual training loss curves; leave them fixed until the model trains stably.

Recommended sequence:
1. Add gradient checkpointing to Perceiver cross-attention.
2. If still unstable, reduce Perceiver_latent_tokens or Perceiver_cross_attention_layers.
3. If still unstable, reduce backbone widths.
4. Only after stable multi-step training at batch_size≥8, run a focused Optuna search over the remaining searchable parameters.

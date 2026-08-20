"""DS-01 "Six-Timeframe Hybrid Temporal Model" per
docs/ML_Forecasting_System_Design/designsets/Tier-1_000.par_branch_mtcn_lstm_perc_gqa_mlp_lgbm(handmade).base.jsonc,
resolved to its "1-base" hyperparameters (the `//` bound comments in that file are Optuna search bounds,
not alternative values for a single run — see PROMPT.md's embedded-search-space convention).

Known, documented deviations from the spec (see docs/ML_Forecasting_System_Design/todo/01-input-data-channels.md and
docs/ML_Forecasting_System_Design/todo/02-training-data-labels.md for the upstream gaps driving these):

- `CANDLE_FEATURE_COLUMNS` is now the full 15-field `candle_dataset` (relative_OHLC's 5 + V + the 8
  `higher_extremum_distance` price/time-distance terms + `extremum_weight`), computed via
  domain/price_action/CausalExtremum.py (Step A/B causal-capped reach) and
  application/dataset_generation/extremum_features.py (Step C/D orchestration), wired in by
  datafeeder_input3_outcome1.py. A few genuinely underspecified edges in the jsonc were resolved as
  judgment calls, documented at the point of implementation (datafeeder_input3_outcome1.py /
  CausalExtremum.py) rather than repeated here in full — summarized:
  - **Peak wins ties** in Step A's `extremum_sign` when a candle's peak-reach and valley-reach are
    exactly equal (not specified in the jsonc either way).
  - **1M/4M/1Y have no native cached series** in this codebase (app_config.timeframes stops at 1W) —
    plus2TF/plus3TF targets that land on 1M/4M/1Y are sourced from the 1W branch's own Step A/B reach
    output instead of a synthesized coarser series (valid because Step A's reach is computed purely
    from elapsed time against TF_MINUTES thresholds, not from literal coarser-candle construction).
  - **Empty eligible-event pool -> 0.0** for both `price_normal_distance` and `time_distance` (the
    early-history "no signal yet" case), rather than NaN — matches the existing NaN-scrub convention
    at the end of `build_dataset()`, which still catches genuine indicator-warmup NaNs (e.g. ATR not
    yet warmed) separately.
- `input_set=3` (not the "1-base" row's `input_set=1`) — variation 1 needs 256 candles on every
  branch including 1W (≈4.9 years of history); the largest verified gap-free cached BTCUSDT range is
  995 days (≈2.7 years). Variation 3 (64 candles on 1D/1W) is a spec-sanctioned option in the same
  input file, not an invented shortcut.
- Perceiver cross-attention's head count isn't in the jsonc (only GQA's is) — reused GQA's own
  heads=8 convention for it, documented here since it's a real choice, not a spec value.
"""

from __future__ import annotations

import math
from typing import cast

import tensorflow as tf
from tensorflow import keras as tf_keras
from tensorflow.keras import layers

# --- input shape (per docs/todos § known deviations above) -----------------------------------------
BRANCH_TIMEFRAMES: list[str] = ["5min", "15min", "1h", "4h", "1D", "1W"]
BRANCH_WINDOW_LENGTHS: dict[str, int] = {  # input_set=3 (see module docstring)
    "5min": 256,
    "15min": 128,
    "1h": 128,
    "4h": 128,
    "1D": 128,
    "1W": 128,
}
CANDLE_FEATURE_COLUMNS: list[str] = [
    "relative_normal_close",
    "rel_high_close",
    "rel_close_low",
    "gap",
    "rel_candle_height",
    "log_volume_sma_ratio",
    "price_normal_distance_plus2tf_peak",
    "price_normal_distance_plus2tf_valley",
    "price_normal_distance_plus3tf_peak",
    "price_normal_distance_plus3tf_valley",
    "time_distance_plus2tf_peak",
    "time_distance_plus2tf_valley",
    "time_distance_plus3tf_peak",
    "time_distance_plus3tf_valley",
    "extremum_weight",
]
CANDLE_FEATURE_DIM = len(CANDLE_FEATURE_COLUMNS)
AUX_FEATURE_DIM = CANDLE_FEATURE_DIM * len(BRANCH_TIMEFRAMES)  # LAST candle only, flattened, all branches

# --- "1-base" resolved hyperparameters (searchable_architecture_parameter_sets."1-base") -----------
TIER1_000_CONFIG: dict[str, object] = {
    "ModernTCN_kernel_size": 7,
    "ModernTCN_depth": 8,
    "ModernTCN_channels": 192,
    "LSTM_layers": 4,
    "LSTM_hidden_units": 384,
    "LSTM_bidirectional": False,
    "Perceiver_latent_tokens": 96,
    "Perceiver_latent_dim": 768,
    "Perceiver_cross_attention_layers": 8,
    "Perceiver_heads": 8,  # not in the jsonc — see module docstring
    "GQA_layers": 8,
    "GQA_heads": 8,
    "GQA_kv_heads": 2,
    "MLP_depth": 4,
    "MLP_width": 512,
    "dropout": 0.1,
    "activation": "gelu",
    "pooling_method": "last_token",
    # 128 (the spec's own resolved value) OOMs on the 8GB RTX 4060 even at 90% VRAM cap with
    # memory_growth on — profiled directly (batch=32 also OOMs). The design doc's own
    # transient_activation_memory estimate is a coarse per-stage formula that doesn't account for
    # ModernTCN's ConvFFN 4x-expansion intermediate tensors (768-wide, not 192-wide) kept for
    # backprop across 8 layers x 6 branches — real profiling (this repo's own "profile_trial_cost()
    # remains ground truth" principle) found the true footprint larger. batch=16 fits a short burst
    # (~7.2GB peak) but OOM'd mid-run on a longer one — BFC allocator fragmentation at ~98% of the
    # 90%-of-8GB cap leaves ~0 slack for step-to-step allocation-pattern variance. batch=8 leaves
    # real headroom. Batch size is the hardware-constraints doc's own first fallback for this case.
    "batch_size": 8,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "scheduler": "cosine",
    "gradient_clip_norm": 1.0,
}


def _ffn_block(d_model: int, dropout_rate: float, name: str) -> tf_keras.Sequential:
    """Standard 4x-expansion Transformer FFN — shared shape for Perceiver's and GQA's post-attention
    sub-layer, per PROMPT.md § Memory sizing convention ("FFN: 2 x d_model x (4 x d_model)")."""
    return tf_keras.Sequential(
        [
            layers.Dense(4 * d_model, activation="gelu", name=f"{name}_expand"),
            layers.Dense(d_model, name=f"{name}_project"),
            layers.Dropout(dropout_rate),
        ],
        name=name,
    )


class ModernTCNBlock(layers.Layer):  # type: ignore[misc]  # TensorFlow's Layer is untyped.
    """Large-kernel depthwise conv + ConvNeXt-style inverted-bottleneck ConvFFN (4x expansion), per
    03-Model & Architecture Engineering.md § local feature extraction / PROMPT.md's param formula.
    Pre-norm + residual around each of the two sub-layers, per the model-wide
    regularization_and_stabilization convention."""

    def __init__(self, channels: int, kernel_size: int, dropout_rate: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization()
        # Keras' DepthwiseConv1D has no 'causal' padding mode (unlike Conv1D) — left-pad explicitly
        # then 'valid'-convolve, the standard causal-conv workaround.
        self.causal_pad = layers.ZeroPadding1D(padding=(kernel_size - 1, 0))
        self.depthwise = layers.DepthwiseConv1D(kernel_size=kernel_size, padding="valid")
        self.drop1 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization()
        self.conv_ffn = _ffn_block(channels, dropout_rate, name=f"{self.name}_conv_ffn")

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = x + self.drop1(self.depthwise(self.causal_pad(self.norm1(x))), training=training)
        x = x + self.conv_ffn(self.norm2(x), training=training)
        return x


class TimeframeBranchEncoder(layers.Layer):  # type: ignore[misc]  # TensorFlow's Layer is untyped.
    """One independent per-timeframe branch: raw-feature projection -> ModernTCN (depth layers) ->
    stacked unidirectional LSTM (LSTM_layers), sequence-preserving throughout (return_sequences=True
    on every LSTM layer — Perceiver fusion needs the full "*_encoded_sequence", not a summary vector).
    """

    def __init__(self, config: dict[str, object], **kwargs: object) -> None:
        super().__init__(**kwargs)
        channels = cast(int, config["ModernTCN_channels"])
        dropout_rate = cast(float, config["dropout"])
        self.input_projection = layers.Dense(channels, name=f"{self.name}_input_projection")
        self.tcn_blocks = [
            ModernTCNBlock(
                channels,
                cast(int, config["ModernTCN_kernel_size"]),
                dropout_rate,
                name=f"{self.name}_tcn{i}",
            )
            for i in range(cast(int, config["ModernTCN_depth"]))
        ]
        self.lstm_layers = [
            layers.LSTM(cast(int, config["LSTM_hidden_units"]), return_sequences=True, name=f"{self.name}_lstm{i}")
            for i in range(cast(int, config["LSTM_layers"]))
        ]

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.input_projection(x)
        for tcn in self.tcn_blocks:
            x = tcn(x, training=training)
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        return x


class GroupedQueryAttention(layers.Layer):  # type: ignore[misc]  # TensorFlow's Layer is untyped.
    """Hand-rolled: Keras' built-in MultiHeadAttention has no kv_heads<heads support (verified against
    the installed keras==3.9.1 signature), so GQA itself must be hand-rolled — the one block in this
    model lib-first genuinely can't cover. Q/K/V/O formula per PROMPT.md § Memory sizing convention.
    """

    def __init__(self, d_model: int, heads: int, kv_heads: int, dropout_rate: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        if heads % kv_heads != 0:
            raise ValueError(f"heads={heads} must be divisible by kv_heads={kv_heads}")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // heads
        self.group_size = heads // kv_heads
        self.q_proj = layers.Dense(d_model, name=f"{self.name}_q_proj")
        self.k_proj = layers.Dense(kv_heads * self.head_dim, name=f"{self.name}_k_proj")
        self.v_proj = layers.Dense(kv_heads * self.head_dim, name=f"{self.name}_v_proj")
        self.o_proj = layers.Dense(d_model, name=f"{self.name}_o_proj")
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        q = tf.transpose(tf.reshape(self.q_proj(x), (batch, seq_len, self.heads, self.head_dim)), [0, 2, 1, 3])
        k = tf.reshape(self.k_proj(x), (batch, seq_len, self.kv_heads, self.head_dim))
        v = tf.reshape(self.v_proj(x), (batch, seq_len, self.kv_heads, self.head_dim))
        k = tf.repeat(tf.transpose(k, [0, 2, 1, 3]), self.group_size, axis=1)
        v = tf.repeat(tf.transpose(v, [0, 2, 1, 3]), self.group_size, axis=1)
        scale = 1.0 / math.sqrt(float(self.head_dim))
        scores = tf.matmul(q, k, transpose_b=True) * scale
        weights = tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1)
        weights = tf.cast(self.dropout(weights, training=training), v.dtype)
        attended = tf.matmul(weights, v)
        attended = tf.reshape(tf.transpose(attended, [0, 2, 1, 3]), (batch, seq_len, self.heads * self.head_dim))
        return self.o_proj(attended)


class PerceiverFusion(layers.Layer):  # type: ignore[misc]  # TensorFlow's Layer is untyped.
    """Learnable latent-token bank cross-attends into the concatenated multi-timeframe sequence,
    stacked Perceiver_cross_attention_layers times (cross-attn only — no self-attention among latents
    inside this block; that role is filled by the separate downstream GQA stage, per
    03-Model & Architecture Engineering.md). Uses Keras' native MultiHeadAttention (gets
    FlashAttention where the backend supports it, per the hardware-constraints "near-free win, on by
    default" guidance) since plain multi-head cross-attention needs no kv_heads grouping.
    """

    def __init__(self, config: dict[str, object], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.hidden_units = cast(int, config["LSTM_hidden_units"])
        self.latent_dim = cast(int, config["Perceiver_latent_dim"])
        self.num_latents = cast(int, config["Perceiver_latent_tokens"])
        self.dropout_rate = cast(float, config["dropout"])
        heads = cast(int, config["Perceiver_heads"])
        self.tf_identity_embeddings = [
            self.add_weight(
                shape=(1, 1, self.hidden_units),
                initializer="random_normal",
                trainable=True,
                name=f"tf_identity_{tf_name}",
            )
            for tf_name in BRANCH_TIMEFRAMES
        ]
        self.latent_bank = self.add_weight(
            shape=(1, self.num_latents, self.latent_dim),
            initializer="random_normal",
            trainable=True,
            name="perceiver_latent_bank",
        )
        n_layers = cast(int, config["Perceiver_cross_attention_layers"])
        self.cross_attn_norms = [layers.LayerNormalization(name=f"perceiver_norm{i}") for i in range(n_layers)]
        self.cross_attns = [
            layers.MultiHeadAttention(
                num_heads=heads,
                key_dim=self.latent_dim // heads,
                dropout=self.dropout_rate,
                name=f"perceiver_cross_attn{i}",
            )
            for i in range(n_layers)
        ]
        self.ffn_norms = [layers.LayerNormalization(name=f"perceiver_ffn_norm{i}") for i in range(n_layers)]
        self.ffns = [_ffn_block(self.latent_dim, self.dropout_rate, name=f"perceiver_ffn{i}") for i in range(n_layers)]

    def call(self, branch_sequences: list[tf.Tensor], training: bool | None = None) -> tf.Tensor:
        # add_weight() variables default to float32 (the mixed-precision "variable dtype") even
        # though branch_sequences are float16 (the "compute dtype") — plain `+` outside a Keras
        # layer's own __call__ gets none of Keras' automatic input-casting, so cast explicitly.
        identified = [
            seq + tf.cast(emb, seq.dtype)
            for seq, emb in zip(branch_sequences, self.tf_identity_embeddings, strict=True)
        ]
        all_timeframe_sequences = layers.Concatenate(axis=1)(identified)
        batch = tf.shape(all_timeframe_sequences)[0]
        latents = tf.tile(tf.cast(self.latent_bank, all_timeframe_sequences.dtype), [batch, 1, 1])
        for norm, attn, ffn_norm, ffn in zip(
            self.cross_attn_norms, self.cross_attns, self.ffn_norms, self.ffns, strict=True
        ):
            latents = latents + attn(
                query=norm(latents),
                value=all_timeframe_sequences,
                key=all_timeframe_sequences,
                training=training,
            )
            latents = latents + ffn(ffn_norm(latents), training=training)
        return cast(tf.Tensor, latents)


class GQAEncoder(layers.Layer):  # type: ignore[misc]  # TensorFlow's Layer is untyped.
    """Refines the Perceiver's output latents: model_dependencies_between_latent_representations /
    capture_global_interactions / refine_cross-timeframe_representation, per Tier-1_000's
    dependency_modeling.attention.GQA.purpose. Pre-norm + residual, GQA_layers stacked blocks."""

    def __init__(self, config: dict[str, object], **kwargs: object) -> None:
        super().__init__(**kwargs)
        d_model = cast(int, config["Perceiver_latent_dim"])  # "GQA model dimension = Perceiver_latent_dim"
        dropout_rate = cast(float, config["dropout"])
        n_layers = cast(int, config["GQA_layers"])
        self.attn_norms = [layers.LayerNormalization(name=f"gqa_norm{i}") for i in range(n_layers)]
        self.attns = [
            GroupedQueryAttention(
                d_model,
                cast(int, config["GQA_heads"]),
                cast(int, config["GQA_kv_heads"]),
                dropout_rate,
                name=f"gqa_attn{i}",
            )
            for i in range(n_layers)
        ]
        self.ffn_norms = [layers.LayerNormalization(name=f"gqa_ffn_norm{i}") for i in range(n_layers)]
        self.ffns = [_ffn_block(d_model, dropout_rate, name=f"gqa_ffn{i}") for i in range(n_layers)]

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        for norm, attn, ffn_norm, ffn in zip(self.attn_norms, self.attns, self.ffn_norms, self.ffns, strict=True):
            x = x + attn(norm(x), training=training)
            x = x + ffn(ffn_norm(x), training=training)
        return x


def _pool(x: tf.Tensor, method: str) -> tf.Tensor:
    if method == "last_token":
        return x[:, -1, :]
    if method == "mean":
        return cast(tf.Tensor, layers.GlobalAveragePooling1D()(x))
    raise ValueError(
        f"pooling_method={method!r} not specified anywhere in the design docs (only "
        f"last_token/mean have a formula — see 03-Model & Architecture Engineering.md)"
    )


class PredictionHead(layers.Layer):  # type: ignore[misc]  # TensorFlow's Layer is untyped.
    """MLP trunk (fusion_concatenation of the pooled deep representation + auxiliary_features) ->
    action_head (3-class softmax) + mean_std_pairs for [mfe, rer] (heteroscedastic, Gaussian NLL —
    outcome_set=1 in Tier-1_000.action_mfe_rer(handmade).outcome.jsonc, not the skew/kurtosis
    outcome_set=2)."""

    def __init__(self, config: dict[str, object], **kwargs: object) -> None:
        super().__init__(**kwargs)
        width = cast(int, config["MLP_width"])
        dropout_rate = cast(float, config["dropout"])
        self.trunk = tf_keras.Sequential(
            [
                layer
                for i in range(cast(int, config["MLP_depth"]))
                for layer in (
                    layers.Dense(width, activation="gelu", name=f"mlp_dense{i}"),
                    layers.Dropout(dropout_rate),
                )
            ],
            name="mlp_trunk",
        )
        self.action_head = layers.Dense(3, activation="softmax", dtype="float32", name="action")
        self.mfe_mean = layers.Dense(1, activation="softplus", dtype="float32", name="mfe_mean")
        self.mfe_std = layers.Dense(1, activation="softplus", dtype="float32", name="mfe_std")
        self.rer_mean = layers.Dense(1, activation="sigmoid", dtype="float32", name="rer_mean")
        self.rer_std = layers.Dense(1, activation="softplus", dtype="float32", name="rer_std")

    def call(
        self, pooled: tf.Tensor, auxiliary_features: tf.Tensor, training: bool | None = None
    ) -> dict[str, tf.Tensor]:
        x = self.trunk(layers.Concatenate()([pooled, auxiliary_features]), training=training)
        return {
            "action": self.action_head(x),
            # dtype='float32' here too — under mixed_float16 a Concatenate with no explicit dtype casts
            # its float32 Dense inputs down to the global compute dtype, silently losing the precision
            # the heads' own dtype='float32' was meant to guarantee for the loss computation.
            "mfe_params": layers.Concatenate(name="mfe_params", dtype="float32")([self.mfe_mean(x), self.mfe_std(x)]),
            "rer_params": layers.Concatenate(name="rer_params", dtype="float32")([self.rer_mean(x), self.rer_std(x)]),
        }


class Tier1000Model(tf_keras.Model):  # type: ignore[misc]  # TensorFlow's Model is untyped.
    """DS-01: six ModernTCN+LSTM timeframe branches -> Perceiver latent-bottleneck fusion -> GQA ->
    pooling -> MLP dual-head prediction. See module docstring for resolved config and deviations."""

    def __init__(self, config: dict[str, object] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.config = config or TIER1_000_CONFIG
        self.branch_encoders = {
            tf_name: TimeframeBranchEncoder(self.config, name=f"branch_{tf_name}") for tf_name in BRANCH_TIMEFRAMES
        }
        self.fusion = PerceiverFusion(self.config, name="perceiver_fusion")
        self.gqa = GQAEncoder(self.config, name="gqa_encoder")
        self.prediction_head = PredictionHead(self.config, name="prediction_head")
        self.pooling_method = cast(str, self.config["pooling_method"])

    def call(self, inputs: dict[str, tf.Tensor], training: bool | None = None) -> dict[str, tf.Tensor]:
        branch_sequences = [
            self.branch_encoders[tf_name](inputs[tf_name], training=training) for tf_name in BRANCH_TIMEFRAMES
        ]
        latents = self.fusion(branch_sequences, training=training)
        latents = self.gqa(latents, training=training)
        pooled = _pool(latents, self.pooling_method)
        return cast(dict[str, tf.Tensor], self.prediction_head(pooled, inputs["auxiliary_features"], training=training))


def gaussian_nll_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian NLL for a jointly-trained (mean, std) pair — mfe_params/rer_params' own loss, per
    Tier-1_000.action_mfe_rer(handmade).outcome.jsonc outcome_set=1 ("mean+std trained jointly per
    pair, not two independent MSE heads"). y_pred: (batch, 2) = [mean, std]; y_true: (batch, 1)
    realized value.
    """
    mean = y_pred[..., 0]
    std = tf.maximum(y_pred[..., 1], 1e-3)  # numerical floor against log(0)/div-by-0
    target = tf.reshape(tf.cast(y_true, mean.dtype), tf.shape(mean))
    return tf.reduce_mean(0.5 * tf.math.log(2.0 * math.pi) + tf.math.log(std) + 0.5 * tf.square((target - mean) / std))


def build_tier1000_model(config: dict[str, object] | None = None) -> Tier1000Model:
    config = config or TIER1_000_CONFIG
    policy = tf_keras.mixed_precision.Policy("mixed_float16")  # training.optimization.mixed_precision=enabled
    tf_keras.mixed_precision.set_global_policy(policy)
    model = Tier1000Model(config)
    optimizer = tf_keras.optimizers.AdamW(
        learning_rate=cast(float, config["learning_rate"]),
        weight_decay=cast(float, config["weight_decay"]),
        clipnorm=cast(float, config["gradient_clip_norm"]),
    )
    optimizer = tf_keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss={
            "action": tf_keras.losses.CategoricalCrossentropy(),
            "mfe_params": gaussian_nll_loss,
            "rer_params": gaussian_nll_loss,
        },
        loss_weights={"action": 1.0, "mfe_params": 1.0, "rer_params": 1.0},
    )
    return model

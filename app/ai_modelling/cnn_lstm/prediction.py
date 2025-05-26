"""
Prediction script for the CNN-LSTM P&L model.
Assumes the training code in `model.py` saved a *.keras* file next to the dataset
folders (see `train_model()`).

Usage
-----
python prediction.py
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import pandas as pd
from tensorflow import keras as tf_keras

from Config import app_config
from ai_modelling.base import dataset_folder, master_x_shape, overlapped_quarters
from ai_modelling.cnn_lstm_attention.cnn_lstm_attention_model import MultiHeadAttentionLayer
from ai_modelling.dataset_generator.training_datasets import train_data_of_mt_n_profit
from ai_modelling.dataset_generator.zip_pkl_batch import load_validators_zip_pkl, load_single_batch_zip_pkl
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.functions import date_range_to_string


def choose_dataset_files(x_shape: Dict[str, tuple[int, int]]) -> tuple[str, str, str]:
    """
    Pick *one* dataset-zip / validator-zip pair from the generated batches.
    They share the same index so their rows match one-to-one.
    """
    folder_name = dataset_folder(x_shape, batch_size=400)
    folder_path = os.path.join(app_config.path_of_data, folder_name)

    batch_files = sorted(
        f for f in os.listdir(folder_path) if f.startswith("dataset-") and f.endswith(".zip")
    )
    val_files = sorted(
        f for f in os.listdir(folder_path) if f.startswith("validators-") and f.endswith(".zip")
    )

    if not batch_files:
        raise FileNotFoundError("No *.zip* batches found in the dataset folder")

    # keep indices aligned
    idx = random.randint(0, len(batch_files) - 1)
    return folder_path, batch_files[idx], val_files[idx]


def expand_batch_dim(sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Keras expects a batch-axis → (1, T, C) instead of (T, C)."""
    return {k: np.expand_dims(v, 0) for k, v in sample.items()}


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def predict_once(x_shape: Dict[str, tuple[int, int]]) -> None:
    """
    * Loads the most recent (or a random) batch
    * Picks one random row
    * Runs the model and prints prediction vs. ground-truth
    """
    from tensorflow import convert_to_tensor, float16 as tf_float16
    model_path = os.path.join(
        app_config.path_of_data,
        "cnn_lstm_attention.mt_pnl_n_ind.cnn_f64c3k1.lstm_u512-256.dense_u128.drop_r0.3.keras",
        # "cnn_lstm.mt_pnl_n_ind.cnn_f64c4k2.lstm_u512-256.dense_u128.drop_r0.3.keras",
        # "cnn_lstm.mt_pnl_n_ind.cnn_f48c3k2.lstm_u256-128.dense_u128.drop_r0.3 - y_clipped_and_scaled.keras",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    model: tf_keras.Model = tf_keras.models.load_model(
        model_path,
        custom_objects={'MultiHeadAttentionLayer': MultiHeadAttentionLayer, },
        # custom_objects={"CNNLSTMModel": CNNLSTMModel, "CNNLSTMLayer": CNNLSTMLayer},
    )
    # model_compile(model)  # harmless even if only predicting

    # ---------------------------------------------------------------- dataset
    start = pd.to_datetime('1-01-25')
    end = pd.to_datetime('04-23-25')
    quarters = overlapped_quarters(date_range_to_string(start=start, end=end))
    random.shuffle(quarters)
    start, end = quarters[0]
    app_config.processing_date_range = date_range_to_string(start=start, end=end)
    symbol = 'BTCUSDT'
    app_config.under_process_symbol = symbol
    mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)
    Xs, ys, X_dfs, y_dfs, y_timeframe, y_debug_dfs = (
        train_data_of_mt_n_profit(
            structure_tf='4h', mt_ohlcv=mt_ohlcv, x_shape=x_shape, batch_size=10, dataset_batches=1,
            forecast_trigger_bars=3 * 4 * 4 * 4 * 1, ))

    row = random.randrange(len(ys))
    Xs = {
        'structure': convert_to_tensor(Xs['structure'], dtype=tf_float16),
        'pattern': convert_to_tensor(Xs['pattern'], dtype=tf_float16),
        'trigger': convert_to_tensor(Xs['trigger'], dtype=tf_float16),
        'double': convert_to_tensor(Xs['double'], dtype=tf_float16),
        # 'structure-indicators': convert_to_tensor(Xs['structure-indicators'], dtype=tf_float16),
        # 'pattern-indicators': convert_to_tensor(Xs['pattern-indicators'], dtype=tf_float16),
        # 'trigger-indicators': convert_to_tensor(Xs['trigger-indicators'], dtype=tf_float16),
        # 'double-indicators': convert_to_tensor(Xs['double-indicators'], dtype=tf_float16),
    }
    preds = model.predict(Xs, verbose=1)
    df = pd.DataFrame(
        preds, columns=["p_short_s", "p_long_s"], dtype=np.float32
    )
    df["y_short_s"] = ys[:, 0]
    df["y_long_s"] = ys[:, 1]

    df["ydf_short_s"] = pd.Series({d.index[0]: d["short_signal"].iloc[0] for d in y_dfs})
    df["ydf_long_s"] = pd.Series({d.index[0]: d["long_signal"].iloc[0] for d in y_dfs})
    df['final'] = False
    df.loc[((df["ydf_long_s"] > df["ydf_short_s"]) & (df['p_long_s'] > df['p_short_s'])
            | (df["ydf_long_s"] < df["ydf_short_s"]) & (df['p_long_s'] < df['p_short_s'])), 'final'] = True
    accuracy = len(df[df['final'].eq(True)]) / len(df)
    # ----------------------------------------------------------- show result
    print(f"\nSelected row #{row} – time-frame: {y_timeframe}")
    # print(f"Ground-truth  : {gt_y}")
    print(f"Model predicts: {preds}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    predict_once(master_x_shape)

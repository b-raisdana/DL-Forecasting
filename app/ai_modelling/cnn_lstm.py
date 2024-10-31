import os
import sys
from datetime import timedelta, datetime
from random import shuffle, seed
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, LeakyReLU, Flatten, Dense, Concatenate, Dropout, LSTM, BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model, load_model

from app.Config import config
from app.PreProcessing.encoding.rolling_mean_std import read_multi_timeframe_rolling_mean_std_ohlcv
from app.data_processing.ohlcv import read_multi_timeframe_ohlcv
from app.helper.data_preparation import single_timeframe
from app.helper.helper import date_range, log_d, date_range_to_string, profile_it
from app.training.trainer import mt_train_n_test

print('tensorflow:' + tf.__version__)

cnn_lstd_model_x_lengths = {
    'structure': (128, 5),
    'pattern': (256, 5),
    'trigger': (256, 5),
    'double': (256, 5),
}


class CustomEpochLogger(Callback):
    # def __init__(self, model=None):
    #     super().__init__()
    #     self.model = model  # Save the model instance

    def on_epoch_end(self, epoch, logs=None):
        # training_loss = logs.get('loss')
        # validation_loss = logs.get('val_loss')
        log_d(
            f" {config.under_process_symbol}:{config.processing_date_range}"
            # f"Epoch {epoch + 1}/{self.params['epochs']} - "
            # f"Training Loss: {training_loss:.4f} - "
            # f"Validation Loss: {validation_loss:.4f}"
        )

    def set_model(self, model):
        super().set_model(model)  # Call the parent class method
        # self.model = model  # Update the model instance if needed

@profile_it
def train_model(input_x: Dict[str, pd.DataFrame], input_y: pd.DataFrame, x_shapes, batch_size, model=None, filters=64,
                lstm_units_list: list = None, dense_units=64, cnn_count=3, cnn_kernel_growing_steps=2,
                dropout_rate=0.3, rebuild_model: bool = False, epochs=500):
    """
    Check if the model is already trained or partially trained. If not, build a new model.
    Continue training the model and save the trained model to 'cnn_lstm_model.h5' after each session.

    Args:
        input_x (Dict[str, pd.DataFrame]): Dictionary containing time-series data for the model inputs.
            The dictionary should have the following keys, each corresponding to a specific input component:
            - 'structure': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'pattern': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'trigger': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'double': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            The number of time steps (n_samples) must be the same for all keys in `X` and should match the length of `y`.

        input_y (pd.DataFrame): The target output, a DataFrame with the shape (n_samples, 1), where each entry corresponds to the target value for a time step.

        x_shapes (dict): Dictionary containing the input shapes for the different branches of the model. The input shapes should correspond to the data for 'structure', 'pattern', 'trigger', and 'double' as specified in `X`.

        model (optional): A pre-trained model to load. If not provided, a new model will be built.

    Returns:
        model (tf.keras.Model): The trained CNN-LSTM model.

    Raises:
        RuntimeError: If the lengths of the inputs in `X` or `y` are not the same, a RuntimeError will be raised.
    """

    # Verify structure of input_x
    for key in ['structure', 'pattern', 'trigger', 'double']:
        if key not in input_x:
            raise ValueError(f"Missing key '{key}' in input_x.")
        if input_x[key].shape[1:] != x_shapes[key]:
            raise ValueError(f"Shape mismatch for {key}: expected {x_shapes[key]}, got {input_x[key].shape[1:]}")

    # Check that all input lengths match
    input_lens = {
        'structure': len(input_x['structure']),
        'pattern': len(input_x['pattern']),
        'trigger': len(input_x['trigger']),
        'double': len(input_x['double']),
        'y': len(input_y),
    }
    unique_lengths = set(input_lens.values())
    if len(unique_lengths) > 1:
        raise RuntimeError(f'Batch sizes should be the same. input lengths: {input_lens}')

    model_path_h5 = os.path.join(config.path_of_data, 'cnn_lstm_model.h5')
    model_path_keras = os.path.join(config.path_of_data, 'cnn_lstm_model.keras')
    # Check if the model already exists, load if it does
    if model is None:
        if not rebuild_model and os.path.exists(model_path_keras):
            log_d("Loading existing keras model from disk...")
            model = load_model(model_path_keras)
        elif not rebuild_model and os.path.exists(model_path_h5):
            log_d("Loading existing h5 model from disk...")
            model = load_model(model_path_h5)
        else:
            log_d("Building new model...")
            model = build_model(x_shapes=x_shapes, y_shape=(input_y.shape[1], input_y.shape[2]), filters=filters,
                                lstm_units_list=lstm_units_list, dense_units=dense_units, cnn_count=cnn_count,
                                cnn_kernel_growing_steps=cnn_kernel_growing_steps, dropout_rate=dropout_rate)

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    epoch_logger = CustomEpochLogger()
    history = model.fit(
        x={'structure_model_input': input_x['structure'],
           'pattern_model_input': input_x['pattern'],
           'trigger_model_input': input_x['trigger'],
           'double_model_input': input_x['double']},
        y=input_y, epochs=epochs, batch_size=batch_size, validation_split=0.2,
        # Use a portion of your data for validation
        callbacks=[early_stopping, epoch_logger])
    log_d(history)
    model.save(model_path_keras)
    log_d("Model saved to disk.")

    return model


@profile_it
def create_cnn_lstm(x_shape, model_prefix, filters=64, lstm_units_list: list = None, dense_units=64, cnn_count=2,
                    cnn_kernel_growing_steps=2, dropout_rate=0.3):
    if lstm_units_list is None:
        lstm_units_list = [64, 64]
    input_layer = Input(shape=x_shape, name=f'{model_prefix}_input')

    # CNN Layers with growing filters and kernel sizes
    conv = input_layer
    for i in range(cnn_count):
        conv = Conv1D(filters=filters * (i + 1), kernel_size=3 + i * cnn_kernel_growing_steps, padding='same',
                      name=f'{model_prefix}_conv{i + 1}')(conv)
        conv = LeakyReLU(name=f'{model_prefix}_leaky_relu{i + 1}')(conv)
        conv = Dropout(dropout_rate, name=f'{model_prefix}_dropout_conv{i + 1}')(conv)
        conv = BatchNormalization(name=f'{model_prefix}_batch_norm_conv{i + 1}')(conv)

    flatten = Flatten(name=f'{model_prefix}_flatten')(conv)

    # Reshape for LSTM (LSTM expects 3D input: (batch_size, timesteps, features))
    # lstm_input = ExpandDimsLayer(axis=1)(flatten)
    lstm_input = Reshape((1, flatten.shape[-1]))(flatten)

    # Stack multiple LSTM layers with varying units
    for i, lstm_units in enumerate(lstm_units_list):
        return_seq = True if i < len(lstm_units_list) - 1 else False  # Only last LSTM should not return sequences
        lstm_input = LSTM(lstm_units, return_sequences=return_seq, name=f'{model_prefix}_lstm{i + 1}')(lstm_input)
        lstm_input = Dropout(dropout_rate, name=f'{model_prefix}_dropout_lstm{i + 1}')(lstm_input)

    # Dense layers
    dense = Dense(dense_units, name=f'{model_prefix}_dense1')(lstm_input)
    dense = LeakyReLU(name=f'{model_prefix}_leaky_relu_dense1')(dense)
    dense = Dropout(dropout_rate, name=f'{model_prefix}_dropout_dense1')(dense)

    dense = Dense(dense_units // 2, name=f'{model_prefix}_dense2')(dense)
    dense = LeakyReLU(name=f'{model_prefix}_leaky_relu_dense2')(dense)
    dense = Dropout(dropout_rate, name=f'{model_prefix}_dropout_dense2')(dense)

    # Final output layer for 40 outputs (20 pairs of high and low)
    output = Dense(1, activation='linear')(dense)  # Predicting 40 values (20 highs and 20 lows)

    # Model setup
    model = Model(inputs=input_layer, outputs=output, name=f'{model_prefix}_model')
    model.compile(optimizer='adam', loss='mse')
    return model


@profile_it
def build_model(x_shapes, y_shape: tuple[int, int], filters=64, lstm_units_list: list = None, dense_units=64,
                cnn_count=2,
                cnn_kernel_growing_steps=2, dropout_rate=0.3):
    structure_model = create_cnn_lstm(x_shapes['structure'], 'structure_model', filters,
                                      lstm_units_list, dense_units, cnn_count,
                                      cnn_kernel_growing_steps, dropout_rate)
    pattern_model = create_cnn_lstm(x_shapes['pattern'], 'pattern_model', filters,
                                    lstm_units_list, dense_units, cnn_count,
                                    cnn_kernel_growing_steps, dropout_rate)
    trigger_model = create_cnn_lstm(x_shapes['trigger'], 'trigger_model', filters,
                                    lstm_units_list, dense_units, cnn_count,
                                    cnn_kernel_growing_steps, dropout_rate)
    double_model = create_cnn_lstm(x_shapes['double'], 'double_model', filters,
                                   lstm_units_list, dense_units, cnn_count,
                                   cnn_kernel_growing_steps, dropout_rate)

    combined_output = Concatenate()(
        [structure_model.output, pattern_model.output, trigger_model.output, double_model.output])

    # Add an additional Dense layer with ReLU activation
    combined_dense = Dense(128)(combined_output)
    combined_dense = LeakyReLU()(combined_dense)

    # Final output layer (for regression tasks, use linear activation; for classification, consider sigmoid/softmax)
    # Final output layer for 40 outputs (20 pairs of high and low)
    final_output = Dense(np.prod(np.array(y_shape)), activation='linear')(
        combined_dense)  # Predicting 40 values (20 highs and 20 lows)

    # Optionally, reshape to (20, 2) to make it clear that we have 20 high-low pairs
    final_output = Reshape(y_shape)(final_output)

    # Define the final model
    inputs = {
        'structure_model_input': structure_model.input,
        'pattern_model_input': pattern_model.input,
        'trigger_model_input': trigger_model.input,
        'double_model_input': double_model.input}
    model = Model(inputs=inputs,
                  outputs=final_output)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def ceil_to_slide(t_date: datetime, slide: timedelta):
    if (t_date - datetime(t_date.year, t_date.month, t_date.day, tzinfo=t_date.tzinfo)) > timedelta(0):
        t_date = datetime(t_date.year, t_date.month, t_date.day + 1, tzinfo=t_date.tzinfo)
    days = (t_date - datetime(t_date.year, 1, 1, tzinfo=t_date.tzinfo)).days
    rounded_days = (days // slide.days) * slide.days + (slide.days if days % slide.days > 0 else 0)
    return datetime(t_date.year, 1, 1, tzinfo=t_date.tzinfo) + rounded_days * timedelta(days=1)


def overlapped_quarters(i_date_range, length=timedelta(days=30 * 3), slide=timedelta(days=30 * 1.5)):
    if i_date_range is None:
        i_date_range = config.processing_date_range
    start, end = date_range(i_date_range)
    rounded_start = ceil_to_slide(start, slide)
    list_of_periods = [(p_start, p_start + length) for p_start in
                       pd.date_range(rounded_start, end - length, freq=slide)]
    return list_of_periods


if __name__ == "__main__":
    print("Python:" + sys.version)
    config.processing_date_range = "22-12-29.00-00T24-10-24.00-00"
    seed(42)
    np.random.seed(42)

    while True:
        quarters = overlapped_quarters(config.processing_date_range)
        shuffle(quarters)
        for start, end in quarters:
            log_d(f'quarter start:{start} end:{end}##########################################')
            config.processing_date_range = date_range_to_string(start=start, end=end)
            for symbol in [
                'BTCUSDT',
                'ETHUSDT',
                'BNBUSDT',
                'EOSUSDT',
                'TRXUSDT',
                # 'TONUSDT',
                'SOLUSDT',
            ]:
                log_d(f'Symbol:{symbol}##########################################')
                config.under_process_symbol = symbol
                n_mt_ohlcv = read_multi_timeframe_rolling_mean_std_ohlcv(config.processing_date_range)
                mt_ohlcv = read_multi_timeframe_ohlcv(config.processing_date_range)
                base_ohlcv = single_timeframe(mt_ohlcv, '15min')
                batch_size = 128
                X, y, X_df, y_df = mt_train_n_test('4h', n_mt_ohlcv, cnn_lstd_model_x_lengths, batch_size)

                # plot_mt_train_n_test(X_df, y_df, 3, base_ohlcv)
                nop = 1
                t_model = train_model(X, y, cnn_lstd_model_x_lengths, batch_size)

"""
Potential Areas of Improvement for Professional Price Forecasting:

    Excessive Use of CNN Layers:
        While CNNs can capture local patterns in time-series data, the use of multiple convolutional layers might not be necessary for financial time-series forecasting. Generally, financial time-series models rely more heavily on recurrent structures like LSTMs or GRUs, rather than deep CNN architectures.
        You might consider reducing the number of CNN layers or simplifying the network to focus more on temporal dependencies.

    More Complex LSTM or GRU Structures:
        Instead of a single LSTM layer, you could consider stacking multiple LSTM layers or using GRU (Gated Recurrent Units), which is a simpler alternative but can sometimes perform better in price forecasting tasks.
        You could also experiment with Bidirectional LSTMs or Attention Mechanisms to give the model more flexibility in capturing dependencies both forward and backward in time.

    Incorporating External Features:
        Financial markets are often influenced by factors beyond just historical prices, such as trading volume, economic indicators, sentiment data, and news. You might want to integrate external features (such as trading volume, market sentiment, or macroeconomic variables) into your model.
        This could be done via multi-input models where different features (price, volume, sentiment) are processed separately and combined before the final prediction layer.

    Model Interpretability:
        For professional models, interpretability is important. You may want to ensure that the model's decisions are explainable, especially when dealing with financial data.
        Consider techniques like SHAP (Shapley Additive Explanations) or LIME (Local Interpretable Model-agnostic Explanations) for model interpretability, which can help you understand the decision-making process of your model.

    Advanced Time-Series Models:
        While CNN-LSTM models can perform well, there are also models like Transformer-based architectures (e.g., Temporal Fusion Transformers) or even ARIMA (AutoRegressive Integrated Moving Average) models that are tailored specifically for time-series forecasting tasks.
        XGBoost and LightGBM models have also been shown to perform well in certain forecasting scenarios, where you can create lagged features and use tree-based models.

Stacked LSTM Layers:

    You could experiment with a stacked LSTM, which would allow the model to capture more complex patterns over time.


lstm_1 = LSTM(lstm_units, return_sequences=True, name=f'{model_prefix}_lstm_1')(tf.expand_dims(flatten, axis=1))
lstm_2 = LSTM(lstm_units, return_sequences=False, name=f'{model_prefix}_lstm_2')(lstm_1)

Attention Mechanism:

    Adding an attention mechanism might help the model focus on more important time steps when making predictions. This is especially useful when the model is trying to predicting prices based on historical data with varying importance at different points in time.

Incorporate Financial Indicators:

    If you are working with stock or cryptocurrency prices, consider adding technical indicators like RSI, MACD, or Bollinger Bands as additional features. These indicators capture market trends and could improve the model's predictive power.

Experiment with More Complex Architectures:

    You could also experiment with Transformer-based models such as the Temporal Fusion Transformer (TFT), which have shown significant promise in time-series forecasting tasks, especially in financial data.

Hyperparameter Tuning:

    Use Grid Search or Random Search to fine-tune hyperparameters like filters, lstm_units, dropout_rate, and the number of CNN layers. This ensures the model is not underfitting or overfitting.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Flatten, Dense, Concatenate, LSTM
from tensorflow.python.keras.models import Model, load_model

from Config import config
from PreProcessing.encoding.rolling_mean_std import read_multi_timeframe_rolling_mean_std_ohlcv
from data_processing.fragmented_data import data_path
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.data_preparation import single_timeframe
from helper.helper import date_range
from training.trainer import mt_train_n_test

cnn_lstd_model_input_lengths = {
    'structure': 128,
    'pattern': 256,
    'trigger': 256,
    'double': 256,
}


def train_model(X: Dict[str:pd.DataFrame], y: pd.DataFrame, input_shapes, model=None):
    """
    Check if the model is already trained or partially trained. If not, build a new model.
    Continue training the model and save the trained model to 'cnn_lstm_model.h5' after each session.

    Args:
        X (Dict[str, pd.DataFrame]): Dictionary containing time-series data for the model inputs.
            The dictionary should have the following keys, each corresponding to a specific input component:
            - 'structure': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'pattern': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'trigger': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'double': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            The number of time steps (n_samples) must be the same for all keys in `X` and should match the length of `y`.

        y (pd.DataFrame): The target output, a DataFrame with the shape (n_samples, 1), where each entry corresponds to the target value for a time step.

        input_shapes (dict): Dictionary containing the input shapes for the different branches of the model. The input shapes should correspond to the data for 'structure', 'pattern', 'trigger', and 'double' as specified in `X`.

        model (optional): A pre-trained model to load. If not provided, a new model will be built.

    Returns:
        model (tf.keras.Model): The trained CNN-LSTM model.

    Raises:
        RuntimeError: If the lengths of the inputs in `X` or `y` are not the same, a RuntimeError will be raised.
    """
    structure_x = np.array(X['structure'][['n_open', 'n_high', 'n_low', 'n_close', 'n_volume', ]])
    pattern_x = np.array(X['pattern'][['n_open', 'n_high', 'n_low', 'n_close', 'n_volume', ]])
    trigger_x = np.array(X['trigger'][['n_open', 'n_high', 'n_low', 'n_close', 'n_volume', ]])
    double_x = np.array(X['double'][['n_open', 'n_high', 'n_low', 'n_close', 'n_volume', ]])

    # Check that all input lengths match
    input_lens = {
        'structure': len(X['structure']),
        'pattern': len(X['pattern']),
        'trigger': len(X['trigger']),
        'double': len(X['double']),
        'y': len(y),
    }

    unique_lengths = set(input_lens.values())
    if len(unique_lengths) > 1:
        raise RuntimeError(f'Batch sizes should be the same. input lengths: {input_lens}')

    # Check if the model already exists, load if it does
    if model is None:
        model_path = os.path.join(data_path(), 'cnn_lstm_model.h5')
        if os.path.exists(model_path):
            print("Loading existing model from disk...")
            model = load_model(model_path)
        else:
            print("Building new model...")
            model = build_model(input_shapes)

    # Train the model
    history = model.fit([structure_x, pattern_x, trigger_x, double_x], y,
                        epochs=10, batch_size=len(y))
    print(history)
    # Save the model after each training session to avoid losing progress
    model.save(model_path)
    print("Model saved to disk.")

    return model


def create_cnn_lstm(input_shape, model_prefix, filters=64, lstm_units=64, dense_units=64, cnn_count=3,
                    cnn_kernel_growing_steps=2, dropout_rate=0.3):
    input_layer = Input(shape=input_shape, name=f'{model_prefix}_input')

    # CNN Layers with growing filters and kernel sizes
    conv = input_layer
    for i in range(cnn_count):
        conv = Conv1D(filters=filters * (i + 1), kernel_size=3 + i * cnn_kernel_growing_steps, padding='same',
                      name=f'{model_prefix}_conv{i + 1}')(conv)
        conv = LeakyReLU(name=f'{model_prefix}_leaky_relu{i + 1}')(conv)
        conv = Dropout(dropout_rate, name=f'{model_prefix}_dropout_conv{i + 1}')(conv)
        conv = BatchNormalization(name=f'{model_prefix}_batchnorm_conv{i + 1}')(conv)

    # Flatten the CNN output
    flatten = Flatten(name=f'{model_prefix}_flatten')(conv)

    # LSTM Layer with optional sequence return
    lstm = LSTM(lstm_units, return_sequences=False, name=f'{model_prefix}_lstm')(tf.expand_dims(flatten, axis=1))
    lstm = Dropout(dropout_rate, name=f'{model_prefix}_dropout_lstm')(lstm)

    # Dense layers with increasing complexity
    dense = Dense(dense_units, name=f'{model_prefix}_dense1')(lstm)
    dense = LeakyReLU(name=f'{model_prefix}_leaky_relu_dense1')(dense)
    dense = Dropout(dropout_rate, name=f'{model_prefix}_dropout_dense1')(dense)

    dense = Dense(dense_units // 2, name=f'{model_prefix}_dense2')(dense)
    dense = LeakyReLU(name=f'{model_prefix}_leaky_relu_dense2')(dense)
    dense = Dropout(dropout_rate, name=f'{model_prefix}_dropout_dense2')(dense)

    # Final output layer
    output_layer = Dense(1, activation='linear', name=f'{model_prefix}_output')(dense)

    # Model setup
    model = Model(inputs=input_layer, outputs=output_layer, name=f'{model_prefix}_model')

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    return model
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

    Adding an attention mechanism might help the model focus on more important time steps when making predictions. This is especially useful when the model is trying to predict prices based on historical data with varying importance at different points in time.

Incorporate Financial Indicators:

    If you are working with stock or cryptocurrency prices, consider adding technical indicators like RSI, MACD, or Bollinger Bands as additional features. These indicators capture market trends and could improve the model's predictive power.

Experiment with More Complex Architectures:

    You could also experiment with Transformer-based models such as the Temporal Fusion Transformer (TFT), which have shown significant promise in time-series forecasting tasks, especially in financial data.

Hyperparameter Tuning:

    Use Grid Search or Random Search to fine-tune hyperparameters like filters, lstm_units, dropout_rate, and the number of CNN layers. This ensures the model is not underfitting or overfitting.
"""


def build_model(input_shapes):
    structure_model = create_cnn_lstm((input_shapes['structure'], 5), 'structure_model')
    pattern_model = create_cnn_lstm((input_shapes['pattern'], 5), 'pattern_model')
    trigger_model = create_cnn_lstm((input_shapes['trigger'], 5), 'trigger_model')
    double_model = create_cnn_lstm((input_shapes['double'], 5), 'double_model')

    combined_output = Concatenate()(
        [structure_model.output, pattern_model.output, trigger_model.output, double_model.output])

    # Add an additional Dense layer with ReLU activation
    combined_dense = Dense(128)(combined_output)
    combined_dense = LeakyReLU()(combined_dense)

    # Final output layer (for regression tasks, use linear activation; for classification, consider sigmoid/softmax)
    final_output = Dense(1, activation='linear')(combined_dense)

    # Define the final model
    model = Model(inputs=[structure_model.input, pattern_model.input, trigger_model.input, double_model.input],
                  outputs=final_output)

    # Compile the model with mean squared error loss for regression tasks
    model.compile(optimizer='adam', loss='mse')

    # Model summary
    model.summary()

    return model


# config.processing_date_range = date_range_to_string(start=pd.to_datetime('03-01-24'),
#                                                     end=pd.to_datetime('09-01-24'))
# # devided by rolling mean, std
# n_mt_ohlcv = pd.read_csv(
#     os.path.join(r"C:\Code\dl-forcasting\data\Kucoin\Spot\BTCUSDT",
#                  f"n_mt_ohlcv.{config.processing_date_range}.csv.zip"), compression='zip')
# n_mt_ohlcv
# config.processing_date_range = "24-03-01.00-00T24-09-01.00-00"
config.processing_date_range = "24-03-01.00-00T24-06-01.00-00"
t = date_range(config.processing_date_range)
n_mt_ohlcv = read_multi_timeframe_rolling_mean_std_ohlcv(config.processing_date_range)
mt_ohlcv = read_multi_timeframe_ohlcv(config.processing_date_range)
base_ohlcv = single_timeframe(mt_ohlcv, '15min')
X, y, X_df, y_df = mt_train_n_test('4h', n_mt_ohlcv, cnn_lstd_model_input_lengths, batch_size=10)

# plot_mt_train_n_test(X_df, y_df, 3, base_ohlcv)
nop = 1
t_model = train_model(X, y, cnn_lstd_model_input_lengths)

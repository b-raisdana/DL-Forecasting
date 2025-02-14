import os
from typing import Dict

import numpy as np
import pandas as pd

from Config import app_config
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.training_datasets import x_shape_assertion
from helper.br_py.logging import log_d
from helper.br_py.profiling import profile_it


@profile_it
def build_model(x_shape, y_shape: int, filters=64, lstm_units_list: list = None, dense_units=64,
                cnn_count=2, cnn_kernel_growing_steps=2, dropout_rate=0.3):
    from tensorflow.keras.layers import Dense
    from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
    from tensorflow.python.keras.layers.core import Reshape
    from tensorflow.python.keras.layers.merge import Concatenate
    from tensorflow.python.keras.models import Model

    import importlib

    tf_version = importlib.metadata.version("tensorflow")
    log_d('tensorflow:' + tf_version)
    structure_model = create_cnn_lstm(x_shape['structure'], 'structure_model', filters,
                                      lstm_units_list, dense_units, cnn_count,
                                      cnn_kernel_growing_steps, dropout_rate)
    pattern_model = create_cnn_lstm(x_shape['pattern'], 'pattern_model', filters,
                                    lstm_units_list, dense_units, cnn_count,
                                    cnn_kernel_growing_steps, dropout_rate)
    trigger_model = create_cnn_lstm(x_shape['trigger'], 'trigger_model', filters,
                                    lstm_units_list, dense_units, cnn_count,
                                    cnn_kernel_growing_steps, dropout_rate)
    double_model = create_cnn_lstm(x_shape['double'], 'double_model', filters,
                                   lstm_units_list, dense_units, cnn_count,
                                   cnn_kernel_growing_steps, dropout_rate)
    structure_indicators_model = create_cnn_lstm(x_shape['indicators'], 'structure_indicators_model', filters,
                                                 lstm_units_list, dense_units, cnn_count,
                                                 cnn_kernel_growing_steps, dropout_rate)
    pattern_indicators_model = create_cnn_lstm(x_shape['indicators'], 'pattern_indicators_model', filters,
                                               lstm_units_list, dense_units, cnn_count,
                                               cnn_kernel_growing_steps, dropout_rate)
    trigger_indicators_model = create_cnn_lstm(x_shape['indicators'], 'trigger_indicators_model', filters,
                                               lstm_units_list, dense_units, cnn_count,
                                               cnn_kernel_growing_steps, dropout_rate)
    double_indicators_model = create_cnn_lstm(x_shape['indicators'], 'double_indicators_model', filters,
                                              lstm_units_list, dense_units, cnn_count,
                                              cnn_kernel_growing_steps, dropout_rate)

    combined_output = Concatenate()(
        [structure_model.output, pattern_model.output, trigger_model.output, double_model.output,
         structure_indicators_model.output, pattern_indicators_model.output,
         trigger_indicators_model.output, double_indicators_model.output])

    # Add a Dense layer with ReLU activation
    combined_dense = Dense(128)(combined_output)
    combined_dense = LeakyReLU()(combined_dense)

    final_output = Dense(np.prod(np.array(y_shape)), activation='linear')(combined_dense)

    final_output = Reshape((y_shape,))(final_output)

    # Define the final model
    inputs = {
        'structure_model_input': structure_model.input,
        'pattern_model_input': pattern_model.input,
        'trigger_model_input': trigger_model.input,
        'double_model_input': double_model.input,
        'structure_indicators_model_input': structure_indicators_model.input,
        'pattern_indicators_model_input': pattern_indicators_model.input,
        'trigger_indicators_model_input': trigger_indicators_model.input,
        'double_indicators_model_input': double_indicators_model.input,
    }
    model = Model(inputs=inputs,
                  outputs=final_output)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


@profile_it
def create_cnn_lstm(x_shape, model_prefix, filters=64, lstm_units_list: list = None, dense_units=64, cnn_count=2,
                    cnn_kernel_growing_steps=2, dropout_rate=0.05):  # dropout_rate=0.1
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.python.keras.layers.convolutional import Conv1D
    from tensorflow.python.keras.engine.input_layer import Input
    from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
    from tensorflow.python.keras.layers.core import Flatten, Reshape
    from tensorflow.python.keras.layers.recurrent import LSTM
    from tensorflow.python.keras.models import Model

    if lstm_units_list is None:
        lstm_units_list = [64, 64]  # 256, 128
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


@profile_it(extract_params=False)
def train_model(input_x: Dict[str, pd.DataFrame], input_y: pd.DataFrame, x_shape, batch_size, model=None, filters=64,
                lstm_units_list: list = None, dense_units=64, cnn_count=3, cnn_kernel_growing_steps=2,
                dropout_rate=0.3, rebuild_model: bool = False, epochs=500):
    """
    Check if the model is already trained or partially trained. If not, build a new model.
    Continue training_data the model and save the trained model to 'cnn_lstm_model.h5' after each session.

    Args:
        input_x (Dict[str, pd.DataFrame]): Dictionary containing time-series data for the model inputs.
            The dictionary should have the following keys, each corresponding to a specific input component:
            - 'structure': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'pattern': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'trigger': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            - 'double': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
            The number of time steps (n_samples) must be the same for all keys in `X` and should match the length of `y`.

        input_y (pd.DataFrame): The target output, a DataFrame with the shape (n_samples, 1), where each entry corresponds to the target value for a time step.

        x_shape (dict): Dictionary containing the input shapes for the different branches of the model. The input shapes should correspond to the data for 'structure', 'pattern', 'trigger', and 'double' as specified in `X`.

        model (optional): A pre-trained model to load. If not provided, a new model will be built.

    Returns:
        model (tf.keras.Model): The trained CNN-LSTM model.

    Raises:
        RuntimeError: If the lengths of the inputs in `X` or `y` are not the same, a RuntimeError will be raised.
    """
    from tensorflow.python.keras.callbacks import Callback
    from tensorflow.python.keras.saving.save import load_model
    from tensorflow.python.keras.callbacks import EarlyStopping

    class CustomEpochLogger(Callback):
        # def __init__(self, model=None):
        #     super().__init__()
        #     self.model = model  # Save the model instance

        def on_epoch_end(self, epoch, logs=None):
            # training_loss = logs.get('loss')
            # validation_loss = logs.get('val_loss')
            log_d(
                f" {app_config.under_process_symbol}:{app_config.processing_date_range}"
                # f"Epoch {epoch + 1}/{self.params['epochs']} - "
                # f"Training Loss: {training_loss:.4f} - "
                # f"Validation Loss: {validation_loss:.4f}"
            )

        def set_model(self, model):
            super().set_model(model)  # Call the parent class method
            # self.model = model  # Update the model instance if needed

    x_shape_assertion(input_x, batch_size, x_shape)

    model_path_h5 = os.path.join(app_config.path_of_data, 'cnn_lstm_model.mt_profit_n_loss_n_indicators.h5')
    model_path_keras = os.path.join(app_config.path_of_data, 'cnn_lstm_model.mt_profit_n_loss_n_indicators.keras')
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
            model = build_model(x_shape=x_shape, y_shape=input_y.shape[1], filters=filters,
                                lstm_units_list=lstm_units_list, dense_units=dense_units, cnn_count=cnn_count,
                                cnn_kernel_growing_steps=cnn_kernel_growing_steps, dropout_rate=dropout_rate)

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    epoch_logger = CustomEpochLogger()
    history = model.fit(
        x={'structure_model_input': input_x['structure'],
           'pattern_model_input': input_x['pattern'],
           'trigger_model_input': input_x['trigger'],
           'double_model_input': input_x['double'],
           'structure_indicators_model_input': input_x['structure-indicators'],
           'pattern_indicators_model_input': input_x['pattern-indicators'],
           'trigger_indicators_model_input': input_x['trigger-indicators'],
           'double_indicators_model_input': input_x['double-indicators'],
           },
        y=input_y, epochs=epochs, batch_size=batch_size, validation_split=0.2,
        # Use a portion of your data for validation
        callbacks=[early_stopping, epoch_logger])
    log_d(history)
    model.save(model_path_keras)
    log_d("Model saved to disk.")

    return model

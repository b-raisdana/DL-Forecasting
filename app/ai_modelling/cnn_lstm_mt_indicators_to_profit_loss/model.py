import os
from typing import Dict

import pandas as pd

from Config import app_config
from helper.br_py.logging import log_d


def train_model(input_x: Dict[str, pd.DataFrame], input_y: pd.DataFrame, x_shape, batch_size, model=None,
                cnn_filters=64,
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
    from tensorflow import keras as tf_keras

    from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.cnn_lstm_model import CNNLSTMModel
    from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.training_datasets import x_shape_assertion

    class CustomEpochLogger(tf_keras.callbacks.Callback):
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
    model_name = (f"cnn_lstm.mt_pnl_n_ind"
                  f".cnn_f{cnn_filters}c{cnn_count}k{cnn_kernel_growing_steps}."
                  f"lstm_u{"-".join([str(i) for i in lstm_units_list])}.dense_u{dense_units}.drop_r{dropout_rate}")
    # model_path_h5 = os.path.join(app_config.path_of_data, f'{model_name}.h5')
    model_path_keras = os.path.join(app_config.path_of_data, f'{model_name}.keras')
    # Check if the model already exists, load if it does
    if model is None:
        if not rebuild_model and os.path.exists(model_path_keras):
            log_d("Loading existing keras model from disk...")
            model = tf_keras.models.load_model(model_path_keras)
        # elif not rebuild_model and os.path.exists(model_path_h5):
        #     log_d("Loading existing h5 model from disk...")
        #     model = tf_keras.models.load_model(model_path_h5)
        else:
            log_d("Building new model...")
            model = CNNLSTMModel(y_shape=input_y.shape[1:], cnn_filters=cnn_filters, lstm_units_list=lstm_units_list,
                                 dense_units=dense_units, cnn_count=cnn_count,
                                 cnn_kernel_growing_steps=cnn_kernel_growing_steps, dropout_rate=dropout_rate)
            model.compile(loss='mse')
            batch_shape = {k: (batch_size,) + v for k, v in x_shape.items()}
            model.build(input_shape=batch_shape)
            model.summary()

    # Train the model
    early_stopping: callable = tf_keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                                                restore_best_weights=True)
    epoch_logger: callable = CustomEpochLogger()
    history = model.fit(x={
        'structure': input_x['structure'],
        'pattern': input_x['pattern'],
        'trigger': input_x['trigger'],
        'double': input_x['double'],
        'structure_indicators': input_x['structure-indicators'],
        'pattern_indicators': input_x['pattern-indicators'],
        'trigger_indicators': input_x['trigger-indicators'],
        'double_indicators': input_x['double-indicators'],
    },
        y=input_y, epochs=epochs, batch_size=batch_size, validation_split=0.2,
        # Use a portion of your data for validation
        callbacks=[early_stopping, epoch_logger])
    log_d(history)
    model.save(model_path_keras)
    log_d("Model saved to disk.")

    return model

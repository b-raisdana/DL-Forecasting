import logging
import os
import sys
from datetime import datetime
from random import shuffle
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Config import app_config
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.base import overlapped_quarters, master_x_shape, load_batch_zip
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.cnn_lstm_model import CNNLSTMLayer
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.training_datasets import ndarray_stats
from helper.br_py.br_py.base import sync_br_lib_init
from helper.br_py.br_py.do_log import log_d
from helper.functions import date_range_to_string


def train_model(input_x: Dict[str, pd.DataFrame], input_y: pd.DataFrame, x_shape: Dict[str, Tuple[int, int]],
                batch_size,
                model=None, cnn_filters=64, lstm_units_list: list = None, dense_units=64, cnn_count=3,
                cnn_kernel_growing_steps=2, dropout_rate=0.3, rebuild_model: bool = False, epochs=500,
                steps_per_epoch=10):
    """
    Check if the model is already trained or partially trained. If not, build a new model.
    Continue training_data the model and save the trained model to 'cnn_lstm_model.h5' after each session.

    input_x (Dict[str, pd.DataFrame]): Dictionary containing time-series data for the model inputs.
        The dictionary should have the following keys, each corresponding to a specific input component:
        - 'structure': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
        - 'pattern': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
        - 'trigger': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
        - 'double': DataFrame with shape (n_samples, 5), where the columns represent [n_open, n_high, n_low, n_close, n_volume].
        The number of time steps (n_samples) must be the same for all keys in `X` and should match the length of `y`.

    input_y (pd.DataFrame): The target output, a DataFrame with the shape (n_samples, 1), where each entry corresponds to the target value for a time step.

    x_shape (dict): Dictionary containing the input shapes for the different branches of the model. The input shapes should correspond to the data for 'structure', 'pattern', 'trigger', and 'double' as specified in `X`.

    Raises:
        RuntimeError: If the lengths of the inputs in `X` or `y` are not the same, a RuntimeError will be raised.
    """
    from tensorflow import keras as tf_keras
    from tensorflow import config as tf_config

    from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.cnn_lstm_model import CNNLSTMModel

    def model_compile(t_model: tf_keras.Model) -> tf_keras.Model:
        opt = tf_keras.optimizers.RMSprop(clipvalue=1.0)
        opt = tf_keras.mixed_precision.LossScaleOptimizer(opt)
        t_model.compile(loss='mse', optimizer=opt)
        # t_model.compile(loss=tf_keras.losses.Huber(), optimizer=opt)
        return t_model

    class CustomEpochLogger(tf_keras.callbacks.Callback):
        def on_epoch_start(self, epoch, logs=None):
            print(f"\n>>> Epoch {epoch} started at {datetime.now().strftime('%H:%M:%S.%f')}")

        def on_epoch_end(self, epoch, logs=None):
            training_loss = logs.get('loss')
            validation_loss = logs.get('val_loss')
            print(
                f"\n<<<Ends@{datetime.now().strftime('%H%M%S')}:{app_config.under_process_symbol}:{app_config.processing_date_range}/TrainL:{training_loss}/ValidL:{validation_loss}"
            )

        def set_model(self, model):
            super().set_model(model)

    input_y = input_y.astype('float32')

    # policy = tf_keras.mixed_precision.Policy('mixed_float16')
    # tf_keras.mixed_precision.set_global_policy(policy)
    # Check if a GPU is available
    if len(tf_config.list_physical_devices('GPU')) == 0:
        print("No GPU found, using CPU.")
    else:
        print("GPU found, using GPU.")
    model_name = (f"cnn_lstm.mt_pnl_n_ind"
                  f".cnn_f{cnn_filters}c{cnn_count}k{cnn_kernel_growing_steps}."
                  f"lstm_u{'-'.join([str(i) for i in lstm_units_list])}.dense_u{dense_units}.drop_r{dropout_rate}")
    # model_path_h5 = os.path.join(app_config.path_of_data, f'{model_name}.h5')
    model_path_keras = os.path.join(app_config.path_of_data, f'{model_name}.keras')
    # Check if the model already exists, load if it does
    if model is not None:
        raise NotImplementedError
    if not rebuild_model and os.path.exists(model_path_keras):
        log_d("Loading existing keras model from disk...")
        model = tf_keras.models.load_model(model_path_keras,
                                           custom_objects={'CNNLSTMModel': CNNLSTMModel, 'CNNLSTMLayer': CNNLSTMLayer})
        model_compile(model)  # Re-compile without loading optimizer state
        # tf_keras.utils.plot_model(model, to_file=os.path.join(app_config.path_of_plots,
        #                                                       f"model-plot.{int(datetime.now().timestamp())}.jpg")
        #                           , show_shapes=True, show_layer_names=True)
    # elif not rebuild_model and os.path.exists(model_path_h5):
    #     log_d("Loading existing h5 model from disk...")
    #     model = tf_keras.models.load_model(model_path_h5)
    else:
        log_d("Building new model...")
        model = CNNLSTMModel(y_shape=input_y.shape[1:], cnn_filters=cnn_filters, lstm_units_list=lstm_units_list,
                             dense_units=dense_units, cnn_count=cnn_count,
                             cnn_kernel_growing_steps=cnn_kernel_growing_steps, dropout_rate=dropout_rate)
        model_compile(model)
        batch_shape = {k: (batch_size,) + v for k, v in x_shape.items()}
        model.build(input_shape=batch_shape)
        model.summary()
        # tf_keras.utils.plot_model(model, to_file=os.path.join(app_config.path_of_plots,
        #                                                       f"model-plot.{int(datetime.now().timestamp())}")
        #                           , show_shapes=True, show_layer_names=True)

    # Train the model
    early_stopping: callable = tf_keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                                                restore_best_weights=True)
    epoch_logger: callable = CustomEpochLogger()
    tensorboard = setup_tensorboard()
    stats = ndarray_stats(np.array(input_y), ['long_signal', 'short_signal', 'min_low', 'max_high', 'long_profit',
       'short_profit', 'long_risk', 'short_risk', 'long_drawdown',
       'short_drawdown', 'long_drawdown', 'short_drawdown'])
    print("input_y stats:")
    print(stats)
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
        y=input_y, epochs=epochs, batch_size=int(batch_size / steps_per_epoch), validation_split=0.2,
        steps_per_epoch=steps_per_epoch,
        # Use a portion of your data for validation
        callbacks=[early_stopping, epoch_logger, tensorboard])
    log_d(history)
    model.save(model_path_keras)
    log_d("Model saved to disk.")

    return model


def setup_gpu():
    from tensorflow import config as tf_config
    from tensorflow import data as tf_data

    tf_data.experimental.enable_debug_mode()
    tf_config.run_functions_eagerly(True)

    physical_devices = tf_config.list_physical_devices('GPU')
    expanded_memory_size = 0.95 * 8 * 1024
    for device in physical_devices:
        tf_config.experimental.set_memory_growth(device, True)
        tf_config.set_logical_device_configuration(
            device=device,
            logical_devices=[
                tf_config.LogicalDeviceConfiguration(memory_limit=expanded_memory_size)
            ]
        )


def setup_tensorboard():
    from tensorflow import keras as tf_keras
    this_run_folder = "TFB-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    this_run_log_path = os.path.join(app_config.path_of_logs, this_run_folder)
    os.mkdir(this_run_log_path)
    tensorboard = tf_keras.callbacks.TensorBoard(log_dir=this_run_log_path, write_graph=True, write_images=True,
                                                 histogram_freq=1)
    return tensorboard


def run_trainer():
    log_d("Starting")
    sync_br_lib_init(path_of_logs='logs', root_path=app_config.root_path, log_to_file_level=logging.DEBUG,
                     log_to_std_out_level=logging.DEBUG)
    # parser = argparse.ArgumentParser(description="Script for processing OHLCV data.")
    # args = parser.parse_args()
    app_config.processing_date_range = date_range_to_string(start=pd.to_datetime('03-01-24'),
                                                            end=pd.to_datetime('09-01-24'))
    quarters = overlapped_quarters(app_config.processing_date_range)
    batch_size = 20
    # parser.add_argument("--do_not_fetch_prices", action="store_true", default=False,
    #                     help="Flag to indicate if prices should not be fetched (default: False).")
    print("Python:" + sys.version)
    # Apply config from arguments
    app_config.processing_date_range = "22-08-15.00-00T24-10-30.00-00"
    # config.do_not_fetch_prices = args.do_not_fetch_prices
    # seed(42)
    # np.random.seed(42)
    setup_gpu()
    while True:
        shuffle(quarters)
        for start, end in quarters:
            log_d(f'quarter start:{start} end:{end}##########################################')
            app_config.processing_date_range = date_range_to_string(start=start, end=end)
            for symbol in [
                'BTCUSDT',
                # # # 'ETHUSDT',
                # 'BNBUSDT',
                # 'EOSUSDT',
                # # 'TRXUSDT',
                # 'TONUSDT',
                # # 'SOLUSDT',
            ]:
                log_d(f'Symbol:{symbol}##########################################')
                app_config.under_process_symbol = symbol

                Xs, ys = load_batch_zip(master_x_shape, batch_size)
                train_model(input_x=Xs, input_y=ys, x_shape=master_x_shape, batch_size=batch_size, cnn_filters=16,
                            lstm_units_list=[64 * 12, 8 * 12], dense_units=32 * 12, cnn_count=1 * 12,
                            cnn_kernel_growing_steps=2,
                            dropout_rate=0.3, rebuild_model=False, epochs=10, steps_per_epoch=5)


if __name__ == "__main__":
    run_trainer()

# todo: check input_y data:
#  + there are duplicate columns: ['long_signal', 'short_signal', 'min_low', 'max_high', 'long_profit',
#        'short_profit', 'long_risk', 'short_risk', 'long_drawdown',
#        'short_drawdown', 'long_drawdown', 'short_drawdown']
#  + are all of these required?
#  + does loss function calculates deviation correctly?
#  + optimize GPU usage currently is about 5%
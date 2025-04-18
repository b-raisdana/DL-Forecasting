import logging
import os
import sys
from datetime import datetime
from math import isclose
from typing import Dict, Tuple, Literal, Generator

from tensorflow.data import Dataset

from Config import app_config
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.base import master_x_shape, infinite_load_batch_zip
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.cnn_lstm_model import CNNLSTMLayer
from helper.br_py.br_py.base import sync_br_lib_init
from helper.br_py.br_py.do_log import log_d


def train_model(
        train_dataset: Dataset | Generator, val_dataset: Dataset | Generator,
        x_shape: Dict[str, Tuple[int, int]],
        y_len: int,
        batch_size,
        model=None, cnn_filters=64, lstm_units_list: list = None, dense_units=64, cnn_count=3,
        cnn_kernel_growing_steps=2, dropout_rate=0.3, rebuild_model: bool = False, epochs=500,
        dataset_mode: bool = True,
):
    import gc
    from tensorflow import keras as tf_keras
    from tensorflow import config as tf_config

    from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.cnn_lstm_model import CNNLSTMModel

    def model_compile(t_model: tf_keras.Model) -> tf_keras.Model:
        opt = tf_keras.optimizers.RMSprop(clipnorm=1.0)
        opt = tf_keras.mixed_precision.LossScaleOptimizer(opt)
        t_model.compile(loss=tf_keras.losses.Huber(), optimizer=opt)
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

    policy = tf_keras.mixed_precision.Policy('mixed_float16')
    tf_keras.mixed_precision.set_global_policy(policy)
    # Check if a GPU is available
    if len(tf_config.list_physical_devices('GPU')) == 0:
        print("No GPU found, using CPU.")
    else:
        print("GPU found, using GPU.")
    model_name = (f"cnn_lstm.mt_pnl_n_ind"
                  f".cnn_f{cnn_filters}c{cnn_count}k{cnn_kernel_growing_steps}."
                  f"lstm_u{'-'.join([str(i) for i in lstm_units_list])}.dense_u{dense_units}.drop_r{dropout_rate}")
    model_path_keras = os.path.join(app_config.path_of_data, f'{model_name}.keras')
    # Check if the model already exists, load if it does
    if model is None:
        if not rebuild_model and os.path.exists(model_path_keras):
            log_d("Loading existing keras model from disk...")
            model = tf_keras.models.load_model(model_path_keras,
                                               custom_objects={'CNNLSTMModel': CNNLSTMModel,
                                                               'CNNLSTMLayer': CNNLSTMLayer})
            model_compile(model)  # Re-compile without loading optimizer state
        else:
            log_d("Building new model...")
            model = CNNLSTMModel(y_len=y_len, cnn_filters=cnn_filters, lstm_units_list=lstm_units_list,
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
    if dataset_mode:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=128,
            validation_steps=16,
            callbacks=[early_stopping, epoch_logger, tensorboard])
    else:
        Xs, ys = next(train_dataset)
        validation_data = next(val_dataset)
        history = model.fit(
            x=Xs,
            y=ys,
            validation_data=validation_data,
            # validation_split=0.2,
            epochs=epochs,
            # batch_size=int(batch_size / steps_per_epoch),
            # Use a portion of your data for validation
            callbacks=[early_stopping, epoch_logger, tensorboard])
    log_d(history)
    model.save(model_path_keras)
    log_d("Model saved to disk.")
    # tf_keras.backend.clear_session()
    # gc.collect()
    return model


def setup_gpu():
    from tensorflow import config as tf_config

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
                                                 histogram_freq=0)
    return tensorboard


from threading import Thread
from queue import Queue


class PrefetchGenerator:
    def __init__(self, generator_fn, max_prefetch=10):
        self.generator = generator_fn()
        self.queue = Queue(max_prefetch)
        self.thread = Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item


def dataset_generator( batch_size: int):
    """
    Yields individual samples (not batches). Use `.batch(...)` later in tf.data pipeline.
    Mode must be either 'train' or 'val'.
    """
    from tensorflow import convert_to_tensor, float32 as tf_float32
    # if not isclose(batch_size / ((1 / val_rate) - 1), int(batch_size / ((1 / val_rate) - 1))):
    #     raise ValueError(
    #         f"Batch size * validation rate should result into integer. "
    #         f"nearest batch_size={int(batch_size / ((1 / val_rate) - 1))*((1 / val_rate) - 1)}.")
    # if mode == 'train':
    #     samples_to_fetch = batch_size  # int(batch_size * (1 - val_rate))
    # elif mode == 'val':
    #     samples_to_fetch = int(batch_size / ((1 / val_rate) - 1))
    # else:
    #     raise RuntimeError(f"Unrecognized mode: {mode}")
    # This can be any source of data — split based on mode
    loader = infinite_load_batch_zip(x_shape=master_x_shape, batch_size=batch_size)
    while True:
        # Load full batch once
        Xs, ys = next(loader)
        input_y = (ys.clip(max=4) / 4).astype('float32')

        yield {
            'structure': convert_to_tensor(Xs['structure'], dtype=tf_float32),
            'pattern': convert_to_tensor(Xs['pattern'], dtype=tf_float32),
            'trigger': convert_to_tensor(Xs['trigger'], dtype=tf_float32),
            'double': convert_to_tensor(Xs['double'], dtype=tf_float32),
            'structure-indicators': convert_to_tensor(Xs['structure-indicators'], dtype=tf_float32),
            'pattern-indicators': convert_to_tensor(Xs['pattern-indicators'], dtype=tf_float32),
            'trigger-indicators': convert_to_tensor(Xs['trigger-indicators'], dtype=tf_float32),
            'double-indicators': convert_to_tensor(Xs['double-indicators'], dtype=tf_float32),
        }, convert_to_tensor(input_y, dtype=tf_float32)


def run_trainer():
    from tensorflow import data as tf_data
    from tensorflow import TensorSpec, float32 as tf_float32
    log_d("Starting")
    sync_br_lib_init(path_of_logs='logs', root_path=app_config.root_path, log_to_file_level=logging.DEBUG,
                     log_to_std_out_level=logging.DEBUG)
    batch_size = 100 * 8
    use_dataset = True
    if use_dataset:
        batch_size = int(batch_size / 3.125)
    print("Python:" + sys.version)
    model = None
    setup_gpu()
    x_input = {key: TensorSpec(shape=(batch_size,) + shape, dtype=tf_float32, name=key) for key, shape in
               master_x_shape.items() if
               key != 'indicators'}
    x_input.update(
        {f"{key}-indicators": TensorSpec(shape=(batch_size,) + master_x_shape['indicators'], dtype=tf_float32,
                                         name=f"{key}-indicators") for key, _ in
         master_x_shape.items() if key != 'indicators'})
    threading_options = tf_data.Options()
    threading_options.experimental_threading.private_threadpool_size = 8
    threading_options.experimental_threading.max_intra_op_parallelism = 1
    while True:
        if use_dataset:
            train_dataset = tf_data.Dataset.from_generator(
                lambda: PrefetchGenerator(lambda: dataset_generator(batch_size=batch_size)),
                output_signature=(
                    x_input,
                    TensorSpec(shape=(None, 2), dtype=tf_float32)
                )
            ).prefetch(buffer_size=2)
            train_dataset = train_dataset.with_options(threading_options)

            val_dataset = tf_data.Dataset.from_generator(
                lambda: PrefetchGenerator(lambda: dataset_generator(batch_size=batch_size)),
                output_signature=(
                    x_input,
                    TensorSpec(shape=(None, 2), dtype=tf_float32)
                )
            ).prefetch(buffer_size=2)
            val_dataset = val_dataset.with_options(threading_options)
        else:
            train_dataset = dataset_generator(mode='train', batch_size=batch_size)
            val_dataset = dataset_generator(mode='val', batch_size=batch_size)
        model = train_model(train_dataset, val_dataset, x_shape=master_x_shape, batch_size=batch_size, cnn_filters=48,
                    lstm_units_list=[256, 128], dense_units=128, cnn_count=3,
                    cnn_kernel_growing_steps=2,
                    dropout_rate=0.3, rebuild_model=False, epochs=10, model=model, y_len=2, dataset_mode=use_dataset)


if __name__ == "__main__":
    run_trainer()

# todo: check input_y data:
#  + 1. Use tf.data.experimental.copy_to_device("/GPU:0")
# To move data to GPU as early as possible in the pipeline:

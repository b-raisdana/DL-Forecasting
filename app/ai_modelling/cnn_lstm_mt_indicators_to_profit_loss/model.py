import logging
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, Generator
import numpy as np
from tensorflow.data import Dataset

from Config import app_config
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.base import master_x_shape, batch_zip_generator
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.cnn_lstm_model import CNNLSTMLayer
from helper.br_py.br_py.base import sync_br_lib_init
from helper.br_py.br_py.do_log import log_d


def model_compile(t_model):
    from tensorflow import keras as tf_keras
    # tf_keras.utils.enable_checkpointing(t_model)
    opt = tf_keras.optimizers.RMSprop(clipnorm=1.0)
    opt = tf_keras.mixed_precision.LossScaleOptimizer(opt)
    # t_model.compile(loss=tf_keras.losses.Huber(), optimizer=opt)
    t_model.compile(loss=tf_keras.losses.MeanSquaredError(), optimizer=opt)
    return t_model


def train_model(
        train_dataset: Dataset | Generator, val_dataset: Dataset | Generator,
        x_shape: Dict[str, Tuple[int, int]],
        y_len: int,
        batch_size,
        model=None, cnn_filters=64, lstm_units_list: list = None, dense_units=64, cnn_count=3,
        cnn_kernel_growing_steps=2, dropout_rate=0.3, rebuild_model: bool = False, epochs=500,
        dataset_mode: bool = True,
        steps_per_epoch=20,
        validation_steps=4, save_freq=None,
):
    import gc
    from tensorflow import keras as tf_keras
    from tensorflow import config as tf_config

    from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.cnn_lstm_model import CNNLSTMModel

    if save_freq is None:
        save_freq = epochs

    class CustomEpochLogger(tf_keras.callbacks.Callback):
        def on_epoch_start(self, epoch, logs=None):
            print(f"\n>>> Epoch {epoch} started at {datetime.now().strftime('%H:%M:%S.%f')}")

        def on_epoch_end(self, epoch, logs=None):
            from GPUtil import getGPUs

            training_loss = logs.get('loss')
            validation_loss = logs.get('val_loss')
            if np.isnan(training_loss) or np.isnan(validation_loss):
                raise RuntimeError(f"nan found! training_loss:{training_loss}, validation_loss:{training_loss}")
            mem_info = tf_config.experimental.get_memory_info("GPU:0")
            gpu = getGPUs()[0]
            print(
                f"\n<<<Ends@{datetime.now().strftime('%H%M%S')}:{app_config.under_process_symbol}:{app_config.processing_date_range}/TrainL:{training_loss}/ValidL:{validation_loss}"
            )
            print(
                f"[GPU Memory] Epoch {epoch}: current = {mem_info['current'] / (1024 ** 2):.2f} MB | peak = {mem_info['peak'] / (1024 ** 2):.2f} MB")
            print(f"[Actual GPU] mem_used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.2f}%)")
            if gpu.memoryUtil > 0.95:
                raise MemoryError(f"gpu.memoryUtil{gpu.memoryUtil} > 0.95")

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
        tf_config.optimizer.set_jit('autoclustering')  # Or simply True
        if not rebuild_model and os.path.exists(model_path_keras):
            log_d("Loading existing keras model from disk...")
            model = tf_keras.models.load_model(model_path_keras,
                                               custom_objects={'CNNLSTMModel': CNNLSTMModel,
                                                               'CNNLSTMLayer': CNNLSTMLayer})
            # todo: model = tf_keras.models.clone_model(model, clone_function=tf_keras.experimental.grad_checkpoint)
            model_compile(model)  # Re-compile without loading optimizer state
        else:
            log_d("Building new model...")
            model = CNNLSTMModel(y_len=y_len, cnn_filters=cnn_filters, lstm_units_list=lstm_units_list,
                                 dense_units=dense_units, cnn_count=cnn_count,
                                 cnn_kernel_growing_steps=cnn_kernel_growing_steps, dropout_rate=dropout_rate)

            if hasattr(tf_keras.utils, "enable_gradient_checkpointing"):  # TF 2.18+ official API
                tf_keras.utils.enable_gradient_checkpointing(model)

            elif hasattr(tf_keras.utils, "enable_checkpointing"):  # TF 2.13-2.17 fallback
                tf_keras.utils.enable_checkpointing(model)
            # todo: model = tf_keras.models.clone_model(model, clone_function=tf_keras.experimental.grad_checkpoint)
            model_compile(model)
            batch_shape = {k: (batch_size,) + v for k, v in x_shape.items()}
            model.build(input_shape=batch_shape)
            model.summary()
            # tf_keras.utils.plot_model(model, to_file=os.path.join(app_config.path_of_plots,
            #                                                       f"model-plot.{int(datetime.now().timestamp())}")
            #                           , show_shapes=True, show_layer_names=True)
        print("XLA Enabled:", tf_config.optimizer.get_jit())
    # Train the model
    early_stopping: callable = tf_keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                                                restore_best_weights=True)
    epoch_logger: callable = CustomEpochLogger()
    checkpoint_cb = tf_keras.callbacks.ModelCheckpoint(
        filepath=model_path_keras,
        save_freq=save_freq,  # integer = # batches
        verbose=1,
        save_weights_only=False,
    )
    tensorboard = setup_tensorboard()
    model.fit(train_dataset.take(1), steps_per_epoch=1, epochs=1, verbose=0)
    # if dataset_mode:
    try:
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[early_stopping, epoch_logger,
                       # tensorboard,
                       checkpoint_cb])
    except MemoryError as e:
        log_d(f"Memory error:{str(e)}")
        model.save(model_path_keras)
        log_d("Model saved to disk.")
        sys.exit(5)
    except RuntimeError as e:
        log_d(f"RuntimeError:{str(e)}")
        sys.exit(6)
    # else:
    #     Xs, ys = next(train_dataset)
    #     validation_data = next(val_dataset)
    #     history = model.fit(
    #         x=Xs, y=ys, validation_data=validation_data,
    #         # validation_split=0.2,
    #         epochs=epochs,
    #         # batch_size=int(batch_size / steps_per_epoch),
    #         # Use a portion of your data for validation
    #         callbacks=[early_stopping, epoch_logger, tensorboard, checkpoint_cb])
    # log_d(history)
    model.save(model_path_keras)
    log_d("Model saved to disk.")
    tf_keras.backend.clear_session()
    model.reset_metrics()
    model.train_function = None
    model.test_function = None
    model.predict_function = None
    gc.collect()
    return model


def setup_gpu():
    from tensorflow import config as tf_config

    physical_devices = tf_config.list_physical_devices('GPU')
    expanded_memory_size = 0.90 * 8 * 1024
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


_dataset_x_shapes = None
_dataset_y_shape = None


def check_dataset_shape(x_dataset, y_dataset):
    global _dataset_x_shapes, _dataset_y_shape
    if _dataset_x_shapes is None:
        _dataset_x_shapes = {k: v.shape for k, v in x_dataset.items()}
        _dataset_y_shape = y_dataset.shape
        print("[Shape of batch initialized]")
        # for k, v in _dataset_x_shapes.items():
        #     print(f"  {k}: {v}")
        # print(f"  y: {_dataset_y_shape}")
    else:
        for k, v in x_dataset.items():
            if v.shape != _dataset_x_shapes[k]:
                print(f"❌ Shape mismatch for input '{k}': got {v.shape}, expected {_dataset_x_shapes[k]}")
                sys.exit(1)
        if y_dataset.shape != _dataset_y_shape:
            print(f"❌ Shape mismatch for target 'ys': got {y_dataset.shape}, expected {_dataset_y_shape}")
            sys.exit(1)


def dataset_generator(batch_size: int):
    from tensorflow import convert_to_tensor, float16 as tf_float16
    loader = batch_zip_generator(x_shape=master_x_shape, batch_size=batch_size)
    while True:
        # Load full batch once
        Xs, ys = next(loader)
        # input_y = (ys.clip(max=10)).astype('float16')
        input_y = np.asarray((ys.clip(min=0, max=10)), dtype=np.float16)
        # print(f"y stats:{ndarray_stats(input_y, ['short_signal', 'long_signal'])}")
        final_x = {
            'structure': convert_to_tensor(np.asarray(Xs['structure'], dtype=np.float16), dtype=tf_float16),
            'pattern': convert_to_tensor(np.asarray(Xs['pattern'], dtype=np.float16), dtype=tf_float16),
            'trigger': convert_to_tensor(np.asarray(Xs['trigger'], dtype=np.float16), dtype=tf_float16),
            'double': convert_to_tensor(np.asarray(Xs['double'], dtype=np.float16), dtype=tf_float16),
            # 'structure-indicators': convert_to_tensor(np.asarray(Xs['structure-indicators'], dtype=np.float16),
            #                                           dtype=tf_float16),
            # 'pattern-indicators': convert_to_tensor(np.asarray(Xs['pattern-indicators'], dtype=np.float16),
            #                                         dtype=tf_float16),
            # 'trigger-indicators': convert_to_tensor(np.asarray(Xs['trigger-indicators'], dtype=np.float16),
            #                                         dtype=tf_float16),
            # 'double-indicators': convert_to_tensor(np.asarray(Xs['double-indicators'], dtype=np.float16),
            #                                        dtype=tf_float16),
        }
        final_y = convert_to_tensor(input_y, dtype=tf_float16)
        check_dataset_shape(final_x, final_y)
        yield final_x, final_y


def run_trainer(round_counter: int):
    from tensorflow import data as tf_data
    from tensorflow import TensorSpec, float32 as tf_float32
    log_d("Starting")
    sync_br_lib_init(path_of_logs='logs', root_path=app_config.root_path, log_to_file_level=logging.DEBUG,
                     log_to_std_out_level=logging.DEBUG)
    batch_size = 80
    use_dataset = True
    if use_dataset:
        batch_size = int(batch_size / 3.125)
    print("Python:" + sys.version)
    model = None
    setup_gpu()
    x_input = {key: TensorSpec(shape=(batch_size,) + shape, dtype=tf_float32, name=key) for key, shape in
               master_x_shape.items() if
               key != 'indicators'}
    threading_options = tf_data.Options()
    threading_options.experimental_threading.private_threadpool_size = 2
    threading_options.experimental_threading.max_intra_op_parallelism = 2
    # if use_dataset:
    train_dataset = tf_data.Dataset.from_generator(
        lambda: PrefetchGenerator(lambda: dataset_generator(batch_size=batch_size)),
        output_signature=(
            x_input,
            TensorSpec(shape=(None, 2), dtype=tf_float32)
        )
    ).prefetch(buffer_size=2)
    train_dataset = train_dataset.with_options(threading_options)
    train_dataset = train_dataset.apply(tf_data.experimental.copy_to_device("/GPU:0"))

    val_dataset = tf_data.Dataset.from_generator(
        lambda: PrefetchGenerator(lambda: dataset_generator(batch_size=batch_size)),
        output_signature=(
            x_input,
            TensorSpec(shape=(None, 2), dtype=tf_float32)
        )
    ).prefetch(buffer_size=2)
    val_dataset = val_dataset.with_options(threading_options)
    val_dataset = val_dataset.apply(tf_data.experimental.copy_to_device("/GPU:0"))
    # else:
    #     train_dataset = dataset_generator(mode='train', batch_size=batch_size)
    #     val_dataset = dataset_generator(mode='val', batch_size=batch_size)
    print(f'Round:{round_counter}')
    for i in range(100):
        model = train_model(train_dataset, val_dataset, x_shape=master_x_shape, batch_size=batch_size, cnn_filters=64,
                            lstm_units_list=[512, 256], dense_units=128, cnn_count=4,
                            cnn_kernel_growing_steps=2,
                            dropout_rate=0.3, rebuild_model=False, epochs=10, model=model, y_len=2,
                            dataset_mode=use_dataset,
                            steps_per_epoch=100,
                            validation_steps=20,
                            save_freq=1000,
                            )


if __name__ == "__main__":
    import multiprocessing as mp
    import time

    mp.set_start_method("spawn", force=True)  # Linux defaults to fork; switch to spawn.
    training_round_counter = 0

    try:
        while True:  # infinite loop

            training_round_counter += 1

            p = mp.Process(target=run_trainer,
                           args=(training_round_counter,))  # 1️⃣ start a brand‑new process
            p.start()
            p.join()  # 2️⃣ block until it finishes

            # # Optional health‑check / cooldown
            if p.exitcode != 0:
                print(f"[wrapper] trainer crashed (exit {p.exitcode}) – restarting …")
                if p.exitcode != 5:
                    exit(p.exitcode)
                time.sleep(5)  # keep GPU fans happy, give OS time to reclaim memory
    except KeyboardInterrupt:
        print("\nStopped by user. Bye! 👋")
    except Exception as e:
        raise e
# todo: check input_y data:
#  + 1. Use tf.data.experimental.copy_to_device("/GPU:0")
# To move data to GPU as early as possible in the pipeline:

import gc
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, Tuple, Generator

from tensorflow import config as tf_config
from tensorflow import data as tf_data
from tensorflow import keras as tf_keras
from tensorflow.data import Dataset

from Config import app_config
from ai_modelling.base import setup_gpu, setup_tensorboard, CustomEpochLogger, pre_train_model, model_compile, \
    build_model, master_x_shape
from ai_modelling.dataset_generator.npz_batch import build_npz_dataset
from ai_modelling.cnn_lstm.cnn_lstm_model import CNNLSTMLayer
from ai_modelling.cnn_lstm.cnn_lstm_model import CNNLSTMModel
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
        steps_per_epoch=20,
        validation_steps=4, save_freq=None,
):
    if save_freq is None:
        save_freq = epochs

    pre_train_model()
    model_name = (f"cnn_lstm_attention.mt_pnl_n_ind"
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

            build_model(batch_size, model, x_shape)
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


# class PrefetchGenerator:
#     def __init__(self, generator_fn, max_prefetch=10):
#         self.generator = generator_fn()
#         self.queue = Queue(max_prefetch)
#         self.thread = Thread(target=self._run)
#         self.thread.daemon = True
#         self.thread.start()
#
#     def _run(self):
#         for item in self.generator:
#             self.queue.put(item)
#         self.queue.put(None)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         item = self.queue.get()
#         if item is None:
#             raise StopIteration
#         return item


def run_trainer(round_counter: int):
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
    # x_input = {key: TensorSpec(shape=(batch_size,) + shape, dtype=tf_float32, name=key) for key, shape in
    #            master_x_shape.items() if
    #            key != 'indicators'}
    threading_options = tf_data.Options()
    threading_options.threading.private_threadpool_size = 4
    threading_options.threading.max_intra_op_parallelism = 4
    # start = pd.to_datetime('03-01-24')
    # end = pd.to_datetime('09-01-24')
    # if use_dataset:
    # train_dataset = tf_data.Dataset.from_generator(
    #     lambda: dataset_generator(batch_size=batch_size, ),
    #     output_signature=(
    #         x_input,
    #         TensorSpec(shape=(None, 2), dtype=tf_float32)
    #     )
    # ).prefetch(buffer_size=4)
    # train_dataset = train_dataset.with_options(threading_options)
    # train_dataset = train_dataset.apply(tf_data.experimental.copy_to_device("/GPU:0"))

    train_dataset = build_npz_dataset(app_config.path_of_data, batch_size=80)
    # else:
    #     train_dataset = dataset_generator(mode='train', batch_size=batch_size)
    #     val_dataset = dataset_generator(mode='val', batch_size=batch_size)
    print(f'Round:{round_counter}')
    for i in range(100):
        model = train_model(train_dataset, train_dataset, x_shape=master_x_shape, batch_size=batch_size, cnn_filters=64,
                            lstm_units_list=[512, 256], dense_units=128, cnn_count=4,
                            cnn_kernel_growing_steps=2,
                            dropout_rate=0.3, rebuild_model=False, epochs=10, model=model, y_len=2,
                            dataset_mode=use_dataset,
                            steps_per_epoch=100,
                            validation_steps=20,
                            save_freq=1000,
                            )


if __name__ == "__main__":

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

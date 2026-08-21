import json
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from GPUtil import getGPUs

from tensorflow import keras as tf_keras
from tensorflow import config as tf_config

from Config import app_config
from helper.br_py.br_py.do_log import log_d
from helper.functions import date_range


def setup_gpu():
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
    this_run_folder = "TFB-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    this_run_log_path = os.path.join(app_config.path_of_logs, this_run_folder)
    os.mkdir(this_run_log_path)
    tensorboard = tf_keras.callbacks.TensorBoard(log_dir=this_run_log_path, write_graph=True, write_images=True,
                                                 histogram_freq=0)
    return tensorboard


def ceil_start_of_slide(t_date: datetime, slide: timedelta):
    if (t_date - datetime(t_date.year, t_date.month, t_date.day, tzinfo=t_date.tzinfo)) > timedelta(0):
        t_date = datetime(t_date.year, t_date.month, t_date.day + 1, tzinfo=t_date.tzinfo)
    days = (t_date - datetime(t_date.year, 1, 1, tzinfo=t_date.tzinfo)).days
    rounded_days = (days // slide.days) * slide.days + (slide.days if days % slide.days > 0 else 0)
    return datetime(t_date.year, 1, 1, tzinfo=t_date.tzinfo) + rounded_days * timedelta(days=1)


def dataset_folder(x_shape: Dict[str, Tuple[int, int]], batch_size: int, create: bool = False) -> str:
    serialized = json.dumps({"x_shape": x_shape, "batch_size": batch_size})
    folder_name = sanitize_filename(serialized)
    folder_path = os.path.join(app_config.path_of_data, folder_name)
    if create and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_name


def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[\s]', '', filename)
    filename = re.sub(r'[{}\[\]<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'_+', '_', filename)  # collapse multiple underscores
    filename = re.sub(r'^_', '', filename)  # collapse multiple underscores
    filename = re.sub(r'_$', '', filename)  # collapse multiple underscores
    filename = filename.replace('_,_', '_')  # collapse multiple underscores
    return filename


class CustomEpochLogger(tf_keras.callbacks.Callback):
    def on_epoch_start(self, epoch, logs=None):
        print(f"\n>>> Epoch {epoch} started at {datetime.now().strftime('%H:%M:%S.%f')}")

    def on_epoch_end(self, epoch, logs=None):

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


def pre_train_model():
    policy = tf_keras.mixed_precision.Policy('mixed_float16')
    tf_keras.mixed_precision.set_global_policy(policy)
    # Check if a GPU is available
    if len(tf_config.list_physical_devices('GPU')) == 0:
        print("No GPU found, using CPU.")
    else:
        print("GPU found, using GPU.")
    tf_config.optimizer.set_jit('autoclustering')  # Or simply True


def overlapped_quarters(i_date_range, length=timedelta(days=30 * 3), slide=timedelta(days=30 * 1.5)):
    if i_date_range is None:
        i_date_range = app_config.processing_date_range
    start, end = date_range(i_date_range)
    rounded_start = ceil_start_of_slide(start, slide)
    list_of_periods = [(p_start, p_start + length) for p_start in
                       pd.date_range(rounded_start, end - length, freq=slide)]
    return list_of_periods


def model_compile(t_model):
    # tf_keras.utils.enable_checkpointing(t_model)
    opt = tf_keras.optimizers.RMSprop(clipnorm=1.0)
    opt = tf_keras.mixed_precision.LossScaleOptimizer(opt)
    # t_model.compile(loss=tf_keras.losses.Huber(), optimizer=opt)
    t_model.compile(loss=tf_keras.losses.MeanSquaredError(), optimizer=opt)
    return t_model


def build_model(batch_size, model, x_shape):
    if hasattr(tf_keras.utils, "enable_gradient_checkpointing"):  # TF 2.18+ official API
        tf_keras.utils.enable_gradient_checkpointing(model)
    elif hasattr(tf_keras.utils, "enable_checkpointing"):  # TF 2.13-2.17 fallback
        tf_keras.utils.enable_checkpointing(model)
    else:
        log_d("No gradient checkpointing detected")
    # todo: model = tf_keras.models.clone_model(model, clone_function=tf_keras.experimental.grad_checkpoint)
    model_compile(model)
    batch_shape = {k: (batch_size,) + v for k, v in x_shape.items()}
    model.build(input_shape=batch_shape)
    model.summary()
    # tf_keras.utils.plot_model(model, to_file=os.path.join(app_config.path_of_plots,
    #                                                       f"model-plot.{int(datetime.now().timestamp())}")
    #                           , show_shapes=True, show_layer_names=True)


def check_dataset_shape_change(x_dataset, y_dataset):
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


master_x_shape = {
    # (sequence_length , price/OHLC cols + 12 indicator cols)
    'structure': (127, 17),
    'pattern': (253, 17),
    'trigger': (254, 17),
    'double': (255, 17),
}
_dataset_x_shapes = None
_dataset_y_shape = None

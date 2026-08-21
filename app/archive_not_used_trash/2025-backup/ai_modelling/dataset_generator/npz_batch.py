import logging
import multiprocessing as mp
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from Config import app_config
from ai_modelling.base import overlapped_quarters, master_x_shape
from ai_modelling.dataset_generator.training_datasets import train_data_of_mt_n_profit
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.do_log import log_d, log_w, log_i
from helper.functions import date_range_to_string


def npz_file_dataset(cache_folder, poll_interval_sec=1.0):
    def file_gen():
        while True:
            npz_files = sorted([
                os.path.join(cache_folder, f)
                for f in os.listdir(cache_folder)
                if f.startswith(CACHE_PREFIX) and f.endswith('.npz')
            ])
            if not npz_files:
                time.sleep(poll_interval_sec)
                continue
            for f in npz_files:
                yield f.encode()  # required by tf.py_function

    return tf.data.Dataset.from_generator(
        file_gen,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    )


def load_and_delete_npz(path_tensor):
    def _load_npz(path_bytes):
        path = path_bytes.decode()
        data = np.load(path)
        xs_list = [data[k].astype(np.float32) for k in master_x_shape]
        ys = data['ys'].astype(np.float32)
        data.close()
        os.remove(path)
        return (*xs_list, ys)

    out_types = [tf.float32] * len(master_x_shape) + [tf.float32]
    out_shapes = [tf.TensorShape((None, *master_x_shape[k])) for k in master_x_shape] + [tf.TensorShape((None, 2))]

    raw = tf.py_function(
        func=_load_npz,
        inp=[path_tensor],
        Tout=out_types
    )

    # Apply shapes manually
    for i, s in enumerate(out_shapes):
        raw[i].set_shape(s)
    return raw


def npz_cache_files_clean_up():
    for fname in os.listdir(app_config.path_of_data):
        if fname.startswith(CACHE_PREFIX + ".") and fname.endswith(f".{CACHE_EXT}"):
            try:
                os.remove(os.path.join(app_config.path_of_data, fname))
                log_i(f"Removed old cache file: {fname}")
            except Exception as e:
                logging.warning(f"Could not remove file {fname}: {e}")


def npz_cache_generator(start: datetime, end: datetime, batch_size: int = 400,
                        forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1, verbose: bool = True):
    quarters = overlapped_quarters(date_range_to_string(start=start, end=end))
    # 2. Continuous generation loop
    log_i("Cache Generator started. Monitoring folder for cache refill...")
    while True:
        while True:
            random.shuffle(quarters)
            for start, end in quarters:
                if verbose:  log_d(f'quarter start:{start} end:{end}##########################################')
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
                    if verbose: log_d(f'Symbol:{symbol}##########################################')
                    app_config.under_process_symbol = symbol
                    mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)  # Load once per quarter
                    for _ in range(100):
                        try:
                            # Count existing cache files
                            cache_files = [f for f in os.listdir(app_config.path_of_data)
                                           if f.startswith(CACHE_PREFIX + ".") and f.endswith(f".{CACHE_EXT}")]
                            while len(cache_files) >= CACHE_THRESHOLD:
                                print(f"Cache is full (>= threshold{CACHE_THRESHOLD}), no new file needed right now. Sleep briefly.")
                                time.sleep(0.5)  # small delay to avoid busy-wait
                                cache_files = [f for f in os.listdir(app_config.path_of_data)
                                               if f.startswith(CACHE_PREFIX + ".") and f.endswith(f".{CACHE_EXT}")]
                            # Generate a new batch of training data
                            Xs, ys, *rest = train_data_of_mt_n_profit(
                                structure_tf='4h',
                                mt_ohlcv=mt_ohlcv,  # or provide pre-loaded data if required
                                x_shape=master_x_shape,
                                batch_size=batch_size,
                                dataset_batches=1,
                                forecast_trigger_bars=forecast_trigger_bars,
                                verbose=False
                            )
                            # Create a unique timestamp for the file name
                            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")[:-4]
                            tmp_path = os.path.join(app_config.path_of_data, f"{CACHE_PREFIX}.{timestamp}.tmp")
                            final_path = os.path.join(app_config.path_of_data,
                                                      f"{CACHE_PREFIX}.{timestamp}.{CACHE_EXT}")
                            try:
                                # Save features and labels to a temporary NPZ file
                                # np.savez(tmp_path, **Xs, ys=ys)  # store each array under its key, plus 'ys'
                                with open(tmp_path, 'wb') as f:
                                    np.savez(f, **Xs, ys=ys)
                                    f.flush()
                                    os.fsync(f.fileno())
                                os.replace(tmp_path, final_path)  # atomic move to final name
                                log_i(
                                    f"Generated cache file: {os.path.basename(final_path)} (batch size={len(ys)})")
                            except Exception as e:
                                logging.error(f"Failed to write cache file {final_path}: {e}")
                                # Clean up temp file if exists
                                try:
                                    os.remove(tmp_path)
                                except OSError:
                                    pass
                            # loop continues to potentially generate more if still below threshold...
                        except Exception as e:
                            logging.error(f"Unexpected error in generator loop: {e}", exc_info=True)
                            time.sleep(5)


def build_npz_dataset(cache_folder, batch_size=80):
    file_ds = npz_file_dataset(cache_folder)

    def unpack_and_reconstruct(*xs_and_ys):
        xs_list = xs_and_ys[:-1]
        ys = xs_and_ys[-1]
        xs_dict = dict(zip(master_x_shape.keys(), xs_list))
        return xs_dict, ys

    unpacked_ds = file_ds.map(load_and_delete_npz, num_parallel_calls=tf.data.AUTOTUNE)
    unpacked_ds = unpacked_ds.map(unpack_and_reconstruct)

    def unbatch_fn(xs, ys):
        return tf.data.Dataset.from_tensor_slices((xs, ys))

    flat_ds = unpacked_ds.flat_map(unbatch_fn)
    return flat_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def npz_dataset_generator(batch_size: int):
    """Generator function that yields (Xs, ys) batches from cache files indefinitely."""
    cached_xs = {}
    cached_ys = None
    while True:
        while cached_ys is None or len(cached_ys) < batch_size:
            # List all cache files
            cache_files = [f for f in os.listdir(app_config.path_of_data)
                           if f.startswith(CACHE_PREFIX + ".") and f.endswith(f".{CACHE_EXT}")]
            if not cache_files:
                # No data available, wait and try again
                time.sleep(0.5)
                log_w("Cache folder empty!")
                continue
            # Determine the oldest file (by timestamp in name)
            cache_files.sort()
            oldest_file = cache_files[0]
            file_path = os.path.join(app_config.path_of_data, oldest_file)
            try:
                # Load the .npz file
                data = np.load(file_path)
            except Exception as e:
                logging.error(f"Error loading file {oldest_file}: {e}")
                # If file was removed or corrupted, skip it
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                continue
            # Extract Xs and ys from the loaded data
            try:
                for key in master_x_shape.keys():
                    if key in cached_xs:
                        cached_xs[key] = np.concatenate([cached_xs[key], data[key]], axis=0)
                    else:
                        cached_xs[key] = data[key]
                if cached_ys is None:
                    cached_ys = data['ys']
                else:
                    cached_ys = np.concatenate([cached_ys, data['ys']], axis=0)
            except Exception as e:
                logging.error(f"Data format error in {oldest_file}: {e}")
                # Remove problematic file and skip
                data.close()
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                continue
            # Now that data is in memory, delete the file to free space
            try:
                data.close()
                os.remove(file_path)
                log_i(f"Consumed and removed file: {oldest_file}")
            except Exception as e:
                logging.warning(f"Could not delete file {oldest_file}: {e}")
        while len(cached_ys) >= batch_size:
            picked_xs = {}
            for key in cached_xs:
                picked_xs[key] = cached_xs[key][:batch_size]
                cached_xs[key] = cached_xs[key][batch_size:]
            picked_ys = cached_ys[:batch_size]
            cached_ys = cached_ys[batch_size:]
            # print(f"\nSize of cached_ys={len(cached_ys)}\n")
            yield picked_xs, picked_ys


CACHE_PREFIX = "tf_input_cache"
CACHE_EXT = "npz"
CACHE_THRESHOLD = 40  # Maintain up to 8 files in the cache

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [CacheGen-%(process)d] %(message)s")


    # Define worker wrapper
    def worker():
        start = pd.to_datetime('03-01-24')
        end = pd.to_datetime('09-01-24')
        npz_cache_generator(start=start, end=end)


    num_workers = 10
    processes = []

    for i in range(num_workers):
        p = mp.Process(target=worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
import logging
import os
import random
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import tensorflow as tf

from Config import app_config
from ai_modelling.base import overlapped_quarters, master_x_shape
from ai_modelling.dataset_generator.training_datasets import train_data_of_mt_n_profit
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.do_log import log_d, log_w
from helper.functions import date_range_to_string

_dataset_cache: List[Dict[str, np.ndarray]] = []



def ram_cache_generator(start: datetime, end: datetime, batch_size: int = 400,
                        forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1, verbose: bool = True):
    quarters = overlapped_quarters(date_range_to_string(start=start, end=end))
    # 2. Continuous generation loop
    logging.info("Cache Generator started. Monitoring folder for cache refill...")
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
                            # cache_files = [f for f in os.listdir(app_config.path_of_data)
                            #                if f.startswith(CACHE_PREFIX + ".") and f.endswith(f".{CACHE_EXT}")]
                            while len(_dataset_cache) >= CACHE_THRESHOLD:
                                print("Cache is full (>= threshold), no new file needed right now. Sleep briefly.")
                                time.sleep(0.5)  # small delay to avoid busy-wait
                                # cache_files = [f for f in os.listdir(app_config.path_of_data)
                                #                if f.startswith(CACHE_PREFIX + ".") and f.endswith(f".{CACHE_EXT}")]
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
                            # timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")[:-4]
                            # tmp_path = os.path.join(app_config.path_of_data, f"{CACHE_PREFIX}.{timestamp}.tmp")
                            # final_path = os.path.join(app_config.path_of_data,
                            #                           f"{CACHE_PREFIX}.{timestamp}.{CACHE_EXT}")
                            # try:
                            # Save features and labels to a temporary NPZ file
                            # np.savez(tmp_path, **Xs, ys=ys)  # store each array under its key, plus 'ys'
                            _dataset_cache.append({'Xs': Xs, 'ys': ys, })
                            # with open(tmp_path, 'wb') as f:
                            #     np.savez(f, **Xs, ys=ys)
                            #     f.flush()
                            #     os.fsync(f.fileno())
                            # os.replace(tmp_path, final_path)  # atomic move to final name
                            logging.info(
                                f"Generated cache(={len(_dataset_cache)}) (batch size={len(ys)})")
                            # except Exception as e:
                            #     logging.error(f"Failed to write cache file {final_path}: {e}")
                            #     # Clean up temp file if exists
                            #     try:
                            #         os.remove(tmp_path)
                            #     except OSError:
                            #         pass
                            # loop continues to potentially generate more if still below threshold...
                        except Exception as e:
                            logging.error(f"Unexpected error in generator loop: {e}", exc_info=True)
                            time.sleep(5)


def build_ram_dataset(batch_size=80):
    """Builds a graph-optimizable tf.data.Dataset from the global _dataset_cache using pick_from_cached_datasets."""
    def generator():
        yield from pick_from_cached_datasets(batch_size=batch_size)

    output_signature = (
        {k: tf.TensorSpec(shape=(batch_size, *shape), dtype=tf.float32)
         for k, shape in master_x_shape.items()},
        tf.TensorSpec(shape=(batch_size, 2), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def pick_from_cached_datasets(batch_size: int):
    """Generator function that yields (Xs, ys) batches from cache files indefinitely."""
    cached_xs = {}
    cached_ys = None
    while True:
        while cached_ys is None or len(cached_ys) < batch_size:
            if not _dataset_cache:
                # No data available, wait and try again
                time.sleep(0.5)
                log_w("Cache folder empty!")
                continue
            while _dataset_cache:
                dataset = _dataset_cache.pop()
                for key in master_x_shape.keys():
                    if key in cached_xs:
                        cached_xs[key] = np.concatenate([cached_xs[key], dataset['Xs'][key]], axis=0)
                    else:
                        cached_xs[key] = dataset['Xs'][key]
                if cached_ys is None:
                    cached_ys = dataset['ys']
                else:
                    cached_ys = np.concatenate([cached_ys, dataset['ys']], axis=0)

        while len(cached_ys) >= batch_size:
            picked_xs = {}
            for key in cached_xs:
                picked_xs[key] = cached_xs[key][:batch_size]
                cached_xs[key] = cached_xs[key][batch_size:]
            picked_ys = cached_ys[:batch_size]
            cached_ys = cached_ys[batch_size:]
            # print(f"\nSize of cached_ys={len(cached_ys)}\n")
            yield picked_xs, picked_ys


# CACHE_PREFIX = "tf_input_cache"
CACHE_THRESHOLD = 40  # Maintain up to 8 files in the cache

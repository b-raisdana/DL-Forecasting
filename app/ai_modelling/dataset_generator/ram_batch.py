import logging
import multiprocessing as mp
import multiprocessing.managers as mpm
import os
import random
import threading as th
import time
from datetime import datetime
from multiprocessing import shared_memory, queues
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Config import app_config
from ai_modelling.base import overlapped_quarters, master_x_shape
from ai_modelling.dataset_generator.training_datasets import train_data_of_mt_n_profit
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.do_log import log_d, log_e
from helper.functions import date_range_to_string


def shm_from_array(arr: np.ndarray) -> Tuple[str, Tuple[int, ...], str]:
    """Copy a NumPy array into a fresh SharedMemory block and return (name, shape, dtype)."""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, arr.dtype, buffer=shm.buf)[:] = arr
    return shm.name, arr.shape, str(arr.dtype)  # caller responsible for shm.unlink() later


def array_from_shm(name: str, shape, dtype) -> np.ndarray:
    """Map an existing SharedMemory block into a NumPy array (no copy)."""
    shm = shared_memory.SharedMemory(name=name)
    return np.ndarray(shape, np.dtype(dtype), buffer=shm.buf), shm


def ram_dataset_producer(meta_q: queues.Queue,
                         start: datetime, end: datetime,
                         batch_size: int = 400,
                         forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1,
                         verbose: bool = True):
    quarters = overlapped_quarters(date_range_to_string(start=start, end=end))
    logging.info("Producer started")
    while True:
        random.shuffle(quarters)
        for q_start, q_end in quarters:
            if verbose: log_d(f'quarter {q_start} → {q_end}')
            app_config.processing_date_range = date_range_to_string(start=q_start, end=q_end)
            for symbol in ['BTCUSDT']:
                if verbose: log_d(f'Symbol {symbol}')
                app_config.under_process_symbol = symbol
                mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)
                for _ in range(100):
                    if meta_q.qsize() >= CACHE_THRESHOLD:
                        print(
                            f"RAM Cache is full ({meta_q.qsize()}>= {CACHE_THRESHOLD}), no new file needed right now. Sleep briefly.")
                        time.sleep(0.5)
                        continue
                    Xs, ys, *_ = train_data_of_mt_n_profit(
                        structure_tf='4h',
                        mt_ohlcv=mt_ohlcv,
                        x_shape=master_x_shape,
                        batch_size=batch_size,
                        dataset_batches=1,
                        forecast_trigger_bars=forecast_trigger_bars,
                        verbose=False)

                    # put each branch of Xs plus ys in shared memory
                    xs_meta = {k: shm_from_array(v) for k, v in Xs.items()}
                    ys_meta = shm_from_array(ys)

                    # put only metadata in the manager queue
                    meta_q.put((xs_meta, ys_meta))
                    logging.info(f'put batch #{meta_q.qsize()} (size={len(ys)})')


def ram_dataset_consumer(meta_q: queues.Queue, batch_size: int):
    cached_xs: Dict[str, np.ndarray] = {}
    cached_ys: np.ndarray | None = None

    while True:
        # refill local cache until we have at least one full batch
        while cached_ys is None or len(cached_ys) < batch_size:
            # check_queue(meta_q)
            xs_meta, ys_meta = meta_q.get()
            # attach shared-mem arrays, then immediately close & unlink *after* copy
            ys_arr, ys_shm = array_from_shm(*ys_meta)
            if cached_ys is None:
                cached_ys = ys_arr.copy()
            else:
                cached_ys = np.concatenate([cached_ys, ys_arr], axis=0)
            ys_shm.close();
            ys_shm.unlink()

            for k, meta in xs_meta.items():
                arr, shm = array_from_shm(*meta)
                if k in cached_xs:
                    cached_xs[k] = np.concatenate([cached_xs[k], arr], axis=0)
                else:
                    cached_xs[k] = arr.copy()
                shm.close();
                shm.unlink()

        # yield one batch
        picked_xs = {k: v[:batch_size] for k, v in cached_xs.items()}
        cached_xs = {k: v[batch_size:] for k, v in cached_xs.items()}

        picked_ys, cached_ys = cached_ys[:batch_size], cached_ys[batch_size:]
        yield picked_xs, picked_ys

def check_queue(meta_q):
    try:
        size = meta_q.qsize()  # remote call
        log_d(f"meta_queue size = {size}")
    except Exception as e:
        log_e("Error while querying queue:", e)
        mgr = MyManager(address=("127.0.0.1", 50055), authkey=b"secret123")
        mgr.connect()  # dial the manager
        meta_q = mgr.get_meta_queue()  # proxy object
    return meta_q

def build_ram_dataset(batch_size=80):
    import tensorflow as tf

    mgr = MyManager(address=("127.0.0.1", 50055), authkey=b"secret123")
    mgr.connect()  # dial the manager
    meta_q = mgr.get_meta_queue()
    check_queue(meta_q)

    def gen():
        yield from ram_dataset_consumer(meta_q, batch_size)

    output_signature = (
        {k: tf.TensorSpec(shape=(batch_size, *shape), dtype=tf.float32)
         for k, shape in master_x_shape.items()},
        tf.TensorSpec(shape=(batch_size, 2), dtype=tf.float32)
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.apply(tf.data.experimental.copy_to_device("/GPU:0"))
    return ds.prefetch(4)


# CACHE_PREFIX = "tf_input_cache"
CACHE_THRESHOLD = 80  # Maintain up to 8 files in the cache

meta_q = mp.Queue(maxsize=CACHE_THRESHOLD)

class MyManager(mpm.SyncManager):
    pass
MyManager.register("get_meta_queue", callable=lambda: meta_q)

def run_producer():
    # Register the actual meta_q

    mgr = MyManager(address=("127.0.0.1", 50055), authkey=b"secret123")
    mgr.start()

    print("Server PID", os.getpid())
    print("Address", mgr.address)

    def prod_worker():
        ram_dataset_producer(
            meta_q=meta_q,
            start=pd.to_datetime('2024-03-01'),
            end=pd.to_datetime('2024-09-01')
        )

    num_workers = 15
    processes = []

    for i in range(num_workers):
        p = mp.Process(target=prod_worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    threads = [th.Thread(target=prod_worker, daemon=True) for _ in range(num_workers)]
    for t in threads: t.start()
    for t in threads: t.join()


if __name__ == '__main__':
    run_producer()
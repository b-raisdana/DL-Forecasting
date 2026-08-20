import multiprocessing as mp
import multiprocessing.managers as mpm
import os
import random
import threading as th
import time
from collections.abc import Iterator
from datetime import datetime
from multiprocessing import queues, shared_memory
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from archive_not_used_trash.application.dataset_generation.training_datasets import train_data_of_mt_n_profit
from config import app_config
from domain.ohlcv.ohlcv import read_multi_timeframe_ohlcv
from domain.schemas.common.MultiTimeframe import MultiTimeframe
from helper.functions import date_range_to_string
from helper.logging.do_log import log_d, log_e, log_i
from pandera import typing as pt

from application.model_implementations.shared.base import master_x_shape
from archive_not_used_trash.application.model_implementations.shared.base import overlapped_quarters

if TYPE_CHECKING:
    import tensorflow as tf

# (name, shape, dtype) triple identifying a SharedMemory-backed array — what shm_from_array()
# hands back and array_from_shm() consumes to reattach it in another process.
_ShmMeta = tuple[str, tuple[int, ...], str]
_MetaQueue = "queues.Queue[tuple[dict[str, _ShmMeta], _ShmMeta]]"


def shm_from_array(arr: npt.NDArray[np.generic]) -> _ShmMeta:
    """Copy a NumPy array into a fresh SharedMemory block and return (name, shape, dtype)."""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, arr.dtype, buffer=shm.buf)[:] = arr
    return shm.name, arr.shape, str(arr.dtype)  # caller responsible for shm.unlink() later


def array_from_shm(
    name: str, shape: tuple[int, ...], dtype: str
) -> tuple[npt.NDArray[np.generic], shared_memory.SharedMemory]:
    """Map an existing SharedMemory block into a NumPy array (no copy)."""
    shm = shared_memory.SharedMemory(name=name)
    return np.ndarray(shape, np.dtype(dtype), buffer=shm.buf), shm


def ram_dataset_producer(
    meta_q: "queues.Queue[tuple[dict[str, _ShmMeta], _ShmMeta]]",
    start: datetime,
    end: datetime,
    batch_size: int = 400,
    forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1,
    verbose: bool = True,
) -> None:
    quarters = overlapped_quarters(date_range_to_string(start=start, end=end))
    log_i("Producer started")
    while True:
        random.shuffle(quarters)
        for q_start, q_end in quarters:
            if verbose:
                log_d(f"quarter {q_start} → {q_end}")
            app_config.processing_date_range = date_range_to_string(start=q_start, end=q_end)
            symbol_list = list(app_config.SYMBOLS)
            random.shuffle(symbol_list)
            for symbol in symbol_list:
                if verbose:
                    log_d(f"Symbol {symbol}")
                app_config.under_process_symbol = symbol
                mt_ohlcv = cast(
                    "pt.DataFrame[MultiTimeframe]", read_multi_timeframe_ohlcv(app_config.processing_date_range)
                )
                for _ in range(int(100 / NUM_WORKERS)):
                    while meta_q.qsize() >= CACHE_THRESHOLD:
                        # log_d(
                        #     f"RAM Cache is full ({meta_q.qsize()} >= {CACHE_THRESHOLD}); sleeping briefly."
                        # )
                        time.sleep(0.5)
                    Xs, ys, *_ = train_data_of_mt_n_profit(
                        structure_tf="4h",
                        mt_ohlcv=mt_ohlcv,
                        x_shape=master_x_shape,
                        batch_size=batch_size,
                        dataset_batches=1,
                        forecast_trigger_bars=forecast_trigger_bars,
                        verbose=False,
                    )

                    # put each branch of Xs plus ys in shared memory
                    xs_meta = {k: shm_from_array(v) for k, v in Xs.items()}
                    ys_meta = shm_from_array(ys)

                    # put only metadata in the manager queue
                    meta_q.put((xs_meta, ys_meta))
                    log_d(
                        f"put {symbol} batch for {app_config.processing_date_range} #{meta_q.qsize()} (size={len(ys)})"
                    )


def ram_dataset_consumer(
    meta_q: "queues.Queue[tuple[dict[str, _ShmMeta], _ShmMeta]]", batch_size: int
) -> Iterator[tuple[dict[str, npt.NDArray[np.generic]], npt.NDArray[np.generic]]]:
    while True:
        try:
            mgr = MyManager(address=("127.0.0.1", 50055), authkey=b"secret123")
            mgr.connect()  # dial the manager
            meta_q = mgr.get_meta_queue()
            check_queue(meta_q)

            cached_xs: dict[str, npt.NDArray[np.generic]] = {}
            cached_ys: npt.NDArray[np.generic] | None = None
            while True:
                # refill local cache until we have at least one full batch
                while cached_ys is None or len(cached_ys) < batch_size:
                    # check_queue(meta_q)
                    while meta_q.qsize() == 0:
                        log_d("Queue is empty!")
                        time.sleep(3)
                    xs_meta, ys_meta = meta_q.get()
                    # attach shared-mem arrays, then immediately close & unlink *after* copy
                    ys_arr, ys_shm = array_from_shm(*ys_meta)
                    cached_ys = ys_arr.copy() if cached_ys is None else np.concatenate([cached_ys, ys_arr], axis=0)
                    ys_shm.close()
                    ys_shm.unlink()

                    for k, meta in xs_meta.items():
                        arr, shm = array_from_shm(*meta)
                        if k in cached_xs:
                            cached_xs[k] = np.concatenate([cached_xs[k], arr], axis=0)
                        else:
                            cached_xs[k] = arr.copy()
                        shm.close()
                        shm.unlink()

                # yield one batch
                picked_xs = {k: v[:batch_size] for k, v in cached_xs.items()}
                cached_xs = {k: v[batch_size:] for k, v in cached_xs.items()}

                picked_ys, cached_ys = cached_ys[:batch_size], cached_ys[batch_size:]
                yield picked_xs, picked_ys
        except (
            ConnectionRefusedError,
            ConnectionAbortedError,
            ConnectionResetError,
            ConnectionError,
            FileNotFoundError,
            EOFError,
        ) as e:
            log_d(f"Queue {type(e).__name__}!")
            time.sleep(3)
            continue


def check_queue(
    meta_q: "queues.Queue[tuple[dict[str, _ShmMeta], _ShmMeta]]",
) -> "queues.Queue[tuple[dict[str, _ShmMeta], _ShmMeta]]":
    try:
        size = meta_q.qsize()  # remote call
        log_d(f"meta_queue size = {size}")
    except Exception as e:
        log_e(f"Error while querying queue: {e}")
        mgr = MyManager(address=("127.0.0.1", 50055), authkey=b"secret123")
        mgr.connect()  # dial the manager
        meta_q = mgr.get_meta_queue()  # proxy object
    return meta_q


def build_ram_dataset(batch_size: int = 80) -> "tf.data.Dataset":
    import tensorflow as tf

    def gen() -> Iterator[tuple[dict[str, npt.NDArray[np.generic]], npt.NDArray[np.generic]]]:
        yield from ram_dataset_consumer(meta_q, batch_size)

    output_signature = (
        {k: tf.TensorSpec(shape=(batch_size, *shape), dtype=tf.float32) for k, shape in master_x_shape.items()},
        tf.TensorSpec(shape=(batch_size, 2), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.apply(tf.data.experimental.copy_to_device("/GPU:0"))
    return ds.prefetch(4)


NUM_WORKERS = 15
# CACHE_PREFIX = "tf_input_cache"
CACHE_THRESHOLD = 200

meta_q: "queues.Queue[tuple[dict[str, _ShmMeta], _ShmMeta]]" = mp.Queue(maxsize=CACHE_THRESHOLD)


class MyManager(mpm.SyncManager):
    # get_meta_queue is registered dynamically below (MyManager.register(...)); this stub only
    # declares its shape for type-checking callers like ram_dataset_consumer()/check_queue().
    def get_meta_queue(self) -> "queues.Queue[tuple[dict[str, _ShmMeta], _ShmMeta]]":
        raise NotImplementedError  # overwritten by MyManager.register() below


MyManager.register("get_meta_queue", callable=lambda: meta_q)


def run_producer() -> None:
    # Register the actual meta_q

    mgr = MyManager(address=("127.0.0.1", 50055), authkey=b"secret123")
    mgr.start()

    print("Server PID", os.getpid())
    print("Address", mgr.address)

    def prod_worker() -> None:
        ram_dataset_producer(meta_q=meta_q, start=pd.to_datetime("2024-03-01"), end=pd.to_datetime("2024-09-01"))

    processes = []

    for _i in range(NUM_WORKERS):
        p = mp.Process(target=prod_worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    threads = [th.Thread(target=prod_worker, daemon=True) for _ in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    run_producer()

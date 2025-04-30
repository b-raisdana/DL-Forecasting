import queue
import random
import threading
from datetime import datetime
from typing import Iterator, Dict, Tuple

import numpy as np

from Config import app_config
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.base import overlapped_quarters
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.do_log import log_d
from helper.functions import date_range_to_string
from training_datasets import train_data_of_mt_n_profit


def dataset_streamer(
        s_batch_size: int, start: datetime, end: datetime,
        x_shape: Dict[str, Tuple[int, int]], cache_size: int = 4,
        verbose=False, train_data_batch_size: int = 2000
) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    data_queue = queue.Queue(maxsize=cache_size)

    def data_loader(t_start, t_end, verbose=False):
        nonlocal data_queue
        try:
            quarters = overlapped_quarters(date_range_to_string(start=t_start, end=t_end))
            random.shuffle(quarters)
            while True:

                for t1_start, t1_end in quarters:
                    app_config.processing_date_range = date_range_to_string(start=t1_start, end=t1_end)
                    for symbol in [
                        'BTCUSDT',
                        # # # 'ETHUSDT',
                        # 'BNBUSDT',
                        # 'EOSUSDT',
                        # # 'TRXUSDT',
                        # 'TONUSDT',
                        # # 'SOLUSDT',
                    ]:
                        app_config.under_process_symbol = symbol
                        # app_config.under_process_symbol to find proper source
                        mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)
                        for i in range(100):
                            if verbose: print("Running train_data_of_mt_n_profit")
                            Xs, ys, _, _, _, _ = train_data_of_mt_n_profit(
                                structure_tf='4h',
                                mt_ohlcv=mt_ohlcv,  # adjust accordingly or preload outside if needed
                                x_shape=x_shape,
                                batch_size=train_data_batch_size,
                                dataset_batches=1,
                                forecast_trigger_bars=3 * 4 * 4 * 4 * 1,
                                verbose=False
                            )
                            try:
                                data_queue.put((Xs, ys), timeout=60)
                            except queue.Full:
                                log_d("Queue is full! Check consumer speed.")
        except queue.Full:
            log_d("Queue is full! Check consumer speed.")
        except Exception as e:
            data_queue.put((e, None))

    # Start the prefetching thread immediately
    thread = threading.Thread(target=data_loader, args=(start, end, verbose), daemon=True)
    thread.start()
    cached_xs = {}
    cached_ys = None
    while True:
        Xs, ys = data_queue.get()
        if isinstance(Xs, Exception):
            raise Xs
        for key, value in Xs.items():
            if key in cached_xs:
                cached_xs[key] = np.concatenate([cached_xs[key], value], axis=0)
            else:
                cached_xs[key] = value
        if cached_ys is None:
            cached_ys = ys
        else:
            cached_ys = np.concatenate([cached_ys, ys], axis=0)
        while len(cached_ys) >= s_batch_size:
            picked_xs = {}
            for key in cached_xs:
                picked_xs[key] = cached_xs[key][:s_batch_size]
                cached_xs[key] = cached_xs[key][s_batch_size:]
            picked_ys = cached_ys[:s_batch_size]
            cached_ys = cached_ys[s_batch_size:]
            # print(f"\nSize of cached_ys={len(cached_ys)}\n")
            yield picked_xs, picked_ys
        data_queue.task_done()


# Example usage:
if __name__ == "__main__":
    import multiprocessing as mp
    import time


    def subprocess_runner(round_counter):
        import tensorflow as tf
        from tensorflow.keras import mixed_precision

        physical_devices = tf.config.list_physical_devices('GPU')
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

        mixed_precision.set_global_policy('mixed_float16')

        # Your training function here
        from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.model import run_trainer
        run_trainer(round_counter)


    mp.set_start_method('spawn', force=True)

    training_round_counter = 0
    while True:
        training_round_counter += 1
        process = mp.Process(target=subprocess_runner, args=(training_round_counter,))
        process.start()
        process.join()

        if process.exitcode != 0:
            print(f"Trainer crashed (exit {process.exitcode}), restarting after cooldown...")
            time.sleep(10)  # cooldown and memory clear interval

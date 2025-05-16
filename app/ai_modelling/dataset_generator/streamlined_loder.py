import queue
import random
import threading
import numpy as np
from datetime import datetime
from typing import Iterator, Dict, Tuple

from Config import app_config
from ai_modelling.base import overlapped_quarters
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.do_log import log_d
from helper.functions import date_range_to_string
from ai_modelling.dataset_generator.training_datasets import train_data_of_mt_n_profit


def dataset_streamer(
        s_batch_size: int, start: datetime, end: datetime,
        x_shape: Dict[str, Tuple[int, int]], cache_size: int = 4,
        verbose=False, train_data_batch_size: int = 2000
) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    data_queue = queue.Queue(maxsize=cache_size)

    def data_loader(t_start, t_end):
        try:
            quarters = overlapped_quarters(date_range_to_string(start=t_start, end=t_end))
            random.shuffle(quarters)

            while True:
                for q_start, q_end in quarters:
                    app_config.processing_date_range = date_range_to_string(q_start, q_end)

                    for symbol in ['BTCUSDT']:
                        app_config.under_process_symbol = symbol
                        mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)  # Load once per quarter
                        for _ in range(100):
                            Xs, ys, _, _, _, _ = train_data_of_mt_n_profit(
                                structure_tf='4h',
                                mt_ohlcv=mt_ohlcv,
                                x_shape=x_shape,
                                batch_size=train_data_batch_size,
                                dataset_batches=1,
                                forecast_trigger_bars=192,
                                verbose=False
                            )
                            data_queue.put((Xs, ys), timeout=60)
        except Exception as e:
            log_d(f"Exception in data_loader: {e}")
            data_queue.put((e, None))

    thread = threading.Thread(target=data_loader, args=(start, end), daemon=True)
    thread.start()

    max_buffer = s_batch_size * 5
    cached_xs = {key: np.empty((max_buffer,) + shape[1:], dtype=np.float32) for key, shape in x_shape.items()}
    y_parameters = 2
    cached_ys = np.empty((max_buffer, y_parameters), dtype=np.float32)
    buffer_idx = 0

    while True:
        Xs, ys = data_queue.get()
        if isinstance(Xs, Exception):
            raise Xs

        batch_len = len(ys)
        if buffer_idx + batch_len > max_buffer:
            buffer_idx = 0  # reset if overflow
            log_d("Buffer overflow reset. Adjust max_buffer if frequent.")

        for key in cached_xs:
            cached_xs[key][buffer_idx:buffer_idx + batch_len] = Xs[key]
        cached_ys[buffer_idx:buffer_idx + batch_len] = ys
        buffer_idx += batch_len

        while buffer_idx >= s_batch_size:
            output_xs = {key: cached_xs[key][:s_batch_size] for key in cached_xs}
            output_ys = cached_ys[:s_batch_size]

            for key in cached_xs:
                cached_xs[key][:buffer_idx - s_batch_size] = cached_xs[key][s_batch_size:buffer_idx]
            cached_ys[:buffer_idx - s_batch_size] = cached_ys[s_batch_size:buffer_idx]
            buffer_idx -= s_batch_size
            if verbose:
                log_d(f"Queue size: {data_queue.qsize()}, Cached ys length: {len(cached_ys)}")
            yield output_xs, output_ys

            data_queue.task_done()

import os
import random
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from Config import app_config
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.base import master_x_shape, overlapped_quarters
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.training_datasets import train_data_of_mt_n_profit
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.do_log import log_d
from helper.functions import date_range_to_string

# Configuration
CACHE_PREFIX = "tf_input_cache"  # Prefix for cache files
CACHE_EXT = "npz"  # File extension for cache files
CACHE_THRESHOLD = 8  # Maintain up to 8 files in the cache



# 1. Initial cleanup of existing cache files
def cache_files_clean_up():
    pattern = f"{CACHE_PREFIX}."  # pattern start for identification
    for fname in os.listdir(app_config.path_of_data):
        if fname.startswith(CACHE_PREFIX + ".") and fname.endswith(f".{CACHE_EXT}"):
            try:
                os.remove(os.path.join(app_config.path_of_data, fname))
                logging.info(f"Removed old cache file: {fname}")
            except Exception as e:
                logging.warning(f"Could not remove file {fname}: {e}")


def cache_generator(start: datetime, end: datetime, batch_size: int = 400,
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
                            cache_files = [f for f in os.listdir(app_config.path_of_data)
                                           if f.startswith(CACHE_PREFIX + ".") and f.endswith(f".{CACHE_EXT}")]
                            while len(cache_files) >= CACHE_THRESHOLD:
                                log_d("Cache is full (>= threshold), no new file needed right now. Sleep briefly.")
                                time.sleep(0.5)  # small delay to avoid busy-wait

                            # Generate a new batch of training data
                            # Here we assume train_data_of_mt_n_profit can be called directly for a batch.
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
                                np.savez(tmp_path, **Xs, ys=ys)  # store each array under its key, plus 'ys'
                                os.replace(tmp_path, final_path)  # atomic move to final name
                                logging.info(
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
                            # Catch-all for any unexpected error to keep generator running
                            logging.error(f"Unexpected error in generator loop: {e}", exc_info=True)
                            time.sleep(5)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [CacheGen] %(message)s")

    # cache_files_clean_up()
    start = pd.to_datetime('03-01-24')
    end = pd.to_datetime('09-01-24')
    cache_generator(start=start, end=end)

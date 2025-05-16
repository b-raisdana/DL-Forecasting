import logging
import os
import pickle
import random
import sys
import zipfile
from datetime import datetime
from typing import Dict, Tuple, Iterator, List

import numpy as np
import pandas as pd

from Config import app_config
from PanderaDFM import MultiTimeframe
from ai_modelling.base import dataset_folder, overlapped_quarters, master_x_shape
from ai_modelling.dataset_generator.training_datasets import train_data_of_mt_n_profit
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.base import sync_br_lib_init
from helper.br_py.br_py.do_log import log_d
from helper.functions import date_range_to_string
from helper.importer import pt


def batch_generator_zip_pkl(x_shape: Dict[str, Tuple[int, int]], batch_size: int) \
        -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    # x_shape_structure_127,5_pattern_253,5_trigger_254,5_double_255,5_indicators_129,12_batch_size_400
    folder_name = 'x_shape_structure_127,5_pattern_253,5_trigger_254,5_double_255,5_indicators_129,12_batch_size_400'  # dataset_folder(x_shape, 400)
    folder_path: str = os.path.join(app_config.path_of_data, folder_name)
    cached_xs = {}
    cached_ys = None
    while True:
        files = [f for f in os.listdir(folder_path) if f.startswith('dataset-') and f.endswith('.zip')]
        if not files or len(files) == 0:
            raise ValueError("No dataset files found!")
        random.shuffle(files)
        for picked_file in files:
            Xs, ys = load_single_batch_zip_pkl(folder_path, picked_file)
            for key, value in Xs.items():
                if key in cached_xs:
                    cached_xs[key] = np.concatenate([cached_xs[key], value], axis=0)
                else:
                    cached_xs[key] = value
            if cached_ys is None:
                cached_ys = ys
            else:
                cached_ys = np.concatenate([cached_ys, ys], axis=0)
            while len(cached_ys) >= batch_size:
                picked_xs = {}
                for key in cached_xs:
                    picked_xs[key] = cached_xs[key][:batch_size]
                    cached_xs[key] = cached_xs[key][batch_size:]
                picked_ys = cached_ys[:batch_size]
                cached_ys = cached_ys[batch_size:]
                # print(f"\nSize of cached_ys={len(cached_ys)}\n")
                yield picked_xs, picked_ys
        log_d(f"All listed files read at-least once.")


def generate_batch_zip_pkl(batch_size: int, mt_ohlcv: pt.DataFrame[MultiTimeframe],
                           x_shape: Dict[str, Tuple[int, int]], save=False):
    Xs, ys, X_dfs, y_dfs, y_timeframe, y_debug_dfs = (
        train_data_of_mt_n_profit(
            structure_tf='4h', mt_ohlcv=mt_ohlcv, x_shape=x_shape, batch_size=batch_size, dataset_batches=1,
            forecast_trigger_bars=3 * 4 * 4 * 4 * 1, ))
    folder_name = dataset_folder(x_shape, batch_size, create=True)
    if save:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_batch_zip_pkl(Xs, ys, folder_name, app_config.under_process_symbol, timestamp)
        save_validators_zip_pkl(X_dfs, y_dfs, y_timeframe, y_debug_dfs, folder_name,
                                app_config.under_process_symbol, timestamp)
    return Xs, ys, X_dfs, y_dfs, y_timeframe, y_debug_dfs
    #     plot_train_data_of_mt_n_profit(X_dfs, y_dfs, y_debug_dfs, i)


def load_single_batch_zip_pkl(folder_path, picked_file):
    file_path = os.path.join(folder_path, picked_file)
    with zipfile.ZipFile(file_path, 'r') as zipf:
        # print(f" File {picked_file} read ")
        with zipf.open('X_dfs.pkl') as f:
            Xs: Dict[str, np.ndarray] = pickle.load(f)
        with zipf.open('ys.pkl') as f:
            ys: np.ndarray = pickle.load(f)
    return Xs, ys


def save_validators_zip_pkl(X_dfs: Dict[str, List[pd.DataFrame]], y_dfs: List[pd.DataFrame], y_timeframe: str,
                            y_debug_dfs: List[pd.DataFrame], folder_name: str, symbol: str, timestamp) -> None:
    zip_file_name = f"validators-{symbol}-{timestamp}.zip"
    zip_file_path = os.path.join(app_config.path_of_data, folder_name, zip_file_name)

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.writestr('X_dfs.pkl', pickle.dumps(X_dfs))
        zipf.writestr('y_dfs.pkl', pickle.dumps(y_dfs))
        zipf.writestr('y_timeframe.pkl', pickle.dumps(y_timeframe))
        zipf.writestr('y_debug_dfs.pkl', pickle.dumps(y_debug_dfs))


def save_batch_zip_pkl(Xs: Dict[str, np.ndarray], ys: np.ndarray, folder_name: str, symbol: str, timestamp) -> None:
    zip_file_name = f"dataset-{symbol}-{timestamp}.zip"
    zip_file_path = os.path.join(app_config.path_of_data, folder_name, zip_file_name)

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # for key in Xs:
        #     zipf.writestr(f'Xs-{key}.npy', Xs[key].tobytes())
        zipf.writestr('X_dfs.pkl', pickle.dumps(Xs))
        zipf.writestr('ys.pkl', pickle.dumps(ys))


def load_validators_zip_pkl(folder_path: str, picked_file: str) \
        -> Tuple[Dict[str, List[pd.DataFrame]], List[pd.DataFrame], str, List[pd.DataFrame]]:
    file_path = os.path.join(folder_path, picked_file)

    with zipfile.ZipFile(file_path, 'r') as zipf:
        with zipf.open('X_dfs.pkl') as f:
            X_dfs: Dict[str, List[pd.DataFrame]] = pickle.load(f)
        with zipf.open('y_dfs.pkl') as f:
            y_dfs: List[pd.DataFrame] = pickle.load(f)
        with zipf.open('y_timeframe.pkl') as f:
            y_timeframe: str = pickle.load(f)
        with zipf.open('y_tester_dfs.pkl') as f:
            y_tester_dfs: List[pd.DataFrame] = pickle.load(f)

    log_d(f"Loaded validators zip file: {picked_file}")
    # Optionally log some details
    log_d(f"X_dfs keys: {list(X_dfs.keys())}, count of y_dfs: {len(y_dfs)}, timeframe: {y_timeframe}")

    return X_dfs, y_dfs, y_timeframe, y_tester_dfs


def training_dataset_main():
    log_d("Starting")
    sync_br_lib_init(path_of_logs='../cnn_lstm/logs', root_path=app_config.root_path, log_to_file_level=logging.DEBUG,
                     log_to_std_out_level=logging.DEBUG)
    # parser = argparse.ArgumentParser(description="Script for processing OHLCV data.")
    # args = parser.parse_args()
    app_config.processing_date_range = date_range_to_string(start=pd.to_datetime('03-01-24'),
                                                            end=pd.to_datetime('09-01-24'))
    quarters = overlapped_quarters(app_config.processing_date_range)
    mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)
    batch_size = 100 * 4

    # parser.add_argument("--do_not_fetch_prices", action="store_true", default=False,
    #                     help="Flag to indicate if prices should not be fetched (default: False).")
    print("Python:" + sys.version)

    # Apply config from arguments
    app_config.processing_date_range = "22-08-15.00-00T24-10-30.00-00"
    # config.do_not_fetch_prices = args.do_not_fetch_prices
    # seed(42)
    # np.random.seed(42)

    while True:
        random.shuffle(quarters)
        for start, end in quarters:
            log_d(f'quarter start:{start} end:{end}##########################################')
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
                log_d(f'Symbol:{symbol}##########################################')
                app_config.under_process_symbol = symbol
                generate_batch_zip_pkl(batch_size, mt_ohlcv, master_x_shape)


if __name__ == "__main__":
    training_dataset_main()


# todo: check the dataset files to check if input_y is reperesting good results?
def batch_generator(start, end, x_shape: Dict[str, Tuple[int, int]], batch_size: int, verbose=False):
    app_config.processing_date_range = date_range_to_string(start=start, end=end)

    quarters = overlapped_quarters(app_config.processing_date_range)
    mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)

    cached_xs = {}
    cached_ys = None
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
                Xs, ys, X_dfs, y_dfs, y_timeframe, y_debug_dfs = (
                    train_data_of_mt_n_profit(
                        structure_tf='4h', mt_ohlcv=mt_ohlcv, x_shape=x_shape, batch_size=1000, dataset_batches=1,
                        forecast_trigger_bars=3 * 4 * 4 * 4 * 1, verbose=verbose))
                for key, value in Xs.items():
                    if key in cached_xs:
                        cached_xs[key] = np.concatenate([cached_xs[key], value], axis=0)
                    else:
                        cached_xs[key] = value
                if cached_ys is None:
                    cached_ys = ys
                else:
                    cached_ys = np.concatenate([cached_ys, ys], axis=0)
                while len(cached_ys) >= batch_size:
                    picked_xs = {}
                    for key in cached_xs:
                        picked_xs[key] = cached_xs[key][:batch_size]
                        cached_xs[key] = cached_xs[key][batch_size:]
                    picked_ys = cached_ys[:batch_size]
                    cached_ys = cached_ys[batch_size:]
                    # print(f"\nSize of cached_ys={len(cached_ys)}\n")
                    yield picked_xs, picked_ys

import json
import os
import pickle
import random
import re
import zipfile
from datetime import timedelta, datetime
from typing import Dict, Tuple, List, Iterator

import numpy as np
import pandas as pd

from Config import app_config
from helper.br_py.br_py.do_log import log_d
from helper.functions import date_range


def overlapped_quarters(i_date_range, length=timedelta(days=30 * 3), slide=timedelta(days=30 * 1.5)):
    if i_date_range is None:
        i_date_range = app_config.processing_date_range
    start, end = date_range(i_date_range)
    rounded_start = ceil_start_of_slide(start, slide)
    list_of_periods = [(p_start, p_start + length) for p_start in
                       pd.date_range(rounded_start, end - length, freq=slide)]
    return list_of_periods


master_x_shape = {
    'structure': (127, 5),
    'pattern': (253, 5),
    'trigger': (254, 5),
    'double': (255, 5),
    'indicators': (129, 12),
}


def save_batch_zip(Xs: Dict[str, np.ndarray], ys: np.ndarray, folder_name: str, symbol: str, timestamp) -> None:
    zip_file_name = f"dataset-{symbol}-{timestamp}.zip"
    zip_file_path = os.path.join(app_config.path_of_data, folder_name, zip_file_name)

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # for key in Xs:
        #     zipf.writestr(f'Xs-{key}.npy', Xs[key].tobytes())
        zipf.writestr('X_dfs.pkl', pickle.dumps(Xs))
        zipf.writestr('ys.pkl', pickle.dumps(ys))


def load_batch_zip(x_shape: Dict[str, Tuple[int, int]], batch_size: int, n: None | int = None) \
        -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    folder_name = dataset_folder(x_shape, 400)
    folder_path: str = os.path.join(app_config.path_of_data, folder_name)
    cached_xs = {}
    cached_ys = None
    while True:
        files = [f for f in os.listdir(folder_path) if f.startswith('dataset-') and f.endswith('.zip')]
        if not files or len(files) == 0:
            raise ValueError("No dataset files found!")
        if n is None:
            random.shuffle(files)
            for picked_file in files:
                Xs, ys = load_single_batch_zip(folder_path, picked_file)
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
                    # print(f"Size of cached_ys={len(cached_ys)}")
                    yield picked_xs, picked_ys
            log_d(f"All listed files read at-least once.")
        else:
            if n >= len(files) or n < 0:
                raise ValueError(f"n must be greater than 0 and less than {len(files)} (number of batch files).")
            picked_file = files[n]
            Xs, ys = load_single_batch_zip(folder_path, picked_file)
            return Xs, ys


def load_single_batch_zip(folder_path, picked_file):
    file_path = os.path.join(folder_path, picked_file)
    with zipfile.ZipFile(file_path, 'r') as zipf:
        # print(f" File {picked_file} read ")
        with zipf.open('X_dfs.pkl') as f:
            Xs: Dict[str, np.ndarray] = pickle.load(f)
        with zipf.open('ys.pkl') as f:
            ys: np.ndarray = pickle.load(f)
    return Xs, ys


def save_validators_zip(X_dfs: Dict[str, List[pd.DataFrame]], y_dfs: List[pd.DataFrame], y_timeframe: str,
                        y_tester_dfs: List[pd.DataFrame], folder_name: str, symbol: str, timestamp) -> None:
    zip_file_name = f"validators-{symbol}-{timestamp}.zip"
    zip_file_path = os.path.join(app_config.path_of_data, folder_name, zip_file_name)

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.writestr('X_dfs.pkl', pickle.dumps(X_dfs))
        zipf.writestr('y_dfs.pkl', pickle.dumps(y_dfs))
        zipf.writestr('y_timeframe.pkl', pickle.dumps(y_timeframe))
        zipf.writestr('y_tester_dfs.pkl', pickle.dumps(y_tester_dfs))


def load_validators_zip(x_shape: Dict[str, Tuple[int, int]], batch_size: int, n: int) \
        -> Tuple[Dict[str, List[pd.DataFrame]], List[pd.DataFrame], str, List[pd.DataFrame]]:
    folder_name = dataset_folder(x_shape, batch_size)
    folder_path: str = os.path.join(app_config.path_of_data, folder_name)

    files = [f for f in os.listdir(folder_path) if f.startswith('validators-') and f.endswith('.zip')]
    if not files:
        raise ValueError("No validators zip files found!")
    if n < 0 or n > len(files) - 1:
        raise ValueError(f"n must be greater than 0 and less than {len(files)} (number of batch files).")
    picked_file = files[n]
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

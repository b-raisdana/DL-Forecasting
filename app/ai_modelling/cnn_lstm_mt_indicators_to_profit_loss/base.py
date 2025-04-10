import json
import os
import random
import re
import zipfile
from datetime import timedelta, datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Config import app_config
from helper.br_py.br_py.do_log import log_d
from helper.functions import date_range, get_size


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


def read_batch_zip(x_shape: Dict[str, Tuple[int, int]], batch_size: int) \
        -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    folder_name = dataset_folder(x_shape, batch_size)
    folder_path: str = os.path.join(app_config.path_of_data, folder_name)

    files = [f for f in os.listdir(folder_path) if f.startswith('dataset-')]
    if not files:
        raise ValueError("No dataset files found!")

    picked_file = random.choice(files)
    file_path = os.path.join(app_config.path_of_data, folder_name, picked_file)
    Xs: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(file_path, 'r') as zipf:
        for name in zipf.namelist():
            if name.startswith('Xs-') and name.endswith('.npy'):
                key: str = name[3:-4]
                with zipf.open(name) as f:
                    arr: np.ndarray = np.frombuffer(f.read(), dtype=np.float32)
                    shape_key = 'indicators' if 'indicators' in key else key
                    Xs[key] = arr.reshape(-1, *x_shape[shape_key])
        with zipf.open('ys.npy') as f:
            ys: np.ndarray = np.frombuffer(f.read(), dtype=np.float32)
            # ys = ys.reshape(-1, *y_shape)

    log_d(f"Xs dataset size: {str(get_size(Xs))}")
    return Xs, ys


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

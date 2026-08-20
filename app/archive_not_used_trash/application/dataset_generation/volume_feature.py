import numpy as np
import pandas as pd
from config import app_config
from helper.importer import ta

__volume_feature_columns = ["volume_atr"]


def volume_feature_columns() -> list[str]:
    return __volume_feature_columns


def log_sma_volume_feature_columns() -> list[str]:
    return ["log_volume_sma_ratio"]

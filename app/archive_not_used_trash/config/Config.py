import base64
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CandleSize(Enum):
    @dataclass
    class MinMax:
        min: float
        max: float

    Spinning = MinMax(min=0.0, max=0.80)
    Standard = MinMax(min=0.80, max=1.20)
    Long = MinMax(min=1.2, max=2.5)
    Spike = MinMax(min=2.5, max=np.inf)


class TREND(Enum):
    BULLISH = "BULLISH_TREND"
    BEARISH = "BEARISH_TREND"
    SIDE = "SIDE_TREND"


class TopTYPE(Enum):
    PEAK = "peak"
    VALLEY = "valley"


_ROOT_PATH = Path(__file__).resolve().parent.parent.parent


app_config = Config()

config_as_json = app_config.model_dump_json()

config_digest = str.translate(
    base64.b64encode(hashlib.md5(config_as_json.encode("utf-8")).digest()).decode("ascii"),
    {
        ord("+"): "",
        ord("/"): "",
        ord("="): "",
    },
)

app_config.path_of_logs.mkdir(parents=True, exist_ok=True)

dump_filename = app_config.path_of_logs / f"Config.{config_digest}.json"

if not dump_filename.exists():
    dump_filename.write_text(config_as_json, encoding="utf-8")

app_config.id = config_digest

import base64
import hashlib
import os
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

import numpy as np
import pandas as pd
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


_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


class Config(BaseSettings):  # type: ignore[explicit-any]
    """Runtime settings. Any field can be overridden via a `DLF_<FIELD_NAME>` env var
    (or a `.env` file), validated against its declared type/bounds on load and on
    every later `app_config.<field> = ...` assignment."""

    model_config = SettingsConfigDict(
        env_prefix="DLF_",
        env_file=".env",
        extra="ignore",
        validate_assignment=True,
    )

    root_path: str = _ROOT_PATH
    # self.processing_date_range = '17-12-24.00-00T17-12-31.23-59'
    processing_date_range: str = "17-12-01.00-00T17-12-31.23-59"
    limit_to_under_process_period: bool = False
    under_process_symbol: str = "BTCUSDT"
    under_process_exchange: str = "Kucoin"
    under_process_market: str = "Spot"
    files_to_load: list[str] = [
        "17-01-01.0-01TO17-12-31.23-59.1min",
        "17-01-01.0-01TO17-12-31.23-59.5min",
        "17-01-01.0-01TO17-12-31.23-59.15min",
        "17-01-01.0-01TO17-12-31.23-59.1h",
        "17-01-01.0-01TO17-12-31.23-59.4h",
        "17-01-01.0-01TO17-12-31.23-59.1D",
        "17-01-01.0-01TO17-12-31.23-59.1W",
    ]
    data_path_preamble: str = "https://raw.githubusercontent.com/b-raisdana/BTC-price/main/"

    timeframe_shifter: dict[str, int] = {
        "structure": 0,
        "pattern": -1,
        "trigger": -2,
        "double": -4,
        "hat_trick": -6,
    }
    timeframes: list[str] = [
        "1min",
        "5min",
        "15min",
        "1h",
        "4h",
        "1D",
        "1W",
    ]
    hat_trick_index: int = 0
    trigger_dept: int = Field(default=16, gt=0)

    max_x_gap: int = Field(default=1, gt=0)

    dept_of_analysis: int = Field(default=3, gt=0)

    end_time: str = "2021-03-01 03:43:00"

    INFINITY_TIME_DELTA: timedelta = timedelta(days=10 * 365)

    path_of_data: str = Field(
        default=os.path.join(_ROOT_PATH, "data"),
        validation_alias="DLF_DATA_ROOT",
    )
    path_of_logs: str = os.path.join(_ROOT_PATH, "logs")
    path_of_test_plots: str = "test_plots"

    # infrastructure.disk_cache windowing (see app/infrastructure/ohlcv/README.md): default
    # calendar-window size for any data_frame_type without an entry in cache_window_freq_overrides,
    # as a pandas period freq alias ("M" calendar month, "D" calendar day, ...).
    default_cache_window_freq: str = "M"
    cache_window_freq_overrides: dict[str, str] = {
        "ohlcv": "D",
        "multi_timeframe_ohlcv": "D",
        "multi_timeframe_ohlcva": "D",
    }
    # Warn if a data_frame_type's cache-file generation rate, extrapolated to 24h, exceeds this many
    # bytes; re-evaluated at most once per cache_generation_monitor_interval_minutes per prefix.
    cache_generation_warn_bytes_per_day: int = Field(default=1_000_000_000, gt=0)
    cache_generation_monitor_interval_minutes: int = Field(default=30, gt=0)

    momentum_trand_strength_factor: float = Field(default=0.70, gt=0)  # CandleSize.Standard.value[0]

    load_data_to_meta_trader: bool = False

    atr_timeperiod: int = Field(default=14, gt=0)
    atr_safe_start_expand_multipliers: int = Field(default=1, gt=0)

    base_pattern_ttl: int = Field(default=4 * 4 * 4 * 4, gt=0)
    base_pattern_number_of_spinning_candles: int = Field(default=2, ge=0)
    base_pattern_candle_min_backward_coverage: float = Field(default=0.8, gt=0, le=1)
    # >1 means make sure the last candle is closed
    base_pattern_index_shift_after_last_candle_in_the_sequence: int = 1
    base_pattern_order_limit_price_margin_percentage: float = Field(default=0.05, ge=0, le=1)  # 5%
    base_pattern_risk_reward_rate: float = Field(default=5, gt=0)  # 500% = average rate of looses to achieve a win.

    ftc_price_range_percentage: float = Field(
        default=0.38, gt=0, le=1
    )  # the FTC will be in the last 38% of the movement.
    # 300% = we expect the profit to be 300% of trading fee to consider the trade profitable.
    trading_fee_safe_side_multiplier: float = Field(default=3, gt=0)
    # base patterns with size of less than n * atr (of base time frame) are not enough big to be back tested.
    base_pattern_small_to_trace_in_base_candles_atr_factor: float = Field(default=3, gt=0)
    initial_cash: float = Field(default=1000.0, gt=0)
    risk_per_order_percent: float = Field(default=0.01, gt=0, le=1)  # 1%
    capital_max_total_risk_percentage: float = Field(default=0.1, gt=0, le=1)  # 10%

    figure_width: int = Field(default=1500, gt=0)
    figure_height: int = Field(default=1000, gt=0)
    figure_font_size: int = Field(default=7, gt=0)

    pivot_number_of_active_hits: int = Field(default=2, gt=0)

    check_assertions: bool = True

    id: str = ""
    GLOBAL_CACHE: dict[str, object] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def structure_timeframes(self) -> list[str]:
        return self.timeframes[2:]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pattern_timeframes(self) -> list[str]:
        return self.timeframes[1:]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trigger_timeframes(self) -> list[str]:
        return self.timeframes[:-2]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_time_delta(self) -> timedelta:
        return pd.to_timedelta(self.timeframes[0])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def path_of_plots(self) -> str:
        return os.path.join(self.path_of_data, "plots")


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

dump_filename = os.path.join(app_config.path_of_logs, f"Config.{config_digest}.json")
if not os.path.exists(app_config.path_of_logs):
    os.makedirs(app_config.path_of_logs)
if not os.path.exists(dump_filename):
    with open(dump_filename, "w+") as config_file:
        config_file.write(config_as_json)

app_config.id = config_digest

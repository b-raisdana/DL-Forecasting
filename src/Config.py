import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

import numpy as np
import pandas as pd


# class CandleSize(Enum):
#     Spinning = (0.0, 0.80)
#     Standard = (0.80, 1.20)
#     Long = (1.20, 2.5)
#     Spike = (2.5, 999999999)


class CandleSize(Enum):
    @dataclass
    class MinMax:
        min: float
        max: float

    Spinning: MinMax = MinMax(min=0.0, max=0.80)
    Standard: MinMax = MinMax(min=0.80, max=1.20)
    Long: MinMax = MinMax(min=1.2, max=2.5)
    Spike: MinMax = MinMax(min=2.5, max=np.inf)


class TREND(Enum):
    BULLISH = 'BULLISH_TREND'
    BEARISH = 'BEARISH_TREND'
    SIDE = 'SIDE_TREND'


class TopTYPE(Enum):
    PEAK = 'peak'
    VALLEY = 'valley'


class Config():
    def __init__(self):
        # self.processing_date_range = '17-12-24.00-00T17-12-31.23-59'
        self.processing_date_range = '17-12-01.00-00T17-12-31.23-59'
        self.limit_to_under_process_period = True
        self.under_process_symbol = 'BTCUSDT'
        self.files_to_load = [
            '17-01-01.0-01TO17-12-31.23-59.1min',
            '17-01-01.0-01TO17-12-31.23-59.5min',
            '17-01-01.0-01TO17-12-31.23-59.15min',
            '17-01-01.0-01TO17-12-31.23-59.1h',
            '17-01-01.0-01TO17-12-31.23-59.4h',
            '17-01-01.0-01TO17-12-31.23-59.1D',
            '17-01-01.0-01TO17-12-31.23-59.1W',
        ]
        self.data_path_preamble = 'https://raw.githubusercontent.com/b-raisdana/BTC-price/main/'

        self.timeframe_shifter = {
            'structure': 0,
            'pattern': -1,
            'trigger': -2,
            'double': -4,
            'hat_trick': -6,
        }
        self.timeframes = [
            '1min',  #: to_offset('1min'),
            '5min',  #: to_offset('5min'),
            '15min',  #: to_offset('15min'),
            '1h',  #: to_offset('1H'),
            '4h',  #: to_offset('4H'),
            '1D',  #: to_offset('1D'),
            '1W',  #: to_offset('1W')
        ]
        self.structure_timeframes = self.timeframes[2:]
        self.pattern_timeframes = self.timeframes[1:]
        self.trigger_timeframes = self.timeframes[:-2]
        self.hat_trick_index = 0
        self.trigger_dept = 16

        self.dept_of_analysis = 3

        self.end_time = '2021-03-01 03:43:00'

        self.INFINITY_TIME_DELTA = timedelta(days=10 * 365)

        self.path_of_data = 'data'
        self.path_of_plots = os.path.join(self.path_of_data, 'plots')
        self.path_of_logs = 'logs'
        self.path_of_test_plots = os.path.join('test_plots')

        self.base_time_delta = pd.to_timedelta(self.timeframes[0])  # timedelta(minutes=1)

        self.momentum_trand_strength_factor = 0.70  # CandleSize.Standard.value[0]

        self.load_data_to_meta_trader = False

        self.atr_timeperiod = 14
        self.atr_safe_start_expand_multipliers = 1

        self.base_pattern_ttl = 4 * 4 * 4 * 4
        self.base_pattern_number_of_spinning_candles = 2
        self.base_pattern_candle_min_backward_coverage = 0.8
        self.base_pattern_index_shift_after_last_candle_in_the_sequence = 1  # >1 means make sure the last candle is closed
        self.base_pattern_order_limit_price_margin_percentage = 0.1
        self.base_pattern_order_limit_price_margin_percentage = 0.05  # 5%
        self.base_pattern_risk_reward_rate = 5  # 500% = average rate of looses to achieve a win.

        self.ftc_price_range_percentage = 0.38  # the FTC will be in the last 38% of the movement.
        # 300% = we expect the profit to be 300% of trading fee to consider the trade profitable.
        self.trading_fee_safe_side_multiplier = 3
        # base patterns with size of less than n * atr (of base time frame) are not enough big to be back tested.
        self.base_pattern_small_to_trace_in_base_candles_atr_factor = 3
        self.initial_cash = 1000.0
        self.risk_per_order_percent = 0.01  # 1%
        self.capital_max_total_risk_percentage = 0.1  # 10%

        self.figure_width = 1500
        self.figure_height = 1000
        # self.figure_width = 1800
        # self.figure_height = 900
        self.figure_font_size = 7

        self.pivot_number_of_active_hits = 2

        self.check_assertions = True

        self.id = ""


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            return o.__dict__
        except AttributeError:
            return str(o)


myEncoder = MyEncoder()
config = Config()
config_as_json = myEncoder.encode(config)
config_digest = str.translate(base64.b64encode(hashlib.md5(config_as_json.encode('utf-8')).digest())
                              .decode('ascii'), {ord('+'): '', ord('/'): '', ord('='): '', })

dump_filename = os.path.join(config.path_of_logs, f'Config.{config_digest}.json')
if not os.path.exists(config.path_of_logs):
    os.makedirs(config.path_of_logs)
if not os.path.exists(dump_filename):
    with open(dump_filename, 'w+') as config_file:
        config_file.write(str(config_as_json))

config.id = config_digest
config.GLOBAL_CACHE = {}


def check_working_directory():
    current_directory = os.path.abspath(os.path.curdir)

    # Check if 'src' and 'data' folders exist
    src_folder_exists = os.path.exists(os.path.join(current_directory, 'src'))
    data_folder_exists = os.path.exists(os.path.join(current_directory, 'data'))
    if not src_folder_exists or not data_folder_exists:
        raise Exception(f"'src' or 'data' folders not found under working directory({current_directory})")


check_working_directory()

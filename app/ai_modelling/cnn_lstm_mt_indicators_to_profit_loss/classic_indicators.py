import warnings

import pandas as pd
import pandas_ta as ta

from helper.importer import ta

warnings.simplefilter(action='ignore', category=FutureWarning)


def add_ichimoku(ohlc):
    org_ichimoku = ta.ichimoku(
        high=ohlc['high'],
        low=ohlc['low'],
        close=ohlc['close'],
    )
    # ichimoku = pd.concat([org_ichimoku[0], org_ichimoku[1]], axis=1)
    ichimoku = org_ichimoku[0]
    # Unpack the components and rename to lowercase
    ohlc['ichi_conv'] = ichimoku['ITS_9']  # Tenkan-sen
    ohlc['ichi_base'] = ichimoku['IKS_26']  # Kijun-sen
    ohlc['ichi_lag'] = ichimoku['ICS_26']  # Chikou Span
    ohlc['ichi_lead_a'] = ichimoku['ISA_9']  # Senkou Span A
    ohlc['ichi_lead_b'] = ichimoku['ISB_26']  # Senkou Span B

    # # Shift the leading spans forward by 26 periods
    # ohlc['leading_span_a'] = ohlc['leading_span_a'].shift(26)
    # ohlc['leading_span_b'] = ohlc['leading_span_b'].shift(26)
    #
    # # Shift the Lagging Span backward by 26 periods
    # ohlc['lagging_span'] = ohlc['lagging_span'].shift(-26)
    return ohlc


def add_bbands(ohlc):
    bbands = ta.bbands(close=ohlc['close'])
    ohlc['bbands_m'] = bbands['BBM_5_2.0']  # Middle Band (Moving Average)
    ohlc['bbands_u'] = bbands['BBU_5_2.0']  # Upper Band
    ohlc['bbands_l'] = bbands['BBL_5_2.0']  # Lower Band
    return ohlc


__classic_indicator_columns = [
    'bbands_u', 'bbands_m', 'bbands_l', 'sc_obv', 'sc_cci', 'rsi', 'mfi',
    'ichi_conv', 'ichi_base', 'ichi_lead_a', 'ichi_lead_b', 'ichi_lag'
]


def classic_indicator_columns():
    return __classic_indicator_columns


__scaleless_indicators = ['sc_cci', 'rsi', 'mfi', 'sc_obv', ]


def scaleless_indicators():
    return __scaleless_indicators


def add_classic_indicators(ohlcv):
    """

    Args:
        ohlcv:

    Returns:

    """
    previous_columns = set(ohlcv.columns)
    obv = ta.obv(close=ohlcv['close'], volume=ohlcv['volume'])
    mu = obv.rolling(288, min_periods=1).mean()
    std = obv.rolling(288, min_periods=1).std(ddof=0).replace(0, 1e-9)
    ohlcv['sc_obv'] = 10 * (obv - mu) / (3 * std)
    cci = ta.cci(high=ohlcv['high'], low=ohlcv['low'], close=ohlcv['close'])
    mu = cci.rolling(288, min_periods=1).mean()
    std = cci.rolling(288, min_periods=1).std(ddof=0).replace(0, 1e-9)
    ohlcv['sc_cci'] = 10 * (cci - mu) / (3 * std)
    ohlcv['rsi'] = ta.rsi(close=ohlcv['close'])
    ohlcv['mfi'] = pd.Series().astype(float)
    ohlcv['mfi'] = \
        ta.mfi(high=ohlcv['high'], low=ohlcv['low'], close=ohlcv['close'], volume=ohlcv['volume']).astype(float)
    ohlcv = add_bbands(ohlcv)
    ohlcv = add_ichimoku(ohlcv)
    final_columns = set(ohlcv.columns)
    added_columns = final_columns - previous_columns
    assert added_columns.difference(classic_indicator_columns()) == set()
    return ohlcv


def zz_bollinger_width(ohlc, position_max_bars):
    bollinger = ta.bbands(ohlc['close'], length=position_max_bars, std=2)
    ohlc['bollinger_width'] = bollinger[f'BBU_{position_max_bars}_2.0'] - bollinger[f'BBL_{position_max_bars}_2.0']
    return ohlc

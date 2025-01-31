import pandas as pd
import pandas_ta as ta

from app.helper.importer import ta


def add_ichimoku(ohlc):
    org_ichimoku = ta.ichimoku(
        high=ohlc['high'],
        low=ohlc['low'],
        close=ohlc['close'],
    )
    # ichimoku = pd.concat([org_ichimoku[0], org_ichimoku[1]], axis=1)
    ichimoku = org_ichimoku[0]
    # Unpack the components and rename to lowercase
    ohlc['ichimoku_conversion'] = ichimoku['ITS_9']  # Tenkan-sen
    ohlc['ichimoku_base'] = ichimoku['IKS_26']  # Kijun-sen
    ohlc['ichimoku_lagging'] = ichimoku['ICS_26']  # Chikou Span
    ohlc['ichimoku_lead_a'] = ichimoku['ISA_9']  # Senkou Span A
    ohlc['ichimoku_lead_b'] = ichimoku['ISB_26']  # Senkou Span B

    # # Shift the leading spans forward by 26 periods
    # ohlc['leading_span_a'] = ohlc['leading_span_a'].shift(26)
    # ohlc['leading_span_b'] = ohlc['leading_span_b'].shift(26)
    #
    # # Shift the Lagging Span backward by 26 periods
    # ohlc['lagging_span'] = ohlc['lagging_span'].shift(-26)
    return ohlc


def add_bbands(ohlc):
    bbands = ta.bbands(ohlc['close'])
    ohlc['bbands_middle'] = bbands['BBM_5_2.0']  # Middle Band (Moving Average)
    ohlc['bbands_upper_band'] = bbands['BBU_5_2.0']  # Upper Band
    ohlc['bbands_lower_band'] = bbands['BBL_5_2.0']  # Lower Band
    return ohlc


classic_indicator_columns = [
    'obv', 'cci', 'rsi', 'mfi', 'bbands_upper', 'bbands_middle', 'bbands_lower',
    'ichimoku_conversion', 'ichimoku_base', 'ichimoku_lead_a', 'ichimoku_lead_b', 'ichimoku_lagging'
]


def add_classic_indicators(ohlcv):
    """

    Args:
        ohlcv:

    Returns:

    """
    previous_columns = set(ohlcv.columns)
    ohlcv['obv'] = ta.obv(ohlcv['close'], ohlcv['volume'])
    ohlcv['cci'] = ta.cci(ohlcv['high'], ohlcv['low'], ohlcv['close'])
    ohlcv['rsi'] = ta.rsi(ohlcv['close'])
    ohlcv['mfi'] = ta.mfi(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
    ohlcv = add_bbands(ohlcv)
    ohlcv = add_ichimoku(ohlcv)
    final_columns = set(ohlcv.columns)
    added_columns = final_columns - previous_columns
    assert added_columns.difference(classic_indicator_columns) == set()
    return ohlcv


def zz_bollinger_width(ohlc, position_max_bars):
    bollinger = ta.bbands(ohlc['close'], length=position_max_bars, std=2)
    ohlc['bollinger_width'] = bollinger[f'BBU_{position_max_bars}_2.0'] - bollinger[f'BBL_{position_max_bars}_2.0']
    return ohlc

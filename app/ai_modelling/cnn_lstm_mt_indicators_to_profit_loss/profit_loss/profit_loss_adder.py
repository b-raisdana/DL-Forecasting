from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
import pandas_ta as ta
from plotly.subplots import make_subplots

from FigurePlotter.plotter import show_and_save_plot
from ai_modelling.training_data.PreProcessing.encoding.rolling_mean_std import reverse_rolling_mean_std
from helper.br_py.profiling import profile_it
from helper.functions import log_d
from helper.importer import go


def max_profit_n_loss(ohlc, position_max_bars, action_delay, rolling_window):
    """
    Calculates and appends additional columns to an OHLC (open-high-low-close) DataFrame to track the worst-case and best-case scenarios
    for long and short positions, as well as distances to these extreme points.

    Parameters:
    -----------
    ohlc : pandas.DataFrame
        A DataFrame containing OHLC data with columns ['open', 'high', 'low', 'close'].

    position_max_bars : int
        The maximum number of bars (time steps) to hold a position.

    action_delay : int
        The delay (in bars) before an action can be taken. Used to compute the worst-case scenario for opening positions.

    rolling_window : int
        The rolling window size (in bars) used to calculate the highest high and lowest low over a specified period.

    Returns:
    --------
    pandas.DataFrame
        The original `ohlc` DataFrame with the following additional columns:
        - `worst_long_open`: The worst-case high for a long position, delayed by `action_delay`.
        - `worst_short_open`: The worst-case low for a short position, delayed by `action_delay`.
        - `max_high`: The highest high within the rolling window, shifted back by `position_max_bars`.
        - `min_low`: The lowest low within the rolling window, shifted back by `position_max_bars`.
        - `max_high_distance`: The relative distance (in bars) to the highest high within the rolling window.
        - `min_low_distance`: The relative distance (in bars) to the lowest low within the rolling window.

    Notes:
    ------
    - This function is designed for financial time-series analysis and assumes the presence of 'high' and 'low' columns in the input DataFrame.
    - The rolling operations include adjustments for shifting values based on `action_delay` and `position_max_bars`.

    Example:
    --------
    >>> import pandas as pd
    >>> ohlc = pd.DataFrame({
    ...     'open': [100, 101, 102],
    ...     'high': [105, 106, 107],
    ...     'low': [95, 96, 97],
    ...     'close': [104, 105, 106]
    ... })
    >>> result = max_profit_n_loss(ohlc, position_max_bars=2, action_delay=1, rolling_window=3)
    >>> print(result)
    """

    ohlc['worst_long_open'] = (
        ohlc['high'].rolling(window=action_delay, min_periods=1).max().shift(1 - action_delay))
    ohlc['worst_short_open'] = (
        ohlc['low'].rolling(window=action_delay, min_periods=1).min().shift(1 - action_delay))
    ohlc['max_high'] = ohlc['high'].rolling(rolling_window, min_periods=1).max().shift(
        -position_max_bars)
    ohlc['min_low'] = ohlc['low'].rolling(rolling_window, min_periods=1).min().shift(
        -position_max_bars)

    ohlc['max_high_distance'] = ohlc['high'].rolling(rolling_window).apply(lambda x: x.argmax(), raw=True).shift(
        -position_max_bars) + action_delay
    ohlc['min_low_distance'] = ohlc['low'].rolling(rolling_window).apply(lambda x: x.argmin(), raw=True).shift(
        -position_max_bars) + action_delay
    return ohlc


def quantile_maxes(ohlc, rolling_window, quantiles):
    """
    Adds quantile-based maximum and minimum values, along with their distances, to an OHLC (open-high-low-close) DataFrame.

    This function calculates the highest high and lowest low values, as well as their relative positions, for various quantile-based
    rolling window sizes. The results are appended as new columns to the input DataFrame.

    Parameters:
    -----------
    ohlc : pandas.DataFrame
        A DataFrame containing OHLC data with at least 'high' and 'low' columns.

    rolling_window : int
        The base size of the rolling window used for quantile calculations.

    quantiles : int
        The number of quantile divisions for the rolling window. For each quantile `i`, a rolling window of
        size `i * rolling_window / quantiles` is used.

    Returns:
    --------
    pandas.DataFrame
        The original `ohlc` DataFrame with additional columns for each quantile `i`:
        - `q{i}_max_high`: The maximum 'high' value within the rolling window of size `i * rolling_window / quantiles`.
        - `q{i}_min_low`: The minimum 'low' value within the rolling window of size `i * rolling_window / quantiles`.
        - `q{i}_max_high_distance`: The relative distance (in bars) to the highest high within the rolling window.
        - `q{i}_min_low_distance`: The relative distance (in bars) to the lowest low within the rolling window.

    Notes:
    ------
    - Rolling windows for quantiles are calculated as `int(i * rolling_window / quantiles)` for each `i` from 1 to `quantiles`.
    - The calculated distances (`q{i}_max_high_distance` and `q{i}_min_low_distance`) are indices relative to the start of the rolling window.

    Example:
    --------
    >>> import pandas as pd
    >>> ohlc = pd.DataFrame({
    ...     'high': [105, 106, 107, 108],
    ...     'low': [95, 96, 97, 98]
    ... })
    >>> result = quantile_maxes(ohlc, rolling_window=4, quantiles=2)
    >>> print(result)

    """

    new_columns = {}
    for i in range(1, quantiles + 1):
        q_rolling_window = int(i * rolling_window / quantiles)
        new_columns[f'q{i}_max_high'] = \
            ohlc['high'].rolling(q_rolling_window, min_periods=q_rolling_window).max().shift(-q_rolling_window)
        new_columns[f'q{i}_min_low'] = \
            ohlc['low'].rolling(q_rolling_window, min_periods=q_rolling_window).min().shift(-q_rolling_window)

        new_columns[f'q{i}_max_high_distance'] = (ohlc['high'].rolling(q_rolling_window)
                                                  .apply(lambda x: x.argmax(), raw=True).shift(-q_rolling_window))
        new_columns[f'q{i}_min_low_distance'] = (ohlc['low'].rolling(q_rolling_window)
                                                 .apply(lambda x: x.argmin(), raw=True).shift(-q_rolling_window))

    new_columns_df = pd.DataFrame(new_columns)
    ohlc = pd.concat([ohlc, new_columns_df], axis=1)
    return ohlc


def long_n_short_drawdown(ohlc, position_max_bars, quantiles, trigger_tf):
    """
    Computes and appends long and short drawdown metrics to an OHLC (open-high-low-close) DataFrame based on quantile-based analysis.

    The function calculates drawdown metrics for both long and short positions by comparing the worst-case scenarios
    with quantile-based minimum and maximum values.

    Parameters:
    -----------
    ohlc : pandas.DataFrame
        A DataFrame containing OHLC data with additional pre-computed columns:
        - `max_high_distance`: Distance (in bars) to the highest high within the rolling window.
        - `min_low_distance`: Distance (in bars) to the lowest low within the rolling window.
        - Quantile-based columns (`q{i}_max_high`, `q{i}_min_low` for `i` in range 1 to `quantiles`).
        - `worst_long_open`: The worst-case high for a long position.
        - `worst_short_open`: The worst-case low for a short position.

    position_max_bars : int
        The maximum number of bars (time steps) to hold a position.

    quantiles : int
        The number of quantile divisions for the rolling window, used for quantile-based calculations.

    trigger_tf : int
        A placeholder parameter, reserved for additional functionality or time-frame specifications
        (not directly used in this function).

    Returns:
    --------
    pandas.DataFrame
        The original `ohlc` DataFrame with the following additional columns:
        - `max_high_quantile`: Quantile index of the maximum high distance.
        - `min_low_quantile`: Quantile index of the minimum low distance.
        - `quantile_long_min_low`: The minimum low value corresponding to the `max_high_quantile`.
        - `long_drawdown`: The percentage drawdown for long positions.
        - `quantile_short_max_high`: The maximum high value corresponding to the `min_low_quantile`.
        - `short_drawdown`: The percentage drawdown for short positions.

    Notes:
    ------
    - Quantile indices are computed by dividing the distance columns by `position_max_bars / quantiles`.
    - The function uses `numpy` for efficient indexing of quantile-based columns.
    - Drawdown is calculated as a percentage of the worst-case open values (`worst_long_open` and `worst_short_open`).

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> ohlc = pd.DataFrame({
    ...     'max_high_distance': [1, 2, 3],
    ...     'min_low_distance': [1, 2, 3],
    ...     'worst_long_open': [100, 110, 120],
    ...     'worst_short_open': [90, 85, 80],
    ...     'q1_max_high': [105, 115, 125],
    ...     'q1_min_low': [95, 90, 85],
    ...     'q2_max_high': [110, 120, 130],
    ...     'q2_min_low': [90, 85, 80]
    ... })
    >>> result = long_n_short_drawdown(ohlc, position_max_bars=2, quantiles=2, trigger_tf=1)
    >>> print(result)

    """

    not_na_indexes = ohlc[~ohlc.isna().any(axis='columns')].index
    ohlc.loc[not_na_indexes, 'max_high_quantile'] = (
            ohlc.loc[not_na_indexes, 'max_high_distance'] / (position_max_bars / quantiles))
    ohlc.loc[not_na_indexes, 'min_low_quantile'] = (
            ohlc.loc[not_na_indexes, 'min_low_distance'] / (position_max_bars / quantiles))
    min_low_np_a = ohlc.loc[not_na_indexes, [f'q{i}_min_low' for i in range(1, quantiles + 1)]].to_numpy()
    ohlc.loc[not_na_indexes, 'quantile_long_min_low'] = min_low_np_a[
        np.arange(len(ohlc.loc[not_na_indexes])), ohlc.loc[not_na_indexes, 'max_high_quantile'].astype(int)]
    # ohlc['long_drawdown'] = \
    #     (ohlc['worst_long_open'] - ohlc['quantile_long_min_low']) / ohlc['worst_long_open']
    ohlc['long_drawdown'] = \
        (ohlc['worst_long_open'] - ohlc['quantile_long_min_low']) / ohlc['atr']
    ohlc['absolute_long_drawdown'] = \
        (ohlc['worst_long_open'] - ohlc['quantile_long_min_low'])
    max_high_np_a = ohlc.loc[not_na_indexes, [f'q{i}_max_high' for i in range(1, quantiles + 1)]].to_numpy()
    ohlc.loc[not_na_indexes, 'quantile_short_max_high'] = max_high_np_a[
        np.arange(len(ohlc.loc[not_na_indexes])), ohlc.loc[not_na_indexes, 'min_low_quantile'].astype(int)]
    # ohlc['short_drawdown'] = \
    #     (ohlc['quantile_short_max_high'] - ohlc['worst_short_open']) / ohlc['worst_short_open']
    ohlc['short_drawdown'] = \
        (ohlc['quantile_short_max_high'] - ohlc['worst_short_open']) / ohlc['atr']
    ohlc['absolute_short_drawdown'] = \
        (ohlc['quantile_short_max_high'] - ohlc['worst_short_open'])

    assert len(ohlc['long_drawdown'].dropna()) > 0
    assert all(ohlc['long_drawdown'].dropna() > 0)

    assert len(ohlc['short_drawdown'].dropna()) > 0
    assert all(ohlc['short_drawdown'].dropna() > 0)

    assert len(ohlc['absolute_long_drawdown'].dropna()) > 0
    assert all(ohlc['absolute_long_drawdown'].dropna() > 0)

    assert len(ohlc['absolute_short_drawdown'].dropna()) > 0
    assert all(ohlc['absolute_short_drawdown'].dropna() > 0)

    return ohlc


def profit_n_loss(ohlc, bar_width_risk_free_rate, order_fee, max_risk):
    """
    Adds profit, loss, and risk metrics for long and short positions to an OHLC (open-high-low-close) DataFrame.

    This function calculates various metrics, including weighted profits, risks, and signals for both long and short
    trading positions, based on input parameters such as risk-free daily rate, order fees, and maximum risk tolerance.

    Parameters:
    -----------
    ohlc : pandas.DataFrame
        A DataFrame containing OHLC data with additional pre-computed columns:
        - `max_high`: The highest high within a rolling window.
        - `min_low`: The lowest low within a rolling window.
        - `worst_long_open`: The worst-case high for a long position.
        - `worst_short_open`: The worst-case low for a short position.
        - `max_high_distance`: The distance (in bars) to the highest high.
        - `min_low_distance`: The distance (in bars) to the lowest low.
        - `long_drawdown`: The percentage drawdown for long positions.
        - `short_drawdown`: The percentage drawdown for short positions.

    risk_free_daily_rate : float
        The risk-free rate expressed as a daily value (e.g., 0.0001 for 0.01% daily rate).

    order_fee : float
        The transaction fee for placing an order, expressed as a proportion of the position size.

    max_risk : float
        The maximum allowable risk value for positions. Positions exceeding this risk are penalized.

    Returns:
    --------
    pandas.DataFrame
        The original `ohlc` DataFrame with the following additional columns:
        - `long_profit`: Absolute profit for long positions (`max_high - worst_long_open`).
        - `short_profit`: Absolute profit for short positions (`worst_short_open - min_low`).
        - `weighted_long_profit`: Adjusted profit for long positions, accounting for risk-free rate and order fees.
        - `weighted_short_profit`: Adjusted profit for short positions, accounting for risk-free rate and order fees.
        - `long_risk`: Risk metric for long positions (`long_drawdown / weighted_long_profit`).
        - `short_risk`: Risk metric for short positions (`short_drawdown / weighted_short_profit`).
        - `long_signal`: Signal strength for long positions, adjusted for risk and weighted profit.
        - `short_signal`: Signal strength for short positions, adjusted for risk and weighted profit.

    Notes:
    ------
    - Positions with a weighted profit ≤ 0 or risk > `max_risk` are penalized by capping their risk at `max_risk`
      and setting their signal strength to zero.
    - The function assumes the presence of specific pre-computed columns; ensure these are calculated beforehand.

    Example:
    --------
    >>> import pandas as pd
    >>> ohlc = pd.DataFrame({
    ...     'max_high': [110, 120, 130],
    ...     'min_low': [90, 85, 80],
    ...     'worst_long_open': [100, 110, 120],
    ...     'worst_short_open': [90, 85, 80],
    ...     'max_high_distance': [1, 2, 3],
    ...     'min_low_distance': [1, 2, 3],
    ...     'long_drawdown': [0.1, 0.2, 0.15],
    ...     'short_drawdown': [0.05, 0.1, 0.08]
    ... })
    >>> result = profit_n_loss(ohlc, bar_width_risk_free_rate=0.0001, order_fee=0.001, max_risk=0.5)
    >>> print(result)

    """

    ohlc['long_profit'] = ohlc['max_high'] - ohlc['worst_long_open']
    ohlc['short_profit'] = ohlc['worst_short_open'] - ohlc['min_low']
    # ohlc['weighted_long_profit'] = (
    #         ohlc['long_profit'] / ohlc['close'] - ohlc['max_high_distance'] * risk_free_daily_rate - order_fee)
    # ohlc['weighted_short_profit'] = (
    #         ohlc['short_profit'] / ohlc['close'] - ohlc['min_low_distance'] * risk_free_daily_rate - order_fee)
    ohlc['weighted_long_profit'] = (
            ohlc['long_profit'] / ohlc['atr'] - ohlc['max_high_distance'] * bar_width_risk_free_rate - order_fee)
    ohlc['weighted_short_profit'] = (
            ohlc['short_profit'] / ohlc['atr'] - ohlc['min_low_distance'] * bar_width_risk_free_rate - order_fee)

    # ohlc['weighted_long_profit'] = (
    #         ohlc['long_profit'] / ohlc['long_sl_distance'] -
    #         ohlc['max_high_distance'] * bar_width_risk_free_rate - order_fee)
    # ohlc['weighted_short_profit'] = (
    #         ohlc['short_profit'] / ohlc['short_sl_distance'] -
    #         ohlc['min_low_distance'] * bar_width_risk_free_rate - order_fee)
    ohlc['long_risk'] = (ohlc['long_drawdown'] / ohlc['weighted_long_profit'])
    ohlc['short_risk'] = (ohlc['short_drawdown'] / ohlc['weighted_short_profit'])

    loser_shorts = ohlc[(ohlc['weighted_short_profit'] <= 0) | (ohlc['short_risk'] > max_risk)].index
    loser_longs = ohlc[(ohlc['weighted_long_profit'] <= 0) | (ohlc['long_risk'] > max_risk)].index
    ohlc.loc[loser_shorts, 'short_risk'] = 1
    ohlc.loc[loser_longs, 'long_risk'] = 1
    # ohlc['range'] =  ohlc['max_high'] - ohlc['min_low']
    ohlc['long_signal'] = ((1 - ohlc['long_risk'].fillna(1)) * (
            ohlc['weighted_long_profit'].fillna(0) / ohlc['long_sl_distance']))
    ohlc.loc[loser_longs, 'long_signal'] = 0
    # ohlc['long_signal'] =  (ohlc['long_signal']) / ta.ema(ohlc['long_signal'], length=position_max_bars)
    ohlc['short_signal'] = (1 - ohlc['short_risk'].fillna(1)) * (
            ohlc['weighted_short_profit'].fillna(0) / ohlc['short_sl_distance'])
    ohlc.loc[loser_shorts, 'short_signal'] = 0
    return ohlc


def singular_stop_loss(series: pd.Series, window: int, mode: Literal['long', 'short'], tops_percent):
    mode_map = {
        'long': 'smallest',
        'short': 'largest',
    }
    try:
        return tops_mean(series, window=window, percent=tops_percent, mode=mode_map[mode])
    except KeyError:
        raise KeyError(f"mod shall be in [{mode_map.keys()}], {mode} is not supported!")


def zz_stop_loss(ohlc, windows: int, tops_percent, sl_atr_distance=2):
    ohlc['long_sl'] = singular_stop_loss(ohlc['low'], window=windows, tops_percent=tops_percent, mode='long')
    ohlc['long_sl'] = np.where(abs(ohlc['long_sl'] - ohlc['low']) < ohlc['atr'] * sl_atr_distance,
                               ohlc['low'] - ohlc['atr'] * sl_atr_distance, ohlc['long_sl'])
    ohlc['short_sl'] = singular_stop_loss(ohlc['high'], window=windows, tops_percent=tops_percent,
                                          mode='short')
    ohlc['short_sl'] = np.where(abs(ohlc['short_sl'] - ohlc['high']) < ohlc['atr'] * sl_atr_distance,
                                ohlc['high'] + ohlc['atr'] * sl_atr_distance, ohlc['short_sl'])
    ohlc['long_sl_distance'] = ohlc['worst_long_open'] - ohlc['long_sl']
    ohlc['short_sl_distance'] = ohlc['short_sl'] - ohlc['worst_short_open']
    return ohlc


def stop_loss(ohlc):
    ohlc['long_sl_distance'] = np.maximum(1, ohlc[
        'long_drawdown'])  # ohlc[['absolute_long_drawdown', 'atr']].max(axis='columns') / ohlc['atr']
    ohlc['short_sl_distance'] = np.maximum(1, ohlc[
        'short_drawdown'])  # ohlc[['absolute_short_drawdown', 'atr']].max(axis='columns') / ohlc['atr']
    return ohlc


@profile_it
def add_long_n_short_profit(ohlc,
                            position_max_bars=3 * 4 * 4 * 4 * 1,  # 768 5-minute intervals = 16 hours
                            action_delay=2,
                            risk_free_daily_rate=(0.10 / 365),  # for 10% annual rate
                            order_fee=0.005,
                            quantiles=50,
                            max_risk=0.1,
                            trigger_tf='15min'):
    """
    Calculates and appends long and short profit, loss, and risk metrics to an OHLC (open-high-low-close) DataFrame.

    This function integrates multiple computations, including maximum profit and loss, quantile-based metrics, drawdowns,
    and risk-adjusted profits for long and short positions. The results are returned as an augmented DataFrame.

    Parameters:
    -----------
    ohlc : pandas.DataFrame
        A DataFrame containing OHLC data with at least 'high', 'low', and 'close' columns.

    position_max_bars : int, optional
        The maximum number of bars (time steps) to hold a position. Default is `3 * 4 * 4 * 4 * 1` (768 5-minute intervals, equivalent to 16 hours).

    action_delay : int, optional
        The delay (in bars) before an action can be taken. Default is 2.

    risk_free_daily_rate : float, optional
        The risk-free rate expressed as a daily value. Default is 0.

    order_fee : float, optional
        The transaction fee for placing an order, expressed as a proportion of the position size. Default is 0.005 (0.5%).

    quantiles : int, optional
        The number of quantile divisions for the rolling window calculations. Default is 50.

    max_risk : float, optional
        The maximum allowable risk value for positions. Positions exceeding this risk are penalized. Default is 0.1.

    trigger_tf : str, optional
        The time frame for triggering actions, used in drawdown calculations. Default is `'15min'`.

    Returns:
    --------
    pandas.DataFrame
        The original `ohlc` DataFrame with additional columns for profit, loss, and risk metrics. These include:
        - Metrics from `max_profit_n_loss`: Worst-case scenarios for long and short positions.
        - Metrics from `quantile_maxes`: Quantile-based max/min values and distances.
        - Metrics from `long_n_short_drawdown`: Drawdown and risk-related calculations.
        - Metrics from `add_profit_n_loss`: Weighted profits, risks, and trading signals for long and short positions.

    Workflow:
    ---------
    1. Computes maximum profit and loss metrics using `max_profit_n_loss`.
    2. Calculates quantile-based metrics with `quantile_maxes`.
    3. Adds drawdown and risk calculations via `long_n_short_drawdown`.
    4. Integrates weighted profit, loss, and signal computations using `add_profit_n_loss`.

    Notes:
    ------
    - Ensure the input DataFrame has at least 'high', 'low', and 'close' columns.
    - The default `position_max_bars` is based on 5-minute intervals; adjust this based on your data frequency.

    Example:
    --------
    >>> import pandas as pd
    >>> ohlc = pd.DataFrame({
    ...     'high': [105, 106, 107, 108],
    ...     'low': [95, 96, 97, 98],
    ...     'close': [100, 101, 102, 103]
    ... })
    >>> result = add_long_n_short_profit(ohlc)
    >>> print(result)

    """

    rolling_window = position_max_bars - action_delay
    if 'atr' not in ohlc.columns:
        log_d('Needs to have ATR 256!')
        ohlc['atr'] = ta.atr(high=ohlc['high'], low=ohlc['low'], close=ohlc['close'], length=256)
    # ohlc['scaler'] = ta.wma(ohlc['close'], length=position_max_bars * 10)
    ohlc = max_profit_n_loss(ohlc, position_max_bars, action_delay, rolling_window)
    ohlc['long_distance_time'] = ohlc['max_high_distance'] * pd.to_timedelta(trigger_tf)
    ohlc['short_distance_time'] = ohlc['min_low_distance'] * pd.to_timedelta(trigger_tf)
    ohlc = quantile_maxes(ohlc, rolling_window, quantiles)
    ohlc = long_n_short_drawdown(ohlc, position_max_bars, quantiles, trigger_tf)
    ohlc = stop_loss(ohlc)  # , int(position_max_bars / 4), tops_percent=0.01)
    ohlc = profit_n_loss(ohlc, order_fee=order_fee, max_risk=max_risk,
                         bar_width_risk_free_rate=risk_free_daily_rate /
                                                  (timedelta(days=1) / pd.to_timedelta(trigger_tf)), )
    return ohlc


def plot_short_profit(t):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.5, 0.5, 0.1])
    fig.add_trace(go.Candlestick(x=t.index, open=t['open'], close=t['close'], high=t['high'], low=t['low']), row=2,
                  col=1)
    fig.add_scatter(x=t.index, y=t['max_high'], mode='lines', line=dict(color='red', width=1), name='max_high', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['min_low'], mode='lines', line=dict(color='blue', width=1), name='min_low', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['quantile_short_max_high'], mode='lines', line=dict(color='magenta', width=1),
                    name='quantile_short_max_high', row=2, col=1)
    fig.add_scatter(x=t.index, y=t['short_risk'], mode='lines', line=dict(color='yellow', width=1),
                    name='short_risk')
    fig.add_scatter(x=t.index, y=t['short_drawdown'], mode='lines', line=dict(color='pink', width=1),
                    name='short_drawdown')
    # fig.add_scatter(x=t.index, y=t['short_profit'], mode='lines', line=dict(color='cyan', width=1),
    #                 name='short_profit')
    fig.add_scatter(x=t.index, y=t['weighted_short_profit'], mode='lines', line=dict(color='blue', width=1),
                    name='weighted_short_profit')
    fig.add_scatter(x=t.index, y=t['short_signal'], mode='lines', line=dict(color='green', width=1),
                    name='short_signal')
    fig.add_scatter(x=t.index, y=t['short_sl'], mode='lines', line=dict(color='orange', width=.5), name='short_sl')
    fig.add_scatter(x=t.index, y=t['long_sl'], mode='lines', line=dict(color='orange', width=.5), name='long_sl')
    fig.update_layout(dragmode="zoom", yaxis2=dict(fixedrange=False), margin=dict(l=0, r=0, t=10, b=10), ).show()


def plot_long_profit(t):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.6])
    fig.add_trace(go.Candlestick(x=t.index, open=t['open'], close=t['close'], high=t['high'], low=t['low']), row=2,
                  col=1)
    fig.add_scatter(x=t.index, y=t['max_high'], mode='lines', line=dict(color='blue', width=1), name='max_high', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['min_low'], mode='lines', line=dict(color='red', width=1), name='min_low', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['quantile_long_min_low'], mode='lines', line=dict(color='magenta', width=1),
                    name='quantile_long_min_low', row=2, col=1)
    fig.add_scatter(x=t.index, y=t['long_risk'], mode='lines', line=dict(color='yellow', width=1),
                    name='long_risk')
    fig.add_scatter(x=t.index, y=t['long_drawdown'], mode='lines', line=dict(color='pink', width=1),
                    name='long_drawdown')
    # fig.add_scatter(x=t.index, y=t['long_profit'], mode='lines', line=dict(color='cyan', width=1),
    #                 name='long_profit')
    fig.add_scatter(x=t.index, y=t['weighted_long_profit'], mode='lines', line=dict(color='blue', width=1),
                    name='weighted_long_profit')
    fig.add_scatter(x=t.index, y=t['long_signal'], mode='lines', line=dict(color='green', width=1),
                    name='long_signal')
    fig.update_layout(dragmode="zoom", yaxis2=dict(fixedrange=False), margin=dict(l=0, r=0, t=10, b=10), ).show()


import math


def scaler(series, target_max):
    max_value = series.max()
    min_value = series.min()
    org_scale = target_max / max(abs(max_value), abs(min_value))

    kilo_power_map = {
        0: '',
        1: 'K',
        2: 'M',
        3: 'B',
    }

    if org_scale > 1:
        leading_number, remained_zeros, round_scale, unit = scale_rounder(kilo_power_map, org_scale)
        scale_mark = f'x{leading_number}' + "0" * remained_zeros + unit
    else:
        org_scale = 1 / org_scale
        leading_number, remained_zeros, round_scale, unit = scale_rounder(kilo_power_map, org_scale)
        round_scale = 1 / round_scale
        scale_mark = f'/{leading_number}' + "0" * remained_zeros + unit

    return scale_mark, round_scale


def scale_rounder(kilo_power_map, org_scale):
    zero_digits = int(math.log10(org_scale))
    leading_number = int(org_scale / math.pow(10, zero_digits))
    round_scale = leading_number * math.pow(10, zero_digits)
    unit_index = int(zero_digits / 3)
    unit = kilo_power_map[unit_index]
    remained_zeros = zero_digits - unit_index * 3
    return leading_number, remained_zeros, round_scale, unit


def zz_plot_long_n_short_profit(t, samples=[]):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.3, 0.7, 0.3], )
    add_scaled_scatter(fig, t, 'long_profit', 'blue')
    add_scaled_scatter(fig, t, 'short_profit', 'red')
    add_scaled_scatter(fig, t, 'long_risk', 'LightSkyBlue', fill_na=1)
    add_scaled_scatter(fig, t, 'short_risk', 'yellow')
    add_scaled_scatter(fig, t, 'long_signal', 'green')
    add_scaled_scatter(fig, t, 'short_signal', 'orange')
    fig.add_scatter(x=t.index, y=t['min_low'], mode='lines', line=dict(color='red', width=1), name='min_low', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['max_high'], mode='lines', line=dict(color='blue', width=1), name='max_high', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['long_sl'], mode='lines', line=dict(color='orange', width=1), name='long_sl', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['short_sl'], mode='lines', line=dict(color='orange', width=1), name='short_sl',
                    row=2,
                    col=1)
    fig.add_trace(
        go.Candlestick(x=t.index, open=t['open'], close=t['close'], high=t['high'], low=t['low']), row=2, col=1)
    if len(samples) > 0:
        showlegend = True
        for sample in samples:
            # x = [sample.replace(tzinfo=None), sample.replace(tzinfo=None) + position_max_bars * timedelta(days=1)]
            # y = [float(t.loc[sample, 'close']), float(t.loc[sample, 'close'])]
            # fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='gray', width=1), name='position_max_bars', row=2, col=1,
            #                 legendgroup='position_max_bars', showlegend=showlegend)
            x = [sample.replace(tzinfo=None),
                 sample.replace(tzinfo=None) + t.loc[sample, 'max_high_distance'] * timedelta(days=1)]
            y = [float(t.loc[sample, 'high']), float(t.loc[sample, 'max_high'])]
            fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='cyan', width=1), name='max_high', row=2, col=1,
                            legendgroup='max_high', showlegend=showlegend)
            x = [sample.replace(tzinfo=None),
                 sample.replace(tzinfo=None) + t.loc[sample, 'min_low_distance'] * timedelta(days=1)]
            y = [float(t.loc[sample, 'low']), float(t.loc[sample, 'min_low'])]
            fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='pink', width=1), name='min_low', row=2, col=1,
                            legendgroup='min_low', showlegend=showlegend)
            # x = [sample.replace(tzinfo=None),
            #      sample.replace(tzinfo=None) + t.loc[sample, 'min_low_distance'] * timedelta(days=1)]
            # y = [float(t.loc[sample, 'worst_short_open']),
            #      float(t.loc[sample, 'worst_short_open'] + t.loc[sample, 'short_drawdown'])]
            # fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='orange', width=1), name='short_drawdown', row=2, col=1,
            #                 legendgroup='short_drawdown', showlegend=showlegend)
    fig.update_layout(dragmode="zoom", yaxis2=dict(title="Price", fixedrange=False),
                      margin=dict(l=0, r=0, t=10, b=10), ).show()
    return fig


def add_scaled_scatter(fig, t, name, color, fill_na=0):
    sc_mark, sc = scaler(t[name].fillna(fill_na), target_max=100)
    fig.add_scatter(x=t.index, y=t[name] * float(sc), mode='lines', line=dict(color=color, width=1),
                    name=f'{name}{sc_mark}')


def zz_plot_mt_train_n_test(x, y, n, base_ohlcv, show=True):
    reconstructed_double = reverse_rolling_mean_std(x['double'][n])
    reconstructed_trigger = reverse_rolling_mean_std(x['trigger'][n])
    reconstructed_pattern = reverse_rolling_mean_std(x['pattern'][n])
    reconstructed_structure = reverse_rolling_mean_std(x['structure'][n])
    # Create traces for each slice
    double_trace = go.Candlestick(x=reconstructed_double.index.get_level_values(level='date'),
                                  open=reconstructed_double['open'],
                                  high=reconstructed_double['high'],
                                  low=reconstructed_double['low'],
                                  close=reconstructed_double['close'],
                                  name='Double')
    trigger_trace = go.Candlestick(x=reconstructed_trigger.index.get_level_values(level='date'),
                                   open=reconstructed_trigger['open'],
                                   high=reconstructed_trigger['high'],
                                   low=reconstructed_trigger['low'],
                                   close=reconstructed_trigger['close'],
                                   name='Trigger')
    pattern_trace = go.Candlestick(x=reconstructed_pattern.index.get_level_values(level='date'),
                                   open=reconstructed_pattern['open'],
                                   high=reconstructed_pattern['high'],
                                   low=reconstructed_pattern['low'],
                                   close=reconstructed_pattern['close'],
                                   name='Pattern')
    structure_trace = go.Candlestick(x=reconstructed_structure.index.get_level_values(level='date'),
                                     open=reconstructed_structure['open'],
                                     high=reconstructed_structure['high'],
                                     low=reconstructed_structure['low'],
                                     close=reconstructed_structure['close'],
                                     name='Structure')
    y_trace = go.Candlestick(x=y[n].index.get_level_values(level='date'),
                             open=y[n]['open'],
                             high=y[n]['high'],
                             low=y[n]['low'],
                             close=y[n]['close'],
                             name='y')
    min_date = min(
        x['double'][n].index.get_level_values(level='date').min(),
        x['trigger'][n].index.get_level_values(level='date').min(),
        x['pattern'][n].index.get_level_values(level='date').min(),
        x['structure'][n].index.get_level_values(level='date').min(),
        y[n].index.get_level_values(level='date').min(),
    )
    max_date = max(
        x['double'][n].index.get_level_values(level='date').max(),
        x['trigger'][n].index.get_level_values(level='date').max(),
        x['pattern'][n].index.get_level_values(level='date').max(),
        x['structure'][n].index.get_level_values(level='date').max(),
        y[n].index.get_level_values(level='date').max()
    )
    ohlcv_slice = base_ohlcv[min_date:max_date]
    ohlcv_trace = go.Candlestick(x=ohlcv_slice.index.get_level_values(level='date'),
                                 open=ohlcv_slice['open'],
                                 high=ohlcv_slice['high'],
                                 low=ohlcv_slice['low'],
                                 close=ohlcv_slice['close'],
                                 name='ohlcv'
                                 )
    layout = go.Layout(
        title='X Items and Double Slice Over Date Range',
        xaxis=dict(title='Date', range=[min_date, max_date]),
        yaxis=dict(title='Values', fixedrange=False, ),
        showlegend=True
    )

    # Combine all traces into a figure
    fig = go.Figure(data=[double_trace, trigger_trace, pattern_trace, structure_trace, y_trace, ohlcv_trace],
                    layout=layout)

    if show:
        show_and_save_plot(fig)
    else:
        return fig


def tops_mean(series: pd.Series, window, mode: Literal['smallest', 'largest'], percent):
    top_k = max(int(np.ceil(window * percent)), 1)  # Number of top values to consider
    print(f'top_k={top_k}')
    sort_to_reverse_map = {
        'smallest': False,
        'largest': True,
    }
    return series.rolling(window).apply(lambda x: np.mean(sorted(x, reverse=sort_to_reverse_map[mode])[:top_k]),
                                        raw=True)

from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from app.Config import config
from app.FigurePlotter.plotter import show_and_save_plot
from app.PanderaDFM import MultiTimeframe
from app.ai_modelling.training_data.PreProcessing.encoding.rolling_mean_std import reverse_rolling_mean_std
from app.helper.data_preparation import pattern_timeframe, trigger_timeframe, single_timeframe
from app.helper.helper import profile_it, date_range, log_d
from app.helper.importer import go, pt


def max_profit_n_loss(ohlc, position_max_bars, action_delay, rolling_window):
    ohlc['worst_long_open'] = (
        ohlc['High'].rolling(window=action_delay, min_periods=1).max().shift(1 - action_delay))
    ohlc['worst_short_open'] = (
        ohlc['Low'].rolling(window=action_delay, min_periods=1).min().shift(1 - action_delay))
    ohlc['max_high'] = ohlc['High'].rolling(rolling_window, min_periods=1).max().shift(
        -position_max_bars)
    ohlc['min_low'] = ohlc['Low'].rolling(rolling_window, min_periods=1).min().shift(
        -position_max_bars)

    ohlc['max_high_distance'] = ohlc['High'].rolling(rolling_window).apply(lambda x: x.argmax(), raw=True).shift(
        -position_max_bars) + action_delay
    ohlc['min_low_distance'] = ohlc['Low'].rolling(rolling_window).apply(lambda x: x.argmin(), raw=True).shift(
        -position_max_bars) + action_delay
    return ohlc


def quantile_maxes(ohlc, rolling_window, quantiles):
    new_columns = {}
    for i in range(1, quantiles + 1):
        q_rolling_window = int(i * rolling_window / quantiles)
        new_columns[f'q{i}_max_high'] = \
            ohlc['High'].rolling(q_rolling_window, min_periods=q_rolling_window).max().shift(-q_rolling_window)
        new_columns[f'q{i}_min_low'] = \
            ohlc['Low'].rolling(q_rolling_window, min_periods=q_rolling_window).min().shift(-q_rolling_window)

        new_columns[f'q{i}_max_high_distance'] = (ohlc['High'].rolling(q_rolling_window)
                                           .apply(lambda x: x.argmax(), raw=True).shift(-q_rolling_window))
        new_columns[f'q{i}_min_low_distance'] = (ohlc['Low'].rolling(q_rolling_window)
                                          .apply(lambda x: x.argmin(), raw=True).shift(-q_rolling_window))

    new_columns_df = pd.DataFrame(new_columns)
    ohlc = pd.concat([ohlc, new_columns_df], axis=1)
    return ohlc


def long_n_short_drawdown(ohlc, position_max_bars, quantiles, trigger_tf):
    not_na_indexes = ohlc[~ohlc.isna().any(axis='columns')].index
    ohlc.loc[not_na_indexes, 'max_high_quantile'] = (
            ohlc.loc[not_na_indexes, 'max_high_distance'] / (position_max_bars / quantiles))
    ohlc.loc[not_na_indexes, 'min_low_quantile'] = (
            ohlc.loc[not_na_indexes, 'min_low_distance'] / (position_max_bars / quantiles))
    min_low_np_a = ohlc.loc[not_na_indexes, [f'q{i}_min_low' for i in range(1, quantiles + 1)]].to_numpy()
    quantile_max_high_consistency = ohlc.loc[not_na_indexes, 'max_high_quantile'].astype(int).eq(0)
    consistent_quantile_max_high = quantile_max_high_consistency[~quantile_max_high_consistency].index
    ohlc.loc[not_na_indexes, 'quantile_long_min_low'] = min_low_np_a[
        np.arange(len(ohlc.loc[not_na_indexes])), ohlc.loc[not_na_indexes, 'max_high_quantile'].astype(int)]
    # ohlc.loc[consistent_quantile_max_high, 'pre_quantile_long_min_low'] = min_low_np_a[
    #     np.arange(len(consistent_quantile_max_high)), ohlc.loc[consistent_quantile_max_high, 'max_high_quantile'].astype(
    #         int) - 1]
    # ohlc.loc[consistent_quantile_max_high, 'long_drawdown_consistency'] = (
    #         ohlc.loc[consistent_quantile_max_high, 'pre_quantile_long_min_low']
    #         == ohlc.loc[consistent_quantile_max_high, 'quantile_long_min_low'])
    ohlc['long_drawdown'] = \
        (ohlc['worst_long_open'] - ohlc['quantile_long_min_low']) / ohlc['worst_long_open']
    max_high_np_a = ohlc.loc[not_na_indexes, [f'q{i}_max_high' for i in range(1, quantiles + 1)]].to_numpy()
    quantile_min_low_consistency = ohlc.loc[not_na_indexes, 'min_low_quantile'].astype(int).eq(0)
    # consistent_quantile_min_low = quantile_min_low_consistency[~quantile_min_low_consistency].index
    ohlc.loc[not_na_indexes, 'quantile_short_max_high'] = max_high_np_a[
        np.arange(len(ohlc.loc[not_na_indexes])), ohlc.loc[not_na_indexes, 'min_low_quantile'].astype(int)]
    # ohlc.loc[consistent_quantile_min_low, 'pre_quantile_short_max_high'] = max_high_np_a[
    #     np.arange(len(consistent_quantile_min_low)), ohlc.loc[consistent_quantile_min_low, 'min_low_quantile'].astype(
    #         int) - 1]
    # ohlc.loc[consistent_quantile_min_low, 'short_drawdown_consistency'] = (
    #         ohlc.loc[consistent_quantile_min_low, 'pre_quantile_short_max_high']
    #         == ohlc.loc[consistent_quantile_min_low, 'quantile_short_max_high'])
    ohlc['short_drawdown'] = \
        (ohlc['quantile_short_max_high'] - ohlc['worst_short_open']) / ohlc['worst_short_open']
    return ohlc


def add_profit_n_loss(ohlc, risk_free_daily_rate, order_fee, max_risk):
    ohlc['long_profit'] = ohlc['max_high'] - ohlc['worst_long_open']
    ohlc['short_profit'] = ohlc['worst_short_open'] - ohlc['min_low']
    ohlc['weighted_long_profit'] = (
            ohlc['long_profit'] / ohlc['Close'] - ohlc['max_high_distance'] * risk_free_daily_rate - order_fee)
    ohlc['weighted_short_profit'] = (
            ohlc['short_profit'] / ohlc['Close'] - ohlc['min_low_distance'] * risk_free_daily_rate - order_fee)
    ohlc['long_risk'] = (ohlc['long_drawdown'] / ohlc['weighted_long_profit'])
    ohlc['short_risk'] = (ohlc['short_drawdown'] / ohlc['weighted_short_profit'])

    loser_shorts = ohlc[(ohlc['weighted_short_profit'] <= 0) | (ohlc['short_risk'] > max_risk)].index
    loser_longs = ohlc[(ohlc['weighted_long_profit'] <= 0) | (ohlc['long_risk'] > max_risk)].index
    ohlc.loc[loser_shorts, 'short_risk'] = max_risk
    ohlc.loc[loser_longs, 'long_risk'] = max_risk
    # ohlc['range'] =  ohlc['max_high'] - ohlc['min_low']
    ohlc['long_signal'] = ((1 - ohlc['long_risk'].fillna(1)) * (
            ohlc['weighted_long_profit'].fillna(0)/ohlc['worst_long_open']))
    ohlc.loc[loser_longs, 'long_signal'] = 0
    # ohlc['long_signal'] =  (ohlc['long_signal']) / ta.ema(ohlc['long_signal'], length=position_max_bars)
    ohlc['short_signal'] = (1 - ohlc['short_risk'].fillna(1)) * (
            ohlc['weighted_short_profit'].fillna(0)/ohlc['worst_short_open'])
    ohlc.loc[loser_shorts, 'short_signal'] = 0
    return ohlc


def add_long_n_short_profit(ohlc,
                            position_max_bars=3 * 4 * 4 * 4 * 1,  # 768 5mins = 16h
                            action_delay=2,
                            risk_free_daily_rate=0,  # (0.1 / 365),
                            order_fee=0.005,
                            quantiles=50,
                            max_risk=0.1,
                            trigger_tf = '15min'
                            ):
    rolling_window = position_max_bars - action_delay

    ohlc = max_profit_n_loss(ohlc, position_max_bars, action_delay, rolling_window)
    ohlc = quantile_maxes(ohlc, rolling_window, quantiles)
    ohlc = long_n_short_drawdown(ohlc, position_max_bars, quantiles, trigger_tf)
    ohlc = add_profit_n_loss(ohlc, risk_free_daily_rate, order_fee, max_risk)
    return ohlc


def plot_short_profit(t):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.5, 0.5, 0.1])
    fig.add_trace(go.Candlestick(x=t.index, open=t['Open'], close=t['Close'], high=t['High'], low=t['Low']), row=2,
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
    fig.update_layout(dragmode="zoom", yaxis2=dict(fixedrange=False), margin=dict(l=0, r=0, t=10, b=10), ).show()


def plot_long_profit(t):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.6])
    fig.add_trace(go.Candlestick(x=t.index, open=t['Open'], close=t['Close'], high=t['High'], low=t['Low']), row=2,
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


def plot_long_n_short_profit(t):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.3, 0.7, 0.3], )
    fig.add_scatter(x=t.index, y=t['long_profit'], mode='lines', line=dict(color='blue', width=1), name='long_profit')
    fig.add_scatter(x=t.index, y=t['short_profit'], mode='lines', line=dict(color='red', width=1), name='short_profit')
    fig.add_scatter(x=t.index, y=1000 * t['long_risk'], mode='lines', line=dict(color='LightSkyBlue', width=1),
                    name='long_risk')
    fig.add_scatter(x=t.index, y=1000 * t['short_risk'], mode='lines', line=dict(color='yellow', width=1),
                    name='short_risk')
    fig.add_scatter(x=t.index, y=1000000 * t['long_signal'], mode='lines', line=dict(color='green', width=1),
                    name='long_signal')
    fig.add_scatter(x=t.index, y=1000000 * t['short_signal'], mode='lines', line=dict(color='orange', width=1),
                    name='short_signal')
    fig.add_scatter(x=t.index, y=t['min_low'], mode='lines', line=dict(color='red', width=1), name='min_low', row=2,
                    col=1)
    fig.add_trace(
        go.Candlestick(x=t.index, open=t['Open'], close=t['Close'], high=t['High'], low=t['Low']), row=2, col=1)
    fig.add_scatter(x=t.index, y=t['max_high'], mode='lines', line=dict(color='blue', width=1), name='max_high', row=2,
                    col=1)

    showlegend = True
    # for sample in samples:
    #     # x = [sample.replace(tzinfo=None), sample.replace(tzinfo=None) + position_max_bars * timedelta(days=1)]
    #     # y = [float(t.loc[sample, 'Close']), float(t.loc[sample, 'Close'])]
    #     # fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='gray', width=1), name='position_max_bars', row=2, col=1,
    #     #                 legendgroup='position_max_bars', showlegend=showlegend)
    #     x = [sample.replace(tzinfo=None),
    #          sample.replace(tzinfo=None) + t.loc[sample, 'max_high_distance'] * timedelta(days=1)]
    #     y = [float(t.loc[sample, 'High']), float(t.loc[sample, 'max_high'])]
    #     fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='cyan', width=1), name='max_high', row=2, col=1,
    #                     legendgroup='max_high', showlegend=showlegend)
    #     x = [sample.replace(tzinfo=None),
    #          sample.replace(tzinfo=None) + t.loc[sample, 'min_low_distance'] * timedelta(days=1)]
    #     y = [float(t.loc[sample, 'Low']), float(t.loc[sample, 'min_low'])]
    #     fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='pink', width=1), name='min_low', row=2, col=1,
    #                     legendgroup='min_low', showlegend=showlegend)
    #     # x = [sample.replace(tzinfo=None),
    #     #      sample.replace(tzinfo=None) + t.loc[sample, 'min_low_distance'] * timedelta(days=1)]
    #     # y = [float(t.loc[sample, 'worst_short_open']),
    #     #      float(t.loc[sample, 'worst_short_open'] + t.loc[sample, 'short_drawdown'])]
    #     # fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='orange', width=1), name='short_drawdown', row=2, col=1,
    #     #                 legendgroup='short_drawdown', showlegend=showlegend)
    fig.update_layout(dragmode="zoom", yaxis2=dict(title="Price", fixedrange=False),
                      margin=dict(l=0, r=0, t=10, b=10), ).show()


def plot_mt_train_n_test(x, y, n, base_ohlcv, show=True):
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


@profile_it
def train_data_of_mt_n_profit(structure_tf, mt_ohlcv: pt.DataFrame[MultiTimeframe], x_lengths: dict,
                              batch_size: int, forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1, ):
    """
    Returns:
        (X, ys)
            - X (dict): A dictionary with keys ('double', 'trigger', 'pattern', 'structure') and corresponding
              values as lists of Pandas DataFrames. Each DataFrame represents a slice of the multi-timeframe
              input features.
                - 'double': Input feature data for the double timeframe.
                - 'trigger': Input feature data for the trigger timeframe.
                - 'pattern': Input feature data for the pattern timeframe.
                - 'structure': Input feature data for the structure timeframe.
            - ys (list): A list of Pandas DataFrames representing the forecast targets. Each DataFrame contains
              the predicted values for the future time steps (horizon) from the trigger timeframe.
    """
    training_x_columns = ['n_open', 'n_high', 'n_low', 'n_close', 'n_volume', ]
    # training_y_columns = ['n_high', 'n_low', ]
    training_y_columns = ['long_signal', 'short_signal', 'min_low', 'max_high']
    pattern_tf = pattern_timeframe(structure_tf)
    trigger_tf = trigger_timeframe(structure_tf)
    double_tf = pattern_timeframe(trigger_timeframe(structure_tf))

    length_of_training = (
            x_lengths['structure'][0] * pd.to_timedelta(structure_tf)
            + x_lengths['pattern'][0] * pd.to_timedelta(pattern_tf)
            + x_lengths['trigger'][0] * pd.to_timedelta(trigger_tf)
            + x_lengths['double'][0] * pd.to_timedelta(double_tf)
    )
    train_safe_start = mt_ohlcv.index.get_level_values(
        level='date').min() + length_of_training * 2  # * 2 for simple safe side.
    train_safe_end = \
        mt_ohlcv.index.get_level_values(level='date').max() - forecast_trigger_bars * pd.to_timedelta(trigger_tf)
    # prediction_end = mt_ohlcv.loc[pd.IndexSlice[: (
    #         mt_ohlcv.index.get_level_values(level='data').max()
    #         - forecast_trigger_bars * pd.to_timedelta(trigger_tf))]].copy()

    structure_df = single_timeframe(mt_ohlcv, structure_tf)
    pattern_df = single_timeframe(mt_ohlcv, pattern_tf)
    trigger_df = single_timeframe(mt_ohlcv, trigger_tf)
    double_df = single_timeframe(mt_ohlcv, double_tf)
    prediction_df = add_long_n_short_profit(ohlc=single_timeframe(mt_ohlcv, trigger_tf),
                                            position_max_bars=forecast_trigger_bars, trigger_tf=trigger_tf)

    duration_seconds = (train_safe_end - train_safe_start) / timedelta(seconds=1)
    if duration_seconds <= 0:
        start, end = date_range(config.processing_date_range)
        raise RuntimeError(
            f"Extend date boundary +{-duration_seconds}s({duration_seconds / (60 * 60 * 24)}days, "
            f"start:{start}<{start + duration_seconds * timedelta(seconds=1)} or "
            f"end:{end}>{end - duration_seconds * timedelta(seconds=1)}) to make possible range of end dates positive!")
    x_dfs, y_dfs, y_tester_dfs = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, [], []
    Xs, ys = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, []

    batch_remained = batch_size
    while batch_remained > 0:
        # for relative_double_end in np.random.randint(0, duration_seconds, size=batch_size):
        relative_double_end = np.random.randint(0, duration_seconds)
        double_end: datetime = train_safe_end - relative_double_end * timedelta(seconds=1)
        trigger_end = double_end - x_lengths['double'][0] * pd.to_timedelta(double_tf)
        pattern_end = trigger_end - x_lengths['trigger'][0] * pd.to_timedelta(trigger_tf)
        structure_end = pattern_end - x_lengths['pattern'][0] * pd.to_timedelta(pattern_tf)

        double_slice = double_df.loc[pd.IndexSlice[: double_end], training_x_columns].iloc[-x_lengths['double'][0]:]
        trigger_slice = trigger_df.loc[pd.IndexSlice[: trigger_end], training_x_columns].iloc[-x_lengths['trigger'][0]:]
        pattern_slice = pattern_df.loc[pd.IndexSlice[: pattern_end], training_x_columns].iloc[-x_lengths['pattern'][0]:]
        structure_slice = structure_df.loc[pd.IndexSlice[: structure_end], training_x_columns].iloc[
                          -x_lengths['structure'][0]:]

        try:
            for timeframe, slice_df, relative_tf_name in [(structure_tf, structure_slice, 'structure'),
                                                          (pattern_tf, pattern_slice, 'pattern'),
                                                          (trigger_tf, trigger_slice, 'trigger'),
                                                          (double_tf, double_slice, 'double')]:
                if abs((slice_df.index.max() - slice_df.index.min()) / pd.to_timedelta(timeframe)
                       - (x_lengths[relative_tf_name][0] - 1)) > config.max_x_gap:
                    raise AssertionError(f"Gap of > {config.max_x_gap} bars found in {config.under_process_exchange}"
                                         f"/{config.under_process_symbol}/{timeframe}:"
                                         f"{slice_df.index.min()}-{slice_df.index.max()}")
        except AssertionError as e:
            log_d(e)
            continue
        x_dfs['double'].append(double_slice)
        x_dfs['trigger'].append(trigger_slice)
        x_dfs['pattern'].append(pattern_slice)
        x_dfs['structure'].append(structure_slice)
        Xs['double'].append(np.array(double_slice[training_x_columns]))
        Xs['trigger'].append(np.array(trigger_slice[training_x_columns]))
        Xs['pattern'].append(np.array(pattern_slice[training_x_columns]))
        Xs['structure'].append(np.array(structure_slice[training_x_columns]))

        # ys.append(np.array(future_slice[training_y_columns]))
        # prediction_time = trigger_df[trigger_df.index <= double_end].index.get_level_values(level='date').max()
        y_dfs.append(prediction_df.loc[pd.IndexSlice[:double_end], training_y_columns])
        # y_dfs.append(prediction_df.loc[prediction_time,training_y_columns])
        future_slice = trigger_df.loc[
                       pd.IndexSlice[double_end: double_end + forecast_trigger_bars * pd.to_timedelta(trigger_tf)], :]
        y_tester_dfs.append(future_slice)
        ys.append(np.array(y_dfs[-1]))
        batch_remained -= 1
    Xs['double'] = np.array(Xs['double'])
    Xs['trigger'] = np.array(Xs['trigger'])
    Xs['pattern'] = np.array(Xs['pattern'])
    Xs['structure'] = np.array(Xs['structure'])
    ys = np.array(ys)
    # assert Xs['double'].shape == (batch_size,) + x_lengths['double']
    # assert Xs['trigger'].shape == (batch_size,) + x_lengths['trigger']
    # assert Xs['pattern'].shape == (batch_size,) + x_lengths['pattern']
    # assert Xs['structure'].shape == (batch_size,) + x_lengths['structure']
    # assert ys.shape == (batch_size, forecast_horizon, 2)

    return Xs, ys, x_dfs, y_dfs, trigger_tf, y_tester_dfs

from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.Config import config
from app.FigurePlotter.plotter import show_and_save_plot
from app.PanderaDFM.MultiTimeframe import MultiTimeframe
from app.ai_modelling.training_data.PreProcessing.encoding.rolling_mean_std import reverse_rolling_mean_std
from app.helper.data_preparation import pattern_timeframe, trigger_timeframe, single_timeframe
from app.helper.helper import profile_it, log_d, date_range
from app.helper.importer import pt
from app.ai_modelling.training_data.profit_loss.profit_loss_adder import add_long_n_short_profit


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


def plot_train_data_of_mt_n_profit(x_dfs: dict[str, pd.DataFrame], y_dfs: pd.DataFrame, predictions_tf: str,
                                   n: int, ):
    # fig = plot_mt_train_n_test(x, y, n, base_ohlcv, show=False)
    # prediction
    fig = go.Figure()
    # reconstructed = reconstruct_from_prediction(
    #     last_x_stats={
    #         'mean_high': x_dfs['double'][n].iloc[-1]['mean_high'],
    #         'mean_low': x_dfs['double'][n].iloc[-1]['mean_low'],
    #         'std_high': x_dfs['double'][n].iloc[-1]['std_high'],
    #         'std_low': x_dfs['double'][n].iloc[-1]['std_low'],
    #         'high': x_dfs['double'][n].iloc[-1]['high'],
    #         'low': x_dfs['double'][n].iloc[-1]['low'],
    #     },
    #     last_x_double_datetime=x_dfs['double'][n].index.get_level_values(level='date').max(),
    #     predictions=predictions[n],
    #     predictions_tf=predictions_tf,
    # )  # last_x_stats, last_x_datetime, predictions: np.array, timeframe,
    # fig.add_trace(go.Candlestick(
    #     x=reconstructed.index.get_level_values('date'),
    #     open=reconstructed['low'],
    #     high=reconstructed['high'],
    #     low=reconstructed['low'],
    #     close=reconstructed['high'],
    #     name='Prediction'
    # ))
    # x structure
    ohlcv = x_dfs['structure'][n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        open=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['high'],
        name='Structure'
    ))
    # x pattern
    ohlcv = x_dfs['pattern'][n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        open=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['high'],
        name='Pattern'
    ))
    # x trigger
    ohlcv = x_dfs['trigger'][n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        open=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['high'],
        name='Trigger'
    ))
    # x double
    ohlcv = x_dfs['double'][n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        open=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['high'],
        name='Double'
    ))
    # y
    ohlcv = y_dfs[n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        close=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        open=ohlcv['high'],
        name='Y'
    ))
    show_and_save_plot(fig.update_yaxes(fixedrange=False))

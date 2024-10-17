from datetime import timedelta

import numpy as np
import pandas as pd

from Config import config
from FigurePlotter.plotter import show_and_save_plot
from PanderaDFM.MultiTimeframe import MultiTimeframe
from PreProcessing.encoding.rolling_mean_std import reverse_rolling_mean_std, \
    read_multi_timeframe_rolling_mean_std_ohlcv
from ai_modelling.cnn_lstm import cnn_lstd_model_input_lengths
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.data_preparation import pattern_timeframe, trigger_timeframe, single_timeframe
from helper.helper import date_range
from helper.importer import pt
import plotly.graph_objects as go


def plot_mt_train_n_test(X, y, n, base_ohlcv):
    reconstructed_double = reverse_rolling_mean_std(X['double'][n])
    reconstructed_trigger = reverse_rolling_mean_std(X['trigger'][n])
    reconstructed_pattern = reverse_rolling_mean_std(X['pattern'][n])
    reconstructed_structure = reverse_rolling_mean_std(X['structure'][n])
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
                                   open=reconstructed_double['open'],
                                   high=reconstructed_double['high'],
                                   low=reconstructed_double['low'],
                                   close=reconstructed_double['close'],
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
        X['double'][n].index.get_level_values(level='date').min(),
        X['trigger'][n].index.get_level_values(level='date').min(),
        X['pattern'][n].index.get_level_values(level='date').min(),
        X['structure'][n].index.get_level_values(level='date').min(),
        y[n].index.get_level_values(level='date').min(),
    )
    max_date = max(
        X['double'][n].index.get_level_values(level='date').max(),
        X['trigger'][n].index.get_level_values(level='date').max(),
        X['pattern'][n].index.get_level_values(level='date').max(),
        X['structure'][n].index.get_level_values(level='date').max(),
        y[n].index.get_level_values(level='date').max()
    )
    ohlcv_slice = base_ohlcv[min_date:max_date]
    ohlcv_trace = go.Candlestick(ohlcv_slice.index.get_level_values(level='date'),
                                 open=ohlcv_slice['open'],
                                 high=ohlcv_slice['high'],
                                 low=ohlcv_slice['low'],
                                 close=ohlcv_slice['close'],
                                 name='ohlcv'
                                 )
    layout = go.Layout(
        title='X Items and Double Slice Over Date Range',
        xaxis=dict(title='Date', range=[min_date, max_date]),
        yaxis=dict(title='Values'),
        showlegend=True
    )

    # Combine all traces into a figure
    fig = go.Figure(data=[double_trace, trigger_trace, pattern_trace, structure_trace, y_trace, ohlcv_trace],
                    layout=layout)

    # Show the plot
    show_and_save_plot(fig)


def mt_train_n_test(structure_tf, mt_any: pt.DataFrame[MultiTimeframe], model_input_lengths: dict,
                    batch_size: int, forecast_horizon: int = 20, ):
    """
    Returns:
        X (np.array): Input features.
        y (np.array): Output targets (high and low).
    """
    pattern_tf = pattern_timeframe(structure_tf)
    trigger_tf = trigger_timeframe(structure_tf)
    double_tf = pattern_timeframe(trigger_timeframe(structure_tf))

    structure_df = single_timeframe(mt_any, structure_tf)
    pattern_df = single_timeframe(mt_any, pattern_tf)
    trigger_df = single_timeframe(mt_any, trigger_tf)
    double_df = single_timeframe(mt_any, double_tf)

    length_of_training = (
            model_input_lengths['structure'] * pd.to_timedelta(structure_tf)
            + model_input_lengths['pattern'] * pd.to_timedelta(pattern_tf)
            + model_input_lengths['trigger'] * pd.to_timedelta(trigger_tf)
            + model_input_lengths['double'] * pd.to_timedelta(double_tf)
    )

    input_start = mt_any.index.get_level_values(
        level='date').min() + length_of_training * 2  # * 2 for simple safeside.
    input_end = mt_any.index.get_level_values(level='date').max() - forecast_horizon * pd.to_timedelta(
        trigger_tf)
    duration_seconds = (input_end - input_start) / timedelta(seconds=1)

    X, y = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, []

    for relative_double_end in np.random.randint(0, duration_seconds, size=batch_size):
        double_end = input_end - relative_double_end * timedelta(seconds=1)
        trigger_end = double_end - model_input_lengths['double'] * pd.to_timedelta(double_tf)
        pattern_end = trigger_end - model_input_lengths['trigger'] * pd.to_timedelta(double_tf)
        structure_end = pattern_end - model_input_lengths['pattern'] * pd.to_timedelta(double_tf)

        double_slice = double_df.loc[pd.IndexSlice[: double_end], :].iloc[-model_input_lengths['double']:]
        trigger_slice = trigger_df.loc[pd.IndexSlice[: trigger_end], :].iloc[-model_input_lengths['trigger']:]
        pattern_slice = pattern_df.loc[pd.IndexSlice[: pattern_end], :].iloc[-model_input_lengths['pattern']:]
        structure_slice = structure_df.loc[pd.IndexSlice[: structure_end], :].iloc[-model_input_lengths['structure']:]

        X['double'].append(double_slice)
        X['trigger'].append(trigger_slice)
        X['pattern'].append(pattern_slice)
        X['structure'].append(structure_slice)

        future_slice = trigger_df.loc[pd.IndexSlice[pattern_end:], :].iloc[:-model_input_lengths['pattern']]
        y.append(future_slice)
    return np.array(X), np.array(y)


# n_mt_ohlcv = pd.read_csv(
#     os.path.join(r"C:\Code\dl-forcasting\data\Kucoin\Spot\BTCUSDT",
#                  f"n_mt_ohlcv.{config.processing_date_range}.csv.zip"), parse_dates=['date'], compression='zip')
# n_mt_ohlcv.set_index(['timeframe', 'date'], inplace=True, drop=True)
# n_mt_ohlcv.dtypes, n_mt_ohlcv.index.dtypes
config.processing_date_range = "24-03-01.00-00T24-09-01.00-00"
t = date_range(config.processing_date_range)
n_mt_ohlcv = read_multi_timeframe_rolling_mean_std_ohlcv(config.processing_date_range)
mt_ohlcv = read_multi_timeframe_ohlcv(config.processing_date_range)
base_ohlcv = single_timeframe(mt_ohlcv, '15min')
X, y = mt_train_n_test('4h', n_mt_ohlcv, cnn_lstd_model_input_lengths, batch_size=10)


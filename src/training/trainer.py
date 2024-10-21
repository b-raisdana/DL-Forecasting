from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from FigurePlotter.plotter import show_and_save_plot
from PanderaDFM.MultiTimeframe import MultiTimeframe
from PreProcessing.encoding.rolling_mean_std import reverse_rolling_mean_std

from src.helper.data_preparation import pattern_timeframe, trigger_timeframe, single_timeframe
from src.helper.importer import pt


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

    # Show the plot
    show_and_save_plot(fig)


def mt_train_n_test(structure_tf, mt_any: pt.DataFrame[MultiTimeframe], model_input_lengths: dict,
                    batch_size: int, forecast_horizon: int = 20, ):
    """
    Returns:
        (X, y)
            - X (dict): A dictionary with keys ('double', 'trigger', 'pattern', 'structure') and corresponding
              values as lists of Pandas DataFrames. Each DataFrame represents a slice of the multi-timeframe
              input features.
                - 'double': Input feature data for the double timeframe.
                - 'trigger': Input feature data for the trigger timeframe.
                - 'pattern': Input feature data for the pattern timeframe.
                - 'structure': Input feature data for the structure timeframe.
            - y (list): A list of Pandas DataFrames representing the forecast targets. Each DataFrame contains
              the predicted values for the future time steps (horizon) from the trigger timeframe.
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

    train_end_safe_start = mt_any.index.get_level_values(
        level='date').min() + length_of_training * 2  # * 2 for simple safe side.
    train_end_safe_end = \
        mt_any.index.get_level_values(level='date').max() - forecast_horizon * pd.to_timedelta(trigger_tf)
    duration_seconds = (train_end_safe_end - train_end_safe_start) / timedelta(seconds=1)
    if duration_seconds <=0 :
        raise RuntimeError(f"Extend date boundary +{duration_seconds+1}s to make possible range of end dates positive!")
    X_df, y_df = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, []
    X, y = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, []

    for relative_double_end in np.random.randint(0, duration_seconds, size=batch_size):
        double_end: datetime = train_end_safe_end - relative_double_end * timedelta(seconds=1)
        trigger_end = double_end - model_input_lengths['double'] * pd.to_timedelta(double_tf)
        pattern_end = trigger_end - model_input_lengths['trigger'] * pd.to_timedelta(trigger_tf)
        structure_end = pattern_end - model_input_lengths['pattern'] * pd.to_timedelta(pattern_tf)

        double_slice = double_df.loc[pd.IndexSlice[: double_end], :].iloc[-model_input_lengths['double']:]
        trigger_slice = trigger_df.loc[pd.IndexSlice[: trigger_end], :].iloc[-model_input_lengths['trigger']:]
        pattern_slice = pattern_df.loc[pd.IndexSlice[: pattern_end], :].iloc[-model_input_lengths['pattern']:]
        structure_slice = structure_df.loc[pd.IndexSlice[: structure_end], :].iloc[-model_input_lengths['structure']:]

        X_df['double'].append((double_slice))
        X_df['trigger'].append((trigger_slice))
        X_df['pattern'].append((pattern_slice))
        X_df['structure'].append((structure_slice))
        X['double'].append(np.array(double_slice))
        X['trigger'].append(np.array(trigger_slice))
        X['pattern'].append(np.array(pattern_slice))
        X['structure'].append(np.array(structure_slice))

        future_slice = trigger_df.loc[pd.IndexSlice[double_end:], :].iloc[:forecast_horizon]
        y_df.append((future_slice))
        y.append(np.array(future_slice))
        # concated_X_df = pd.concat([double_slice, trigger_slice, pattern_slice, structure_slice])
        # concated_X_df.to_csv(os.path.join(data_path(),
        #                                   f'mt_X.{double_end.strftime("%y-%m-%d.%H-%M")}.csv.zip'), compression='zip' )
        # concated_X_df.to_parquet(os.path.join(data_path(),
        #                                   f'mt_X.{double_end.strftime("%y-%m-%d.%H-%M")}.parquet'))
        # future_slice.to_csv(os.path.join(data_path(),
        #                                   f'mt_y.{double_end.strftime("%y-%m-%d.%H-%M")}.csv.zip'), compression='zip' )
        # future_slice.to_parquet(os.path.join(data_path(),
        #                                   f'mt_y.{double_end.strftime("%y-%m-%d.%H-%M")}.parquet'))
    return X, y, X_df, y_df





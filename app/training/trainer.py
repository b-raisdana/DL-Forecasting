from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Config import app_config
from FigurePlotter.plotter import show_and_save_plot
from PanderaDFM.MultiTimeframe import MultiTimeframe
from PreProcessing.encoding.rolling_mean_std import reverse_rolling_mean_std
from helper.data_preparation import pattern_timeframe, trigger_timeframe, single_timeframe
from helper.functions import profile_it, log_d, date_range
from helper.importer import pt


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
def mt_train_n_test(structure_tf, mt_any: pt.DataFrame[MultiTimeframe], x_shape: dict,
                    batch_size: int, forecast_horizon: int = 20, ):
    raise NotImplementedError

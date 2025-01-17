from typing import List

import pandas as pd
import plotly.graph_objects as go

from app.FigurePlotter.plotter import show_and_save_plot
from app.ai_modelling.training_data.PreProcessing.encoding.rolling_mean_std import reverse_rolling_mean_std


def plot_train_data_of_mt_n_profit(x_dfs: dict[str, List[pd.DataFrame]], y_dfs: List[pd.DataFrame],
                                   y_tester_dfs: List[pd.DataFrame], n: int, ):
    training_y_columns = ['long_signal', 'short_signal', 'min_low', 'max_high', 'long_profit', 'short_profit',
                          'long_risk', 'short_risk']
    fig = go.Figure()
    ohlcv_slices = [
        ('structure', 'Structure'),
        ('pattern', 'Pattern'),
        ('trigger', 'Trigger'),
        ('double', 'Double')
    ]
    for key, name in ohlcv_slices:
        ohlcv = x_dfs[key][n]
        fig.add_trace(go.Candlestick(
            x=ohlcv.index.get_level_values('date'),
            open=ohlcv['low'],
            high=ohlcv['high'],
            low=ohlcv['low'],
            close=ohlcv['high'],
            name=name
        ))
    ohlcv = y_tester_dfs[n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        close=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        open=ohlcv['high'],
        name='Y'
    ))
    predictions = y_dfs[n][training_y_columns].to_dict()
    fig.add_annotation(
        x=0, y=1, text=', '.join([f"{col}: {val:.2f}" for col, val in predictions.items()]),
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left",
        bgcolor="white",
        opacity=0.7,
        xref="paper",  # Use the "paper" reference to place it relative to the figure
        yref="paper",  # Use the "paper" reference to place it relative to the figure
        borderpad=10  # Add some padding for the border
    )
    show_and_save_plot(fig.update_yaxes(fixedrange=False))
# def plot_mt_train_n_test(x, y, n, base_ohlcv, show=True):
#     reconstructed_double = reverse_rolling_mean_std(x['double'][n])
#     reconstructed_trigger = reverse_rolling_mean_std(x['trigger'][n])
#     reconstructed_pattern = reverse_rolling_mean_std(x['pattern'][n])
#     reconstructed_structure = reverse_rolling_mean_std(x['structure'][n])
#     # Create traces for each slice
#     double_trace = go.Candlestick(x=reconstructed_double.index.get_level_values(level='date'),
#                                   open=reconstructed_double['open'],
#                                   high=reconstructed_double['high'],
#                                   low=reconstructed_double['low'],
#                                   close=reconstructed_double['close'],
#                                   name='Double')
#     trigger_trace = go.Candlestick(x=reconstructed_trigger.index.get_level_values(level='date'),
#                                    open=reconstructed_trigger['open'],
#                                    high=reconstructed_trigger['high'],
#                                    low=reconstructed_trigger['low'],
#                                    close=reconstructed_trigger['close'],
#                                    name='Trigger')
#     pattern_trace = go.Candlestick(x=reconstructed_pattern.index.get_level_values(level='date'),
#                                    open=reconstructed_pattern['open'],
#                                    high=reconstructed_pattern['high'],
#                                    low=reconstructed_pattern['low'],
#                                    close=reconstructed_pattern['close'],
#                                    name='Pattern')
#     structure_trace = go.Candlestick(x=reconstructed_structure.index.get_level_values(level='date'),
#                                      open=reconstructed_structure['open'],
#                                      high=reconstructed_structure['high'],
#                                      low=reconstructed_structure['low'],
#                                      close=reconstructed_structure['close'],
#                                      name='Structure')
#     y_trace = go.Candlestick(x=y[n].index.get_level_values(level='date'),
#                              open=y[n]['open'],
#                              high=y[n]['high'],
#                              low=y[n]['low'],
#                              close=y[n]['close'],
#                              name='y')
#     min_date = min(
#         x['double'][n].index.get_level_values(level='date').min(),
#         x['trigger'][n].index.get_level_values(level='date').min(),
#         x['pattern'][n].index.get_level_values(level='date').min(),
#         x['structure'][n].index.get_level_values(level='date').min(),
#         y[n].index.get_level_values(level='date').min(),
#     )
#     max_date = max(
#         x['double'][n].index.get_level_values(level='date').max(),
#         x['trigger'][n].index.get_level_values(level='date').max(),
#         x['pattern'][n].index.get_level_values(level='date').max(),
#         x['structure'][n].index.get_level_values(level='date').max(),
#         y[n].index.get_level_values(level='date').max()
#     )
#     ohlcv_slice = base_ohlcv[min_date:max_date]
#     ohlcv_trace = go.Candlestick(x=ohlcv_slice.index.get_level_values(level='date'),
#                                  open=ohlcv_slice['open'],
#                                  high=ohlcv_slice['high'],
#                                  low=ohlcv_slice['low'],
#                                  close=ohlcv_slice['close'],
#                                  name='ohlcv'
#                                  )
#     layout = go.Layout(
#         title='X Items and Double Slice Over Date Range',
#         xaxis=dict(title='Date', range=[min_date, max_date]),
#         yaxis=dict(title='Values', fixedrange=False, ),
#         showlegend=True
#     )
#
#     # Combine all traces into a figure
#     fig = go.Figure(data=[double_trace, trigger_trace, pattern_trace, structure_trace, y_trace, ohlcv_trace],
#                     layout=layout)
#
#     if show:
#         show_and_save_plot(fig)
#     else:
#         return fig

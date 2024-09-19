import pandas as pd

from Config import config, TREND
from helper.data_preparation import single_timeframe
from FigurePlotter.plotter import save_figure, file_id, plot_multiple_figures
from Model.TechnicalAnalysis.PeakValley import peaks_only, valleys_only, major_timeframe
from FigurePlotter.PeakValley_plotter import plot_peaks_n_valleys
from helper.helper import measure_time


@measure_time
def plot_single_timeframe_candle_trend(ohlcv: pd.DataFrame, single_timeframe_candle_trend: pd.DataFrame,
                                       single_timeframe_peaks_n_valleys: pd.DataFrame, show=True, save=True,
                                       path_of_plot=config.path_of_plots, name='Single Timeframe Candle Trend'):
    """
    Plot candlesticks with highlighted trends (Bullish, Bearish, Side).

    The function uses the provided DataFrame containing candle trends and highlights the bars of candles based on
    their trend. Bullish candles are displayed with 70% transparent green color, Bearish candles with 70% transparent red,
    and Side candles with 70% transparent grey color.

    Parameters:
        ohlcv (pd.DataFrame): DataFrame containing OHLC data.
        single_timeframe_candle_trend (pd.DataFrame): DataFrame containing candle trend data.
        single_timeframe_peaks_n_valleys (pd.DataFrame): DataFrame containing peaks and valleys data.
        show (bool): If True, the plot will be displayed.
        save (bool): If True, the plot will be saved as an HTML file.
        path_of_plot (str): Path to save the plot.
        name (str): The title of the figure.

    Returns:
        plgo.Figure: The Plotly figure object containing the plot with highlighted trends.
    """
    # Calculate the trend colors
    trend_colors = single_timeframe_candle_trend['bull_bear_side'].map({
        TREND.BULLISH.value: 'rgba(0, 128, 0, 0.7)',  # 70% transparent green for Bullish trend
        TREND.BEARISH.value: 'rgba(255, 0, 0, 0.7)',  # 70% transparent red for Bearish trend
        TREND.SIDE.value: 'rgba(128, 128, 128, 0.7)'  # 70% transparent grey for Side trend
    })

    # Create the figure using plot_peaks_n_valleys function
    fig = plot_peaks_n_valleys(ohlcv,
                               peaks=peaks_only(single_timeframe_peaks_n_valleys),
                               valleys=valleys_only(single_timeframe_peaks_n_valleys),
                               name=f'{name} Peaks n Valleys')

    # Update the bar trace with trend colors
    fig.update_traces(marker=dict(color=trend_colors), selector=dict(type='bar'))

    # Set the title of the figure
    fig.update_layout(title_text=name)

    # Show the figure or write it to an HTML file
    if save: save_figure(fig, name, file_id(ohlcv))
    if show: fig.show()

    return fig


@measure_time
def plot_multi_timeframe_candle_trend(multi_timeframe_candle_trend, multi_timeframe_peaks_n_valleys, ohlcv, show=True,
                                      save=True, path_of_plot=config.path_of_plots):
    figures = []
    _multi_timeframe_peaks = peaks_only(multi_timeframe_peaks_n_valleys)
    _multi_timeframe_valleys = valleys_only(multi_timeframe_peaks_n_valleys)
    for _, timeframe in enumerate(config.timeframes):
        figures.append(
            plot_single_timeframe_candle_trend(ohlcv, single_timeframe(multi_timeframe_candle_trend, timeframe),
                                               major_timeframe(multi_timeframe_peaks_n_valleys, timeframe),
                                               show=True,
                                               save=True,
                                               path_of_plot=path_of_plot, name=f'{timeframe} Candle Trend'))
    plot_multiple_figures(figures, name='multi_timeframe_candle_trend', show=show, save=save,
                          path_of_plot=path_of_plot)

from pathlib import Path

import pandas as pd
from config import app_config
from archive_not_used_trash.config.Config import TREND
from archive_not_used_trash.domain.price_action.PeakValley import major_timeframe, peaks_only, valleys_only
from helper.data_preparation import single_timeframe
from helper.logging import profile_it
from infrastructure.datastore_engine.disk_cache import symbol_data_path
from archive_not_used_trash.market_structure.PeakValley_plotter import plot_peaks_n_valleys
from archive_not_used_trash.presentation.shared.plotter import file_id, plot_multiple_figures, save_figure


@profile_it
def plot_single_timeframe_candle_trend(
    ohlcv: pd.DataFrame,
    single_timeframe_candle_trend: pd.DataFrame,
    single_timeframe_peaks_n_valleys: pd.DataFrame,
    show=True,
    save=True,
    name="Single Timeframe Candle Trend",
):
    """
    Plot candlesticks with highlighted trends (Bullish, Bearish, Side).

    It highlights bars by their candle trends: bullish green, bearish red, and side grey.

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
    trend_colors = single_timeframe_candle_trend["bull_bear_side"].map(
        {
            TREND.BULLISH.value: "rgba(0, 128, 0, 0.7)",  # 70% transparent green for Bullish trend
            TREND.BEARISH.value: "rgba(255, 0, 0, 0.7)",  # 70% transparent red for Bearish trend
            TREND.SIDE.value: "rgba(128, 128, 128, 0.7)",  # 70% transparent grey for Side trend
        }
    )

    # Create the figure using plot_peaks_n_valleys function
    fig = plot_peaks_n_valleys(
        ohlcv,
        peaks=peaks_only(single_timeframe_peaks_n_valleys),
        valleys=valleys_only(single_timeframe_peaks_n_valleys),
        name=f"{name} Peaks n Valleys",
    )

    # Update the bar trace with trend colors
    fig.update_traces(marker={"color": trend_colors}, selector={"type": "bar"})

    # Set the title of the figure
    fig.update_layout(title_text=name)

    # Show the figure or write it to an HTML file
    if save:
        save_figure(fig, name, file_id(ohlcv))
    if show:
        fig.show()

    return fig


@profile_it
def plot_multi_timeframe_candle_trend(
    multi_timeframe_candle_trend, multi_timeframe_peaks_n_valleys, ohlcv, show=True, save=True, path_of_plot=None
):
    if path_of_plot is None:
        # symbol_data_path() is exchange/market/symbol-shaped; plots are namespaced by symbol alone
        # (see presentation.shared.plotter._per_symbol_plot_dir — same fix, not shared across modules
        # to avoid importing a private helper).
        plot_dir = Path(app_config.path_of_plots) / Path(symbol_data_path()).name
        plot_dir.mkdir(parents=True, exist_ok=True)
        path_of_plot = str(plot_dir)

    figures = []
    _multi_timeframe_peaks = peaks_only(multi_timeframe_peaks_n_valleys)
    _multi_timeframe_valleys = valleys_only(multi_timeframe_peaks_n_valleys)
    for _, timeframe in enumerate(app_config.timeframes):
        figures.append(
            plot_single_timeframe_candle_trend(
                ohlcv,
                single_timeframe(multi_timeframe_candle_trend, timeframe),
                major_timeframe(multi_timeframe_peaks_n_valleys, timeframe),
                show=True,
                save=True,
                path_of_plot=path_of_plot,
                name=f"{timeframe} Candle Trend",
            )
        )
    plot_multiple_figures(figures, name="multi_timeframe_candle_trend", show=show, save=save, path_of_plot=path_of_plot)

import pandas as pd
from pandera import typing as pt
from plotly import graph_objects as plgo

from app.Config import app_config
from app.helper.data_preparation import single_timeframe, df_timedelta_to_str
from app.FigurePlotter.OHLVC_plotter import plot_ohlcva
from app.FigurePlotter.plotter import plot_multiple_figures, file_id, timeframe_color, save_figure, update_figure_layout
from app.PanderaDFM.PeakValley import MultiTimeframePeakValley
from app.Model.TechnicalAnalysis.PeakValley import peaks_only, valleys_only, major_timeframe
from app.data_processing.atr import read_multi_timeframe_ohlcva
from app.helper.helper import profile_it


@profile_it
def plot_multi_timeframe_peaks_n_valleys(multi_timeframe_peaks_n_valleys: pt.DataFrame[MultiTimeframePeakValley],
                                         date_range_str: str, show=True, save=True):
    multi_timeframe_ohlcva = read_multi_timeframe_ohlcva(date_range_str)

    figures = []
    _multi_timeframe_peaks = peaks_only(multi_timeframe_peaks_n_valleys)
    _multi_timeframe_valleys = valleys_only(multi_timeframe_peaks_n_valleys)
    for _, timeframe in enumerate(app_config.timeframes):
        figures.append(plot_peaks_n_valleys(single_timeframe(multi_timeframe_ohlcva, timeframe),
                                            peaks=major_timeframe(_multi_timeframe_peaks, timeframe),
                                            valleys=major_timeframe(_multi_timeframe_valleys, timeframe),
                                            name=f'{timeframe} Peaks n Valleys', show=False, save=False))
    fig = plot_multiple_figures(figures, name=f'multi_timeframe_peaks_n_valleys.{file_id(multi_timeframe_ohlcva)}',
                                show=show, save=save)
    return fig


@profile_it
def plot_peaks_n_valleys(ohlcva: pd = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'atr']),
                         peaks: pd = pd.DataFrame(columns=['high', 'timeframe']),
                         valleys: pd = pd.DataFrame(columns=['low', 'timeframe']),
                         name: str = '', show: bool = True, save: bool = True) -> plgo.Figure:
    """
        Plot candlesticks with highlighted peaks and valleys.

        Parameters:
            ohlcva (pd.DataFrame): DataFrame containing OHLC data plus atr.
            peaks (pd.DataFrame): DataFrame containing peaks data.
            valleys (pd.DataFrame): DataFrame containing valleys data.
            name (str): The name of the plot.
            show (bool): Whether to show
            save (bool): Whether to save

        Returns:
            plgo.Figure: The Plotly figure object containing the candlestick plot with peaks and valleys highlighted.
        """
    fig = plot_ohlcva(ohlcva, name=name, save=False, show=False)
    if len(peaks) > 0:
        for timeframe in app_config.timeframes:
            _indexes, _labels = [], []
            timeframe_peaks = single_timeframe(peaks, timeframe)
            [(_indexes.append(_x), _labels.append(
                f"{timeframe}({df_timedelta_to_str(row['strength'])}@{_x.strftime('%m/%d %H:%M')})={int(row['high'])}"))
             for _x, row in timeframe_peaks.iterrows()]
            fig.add_scatter(x=_indexes, y=timeframe_peaks['high'] + 1, mode="markers", name=f'P{timeframe}',
                            marker=dict(symbol="triangle-up", color=timeframe_color(timeframe)),
                            hovertemplate="%{text}", text=_labels)
    if len(valleys) > 0:
        for timeframe in app_config.timeframes:
            timeframe_valleys = single_timeframe(valleys, timeframe)
            _indexes, _labels = [], []
            [(_indexes.append(_x), _labels.append(
                f"{timeframe}({df_timedelta_to_str(row['strength'])}@{_x.strftime('%m/%d %H:%M')})={int(row['low'])}"))
             for _x, row in timeframe_valleys.iterrows()]
            fig.add_scatter(x=_indexes, y=timeframe_valleys['low'] - 1, mode="markers", name=f'V{timeframe}',
                            legendgroup=timeframe,
                            marker=dict(symbol="triangle-down", color=timeframe_color(timeframe)),
                            hovertemplate="%{text}", text=_labels)
        fig.update_layout(hovermode='x unified')
    # fig.update_layout(title_text=name)
    update_figure_layout(fig)
    if show: fig.show()
    if save:
        save_figure(fig, f'peaks_n_valleys.{file_id(ohlcva, name)}', )
    return fig




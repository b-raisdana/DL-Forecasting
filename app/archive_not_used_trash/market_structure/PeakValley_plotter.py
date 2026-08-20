import pandas as pd
from config import app_config
from archive_not_used_trash.domain.price_action.PeakValley import major_timeframe, peaks_only, valleys_only
from archive_not_used_trash.domain.schemas.market_structure.PeakValley import MultiTimeframePeakValley
from helper.data_preparation import single_timeframe
from archive_not_used_trash.helper.data_preparation import df_timedelta_to_str
from helper.logging import profile_it
from infrastructure.ohlcv.ohlcva import read_multi_timeframe_ohlcva
from pandera import typing as pt
from plotly import graph_objects as plgo
from archive_not_used_trash.ohlcv.OHLVC_plotter import plot_ohlcva
from archive_not_used_trash.presentation.shared.plotter import file_id, plot_multiple_figures, save_figure, timeframe_color, update_figure_layout


@profile_it
def plot_multi_timeframe_peaks_n_valleys(
    multi_timeframe_peaks_n_valleys: pt.DataFrame[MultiTimeframePeakValley], date_range_str: str, show=True, save=True
):
    multi_timeframe_ohlcva = read_multi_timeframe_ohlcva(date_range_str)

    figures = []
    _multi_timeframe_peaks = peaks_only(multi_timeframe_peaks_n_valleys)
    _multi_timeframe_valleys = valleys_only(multi_timeframe_peaks_n_valleys)
    for _, timeframe in enumerate(app_config.timeframes):
        figures.append(
            plot_peaks_n_valleys(
                single_timeframe(multi_timeframe_ohlcva, timeframe),
                peaks=major_timeframe(_multi_timeframe_peaks, timeframe),
                valleys=major_timeframe(_multi_timeframe_valleys, timeframe),
                name=f"{timeframe} Peaks n Valleys",
                show=False,
                save=False,
            )
        )
    fig = plot_multiple_figures(
        figures, name=f"multi_timeframe_peaks_n_valleys.{file_id(multi_timeframe_ohlcva)}", show=show, save=save
    )
    return fig


@profile_it
def plot_peaks_n_valleys(
    ohlcva: pd.DataFrame | None = None,
    peaks: pd.DataFrame | None = None,
    valleys: pd.DataFrame | None = None,
    name: str = "",
    show: bool = True,
    save: bool = True,
) -> plgo.Figure:
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
    ohlcva = pd.DataFrame(columns=["open", "high", "low", "close", "atr"]) if ohlcva is None else ohlcva
    peaks = pd.DataFrame(columns=["high", "timeframe"]) if peaks is None else peaks
    valleys = pd.DataFrame(columns=["low", "timeframe"]) if valleys is None else valleys
    fig = plot_ohlcva(ohlcva, name=name, save=False, show=False)
    if len(peaks) > 0:
        for timeframe in app_config.timeframes:
            _indexes, _labels = [], []
            timeframe_peaks = single_timeframe(peaks, timeframe)
            [
                (
                    _indexes.append(_x),
                    _labels.append(
                        f"{timeframe}({df_timedelta_to_str(row['strength'])}@{_x.strftime('%m/%d %H:%M')})="
                        f"{int(row['high'])}"
                    ),
                )
                for _x, row in timeframe_peaks.iterrows()
            ]
            fig.add_scatter(
                x=_indexes,
                y=timeframe_peaks["high"] + 1,
                mode="markers",
                name=f"P{timeframe}",
                marker={"symbol": "triangle-up", "color": timeframe_color(timeframe)},
                hovertemplate="%{text}",
                text=_labels,
            )
    if len(valleys) > 0:
        for timeframe in app_config.timeframes:
            timeframe_valleys = single_timeframe(valleys, timeframe)
            _indexes, _labels = [], []
            [
                (
                    _indexes.append(_x),
                    _labels.append(
                        f"{timeframe}({df_timedelta_to_str(row['strength'])}@{_x.strftime('%m/%d %H:%M')})="
                        f"{int(row['low'])}"
                    ),
                )
                for _x, row in timeframe_valleys.iterrows()
            ]
            fig.add_scatter(
                x=_indexes,
                y=timeframe_valleys["low"] - 1,
                mode="markers",
                name=f"V{timeframe}",
                legendgroup=timeframe,
                marker={"symbol": "triangle-down", "color": timeframe_color(timeframe)},
                hovertemplate="%{text}",
                text=_labels,
            )
        fig.update_layout(hovermode="x unified")
    # fig.update_layout(title_text=name)
    update_figure_layout(fig)
    if show:
        fig.show()
    if save:
        save_figure(
            fig,
            f"peaks_n_valleys.{file_id(ohlcva, name)}",
        )
    return fig

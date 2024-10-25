from typing import List

import pandas as pd
from pandera import typing as pt
from plotly import graph_objects as plgo

from Model.TechnicalAnalysis import PeakValley
from Model.TechnicalAnalysis.BullBearSide import most_two_significant_tops
from Config import TREND, config, TopTYPE
from FigurePlotter.PeakValley_plotter import plot_peaks_n_valleys
from FigurePlotter.plotter import file_id, save_figure, plot_multiple_figures, update_figure_layout
from PanderaDFM.BullBearSide import BullBearSide, bull_bear_side_repr
from PanderaDFM.OHLCV import OHLCV
from Model.TechnicalAnalysis.PeakValley import peaks_only, valleys_only, major_timeframe
from helper.data_preparation import single_timeframe
from helper.helper import profile_it, log, log_w

MAX_NUMBER_OF_PLOT_SCATTERS = 5000


def plot_single_timeframe_bull_bear_side_trends(single_timeframe_ohlcva: pt.DataFrame[OHLCV],
                                                peaks_n_valleys: pt.DataFrame[PeakValley],
                                                boundaries: pt.DataFrame[BullBearSide],
                                                name: str = '', show: bool = True,
                                                html_path: str = '', save: bool = True) -> plgo.Figure:
    fig = plot_peaks_n_valleys(single_timeframe_ohlcva, peaks=peaks_only(peaks_n_valleys),
                               valleys=valleys_only(peaks_n_valleys), name=name, show=False, save=False)
    remained_number_of_scatters = MAX_NUMBER_OF_PLOT_SCATTERS
    for _start, _trend in boundaries.iterrows():
        trend_peaks_n_valleys = peaks_valleys_in_range(peaks_n_valleys, _start, _trend['end'])
        # included_candles = single_timeframe_ohlcva.loc[_start: _trend['end']].index
        trend_peaks = peaks_only(trend_peaks_n_valleys)['high'].reset_index(level='timeframe').sort_index(level='date')
        trend_valleys = valleys_only(trend_peaks_n_valleys)['low'].reset_index(level='timeframe') \
            .sort_index(level='date', ascending=False)
        fill_color = 'green' if _trend['bull_bear_side'] == TREND.BULLISH.value else \
            'red' if _trend['bull_bear_side'] == TREND.BEARISH.value else 'gray'
        text = bull_bear_side_repr(_start, _trend)
        name = (f'{_trend["bull_bear_side"].replace("_TREND", "")}: '
                f'{_start.strftime("%H:%M")}-{_trend["end"].strftime("%H:%M")}')
        legend_group = (f'{_trend["bull_bear_side"].replace("_TREND", "")}: '
                        f'{_start.strftime("%H:%M")}-{_trend["end"].strftime("%H:%M")}')
        if remained_number_of_scatters > 0:
            # draw boundary
            xs = [_start] + trend_peaks.index.get_level_values('date').tolist() + \
                 [_trend['end']] + trend_valleys.index.get_level_values('date').tolist()
            ys = [single_timeframe_ohlcva.loc[_start, 'open']] + trend_peaks['high'].tolist() + \
                 [single_timeframe_ohlcva.loc[_trend['end'], 'close']] + trend_valleys['low'].tolist()
            fig.add_scatter(x=xs, y=ys, fill="toself", fillpattern=dict(fgopacity=0.5, shape='.'), name=name,
                            hovertext= [text] * len(xs),
                            line=dict(color=fill_color, width=0), mode='lines',
                            legendgroup=legend_group,
                            hoverinfo='text', )
            # adds hover info to all candles
            # fig.add_scatter(x=included_candles, y=single_timeframe_ohlcva.loc[included_candles, 'open'], name=name,
            #                 mode='none', showlegend=False, text=text, legendgroup=legend_group, hoverinfo='text')
            # adds movement line
            if 'movement_start_value' in boundaries.columns:
                fig.add_scatter(x=[_trend['movement_start_time'], _trend['movement_end_time']],
                                y=[_trend['movement_start_value'], _trend['movement_end_value']],
                                name=name, line=dict(color=fill_color), legendgroup=legend_group,  text=text,
                                hoverinfo='text', showlegend=False)
            else:
                log(f'movement not found in boundaries:{boundaries.columns}', stack_trace=False)
            remained_number_of_scatters -= 2
        else:
            break
    update_figure_layout(fig)
    if save or html_path != '':
        file_name = f'single_timeframe_bull_bear_side_trends.{file_id(single_timeframe_ohlcva, name)}'
        save_figure(fig, file_name, html_path)

    if show: fig.show()
    return fig


@profile_it
def plot_multi_timeframe_bull_bear_side_trends(multi_timeframe_ohlcva, multi_timeframe_peaks_n_valleys,
                                               _multi_timeframe_bull_bear_side_trends, show: bool = True,
                                               save: bool = True,
                                               timeframe_shortlist: List['str'] = None):
    figures = []
    if timeframe_shortlist is None:
        timeframe_shortlist = config.timeframes
    for timeframe in timeframe_shortlist:
        _figure = plot_single_timeframe_bull_bear_side_trends(
            single_timeframe_ohlcva=single_timeframe(multi_timeframe_ohlcva, timeframe),
            peaks_n_valleys=major_timeframe(multi_timeframe_peaks_n_valleys, timeframe).sort_index(level='date'),
            boundaries=single_timeframe(_multi_timeframe_bull_bear_side_trends, timeframe).sort_index(level='date'),
            show=False, save=False,
            name=f'{timeframe} boundaries')
        figures.append(_figure)
    plot_multiple_figures(figures, name=f'multi_timeframe_bull_bear_side_trends.'
                                        f'{multi_timeframe_ohlcva.index[0][1].strftime("%y-%m-%d.%H-%M")}T'
                                        f'{multi_timeframe_ohlcva.index[-1][1].strftime("%y-%m-%d.%H-%M")}',
                          show=show, save=save)


def peaks_valleys_in_range(peaks_n_valleys: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    return peaks_n_valleys.loc[(peaks_n_valleys.index.get_level_values('date') >= start) &
                               (peaks_n_valleys.index.get_level_values('date') <= end)]


def add_canal_lines(_boundaries, single_timeframe_peaks_n_valleys):
    log_w("Not tested")
    if _boundaries is None:
        return _boundaries
    for i, _boundary in _boundaries.iterrows():
        if _boundary['bull_bear_side'] == TREND.SIDE.value:
            continue
        _peaks = most_two_significant_tops(i, _boundary['end'], single_timeframe_peaks_n_valleys, TopTYPE.PEAK)
        _valleys = most_two_significant_tops(i, _boundary['end'], single_timeframe_peaks_n_valleys, TopTYPE.VALLEY)
        if _boundary['bull_bear_side'] == TREND.BULLISH.value:
            canal_top = _peaks.iloc[0]['high']
            trend_tops = _valleys['low'].to_numpy()
        else:
            canal_top = _valleys.iloc[0]['low']
            trend_tops = _peaks['high'].to_numpy()
        trend_acceleration = (trend_tops[1][1] - trend_tops[0][1]) / (trend_tops[1][0] - trend_tops[0][0])
        trend_base = trend_tops[0][1] - trend_tops[0][0] * trend_acceleration
        canal_acceleration = trend_acceleration
        canal_base = canal_top[0][1] - canal_top[0][0] * canal_acceleration
        _boundaries.iloc[i, ['trend_acceleration', 'trend_base', 'canal_acceleration', 'canal_base']] = \
            [trend_acceleration, trend_base, canal_acceleration, canal_base]
    return _boundaries

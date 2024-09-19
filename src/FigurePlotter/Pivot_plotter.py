from typing import Literal

import pandas as pd
from pandera import typing as pt
from plotly import graph_objects as plgo

from Config import config
from FigurePlotter.BasePattern_plotter import draw_base
from FigurePlotter.OHLVC_plotter import plot_ohlcv, add_atr_scatter
from FigurePlotter.plotter import show_and_save_plot, update_figure_layout
from PanderaDFM.BasePattern import MultiTimeframeBasePattern
from PanderaDFM.OHLCVA import MultiTimeframeOHLCVA
from PanderaDFM.Pivot2 import Pivot2DFM, MultiTimeframePivot2DFM
from Model.TechnicalAnalysis.atr import read_multi_timeframe_ohlcva
from helper.data_preparation import single_timeframe


# @measure_time
def plot_multi_timeframe_pivots(mt_pivots: pt.DataFrame[MultiTimeframePivot2DFM],
                                group_by: Literal['timeframe', 'original_start'],
                                multi_timeframe_ohlcva: pt.DataFrame[MultiTimeframeOHLCVA] = None,
                                show_boundaries:bool = True, show_duplicates: bool = True, show_ftc: bool = True,
                                date_range_str: str = None, show: bool = True, save: bool = True) -> plgo.Figure:
    if date_range_str is None:
        date_range_str = config.processing_date_range
    # Create the figure using plot_peaks_n_valleys function
    if multi_timeframe_ohlcva is None:
        multi_timeframe_ohlcva = read_multi_timeframe_ohlcva(date_range_str)
    end_time = max(multi_timeframe_ohlcva.index.get_level_values('date'))
    base_ohlcv = single_timeframe(multi_timeframe_ohlcva, config.timeframes[0])

    # plot multiple timeframes candle chart all together.
    resistance_timeframes = mt_pivots[mt_pivots['is_resistance']] \
        .index.get_level_values(level='timeframe').unique()
    support_timeframes = mt_pivots[~mt_pivots['is_resistance']] \
        .index.get_level_values(level='timeframe').unique()
    fig = plot_ohlcv(ohlcv=base_ohlcv, show=False, save=False, name=config.timeframes[0])
    for timeframe in config.timeframes[1:]:
        ohlcva = single_timeframe(multi_timeframe_ohlcva, timeframe)
        fig.add_trace(plgo.Candlestick(x=ohlcva.index,
                                       open=ohlcva['open'],
                                       high=ohlcva['high'],
                                       low=ohlcva['low'],
                                       close=ohlcva['close'], name=timeframe, legendgroup=timeframe))
        midpoints = (ohlcva['high'] + ohlcva['low']) / 2
        # if show_boundaries:
        # Add the atr boundaries
        fig = add_atr_scatter(fig, ohlcva.index, midpoints=midpoints,
                              widths=ohlcva['atr'], show_hover=True,
                              name=timeframe, legendgroup=timeframe, showlegend=False)
        if timeframe in resistance_timeframes:
            legend_group = f"Piv{timeframe}R"
            fig.add_scatter(x=[base_ohlcv.index[0]], y=[base_ohlcv['open']],
                            name=legend_group, line=dict(color='red', width=0), mode='lines',
                            legendgroup=legend_group, showlegend=True, hoverinfo='none', )
        if timeframe in support_timeframes:
            legend_group = f"Piv{timeframe}S"
            fig.add_scatter(x=[base_ohlcv.index[0]], y=[base_ohlcv['open']],
                            name=legend_group, line=dict(color='red', width=0), mode='lines',
                            legendgroup=legend_group, showlegend=True, hoverinfo='none', )

    mt_pivots = mt_pivots.sort_index(level='date')
    if not show_duplicates:
        mt_pivots = mt_pivots[mt_pivots.index.get_level_values('date') == mt_pivots.index.get_level_values('original_start') ]
    pivots_have_ftc = 'ftc_list' in mt_pivots.columns
    plotted_ftc = []
    plotted_pivots = []
    for (pivot_timeframe, pivot_start, pivot_original_start), pivot_info in mt_pivots.iterrows():
        pivot_name = Pivot2DFM.name(pivot_original_start, pivot_timeframe, pivot_info)
        if pivot_name not in plotted_pivots:
            plotted_pivots.append(pivot_name)
            if group_by == 'timeframe':
                legend_group = f"Piv{pivot_timeframe}R" if pivot_info['is_resistance'] else f"Piv{pivot_timeframe}S"
                # show_legend = False
            elif group_by == 'original_start':
                legend_group = pivot_name
                # show_legend = True
            else:
                raise ValueError(f"Invalid group_by:{group_by}")
            pivot_description = Pivot2DFM.description(pivot_start, pivot_timeframe, pivot_info)
            # add a dotted line from creating time of level to the activation time
            fig.add_scatter(
                x=[pivot_start, pivot_original_start],
                y=[pivot_info['level'], pivot_info['level']],
                name=pivot_name, line=dict(color='blue', dash='dot', width=0.5), mode='lines',  # +text',
                legendgroup=legend_group, showlegend=False, hoverinfo='none',
            )
            # draw the level line
            level_color = 'orange' if pivot_info['is_resistance'] else 'magenta'
            if pd.isnull(pivot_info['ttl']):
                raise Exception(f'Every level should have a ttl.'
                                f'{pivot_timeframe}, {pivot_start}, {pivot_info} ttl is Null')
            level_end_time = min(pivot_info['ttl'], end_time)
            if pd.notna(pivot_info['deactivated_at']):
                level_end_time = min(pivot_info['deactivated_at'], level_end_time)
            fig.add_scatter(
                x=[pivot_start, level_end_time],
                y=[pivot_info['level'], pivot_info['level']],
                text=[pivot_description] * 2,
                name=pivot_name, line=dict(color=level_color, width=0.5), mode='lines',  # +text',
                # legendgroup=legend_group, showlegend=show_legend, hoverinfo='text',
                legendgroup=legend_group, showlegend=True, hoverinfo='text',
            )
            if show_boundaries:
                # draw the level boundary
                boundary_color = 'blue' if pivot_info['is_resistance'] else 'red'
                fig.add_scatter(
                    x=[pivot_start, level_end_time, level_end_time, pivot_start],
                    y=[pivot_info['external_margin'], pivot_info['external_margin'],
                       pivot_info['internal_margin'], pivot_info['internal_margin']],
                    fill="toself", opacity=0.2,
                    text=[pivot_description, None, None, pivot_description],
                    name=pivot_name, line=dict(color=boundary_color, width=0), mode='lines',  # +text',
                    legendgroup=legend_group, showlegend=False, hoverinfo='text',  # text=pivot_description,
                )
            if pivot_start == pivot_original_start:
                # add movement and return paths
                if (hasattr(pivot_info, 'movement_start_time')
                        and hasattr(pivot_info, 'return_end_time')
                        and hasattr(pivot_info, 'movement_start_value')
                        and hasattr(pivot_info, 'return_end_value')
                ):
                    fig.add_scatter(x=[pivot_info['movement_start_time'], pivot_start, ],
                                    y=[pivot_info['movement_start_value'], pivot_info['level'], ],
                                    name=pivot_name, line=dict(color='green', width=0.5), mode='lines',  # +text',
                                    legendgroup=legend_group, showlegend=False, hoverinfo='none',
                                    )
                    fig.add_scatter(x=[pivot_start, pivot_info['return_end_time']],
                                    y=[pivot_info['level'], pivot_info['return_end_value']],
                                    name=pivot_name, line=dict(color='red', width=0.5), mode='lines',  # +text',
                                    legendgroup=legend_group, showlegend=False, hoverinfo='none',
                                    )
                # add real start
                if (hasattr(pivot_info, 'real_start_time') and hasattr(pivot_info, 'real_start_value')):
                    fig.add_scatter(x=[pivot_info['real_start_time'], pivot_start, ],
                                    y=[pivot_info['real_start_value'], pivot_info['level'], ],
                                    name=pivot_name, line=dict(color='gray', width=0.5), mode='lines',  # +text',
                                    legendgroup=legend_group, showlegend=False, hoverinfo='none',
                                    )
                    if show_ftc:
                        # draw pivot FTC base patterns
                        if pivots_have_ftc:
                            if isinstance(pivot_info['ftc_list'], list):
                                for ftc in pivot_info['ftc_list']:
                                    ftc_name = MultiTimeframeBasePattern.str(ftc['date'], ftc['timeframe'], ftc)
                                    if ftc_name not in plotted_ftc:
                                        real_start = ftc['date'] + \
                                                     pd.to_timedelta(
                                                         ftc[
                                                             'timeframe']) * config.base_pattern_index_shift_after_last_candle_in_the_sequence
                                        ftc['effective_end'] = min(
                                            ftc['end'] if pd.notna(ftc['end']) else ftc['ttl'],
                                            ftc['ttl'] if pd.notna(ftc['ttl']) else ftc['end'],
                                            end_time
                                        )
                                        draw_base(ftc, fig, ftc['date'], real_start, ftc['timeframe'], legend_group)
                                        plotted_ftc.append(ftc_name)
    update_figure_layout(fig)
    show_and_save_plot(fig, save, show, name_without_prefix=f'multi_timeframe_classic_pivots.{date_range_str}')
    return fig

from typing import List

import pandas as pd
from pandera import typing as pt

from app.Config import config
from app.FigurePlotter.BasePattern_plotter import draw_band_activators, draw_base, draw_band_orders_df
from app.FigurePlotter.BullBearSide_plotter import plot_single_timeframe_bull_bear_side_trends
from app.FigurePlotter.plotter import plot_multiple_figures
from app.Model.TechnicalAnalysis.PeakValley import major_timeframe
from app.PanderaDFM.BasePattern import MultiTimeframeBasePattern
from app.PanderaDFM.BullBearSide import MultiTimeframeBullBearSide
from app.PanderaDFM.OHLCVA import MultiTimeframeOHLCVA
from app.PanderaDFM.PeakValley import MultiTimeframePeakValley
from app.helper.data_preparation import single_timeframe
from app.helper.helper import profile_it

MAX_NUMBER_OF_PLOT_SCATTERS = 5000


@profile_it
def plot_multi_timeframe_bbs_n_base_pattern(multi_timeframe_base_pattern: pt.DataFrame[MultiTimeframeBasePattern],
                                            multi_timeframe_ohlcva: pt.DataFrame[MultiTimeframeOHLCVA],
                                            multi_timeframe_bull_bear_side_trends: pt.DataFrame[
                                                MultiTimeframeBullBearSide],
                                            multi_timeframe_peaks_n_valleys: pt.DataFrame[MultiTimeframePeakValley],
                                            orders_df: pd.DataFrame = None, timeframe_shortlist: List['str'] = None,
                                            show: bool = True, save: bool = True) \
        -> None:
    multi_timeframe_ohlcva = multi_timeframe_ohlcva.sort_index(level='date')
    end_of_dates = multi_timeframe_ohlcva.index.get_level_values('date').max()
    end_of_prices = multi_timeframe_ohlcva.loc[(config.timeframes[0], end_of_dates)]['close']
    multi_timeframe_base_pattern['effective_end'] = \
        multi_timeframe_base_pattern[['end', 'ttl']].min(axis=1, skipna=True)
    multi_timeframe_base_pattern.loc[multi_timeframe_base_pattern['effective_end'] > end_of_dates, 'effective_end'] = \
        end_of_dates
    if not multi_timeframe_base_pattern['effective_end'].notna().all():
        raise AssertionError("not _multi_timeframe_base_pattern['effective_end'].notna().all()")

    figures = []
    if timeframe_shortlist is None:
        timeframe_shortlist = config.timeframes
    for timeframe in timeframe_shortlist:
        # draw BBS
        fig = plot_single_timeframe_bull_bear_side_trends(
            single_timeframe_ohlcva=single_timeframe(multi_timeframe_ohlcva, timeframe),
            peaks_n_valleys=major_timeframe(multi_timeframe_peaks_n_valleys, timeframe).sort_index(level='date'),
            boundaries=single_timeframe(multi_timeframe_bull_bear_side_trends, timeframe).sort_index(level='date'),
            show=False, save=False,
            name=f'{timeframe} boundaries')
        # draw Base Pattern for lower timeframes.
        lower_timeframe_base_pattern = multi_timeframe_base_pattern[multi_timeframe_base_pattern.index \
            .get_level_values(level='timeframe').isin(config.timeframes[:config.timeframes.index(timeframe)])]
        for (base_timeframe, base_date), base_info in lower_timeframe_base_pattern.iterrows():
            if base_timeframe not in config.timeframes[:-2]:
                raise AssertionError(f'timeframe({timeframe}) not in config.timeframes[:-2]')
            real_start = base_date + \
                         pd.to_timedelta(
                             base_timeframe) * config.base_pattern_index_shift_after_last_candle_in_the_sequence
            legend_group, text = draw_base(base_info=base_info, fig=fig, index_date=base_date, real_start=real_start,
                                           base_timeframe=base_timeframe)
            draw_band_activators(base_info, fig, legend_group, real_start, text)
            if (orders_df is not None) and (not orders_df.empty):
                draw_band_orders_df(orders_df, fig, end_of_dates, end_of_prices,
                                    base_date, base_timeframe, legend_group)

        figures.append(fig)
    plot_multiple_figures(figures, name=f'multi_timeframe_bull_bear_side_trends.'
                                        f'{multi_timeframe_ohlcva.index[0][1].strftime("%y-%m-%d.%H-%M")}T'
                                        f'{multi_timeframe_ohlcva.index[-1][1].strftime("%y-%m-%d.%H-%M")}',
                          show=show, save=save)

# def draw_band_activators(base_pattern, fig, legend_group, start, text):
#     if base_pattern['below_band_activated'] is not None:
#         # add a vertical line equal to atr of BasePattern at the time price chart goes 1 atr under below edge.
#         xs = [start, base_pattern['below_band_activated']]
#         ys = [base_pattern['internal_low'], base_pattern['internal_high'] + base_pattern['atr']]
#         fig.add_scatter(x=xs, y=ys,
#                         name=legend_group,
#                         # text=text,
#                         line=dict(color='yellow', width=1),
#                         mode='lines',  # +text',
#                         showlegend=False,
#                         legendgroup=legend_group,
#                         hoverinfo='none',
#                         # hovertemplate="%{text}",
#                         )
#     if base_pattern['upper_band_activated'] is not None:
#         # add a vertical line equal to atr of BasePattern at the time price chart goes 1 atr above upper edge.
#         xs = [start, base_pattern['upper_band_activated']]
#         ys = [base_pattern['internal_high'], base_pattern['internal_low'] - base_pattern['atr']]
#         fig.add_scatter(x=xs, y=ys,
#                         name=legend_group,
#                         # text=text,
#                         line=dict(color='yellow', width=1),
#                         mode='lines',  # +text',
#                         showlegend=False,
#                         legendgroup=legend_group,
#                         hoverinfo='none',
#                         # hovertemplate="%{text}",
#                         )


# def draw_band_orders_df(orders_df: pd.DataFrame, fig: plotly.graph_objs.Figure,
#                         end_of_dates: datetime, end_of_prices: float,
#                         base_pattern_date: pd.Timestamp, base_pattern_timeframe: str,
#                         legend_group: str):
#     order_group_ids = orders_df.loc[(
#             (orders_df['reference_date'] == str(base_pattern_date))
#             & (orders_df['reference_timeframe'] == base_pattern_timeframe)
#     ), 'info_order_group_id'].unique().tolist()
#     for order_group_id in order_group_ids:
#         original_order, stop_loss_order, take_profit_order = df_triple_orders_of_group(order_group_id, orders_df)
#         if config.check_assertions and len(original_order) == 0 or len(stop_loss_order) == 0 or len(
#                 take_profit_order) == 0:
#             raise AssertionError("original_order.empty or stop_loss_order.empty or take_profit_order.empty")
#         start = datetime.strptime(original_order['date'], "%Y-%m-%d %H:%M:%S%z")
#         started = (original_order['status'] == "Completed")
#         if started:
#             start_price = float(original_order['created_price'])
#             stop_loss_price = float(stop_loss_order['created_price'])
#             take_profit_price = float(take_profit_order['created_price'])
#             end, end_price, ended = order_end_information(end_of_dates, end_of_prices, stop_loss_order,
#                                                           take_profit_order)
#             side = OrderSide.Buy if float(original_order['created_size']) > 0 else OrderSide.Sell
#             if config.check_assertions:
#                 if side == OrderSide.Buy:
#                     if not float(original_order['created_size']) > 0:
#                         raise AssertionError("not original_order['created_size'] > 0")
#                 else:
#                     if not float(original_order['created_size']) < 0:
#                         raise AssertionError("not original_order['created_size'] < 0")
#             draw_base_pattern_orders(fig, legend_group, side, start, end, start_price, end_price, stop_loss_price,
#                                      take_profit_price, float(original_order['created_size']),
#                                      start.strftime("%m-%d %H:%M"), ended, order_group_id)
#     return fig


# def order_end_information(end_of_dates, end_of_prices, stop_loss_order, take_profit_order):
#     if stop_loss_order['status'] == 'Completed':
#         end = datetime.strptime(stop_loss_order['date'], '%Y-%m-%d %H:%M:%S%z')
#         # end = stop_loss_order['date']
#         end_price = float(stop_loss_order['executed_price'])
#         ended = True
#     elif take_profit_order['status'] == 'Completed':
#         end = datetime.strptime(take_profit_order['date'], '%Y-%m-%d %H:%M:%S%z')
#         # end = take_profit_order['date']
#         end_price = float(take_profit_order['executed_price'])
#         ended = True
#     else:
#         end = end_of_dates
#         end_price = end_of_prices
#         ended = False
#     if config.check_assertions and not end_price > 0:
#         raise AssertionError("not end_price > 0")
#     return end, end_price, ended


# def df_triple_orders_of_group(order_group_id, orders_df):
#     original_order: pd.Series = orders_df[
#         (orders_df['info_order_group_id'] == order_group_id)
#         & (orders_df['info_custom_type'] == BracketOrderType.Original.value)
#         ].sort_index(level='date').iloc[-1].to_dict()
#     stop_loss_order: pd.Series = orders_df[
#         (orders_df['info_order_group_id'] == order_group_id)
#         & (orders_df['info_custom_type'] == BracketOrderType.StopLoss.value)
#         ].sort_index(level='date').iloc[-1].to_dict()
#     take_profit_order: pd.Series = orders_df[
#         (orders_df['info_order_group_id'] == order_group_id)
#         & (orders_df['info_custom_type'] == BracketOrderType.TakeProfit.value)
#         ].sort_index(level='date').iloc[-1].to_dict()
#     return original_order, stop_loss_order, take_profit_order
#
#
# def draw_base_pattern_orders(fig, legend_group: str, side: OrderSide, start: datetime, end: datetime,
#                              start_price: float, end_price: float, stop_loss_price: float,
#                              take_profit_price: float, size: float, name: str, ended: bool, order_group_id: int):
#     pnl_value = (end_price - start_price) * size
#     text = ",".join([
#         f"E{end.strftime('%H:%M')}",
#         f"{start_price:.0f}",
#         f"SL{stop_loss_price:.0f}",
#         f"TP{take_profit_price:.0f}",
#         f"PNL{'???' if not ended else ''}{pnl_value:.1f}",  # /{pnl_percent}"
#     ]
#     )
#     if config.check_assertions and abs(pnl_value) < 0.5:
#         raise AssertionError("abs(pnl_value)")
#     # draw stop-loss box
#     xs = [start, end, end, start]
#     ys = [start_price, start_price, stop_loss_price, stop_loss_price]
#     fill = "toself" if ended else None
#     line_width = 0 if ended else 0.5
#     name = f"{name}={'???' if not ended else ''}{pnl_value:.0f}G{order_group_id}"
#     fig.add_scatter(x=xs, y=ys, fill=fill,  # fillcolor=fill_color,
#                     fillpattern=dict(fgopacity=0.01),
#                     name=name,
#                     line=dict(color='red', width=line_width, dash='dash' if not ended else None),
#                     mode='lines',
#                     showlegend=False,
#                     legendgroup=legend_group,
#                     hovertemplate="%{text}", text=[text] * 4,
#                     )
#     # draw take_profit box
#     xs = [start, end, end, start]
#     ys = [start_price, start_price, take_profit_price, take_profit_price]
#     fig.add_scatter(x=xs, y=ys, fill=fill,  # fillcolor=fill_color,
#                     fillpattern=dict(fgopacity=0.01),
#                     name=name,
#                     line=dict(color='green', width=line_width, dash='dash' if not ended else None),
#                     mode='lines',
#                     showlegend=False,
#                     legendgroup=legend_group,
#                     hovertemplate="%{text}", text=[text] * 4,
#                     )
#     # draw pnl line
#     line_color = 'green' if pnl_value > 0 else 'red'
#     xs = [start, end]
#     ys = [start_price, end_price]
#     fig.add_scatter(x=xs, y=ys,
#                     name=name,
#                     line=dict(color=line_color, width=1, dash='dash' if not ended else None),
#                     mode='lines',
#                     showlegend=False,
#                     hovertemplate="%{text}", text=[text] * 4,
#                     legendgroup=legend_group,
#                     )
#     # add order pointing mark
#     if side == OrderSide.Buy:
#         _y = take_profit_price + 5
#         symbol = "triangle-up"
#     else:  # side == OrderSide.Sell
#         _y = take_profit_price - 5
#         symbol = "triangle-down"
#     color = "green" if pnl_value > 0 else "red"
#     fig.add_scatter(x=[start], y=[_y], mode="markers",
#                     name=name,
#                     marker=dict(symbol=symbol, color=color), text=[text],
#                     hovertemplate="%{text}",
#                     legendgroup=legend_group, )


# def draw_base(base_pattern, fig, index_date, start, timeframe):
#     xs = [start, base_pattern['effective_end'], base_pattern['effective_end'], start]
#     ys = [base_pattern['internal_low'], base_pattern['internal_low'],
#           base_pattern['internal_high'], base_pattern['internal_high']]
#     fill_color = 'blue'
#     legendgroup = MultiTimeframeBasePattern.str(index_date, timeframe, base_pattern)
#     text = MultiTimeframeBasePattern.repr(index_date, timeframe, base_pattern)
#     fig.add_scatter(x=xs, y=ys, fill="toself",  # fillcolor=fill_color,
#                     fillpattern=dict(fgopacity=0.5),
#                     name=legendgroup,
#                     text=text,
#                     line=dict(color=fill_color, width=0),
#                     mode='lines',
#                     legendgroup=legendgroup,
#                     hoverinfo='text'
#                     )
#     return legendgroup, text

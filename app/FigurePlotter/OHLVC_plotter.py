import pandas as pd
import plotly.graph_objects as go
from pandera import typing as pt
from plotly import graph_objects as plgo
from plotly.subplots import make_subplots

from Config import app_config, CandleSize
from FigurePlotter.plotter import plot_multiple_figures, file_id, DEBUG, save_figure, update_figure_layout
from PanderaDFM.OHLCV import OHLCV
from PanderaDFM.OHLCVA import MultiTimeframeOHLCVA
from helper.data_preparation import single_timeframe
from helper.helper import log, profile_it
price_columns = {
    'open': 'gray',
    'high': 'green',
    'low': 'red',
    'close': 'blue',
}
def trace_chart(i_fig, df, name_prefix, fig_row, fig_col):
    for col in price_columns.keys():
        x = df.index.get_level_values('date')
        i_fig.add_trace(
            go.Scatter(x=x, y=df[col], mode='lines', name=f'{name_prefix} {col}', line=dict(color=price_columns[col])),
            row=fig_row, col=fig_col)
    i_fig.add_trace(
        go.Scatter(x=x, y=df['volume'], mode='lines', name=f'{name_prefix} volume', line=dict(color='orange')),
        row=fig_row, col=fig_col)
    return i_fig

@profile_it
def plot_multi_timeframe_ohlcva(multi_timeframe_ohlcva, name: str = '', show: bool = True, save: bool = True) -> None:
    figures = []
    for _, timeframe in enumerate(app_config.timeframes):
        figures.append(plot_ohlcva(single_timeframe(multi_timeframe_ohlcva, timeframe), show=False, save=False,
                                   name=f'{timeframe} ohlcva'))
    plot_multiple_figures(figures, name=f'multi_timeframe_ohlcva.{file_id(multi_timeframe_ohlcva, name)}',
                          save=save, show=show)


@profile_it
def plot_multi_timeframe_ohlcv(multi_timeframe_ohlcv, date_range_str, show: bool = True, save: bool = True):
    figures = []
    for _, timeframe in enumerate(app_config.timeframes):
        figures.append(plot_ohlcv(single_timeframe(multi_timeframe_ohlcv, timeframe), show=False, save=False,
                                  name=f'{timeframe} ohlcv'))
    plot_multiple_figures(figures, name=f'multi_timeframe_ohlcv.{date_range_str}', show=show, save=save)


@profile_it
def plot_ohlcv(ohlcv: pd = pd.DataFrame(columns=['open', 'high', 'low', 'close']),
               # plot_base_ohlcv: bool = False,
               base_ohlcv: pt.DataFrame[OHLCV] = None,
               save: bool = True, name: str = '', show: bool = True) -> plgo.Figure:
    """
        Plot OHLC (Open, High, Low, Close) data as a candlestick chart.

        Parameters:
            ohlcv (pd.DataFrame): A DataFrame containing OHLC data.
            save (bool): If True, the plot is saved as an image file.
            name (str): The name of the plot.
            show (bool): If False, the plot will not be displayed.

        Returns:
            plgo.Figure: The Plotly figure object containing the OHLC candlestick chart.
            :param show:
            :param name:
            :param save:
            :param ohlcv:
            :param base_ohlcv:
            :param plot_base_ohlcv:
        """

    if DEBUG: log(f'data({ohlcv.shape})')
    if DEBUG: log(ohlcv)
    fig = plgo.Figure(data=[plgo.Candlestick(x=ohlcv.index.values,
                                             open=ohlcv['open'], high=ohlcv['high'], low=ohlcv['low'],
                                             close=ohlcv['close']
                                             , name=name
                                             )]).update_yaxes(fixedrange=False).update_layout(yaxis_title=name)
    if base_ohlcv is not None:
        fig.add_candlestick(
            x=base_ohlcv.index.values,
            open=base_ohlcv['open'], high=base_ohlcv['high'], low=base_ohlcv['low'],
            close=base_ohlcv['close']
            , name=app_config.timeframes[0]
        )
    update_figure_layout(fig)
    if show: fig.show()
    if save:
        file_name = f'ohlcv.{file_id(ohlcv, name)}'
        save_figure(fig, file_name)

    return fig


@profile_it
def plot_ohlcva(ohlcva: pd.DataFrame, save: bool = True, show: bool = True, name: str = '',
                base_ohlcv: pt.DataFrame[OHLCV] = None) -> plgo.Figure:
    """
    Plot OHLC data with an additional atr (Average True Range) boundary.

    The function plots OHLC data as a candlestick chart and adds an atr boundary to the plot.
    The boundary's middle is calculated as the average of the candle's open and close,
    and the width of the boundary is equal to the atr value for each data point.

    Parameters:
        ohlcva (pd.DataFrame): A DataFrame containing OHLC data along with the 'atr' column representing the atr values.
        save (bool): If True, the plot is saved as an HTML file.
        show (bool): If True, the plot is displayed in the browser.

    Returns:
        None

    Example:
        # Assuming you have the 'ohlcva' DataFrame with the required columns (open, high, low, close, atr)
        date_range_str = "17-10-06.00-00T17-10-06"
        plot_ohlcva(ohlcva, date_range_str)
        :param save:
        :param base_ohlcv:
        :param show:
        :param ohlcva:
        :param name:
    """
    # Calculate the middle of the boundary (average of open and close)
    midpoints = (ohlcva['high'] + ohlcva['low']) / 2
    # Create a figure using the plot_ohlcv function
    fig = plot_ohlcv(ohlcva[['open', 'high', 'low', 'close']], base_ohlcv, show=False, save=False, name=name)

    # Add the atr boundaries
    fig = add_atr_scatter(fig, ohlcva.index, midpoints=midpoints,
                          widths=CandleSize.Spinning.value.max * ohlcva['atr'],
                          name='Standard')
    fig = add_atr_scatter(fig, ohlcva.index, midpoints=midpoints,
                          widths=CandleSize.Standard.value.max * ohlcva['atr'],
                          name='Long')
    fig = add_atr_scatter(fig, ohlcva.index, midpoints=midpoints,
                          widths=CandleSize.Long.value.max * ohlcva['atr'],
                          name='Spike')

    fig.add_scatter(x=ohlcva.index, y=midpoints,
                    mode='none',
                    showlegend=False,
                    text=[f'atr: {atr:0.1f}' for atr in ohlcva['atr']],
                    hoverinfo='text')

    fig.update_layout(hovermode='x unified')

    # Show the figure or write it to an HTML file
    if save:
        file_name = f'ohlcva.{file_id(ohlcva, name)}'
        save_figure(fig, file_name)

    if show:
        fig.show()
    return fig


@profile_it
def plot_ohlcva_with_subplot(ohlcva: pd.DataFrame, save: bool = True, show: bool = True, name: str = '',
                             base_ohlcv: pt.DataFrame[OHLCV] = None) -> plgo.Figure:
    """
    Plot OHLC data with an additional atr (Average True Range) boundary.

    The function plots OHLC data as a candlestick chart and adds an atr boundary to the plot.
    The boundary's middle is calculated as the average of the candle's open and close,
    and the width of the boundary is equal to the atr value for each data point.

    Parameters:
        ohlcva (pd.DataFrame): A DataFrame containing OHLC data along with the 'atr' column representing the atr values.
        save (bool): If True, the plot is saved as an HTML file.
        show (bool): If True, the plot is displayed in the browser.

    Returns:
        None

    Example:
        # Assuming you have the 'ohlcva' DataFrame with the required columns (open, high, low, close, atr)
        date_range_str = "17-10-06.00-00T17-10-06"
        plot_ohlcva(ohlcva, date_range_str)
        :param save:
        :param base_ohlcv:
        :param show:
        :param ohlcva:
        :param name:
    """
    # Create a figure with 2 rows
    master_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                               row_heights=[0.7, 0.3])  # Adjust row heights as needed
    # Calculate the middle of the boundary (average of open and close)
    midpoints = (ohlcva['high'] + ohlcva['low']) / 2
    # Create a figure using the plot_ohlcv function
    sub_fig = plot_ohlcv(ohlcva[['open', 'high', 'low', 'close']], base_ohlcv, show=False, save=False, name=name)

    # Add the atr boundaries
    sub_fig = add_atr_scatter(sub_fig, ohlcva.index, midpoints=midpoints,
                              widths=CandleSize.Spinning.value.max * ohlcva['atr'],
                              name='Standard')
    sub_fig = add_atr_scatter(sub_fig, ohlcva.index, midpoints=midpoints,
                              widths=CandleSize.Standard.value.max * ohlcva['atr'],
                              name='Long')
    sub_fig = add_atr_scatter(sub_fig, ohlcva.index, midpoints=midpoints,
                              widths=CandleSize.Long.value.max * ohlcva['atr'],
                              name='Spike')

    sub_fig.add_scatter(x=ohlcva.index, y=midpoints,
                        mode='none',
                        showlegend=False,
                        # text=[f'atr: {atr:0.1f}' for atr in ohlcva['atr']],
                        text=f'atr-{name}:' + ohlcva['atr'].apply(int).apply(str),
                        hoverinfo='text')

    sub_fig.update_layout(hovermode='x unified')

    for trace in sub_fig.data:
        master_fig.add_trace(trace, row=1, col=1)

    master_fig.add_trace(go.Scatter(
        x=ohlcva.index,
        y=ohlcva['atr'],
        name='atr',
        mode='lines',  # Use line mode
        line=dict(color='black')), row=2, col=1)

    # Update layout and Aesthetics
    master_fig.update_layout(
        xaxis=dict(
            fixedrange=False,  # Allows zooming on the x-axis
            rangeslider=dict(visible=False)  # Hide range slider for the first subplot (price chart)
        ),
        xaxis2=dict(
            fixedrange=False,  # Allows zooming on the x-axis
            rangeslider=dict(visible=True)  # Show range slider for the second subplot (atr chart)
        ),
        yaxis=dict(
            fixedrange=False  # Allows zooming on the y-axis for the first subplot
        ),
        yaxis2=dict(
            fixedrange=False  # Allows zooming on the y-axis for the second subplot
        ),
        # height=800, width=1200,
        title_text=name)
    master_fig.update_yaxes(title_text="Price", row=1, col=1)
    master_fig.update_yaxes(title_text="atr", row=2, col=1)

    # Show the figure or write it to an HTML file
    if save:
        file_name = f'ohlcva.{file_id(ohlcva, name)}'
        save_figure(master_fig, file_name)

    if show:
        master_fig.show()
    return master_fig


def add_atr_scatter(fig: plgo.Figure, xs: pd.Series, midpoints: pd.Series, widths: pd.Series,
                    transparency: float = 0.2, name: str = 'atr', legendgroup: str = None,
                    showlegend=True, show_hover: bool = False) -> plgo.Figure:
    xs = xs.tolist()
    half_widths = widths.fillna(value=0).div(2)
    upper_band: pd.Series = midpoints + half_widths
    lower_band: pd.Series = midpoints - half_widths
    """
        fig.add_scatter(x=ohlcva.index, y=midpoints,
                        mode='none',
                        showlegend=False,
                        text=[f'atr: {atr:0.1f}' for atr in ohlcva['atr']],
                        hoverinfo='text')
    """
    fig.add_scatter(
        x=xs + xs[::-1],
        y=upper_band.tolist() + lower_band.tolist()[::-1],
        mode='lines',
        # hovertext=widths,
        line=dict(color='gray', dash='solid', width=0.2),
        fill='toself',
        fillcolor=f'rgba(128, 128, 128, {transparency})',  # 50% transparent gray color
        showlegend=showlegend,
        name=name, legendgroup=legendgroup
    )
    if show_hover:
        fig.add_scatter(x=xs, y=midpoints, mode='none',
                        showlegend=False,
                        text=f'atr-{name}:' + widths.apply(int).apply(str),
                        legendgroup=legendgroup,
                        hoverinfo='text')
    return fig


def plot_merged_timeframe_ohlcva(multi_timeframe_ohlcva: pt.DataFrame[MultiTimeframeOHLCVA]):
    fig = plgo.Figure()
    update_figure_layout(fig)
    multi_timeframe_ohlcva_timeframes = multi_timeframe_ohlcva.index.get_level_values('timeframe').unique()
    timeframe_list = [timeframe for timeframe in app_config.timeframes if timeframe in multi_timeframe_ohlcva_timeframes]
    for timeframe in timeframe_list:
        ohlcva = single_timeframe(multi_timeframe_ohlcva, timeframe)
        fig.add_candlestick(x=ohlcva.index.values,
                            open=ohlcva['open'], high=ohlcva['high'], low=ohlcva['low'],
                            close=ohlcva['close']
                            , name=timeframe, legendgroup=timeframe
                            ).update_yaxes(fixedrange=False)
        # Add the atr boundaries
        midpoints = (ohlcva['high'] + ohlcva['low']) / 2
        add_atr_scatter(fig, ohlcva.index, midpoints=midpoints,
                        widths=CandleSize.Spinning.value.max * ohlcva['atr'],
                        name='Standard', legendgroup=timeframe, showlegend=False)

    return fig

import textwrap

import pandas as pd
from domain.technical_analysis.classic_indicators import classic_indicator_columns, scaleless_indicators
from helper.pandera import pandera_validate
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from presentation.shared.plotter import show_and_save_plot


@pandera_validate(allow_pandas_dataframe=True)  # type: ignore[untyped-decorator]
def plot_classic_indicators(fig: go.Figure, x_dfs: dict[str, list[pd.DataFrame]], n: int) -> go.Figure:
    scalable_indicators = list(set(classic_indicator_columns()) - set(scaleless_indicators()))
    for level in ["structure", "pattern", "double", "trigger"]:
        for indicator_column in scaleless_indicators():
            if indicator_column != "sc_obv":
                t = x_dfs[f"{level}-indicators"][n][indicator_column]
                fig.add_scatter(
                    x=t.index, y=t, row=2, col=1, line={"color": "blue"}, name=f"{indicator_column}-{level}"
                )
        for indicator_column in scalable_indicators:
            if indicator_column != "sc_obv":
                t = x_dfs[f"{level}-indicators"][n][indicator_column]
                fig.add_scatter(
                    x=t.index, y=t, row=1, col=1, line={"color": "blue"}, name=f"{indicator_column}-{level}-"
                )
    return fig


@pandera_validate(allow_pandas_dataframe=True)  # type: ignore[untyped-decorator]
def plot_train_data_of_mt_n_profit(
    x_dfs: dict[str, list[pd.DataFrame]], y_dfs: list[pd.DataFrame], y_tester_dfs: list[pd.DataFrame], n: int
) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,  # vertical_spacing=0.02,
        row_heights=[0.65, 0.25],
    )
    plot_mt_charts(fig, n, x_dfs)
    fig = plot_classic_indicators(fig, x_dfs, n)
    plot_prediction_verifier(fig, n, y_tester_dfs)
    plot_prediction(fig, n, y_dfs)
    fig.update_layout(xaxis={"rangeslider": {"visible": False}})
    show_and_save_plot(fig.update_yaxes(fixedrange=False))


@pandera_validate(allow_pandas_dataframe=True)  # type: ignore[untyped-decorator]
def plot_mt_charts(fig: go.Figure, n: int, x_dfs: dict[str, list[pd.DataFrame]]) -> None:
    ohlcv_slices = [("structure", "Structure"), ("pattern", "Pattern"), ("trigger", "Trigger"), ("double", "Double")]
    for key, name in ohlcv_slices:
        ohlcv = x_dfs[key][n]
        fig.add_trace(
            go.Candlestick(
                x=ohlcv.index.get_level_values("date"),
                open=ohlcv["low"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                close=ohlcv["high"],
                name=name,
            )
        )


@pandera_validate(allow_pandas_dataframe=True)  # type: ignore[untyped-decorator]
def plot_prediction_verifier(fig: go.Figure, n: int, y_tester_dfs: list[pd.DataFrame]) -> None:
    ohlcv = y_tester_dfs[n]
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index.get_level_values("date"),
            close=ohlcv["low"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            open=ohlcv["high"],
            name="Y",
        )
    )


@pandera_validate(allow_pandas_dataframe=True)  # type: ignore[untyped-decorator]
def plot_prediction(fig: go.Figure, n: int, y_dfs: list[pd.DataFrame]) -> None:
    predictions = y_dfs[n].to_dict()
    formatted_predictions = textwrap.fill(
        ", ".join(
            [
                f"{col}: {val:.2f}"
                if isinstance(val, (int, float)) and val == val
                else f"{col}: NaN"
                if val != val
                else f"{col}: {val}"
                for col, val in predictions.items()
            ]
        ),
        width=80,
    ).replace("\n", "<br>")
    fig.add_annotation(
        x=0,
        y=1,
        text=formatted_predictions,
        showarrow=False,
        font={"size": 12, "color": "black"},
        align="left",
        bgcolor="white",
        opacity=0.7,
        xref="paper",  # Use the "paper" reference to place it relative to the figure
        yref="paper",  # Use the "paper" reference to place it relative to the figure
        borderpad=10,  # Add some padding for the border
    )

import textwrap
from datetime import timedelta, datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from Config import app_config
from FigurePlotter.plotter import show_and_save_plot
from PanderaDFM import MultiTimeframe
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.profit_loss.profit_loss_adder import \
    add_long_n_short_profit
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.classic_indicators import add_classic_indicators, \
    classic_indicator_columns, scaleless_indicators
from helper.data_preparation import pattern_timeframe, trigger_timeframe, single_timeframe
from helper.functions import profile_it, date_range, log_d
from helper.importer import pt


def slice_indicators(timeframes_df_dict: dict, end_time: datetime, length: int):
    try:
        t_slice = {
            df_name: pd.DataFrame({
                indicator_column: timeframe_df.loc[pd.IndexSlice[:end_time], indicator_column].iloc[-length:]
                for indicator_column in classic_indicator_columns()
            })
            for df_name, timeframe_df in timeframes_df_dict.items()
        }
    except Exception as e:
        nop = 1
        raise e

    return t_slice


def single_timeframe_n_indicators(mt_ohlcv, timeframe):
    ohlcv = single_timeframe(mt_ohlcv, timeframe)
    ohlcv = add_classic_indicators(ohlcv)
    return ohlcv


# @profile_it
def train_data_of_mt_n_profit(structure_tf, mt_ohlcv: pt.DataFrame[MultiTimeframe], x_shape: dict,
                              batch_size: int, forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1,
                              only_actionable: bool = True, ):
    """
    Generates training data for a multi-timeframe trading model. It slices the multi-timeframe OHLCV data into
    batches of input features and forecast targets based on specified timeframes and lengths.

    Parameters:
    -----------
    structure_tf : str
        The timeframe string for the structure-level data (e.g., '1H', '15min').

    mt_ohlcv : pt.DataFrame[MultiTimeframe]
        A multi-timeframe Pandas DataFrame containing OHLCV data indexed by datetime.

    x_shape : dict
        A dictionary specifying the number of bars (time steps) for each timeframe's input slice. Keys must include:
        - 'double': Length for the double timeframe.
        - 'trigger': Length for the trigger timeframe.
        - 'pattern': Length for the pattern timeframe.
        - 'structure': Length for the structure timeframe.

    batch_size : int
        The number of training samples to generate in one batch.

    forecast_trigger_bars : int, optional
        The number of bars ahead to forecast, based on the trigger timeframe. Default is 768 (16 hours for 5-minute bars).

    Returns:
    --------
    tuple
        - Xs (dict): A dictionary of input features for each timeframe ('double', 'trigger', 'pattern', 'structure').
                     Each value is a NumPy array of shape (batch_size, length, num_features).
        - ys (np.ndarray): A NumPy array of shape (batch_size, num_targets) representing the forecast targets.
        - x_dfs (dict): A dictionary of DataFrame slices for each timeframe, useful for debugging or visualization.
        - y_dfs (list): A list of DataFrames containing the forecast target values for each batch.
        - trigger_tf (str): The trigger timeframe string used for forecasting.
        - y_tester_dfs (list): A list of DataFrames containing the future slices for verification.

    Raises:
    -------
    RuntimeError
        If the time range for generating training data is insufficient.

    Workflow:
    ---------
    1. Derives the necessary timeframes (pattern, trigger, double) from the structure timeframe.
    2. Calculates the safe start and end dates for slicing the training data.
    3. Creates time slices for each timeframe, ensuring no gaps larger than a configured threshold.
    4. Builds the input features (Xs) and forecast targets (ys) for each batch.
    5. Returns the generated data along with slices for analysis.

    Example:
    --------
    >>> mt_ohlcv = load_mt_ohlcv_data()
    >>> x_shape = {
    ...     'structure': [50],
    ...     'pattern': [20],
    ...     'trigger': [10],
    ...     'double': [5],
    ... }
    >>> Xs, ys, x_dfs, y_dfs, trigger_tf, y_tester_dfs = train_data_of_mt_n_profit(
    ...     structure_tf='1H', mt_ohlcv=mt_ohlcv, x_shape=x_shape, batch_size=32)
    >>> print(Xs['trigger'].shape, ys.shape)

    Notes:
    ------
    - Ensure `mt_ohlcv` has a multi-index with 'date' as one of the levels.
    - Configure `config.max_x_gap` to set the acceptable gap tolerance in bars for each timeframe.
    """
    training_x_columns = ['open', 'high', 'low', 'close', 'volume', ]
    training_y_columns = ['long_signal', 'short_signal', 'min_low', 'max_high', 'long_profit', 'short_profit',
                          'long_risk', 'short_risk', 'long_drawdown', 'short_drawdown',
                          'long_drawdown', 'short_drawdown', ]
    pattern_tf = pattern_timeframe(structure_tf)
    trigger_tf = trigger_timeframe(structure_tf)
    double_tf = pattern_timeframe(trigger_timeframe(structure_tf))
    dfs = {}
    for df_name, timeframe in [('structure', structure_tf), ('pattern', pattern_tf),
                               ('trigger', trigger_tf), ('double', double_tf)]:
        dfs[df_name] = single_timeframe_n_indicators(mt_ohlcv, timeframe)
    dfs['trigger']['atr'] = ta.atr(high=dfs['trigger']['high'], low=dfs['trigger']['low'],
                                   close=dfs['trigger']['close'], length=256)
    mt_dfs = dfs.copy()
    dfs['prediction'] = add_long_n_short_profit(ohlc=dfs['trigger'], position_max_bars=forecast_trigger_bars,
                                                trigger_tf=trigger_tf)
    train_safe_end, train_safe_start = remove_nas(dfs)
    duration_seconds = (train_safe_end - train_safe_start) / timedelta(seconds=1)
    if duration_seconds <= 0:
        start, end = date_range(app_config.processing_date_range)
        raise RuntimeError(
            f"Extend date boundary +{-duration_seconds}s({duration_seconds / (60 * 60 * 24)}days, "
            f"start:{start}<{start + duration_seconds * timedelta(seconds=1)} or "
            f"end:{end}>{end - duration_seconds * timedelta(seconds=1)}) to make possible range of end dates positive!")
    x_dfs, y_dfs, y_tester_dfs = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, [], []
    Xs, ys = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, []
    for timeframe in ['structure', 'pattern', 'trigger', 'double']:
        x_dfs[f'{timeframe}-indicators'] = []
        Xs[f'{timeframe}-indicators'] = []  # np.ndarray([])
    batch_remained = batch_size
    while batch_remained > 0:
        # for relative_double_end in np.random.randint(0, duration_seconds, size=batch_size):
        double_end, trigger_end, pattern_end, structure_end = \
            batch_ends(duration_seconds, double_tf, trigger_tf, pattern_tf, x_shape, train_safe_end)
        prediction = dfs['prediction'].loc[pd.IndexSlice[:double_end], training_y_columns].iloc[-1]
        if only_actionable:
            if prediction['long_signal'] == 0 and prediction['short_signal'] == 0:
                continue
        double_slice, pattern_slice, structure_slice, trigger_slice, indicators_slice = \
            slicing(mt_dfs, structure_end, pattern_end, trigger_end,
                    double_end, training_x_columns, x_shape)
        prediction_testing_slice = (
            dfs['trigger'].loc[
                pd.IndexSlice[double_end: double_end + forecast_trigger_bars * pd.to_timedelta(trigger_tf)],
                training_x_columns])
        try:
            for timeframe, slice_df, level in \
                    [(structure_tf, structure_slice, 'structure'), (pattern_tf, pattern_slice, 'pattern'),
                     (trigger_tf, trigger_slice, 'trigger'), (double_tf, double_slice, 'double')]:
                if abs((slice_df.index.max() - slice_df.index.min()) / pd.to_timedelta(timeframe)
                       - (x_shape[level][0] - 1)) > app_config.max_x_gap:
                    raise AssertionError(
                        f"Skipping: gap of > {app_config.max_x_gap} bars in {level}/{timeframe}-{app_config.under_process_exchange}"
                        f"/{app_config.under_process_symbol}/{timeframe}:"
                        f"{slice_df.index.min()}-{slice_df.index.max()}")
        except AssertionError as e:
            log_d(e)
            continue
        (sc_double_slice, sc_indicators_slice, sc_pattern_slice, sc_prediction, sc_prediction_testing_slice,
         sc_structure_slice, sc_trigger_slice) = \
            scaling(structure_slice, pattern_slice, trigger_slice, double_slice, prediction, indicators_slice,
                    prediction_testing_slice, training_x_columns)
        if (
                len(np.array(sc_double_slice[training_x_columns])) != x_shape['double'][0]
                or len(np.array(sc_trigger_slice[training_x_columns])) != x_shape['trigger'][0]
                or len(np.array(sc_pattern_slice[training_x_columns])) != x_shape['pattern'][0]
                or len(np.array(sc_structure_slice[training_x_columns])) != x_shape['structure'][0]
        ):
            continue
        x_dfs['double'].append(sc_double_slice[training_x_columns])
        x_dfs['trigger'].append(sc_trigger_slice[training_x_columns])
        x_dfs['pattern'].append(sc_pattern_slice[training_x_columns])
        x_dfs['structure'].append(sc_structure_slice[training_x_columns])
        for timeframe in ['structure', 'pattern', 'trigger', 'double']:
            x_dfs[f'{timeframe}-indicators'].append(sc_indicators_slice[timeframe])
        Xs['double'].append(np.array(x_dfs['double'][-1]))
        Xs['trigger'].append(np.array(x_dfs['trigger'][-1]))
        Xs['pattern'].append(np.array(x_dfs['pattern'][-1]))
        Xs['structure'].append(np.array(x_dfs['structure'][-1]))
        for timeframe in ['structure', 'pattern', 'trigger', 'double']:
            Xs[f'{timeframe}-indicators'].append(np.array(x_dfs[f'{timeframe}-indicators'][-1]))
        y_dfs.append(sc_prediction)
        y_tester_dfs.append(sc_prediction_testing_slice)
        ys.append(np.array(y_dfs[-1]))
        batch_remained -= 1
    # converting list of batches to a combined ndarray
    try:
        Xs['double'] = np.array(Xs['double'])
        Xs['trigger'] = np.array(Xs['trigger'])
        Xs['pattern'] = np.array(Xs['pattern'])
        Xs['structure'] = np.array(Xs['structure'])
        for timeframe in sc_indicators_slice:
            Xs[f'{timeframe}-indicators'] = np.array(Xs[f'{timeframe}-indicators'])
        ys = np.array(ys)
    except Exception as e:
        raise e
    shape_assertion(Xs, x_dfs, y_dfs, y_tester_dfs, ys, x_shape, batch_size, forecast_trigger_bars)
    return Xs, ys, x_dfs, y_dfs, trigger_tf, y_tester_dfs


def shape_assertion(Xs, x_dfs, y_dfs, y_tester_dfs, ys, x_shape: dict, batch_size: int = 120,
                    forecast_trigger_bars: int = 192):
    """
    x_shape = {'double': (255, 5), 'indicators': (129,), 'pattern': (253, 5), 'structure': (127, 5), 'trigger': (254, 5)}

    Args:
        Xs:
        x_dfs:
        y_dfs:
        y_tester_dfs:
        ys:
        batch_size:
        forecast_trigger_bars:
        x_shape:

    Returns:

    """
    b_l = batch_size
    # i_l = x_shape['indicators']
    x_shape_assertion(Xs, b_l, x_shape)
    if get_shape(ys) != (b_l, 12):
        raise AssertionError("get_shape(ys) != (b_l, 12)")
    from deepdiff import DeepDiff
    if DeepDiff(get_shape(x_dfs), {
        'double': [b_l, x_shape['double']], 'trigger': [b_l, x_shape['trigger']],
        'pattern': [b_l, x_shape['pattern']], 'structure': [b_l, x_shape['structure']],
        'structure-indicators': [b_l, x_shape['indicators']],
        'pattern-indicators': [b_l, x_shape['indicators']],
        'trigger-indicators': [b_l, x_shape['indicators']],
        'double-indicators': [b_l, x_shape['indicators']],
    }) != {}:
        raise AssertionError("DeepDiff(get_shape(x_dfs), {")
    if get_shape(y_dfs) != [b_l, (12,)]:
        raise AssertionError("get_shape(y_dfs) != [b_l, (12,)]")
    if get_shape(y_tester_dfs) != [b_l, (forecast_trigger_bars, 5)]:
        raise AssertionError("get_shape(y_tester_dfs) != [b_l, (forecast_trigger_bars, 5)]")


def x_shape_assertion(Xs, batch_size, x_shape):
    i_l = x_shape['indicators'][0]
    b_l = batch_size
    if get_shape(Xs) != {
        'double': (b_l, x_shape['double'][0], 5), 'double-indicators': (b_l, i_l, 12),
        'pattern': (b_l, x_shape['pattern'][0], 5), 'pattern-indicators': (b_l, i_l, 12),
        'structure': (b_l, x_shape['structure'][0], 5), 'structure-indicators': (b_l, i_l, 12),
        'trigger': (b_l, x_shape['trigger'][0], 5), 'trigger-indicators': (b_l, i_l, 12)}:
        raise AssertionError("get_shape(Xs) != {")


def remove_nas(dfs):
    train_safe_start, train_safe_end = (None, None)
    for df_name in ['structure', 'pattern', 'trigger', 'double', 'prediction']:
        df = dfs[df_name]
        not_na_df = df.dropna(how='any')
        not_na_start = not_na_df.index.get_level_values(level='date').min()
        not_na_end = not_na_df.index.get_level_values(level='date').max()
        if train_safe_start is None or train_safe_start < not_na_start:
            train_safe_start = not_na_start
        if train_safe_end is None or train_safe_end > not_na_end:
            train_safe_end = not_na_end
        nop = 1
    for df_name in ['structure', 'pattern', 'trigger', 'double', 'prediction']:
        dfs[df_name] = dfs[df_name].loc[pd.IndexSlice[train_safe_start:train_safe_end, :]]
    return train_safe_end, train_safe_start


def get_shape(obj):
    if isinstance(obj, np.ndarray):
        return obj.shape
    elif isinstance(obj, pd.DataFrame):
        return obj.shape  # (rows, columns)
    elif isinstance(obj, pd.Series):
        return (obj.shape[0],)  # 1D shape
    elif isinstance(obj, (list, tuple)):
        return [len(obj)] + ([get_shape(obj[0])] if obj else [])
    elif isinstance(obj, dict):
        return {k: get_shape(v) for k, v in obj.items()}
    else:
        return None  # Base case for non-iterables


def scaling(structure_slice, pattern_slice, trigger_slice, double_slice, prediction, indicators_slice,
            prediction_testing_slice, training_x_columns):
    price_scale, price_shift, volume_scale, obv_scale, obv_shift = scaler_trainer(
        {'double': double_slice, 'pattern': pattern_slice, 'structure': structure_slice, 'trigger': trigger_slice,
         'trigger-obv': indicators_slice['trigger']['obv'], }, mean_atr=trigger_slice['atr'].mean(),
        close=double_slice.iloc[-1]['close'], )
    sc_double_slice = scale_ohlc(double_slice, price_shift, price_scale, volume_scale)[training_x_columns]
    sc_trigger_slice = scale_ohlc(trigger_slice, price_shift, price_scale, volume_scale)[
        training_x_columns]
    sc_pattern_slice = scale_ohlc(pattern_slice, price_shift, price_scale, volume_scale)[
        training_x_columns]
    sc_structure_slice = scale_ohlc(structure_slice, price_shift, price_scale, volume_scale)[
        training_x_columns]
    sc_indicators_slice = scale_indicators(indicators_slice, price_shift, price_scale, obv_scale, obv_shift)
    sc_prediction = scale_prediction(prediction, price_shift, price_scale, )
    sc_prediction_testing_slice = \
        scale_ohlc(prediction_testing_slice, price_shift, price_scale, volume_scale)
    return sc_double_slice, sc_indicators_slice, sc_pattern_slice, sc_prediction, sc_prediction_testing_slice, sc_structure_slice, sc_trigger_slice


def batch_ends(duration_seconds, double_tf, trigger_tf, pattern_tf, x_shape, train_safe_end):
    relative_double_end = np.random.randint(0, duration_seconds)
    double_end: datetime = train_safe_end - relative_double_end * timedelta(seconds=1)
    trigger_end = double_end - x_shape['double'][0] * pd.to_timedelta(double_tf)
    pattern_end = trigger_end - x_shape['trigger'][0] * pd.to_timedelta(trigger_tf)
    structure_end = pattern_end - x_shape['pattern'][0] * pd.to_timedelta(pattern_tf)
    return double_end, trigger_end, pattern_end, structure_end


def slicing(dfs, structure_end, pattern_end, trigger_end, double_end, training_x_columns, x_shape):
    double_slice = dfs['double'].loc[pd.IndexSlice[: double_end], training_x_columns].iloc[
                   -x_shape['double'][0]:]
    trigger_slice = dfs['trigger'].loc[
                        pd.IndexSlice[: trigger_end], training_x_columns + ['atr']].iloc[
                    -x_shape['trigger'][0]:]
    pattern_slice = dfs['pattern'].loc[pd.IndexSlice[: pattern_end], training_x_columns].iloc[
                    -x_shape['pattern'][0]:]
    structure_slice = dfs['structure'].loc[pd.IndexSlice[: structure_end], training_x_columns].iloc[
                      -x_shape['structure'][0]:]
    indicators_slice = slice_indicators(timeframes_df_dict=dfs, end_time=double_end, length=x_shape['indicators'][0])
    assert ~double_slice.isna().any().any()
    assert ~trigger_slice.isna().any().any()
    assert ~pattern_slice.isna().any().any()
    assert ~structure_slice.isna().any().any()
    assert all([level_indicators.notna().any().any()
                for level, level_indicators in indicators_slice.items()])
    return double_slice, pattern_slice, structure_slice, trigger_slice, indicators_slice


def plot_classic_indicators(fig: go.Figure, x_dfs, n) -> go.Figure:
    scalable_indicators = list(set(classic_indicator_columns()) - set(scaleless_indicators()))
    for level in ['structure', 'pattern', 'double', 'trigger']:
        for indicator_column in scaleless_indicators():
            if indicator_column != 'obv':
                t = x_dfs[f"{level}-indicators"][n][indicator_column]
                fig.add_scatter(x=t.index, y=t, row=2, col=1, line=dict(color='blue'),
                                name=f"{indicator_column}-{level}")
        for indicator_column in scalable_indicators:
            if indicator_column != 'obv':
                t = x_dfs[f"{level}-indicators"][n][indicator_column]
                fig.add_scatter(x=t.index, y=t, row=1, col=1, line=dict(color='blue'),
                                name=f"{indicator_column}-{level}-")
    return fig


def plot_train_data_of_mt_n_profit(x_dfs: dict[str, List[pd.DataFrame]], y_dfs: List[pd.DataFrame],
                                   y_tester_dfs: List[pd.DataFrame], n: int, ):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,  # vertical_spacing=0.02,
                        row_heights=[0.65, 0.25])
    plot_mt_charts(fig, n, x_dfs)
    fig = plot_classic_indicators(fig, x_dfs, n)
    plot_prediction_verifiyer(fig, n, y_tester_dfs)
    plot_prediction(fig, n, y_dfs)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
    show_and_save_plot(fig.update_yaxes(fixedrange=False))


def plot_mt_charts(fig, n, x_dfs):
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


def plot_prediction_verifiyer(fig, n, y_tester_dfs):
    ohlcv = y_tester_dfs[n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        close=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        open=ohlcv['high'],
        name='Y'
    ))


def plot_prediction(fig, n, y_dfs):
    predictions = y_dfs[n].to_dict()
    formatted_predictions = textwrap.fill(', '.join([
        f"{col}: {val:.2f}" if isinstance(val, (int, float)) and not (val != val)
        else f"{col}: NaN" if val != val
        else f"{col}: {val}"
        for col, val in predictions.items()
    ]), width=80).replace('\n', '<br>')
    try:
        fig.add_annotation(
            x=0, y=1, text=formatted_predictions,
            showarrow=False,
            font=dict(size=12, color="black"),
            align="left",
            bgcolor="white",
            opacity=0.7,
            xref="paper",  # Use the "paper" reference to place it relative to the figure
            yref="paper",  # Use the "paper" reference to place it relative to the figure
            borderpad=10  # Add some padding for the border
        )
    except Exception as e:
        raise e


def scale_prediction(prediction, price_scaler_shift, price_scaler_size):
    prediction['min_low'] = (prediction['min_low'] + price_scaler_shift) * price_scaler_size
    prediction['max_high'] = (prediction['max_high'] + price_scaler_shift) * price_scaler_size
    prediction['long_profit'] = (prediction['long_profit']) * price_scaler_size
    prediction['short_profit'] = (prediction['short_profit']) * price_scaler_size
    return prediction


def scale_ohlc(df, price_shift, price_scale, volume_scale):
    df = df.copy()
    for column in ['open', 'high', 'low', 'close']:
        df[column] = (df[column] + price_shift) * price_scale
    df['volume'] = df['volume'] * volume_scale
    return df


def scale_indicators(indicator_df, price_shift, price_scale, obv_scale, obv_shift):
    columns_to_scale = set(classic_indicator_columns()) - set(scaleless_indicators())
    try:
        for timeframe in indicator_df:
            for column in columns_to_scale:
                indicator_df[timeframe][column] = (indicator_df[timeframe][column] + price_shift) * price_scale
            indicator_df[timeframe]['obv'] = (indicator_df[timeframe]['obv'] + obv_shift) * obv_scale
    except Exception as e:
        raise e
    return indicator_df


def scaler_trainer(slices: Dict[str, pd.DataFrame], mean_atr: float, close: float):
    price_scale = (1 / mean_atr)
    price_shift = - close
    t_slice = pd.concat(slices)
    volume_scale = 1 / t_slice['volume'].mean()
    obv_shift = - slices['trigger-obv'].iloc[-1]
    obv_scale = 50 / ((slices['trigger-obv'].max() - slices['trigger-obv'].min()) / 2)
    return price_scale, price_shift, volume_scale, obv_scale, obv_shift


x_shape = {
    'structure': (127, 5),
    'pattern': (253, 5),
    'trigger': (254, 5),
    'double': (255, 5),
    'indicators': (129, 12),
}

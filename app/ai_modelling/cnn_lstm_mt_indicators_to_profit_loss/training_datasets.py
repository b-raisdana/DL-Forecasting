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
from helper.helper import profile_it, date_range, log_d
from helper.importer import pt


def slice_indicators(timeframes_df_dict: dict, end_time):
    # t_slice = {}
    # for timeframe, timeframe_df in timeframes_df_dict:
    #     for indicator_column in classic_indicator_columns:
    #         t_slice[timeframe][indicator_column] = timeframe_df[pd.IndexSlice[indicator_column, :end_time]]
    t_slice = {
        timeframe: {
            indicator_column: timeframe_df[pd.IndexSlice[indicator_column, :end_time]]
            for indicator_column in classic_indicator_columns()
        }
        for timeframe, timeframe_df in timeframes_df_dict
    }
    return t_slice


def single_timeframe_n_indicators(mt_ohlcv, timeframe):
    ohlcv = single_timeframe(mt_ohlcv, timeframe)
    ohlcv = add_classic_indicators(ohlcv)
    return ohlcv


@profile_it
def train_data_of_mt_n_profit(structure_tf, mt_ohlcv: pt.DataFrame[MultiTimeframe], x_lengths: dict,
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

    x_lengths : dict
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
    >>> x_lengths = {
    ...     'structure': [50],
    ...     'pattern': [20],
    ...     'trigger': [10],
    ...     'double': [5],
    ... }
    >>> Xs, ys, x_dfs, y_dfs, trigger_tf, y_tester_dfs = train_data_of_mt_n_profit(
    ...     structure_tf='1H', mt_ohlcv=mt_ohlcv, x_lengths=x_lengths, batch_size=32)
    >>> print(Xs['trigger'].shape, ys.shape)

    Notes:
    ------
    - Ensure `mt_ohlcv` has a multi-index with 'date' as one of the levels.
    - Configure `config.max_x_gap` to set the acceptable gap tolerance in bars for each timeframe.
    """
    # training_x_columns = ['n_open', 'n_high', 'n_low', 'n_close', 'n_volume', ]
    training_x_columns = ['open', 'high', 'low', 'close', 'volume', ]
    training_y_columns = ['long_signal', 'short_signal', 'min_low', 'max_high', 'long_profit', 'short_profit',
                          'long_risk', 'short_risk', 'long_drawdown', 'short_drawdown',
                          'long_drawdown', 'short_drawdown',
                          ]
    pattern_tf = pattern_timeframe(structure_tf)
    trigger_tf = trigger_timeframe(structure_tf)
    double_tf = pattern_timeframe(trigger_timeframe(structure_tf))
    timeframe_list = [structure_tf, trigger_tf, pattern_tf, double_tf]
    # length_of_training = (
    #         x_lengths['structure'][0] * pd.to_timedelta(structure_tf)
    #         + x_lengths['pattern'][0] * pd.to_timedelta(pattern_tf)
    #         + x_lengths['trigger'][0] * pd.to_timedelta(trigger_tf)
    #         + x_lengths['double'][0] * pd.to_timedelta(double_tf)
    # )
    # dfs['structure_df'], dfs['pattern_df'], dfs['trigger_df'], dfs['double_df'] = (pd.DataFrame(),) * 4  # create a tuple of 4 pd.Dataframes
    dfs = {}
    for df_name, timeframe in [('structure_df', structure_tf), ('pattern_df', pattern_tf),
                               ('trigger_df', trigger_tf), ('double_df', double_tf)]:
        dfs[df_name] = single_timeframe_n_indicators(mt_ohlcv, timeframe)
    dfs['trigger_df']['atr'] = ta.atr(high=dfs['trigger_df']['high'],
                                                 low=dfs['trigger_df']['low'],
                                                 close=dfs['trigger_df']['close'], length=256)
    dfs['prediction_df'] = add_long_n_short_profit(ohlc=dfs['trigger_df'],
                                            position_max_bars=forecast_trigger_bars, trigger_tf=trigger_tf)
    # train_safe_start = mt_ohlcv.index.get_level_values(
    #     level='date').min() + length_of_training * 2  # * 2 for simple safe side.
    # train_safe_end = \
    #     mt_ohlcv.index.get_level_values(level='date').max() - forecast_trigger_bars * pd.to_timedelta(trigger_tf)
    train_safe_start, train_safe_end = (None, None)
    for dataframe in ['structure_df', 'pattern_df', 'trigger_df', 'double_df', 'prediction_df']:
        df = dfs[dataframe]
        not_na_df = df.dropna(how='any')
        not_na_start = not_na_df.index.get_level_values(level='date').min()
        not_na_end = not_na_df.index.get_level_values(level='date').max()
        if train_safe_start is None or train_safe_start < not_na_start:
            train_safe_start = not_na_start
        if train_safe_end is None or train_safe_end > not_na_end:
            train_safe_end = not_na_end
    duration_seconds = (train_safe_end - train_safe_start) / timedelta(seconds=1)
    if duration_seconds <= 0:
        start, end = date_range(app_config.processing_date_range)
        raise RuntimeError(
            f"Extend date boundary +{-duration_seconds}s({duration_seconds / (60 * 60 * 24)}days, "
            f"start:{start}<{start + duration_seconds * timedelta(seconds=1)} or "
            f"end:{end}>{end - duration_seconds * timedelta(seconds=1)}) to make possible range of end dates positive!")
    x_dfs, y_dfs, y_tester_dfs = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, [], []
    Xs, ys = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, []
    for timeframe in timeframe_list:
        for indicator_column in classic_indicator_columns():
            Xs[f'{timeframe}-{indicator_column}'] = []  # np.ndarray([])
    batch_remained = batch_size
    while batch_remained > 0:
        # for relative_double_end in np.random.randint(0, duration_seconds, size=batch_size):
        relative_double_end = np.random.randint(0, duration_seconds)
        double_end: datetime = train_safe_end - relative_double_end * timedelta(seconds=1)
        trigger_end = double_end - x_lengths['double'][0] * pd.to_timedelta(double_tf)
        pattern_end = trigger_end - x_lengths['trigger'][0] * pd.to_timedelta(trigger_tf)
        structure_end = pattern_end - x_lengths['pattern'][0] * pd.to_timedelta(pattern_tf)

        prediction = dfs['prediction_df'].loc[pd.IndexSlice[:double_end], training_y_columns].iloc[-1]
        if only_actionable:
            if prediction['long_signal'] == 0 and prediction['short_signal'] == 0:
                continue

        double_slice = dfs['double_df'].loc[pd.IndexSlice[: double_end], training_x_columns].iloc[
                       -x_lengths['double'][0]:]
        trigger_slice = dfs['trigger_df'].loc[
                            pd.IndexSlice[: trigger_end], training_x_columns + ['atr']].iloc[
                        -x_lengths['trigger'][0]:]
        pattern_slice = dfs['pattern_df'].loc[pd.IndexSlice[: pattern_end], training_x_columns].iloc[
                        -x_lengths['pattern'][0]:]
        structure_slice = dfs['structure_df'].loc[pd.IndexSlice[: structure_end], training_x_columns].iloc[
                          -x_lengths['structure'][0]:]
        indicators_slice = slice_indicators(timeframes_df_dict={
            structure_tf: dfs['structure_df'],
            pattern_tf: dfs['pattern_df'],
            trigger_tf: dfs['trigger_df'],
            double_tf: double_tf,
        }, end_time=double_end)
        scaler_price_scale, scaler_price_shift, volume_scale = scaler_trainer(
            {'double': double_slice, 'pattern': pattern_slice, 'structure': structure_slice, 'trigger': trigger_slice},
            mean_atr=trigger_slice['atr'].mean(),
            cloase=double_slice.iloc[-1]['close'],
        )
        trigger_slice = trigger_slice[training_x_columns]
        prediction_testing_slice = (
            dfs['trigger_df'].loc[
                pd.IndexSlice[double_end: double_end + forecast_trigger_bars * pd.to_timedelta(trigger_tf)],
                training_x_columns])
        try:
            for timeframe, slice_df, relative_tf_name in [(structure_tf, structure_slice, 'structure'),
                                                          (pattern_tf, pattern_slice, 'pattern'),
                                                          (trigger_tf, trigger_slice, 'trigger'),
                                                          (double_tf, double_slice, 'double')]:
                if abs((slice_df.index.max() - slice_df.index.min()) / pd.to_timedelta(timeframe)
                       - (x_lengths[relative_tf_name][0] - 1)) > app_config.max_x_gap:
                    raise AssertionError(
                        f"Gap of > {app_config.max_x_gap} bars found in {app_config.under_process_exchange}"
                        f"/{app_config.under_process_symbol}/{timeframe}:"
                        f"{slice_df.index.min()}-{slice_df.index.max()}")
        except AssertionError as e:
            log_d(e)
            continue
        sc_double_slice = scale_ohlc(double_slice, scaler_price_shift, scaler_price_scale, volume_scale)
        sc_trigger_slice = scale_ohlc(trigger_slice, scaler_price_shift, scaler_price_scale, volume_scale)
        sc_pattern_slice = scale_ohlc(pattern_slice, scaler_price_shift, scaler_price_scale, volume_scale)
        sc_structure_slice = scale_ohlc(structure_slice, scaler_price_shift, scaler_price_scale, volume_scale)
        sc_indicators_slice = scale_indicators(indicators_slice, scaler_price_shift, scaler_price_scale, volume_scale)
        sc_prediction = scale_prediction(prediction, scaler_price_shift, scaler_price_scale, )
        x_dfs['double'].append(sc_double_slice)
        x_dfs['trigger'].append(sc_trigger_slice)
        x_dfs['pattern'].append(sc_pattern_slice)
        x_dfs['structure'].append(sc_structure_slice)
        x_dfs['indicators'].append(sc_indicators_slice)
        Xs['double'].append(np.array(sc_double_slice[training_x_columns]))
        Xs['trigger'].append(np.array(sc_trigger_slice[training_x_columns]))
        Xs['pattern'].append(np.array(sc_pattern_slice[training_x_columns]))
        Xs['structure'].append(np.array(sc_structure_slice[training_x_columns]))
        for timeframe in sc_indicators_slice:
            for indicator_column in classic_indicator_columns():
                Xs[f'{timeframe}-{indicator_column}'].append(
                    np.array(sc_indicators_slice[timeframe][indicator_column])
                )
        y_dfs.append(sc_prediction)
        sc_prediction_testing_slice = \
            scale_ohlc(prediction_testing_slice, scaler_price_shift, scaler_price_scale, volume_scale)
        y_tester_dfs.append(sc_prediction_testing_slice)
        ys.append(np.array(y_dfs[-1]))
        batch_remained -= 1
    # Xs['double'] = np.array(Xs['double'])
    # Xs['trigger'] = np.array(Xs['trigger'])
    # Xs['pattern'] = np.array(Xs['pattern'])
    # Xs['structure'] = np.array(Xs['structure'])
    # for timeframe in sc_indicators_slice:
    #     for indicator_column in classic_indicator_columns:
    #         Xs[f'{timeframe}-{indicator_column}'] = np.array(Xs[f'{timeframe}-{indicator_column}'])
    ys = np.array(ys)
    return Xs, ys, x_dfs, y_dfs, trigger_tf, y_tester_dfs


def plot_classic_indicators(fig: go.Figure, x_dfs) -> go.Figure:
    scalable_indicators = list(set(classic_indicator_columns()) - set(scaleless_indicators()))
    for indicator_column in scaleless_indicators():
        fig.add_scatter(x_dfs[f'{indicator_column}'], row=2, line=dict(color='blue'))
    for indicator_column in scalable_indicators:
        fig.add_scatter(x_dfs[f'{indicator_column}'], row=1, line=dict(color='blue'))
    return fig


def plot_train_data_of_mt_n_profit(x_dfs: dict[str, List[pd.DataFrame]], y_dfs: List[pd.DataFrame],
                                   y_tester_dfs: List[pd.DataFrame], n: int, ):
    # training_y_columns = ['long_signal', 'short_signal', 'min_low', 'max_high', 'long_profit', 'short_profit',
    #                       'long_risk', 'short_risk']
    # fig = go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.7, 0.3])
    plot_mt_charts(fig, n, x_dfs)
    fig = plot_classic_indicators(fig, x_dfs)
    plot_prediction_verifiyer(fig, n, y_tester_dfs)
    plot_prediction(fig, n, y_dfs)
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


def scale_indicators(df, price_shift, price_scale, volume_scale):
    columns_to_scale = [
        'bbands_upper', 'bbands_middle', 'bbands_lower',
        'ichimoku_conversion', 'ichimoku_base', 'ichimoku_lead_a', 'ichimoku_lead_b', 'ichimoku_lagging'
    ]
    for timeframe in df:
        for column in columns_to_scale:
            df[column][timeframe] = (df[column][timeframe] + price_shift) * price_scale
    return df


def scaler_trainer(slices: Dict[str, pd.DataFrame], mean_atr: float, cloase: float):
    price_scale = (1 / mean_atr)
    price_shift = - cloase
    t_slice = pd.concat(slices)
    volume_scale = 1 / t_slice['volume'].mean()
    return price_scale, price_shift, volume_scale


model_dataset_lengths = {
    'structure': (128, 5),
    'pattern': (256, 5),
    'trigger': (256, 5),
    'double': (256, 5),
}

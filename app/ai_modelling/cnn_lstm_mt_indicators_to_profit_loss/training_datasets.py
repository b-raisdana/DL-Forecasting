import hashlib
import json
import os
import pickle
import random
import re
import textwrap
import zipfile
from datetime import timedelta, datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from Config import app_config
from FigurePlotter.plotter import show_and_save_plot
from PanderaDFM import MultiTimeframe
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.classic_indicators import add_classic_indicators, \
    classic_indicator_columns, scaleless_indicators
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.profit_loss.profit_loss_adder import \
    add_long_n_short_profit
from helper.br_py.br_py.do_log import log_d
from helper.data_preparation import pattern_timeframe, trigger_timeframe, single_timeframe
from helper.functions import date_range, get_size
from helper.importer import pt


def slice_indicators(timeframes_df_dict: Dict[str, pd.DataFrame], end_time: datetime, length: int) \
        -> Dict[str, pd.DataFrame]:
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


def single_timeframe_n_indicators(mt_ohlcv: pt.DataFrame[MultiTimeframe], timeframe: str) -> pd.DataFrame:
    ohlcv = single_timeframe(mt_ohlcv, timeframe)
    ohlcv = add_classic_indicators(ohlcv)
    return ohlcv


# @profile_it
def train_data_of_mt_n_profit(structure_tf: str, mt_ohlcv: pt.DataFrame[MultiTimeframe],
                              x_shape: Dict[str, Tuple[int, int]], batch_size: int, dataset_batches: int = 100,
                              forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1,
                              only_actionable: bool = True
                              ) \
        -> Tuple[
            Dict[str, np.ndarray], np.ndarray, Dict[str, pd.DataFrame], List[pd.DataFrame], str, List[pd.DataFrame]]:
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
    train_safe_end, train_safe_start = not_na_range(dfs)
    duration_seconds = int((train_safe_end - train_safe_start) / timedelta(seconds=1))
    if duration_seconds <= 0:
        start, end = date_range(app_config.processing_date_range)
        raise RuntimeError(
            f"Extend date boundary +{-duration_seconds}s({duration_seconds / (60 * 60 * 24)}days, "
            f"start:{start}<{start + duration_seconds * timedelta(seconds=1)} or "
            f"end:{end}>{end - duration_seconds * timedelta(seconds=1)}) to make possible range of end dates positive!")
    x_dfs, y_dfs, y_tester_dfs = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, [], []
    Xs: Dict[str, List[pd.DataFrame]]
    Xs, ys = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, []
    for timeframe in ['structure', 'pattern', 'trigger', 'double']:
        x_dfs[f'{timeframe}-indicators'] = []
        Xs[f'{timeframe}-indicators'] = []  # np.ndarray([])
    remained_samples = batch_size * dataset_batches
    while remained_samples > 0:
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
                or get_shape(sc_prediction_testing_slice) != (forecast_trigger_bars, 5)
        ):
            log_d(f'Skipped by:'
                  + ("len(np.array(sc_double_slice[training_x_columns])) != x_shape['double'][0]"
                     if len(np.array(sc_double_slice[training_x_columns])) != x_shape['double'][0] else "")
                  + ("len(np.array(sc_trigger_slice[training_x_columns])) != x_shape['trigger'][0]"
                     if len(np.array(sc_trigger_slice[training_x_columns])) != x_shape['trigger'][0] else "")
                  + ("len(np.array(sc_pattern_slice[training_x_columns])) != x_shape['pattern'][0]"
                     if len(np.array(sc_pattern_slice[training_x_columns])) != x_shape['pattern'][0] else "")
                  + ("len(np.array(sc_structure_slice[training_x_columns])) != x_shape['structure'][0]"
                     if len(np.array(sc_structure_slice[training_x_columns])) != x_shape['structure'][0] else "")
                  + ("get_shape(sc_prediction_testing_slice) != (forecast_trigger_bars, 5)"
                     if get_shape(sc_prediction_testing_slice) != (forecast_trigger_bars, 5) else "")
                  )
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
        remained_samples -= 1
        if (remained_samples % 10) == 0 and remained_samples > 0:
            log_d(f'Remained Samples {remained_samples}/{batch_size}')
    # converting list of batches to a combined ndarray
    try:
        # for key in Xs:
        #     Xs[key] = np.array(Xs[key])
        Xs['double'] = np.array(Xs['double'])
        Xs['trigger'] = np.array(Xs['trigger'])
        Xs['pattern'] = np.array(Xs['pattern'])
        Xs['structure'] = np.array(Xs['structure'])
        for timeframe in sc_indicators_slice:
            Xs[f'{timeframe}-indicators'] = np.array(Xs[f'{timeframe}-indicators'])
        ys = np.array(ys)
    except Exception as e:
        raise e
    shape_assertion(Xs=Xs, x_dfs=x_dfs, y_dfs=y_dfs, y_tester_dfs=y_tester_dfs, ys=ys, x_shape=x_shape,
                    batch_size=batch_size, dataset_batched=dataset_batches, forecast_trigger_bars=forecast_trigger_bars)
    try:
        for key in Xs:
            if np.isnan(Xs[key]).any():
                raise AssertionError(f"Found NA in Xs[{key}]")
        if any([np.isnan(y).any() for y in ys]):
            raise AssertionError(f"Found NA in ys")
    except Exception as e:
        raise e
    return Xs, ys, x_dfs, y_dfs, trigger_tf, y_tester_dfs


def shape_assertion(Xs: Dict[str, np.ndarray], x_dfs: Dict[str, List[pd.DataFrame]], y_dfs: List[pd.DataFrame],
                    y_tester_dfs: List[pd.DataFrame], ys: np.ndarray, x_shape: Dict[str, Tuple[int, int]],
                    batch_size: int = 120, dataset_batched: int = 100, forecast_trigger_bars: int = 192) -> None:
    """
    x_shape = {'double': (255, 5), 'indicators': (129,), 'pattern': (253, 5), 'structure': (127, 5), 'trigger': (254, 5)}
    """
    b_l = batch_size * dataset_batched
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
        raise AssertionError("get_shape(y_tester_dfs) != [b_l, (forecast_trigger_bars, 5)]")  # todo: this happens!


def x_shape_assertion(Xs: Dict[str, np.ndarray], batch_size: int, x_shape: Dict[str, Tuple[int, int]],
                      num_of_indicators: int = 12) -> None:
    i_l = x_shape['indicators'][0]
    b_l = batch_size
    if get_shape(Xs) != {
        'double': (b_l, x_shape['double'][0], 5), 'double-indicators': (b_l, i_l, num_of_indicators),
        'pattern': (b_l, x_shape['pattern'][0], 5), 'pattern-indicators': (b_l, i_l, num_of_indicators),
        'structure': (b_l, x_shape['structure'][0], 5), 'structure-indicators': (b_l, i_l, num_of_indicators),
        'trigger': (b_l, x_shape['trigger'][0], 5), 'trigger-indicators': (b_l, i_l, num_of_indicators)}:
        raise AssertionError("get_shape(Xs) != {")


def not_na_range(dfs: Dict[str, pd.DataFrame]) -> Tuple[datetime, datetime]:
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


def scaling(structure_slice: pd.DataFrame, pattern_slice: pd.DataFrame, trigger_slice: pd.DataFrame,
            double_slice: pd.DataFrame, prediction: pd.DataFrame, indicators_slice: Dict[str, pd.DataFrame],
            prediction_testing_slice: pd.DataFrame, training_x_columns: List[str]) \
        -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame]:
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


def batch_ends(duration_seconds: int, double_tf: str, trigger_tf: str, pattern_tf: str,
               x_shape: Dict[str, Tuple[int, int]],
               train_safe_end: datetime) -> Tuple[datetime, datetime, datetime, datetime]:
    relative_double_end = np.random.randint(0, duration_seconds)
    double_end: datetime = train_safe_end - relative_double_end * timedelta(seconds=1)
    trigger_end = double_end - x_shape['double'][0] * pd.to_timedelta(double_tf)
    pattern_end = trigger_end - x_shape['trigger'][0] * pd.to_timedelta(trigger_tf)
    structure_end = pattern_end - x_shape['pattern'][0] * pd.to_timedelta(pattern_tf)
    return double_end, trigger_end, pattern_end, structure_end


def slicing(dfs: Dict[str, pd.DataFrame], structure_end: datetime, pattern_end: datetime, trigger_end: datetime,
            double_end: datetime, training_x_columns: List[str], x_shape: Dict[str, Tuple[int, int]]
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
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


def plot_classic_indicators(fig: go.Figure, x_dfs: Dict[str, List[pd.DataFrame]], n: int) -> go.Figure:
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


def plot_train_data_of_mt_n_profit(x_dfs: Dict[str, List[pd.DataFrame]], y_dfs: List[pd.DataFrame],
                                   y_tester_dfs: List[pd.DataFrame], n: int) -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,  # vertical_spacing=0.02,
                        row_heights=[0.65, 0.25])
    plot_mt_charts(fig, n, x_dfs)
    fig = plot_classic_indicators(fig, x_dfs, n)
    plot_prediction_verifier(fig, n, y_tester_dfs)
    plot_prediction(fig, n, y_dfs)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
    show_and_save_plot(fig.update_yaxes(fixedrange=False))


def plot_mt_charts(fig: go.Figure, n: int, x_dfs: Dict[str, List[pd.DataFrame]]) -> None:
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


def plot_prediction_verifier(fig: go.Figure, n: int, y_tester_dfs: List[pd.DataFrame]) -> None:
    ohlcv = y_tester_dfs[n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        close=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        open=ohlcv['high'],
        name='Y'
    ))


def plot_prediction(fig: go.Figure, n: int, y_dfs: List[pd.DataFrame]) -> None:
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


def scale_prediction(prediction: pd.DataFrame, price_scaler_shift: float, price_scaler_size: float) -> pd.DataFrame:
    prediction['min_low'] = (prediction['min_low'] + price_scaler_shift) * price_scaler_size
    prediction['max_high'] = (prediction['max_high'] + price_scaler_shift) * price_scaler_size
    prediction['long_profit'] = (prediction['long_profit']) * price_scaler_size
    prediction['short_profit'] = (prediction['short_profit']) * price_scaler_size
    return prediction


def scale_ohlc(df: pd.DataFrame, price_shift: float, price_scale: float, volume_scale: float) -> pd.DataFrame:
    df = df.copy()
    for column in ['open', 'high', 'low', 'close']:
        df[column] = (df[column] + price_shift) * price_scale
    df['volume'] = df['volume'] * volume_scale
    return df


def scale_indicators(indicator_df: Dict[str, pd.DataFrame], price_shift: float, price_scale: float, obv_scale: float,
                     obv_shift: float) -> Dict[str, pd.DataFrame]:
    columns_to_scale = set(classic_indicator_columns()) - set(scaleless_indicators())
    try:
        for timeframe in indicator_df:
            for column in columns_to_scale:
                indicator_df[timeframe][column] = (indicator_df[timeframe][column] + price_shift) * price_scale
            indicator_df[timeframe]['obv'] = (indicator_df[timeframe]['obv'] + obv_shift) * obv_scale
    except Exception as e:
        raise e
    return indicator_df


def scaler_trainer(slices: Dict[str, pd.DataFrame], mean_atr: float, close: float) -> Tuple[
    float, float, float, float, float]:
    price_scale = (1 / mean_atr)
    price_shift = - close
    t_slice = pd.concat(slices)
    volume_scale = 1 / t_slice['volume'].mean()
    obv_shift = - slices['trigger-obv'].iloc[-1]
    obv_scale = 50 / ((slices['trigger-obv'].max() - slices['trigger-obv'].min()) / 2)
    return price_scale, price_shift, volume_scale, obv_scale, obv_shift


master_x_shape = {
    'structure': (127, 5),
    'pattern': (253, 5),
    'trigger': (254, 5),
    'double': (255, 5),
    'indicators': (129, 12),
}


def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[\s]', '', filename)
    filename = re.sub(r'[{}\[\]<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'_+', '_', filename)  # collapse multiple underscores
    filename = re.sub(r'^_', '', filename)  # collapse multiple underscores
    filename = re.sub(r'_$', '', filename)  # collapse multiple underscores
    filename = filename.replace('_,_', '_')  # collapse multiple underscores
    return filename


def dataset_folder(x_shape: Dict[str, Tuple[int, int]], batch_size: int, create: bool = False) -> str:
    serialized = json.dumps({"x_shape": x_shape, "batch_size": batch_size})
    folder_name = sanitize_filename(serialized)
    folder_path = os.path.join(app_config.path_of_data, folder_name)
    if create and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_name


def save_validators_to_zip(X_dfs: Dict[str, List[pd.DataFrame]], y_dfs: List[pd.DataFrame], y_timeframe: str,
                           y_tester_dfs: List[pd.DataFrame], folder_name: str, symbol: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    zip_file_name = f"validators-{symbol}-{timestamp}.zip"
    zip_file_path = os.path.join(app_config.path_of_data, folder_name, zip_file_name)

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.writestr('X_dfs.pkl', pickle.dumps(X_dfs))
        zipf.writestr('y_dfs.pkl', pickle.dumps(y_dfs))
        zipf.writestr('y_timeframe.pkl', pickle.dumps(y_timeframe))
        zipf.writestr('y_tester_dfs.pkl', pickle.dumps(y_tester_dfs))


def generate_batch(batch_size: int, mt_ohlcv: pt.DataFrame[MultiTimeframe],
                   x_shape: Dict[str, Tuple[int, int]]) -> None:
    Xs, ys, X_dfs, y_dfs, y_timeframe, y_tester_dfs = (
        train_data_of_mt_n_profit(
            structure_tf='4h', mt_ohlcv=mt_ohlcv, x_shape=x_shape, batch_size=batch_size, dataset_batches=2,
            forecast_trigger_bars=3 * 4 * 4 * 4 * 1, only_actionable=True, ))
    folder_name = dataset_folder(x_shape, batch_size, create=True)
    save_batch_zip(Xs, ys, folder_name, app_config.under_process_symbol)
    save_validators_to_zip(X_dfs, y_dfs, y_timeframe, y_tester_dfs, folder_name,
                           app_config.under_process_symbol)
    #     plot_train_data_of_mt_n_profit(X_dfs, y_dfs, y_tester_dfs, i)


def save_batch_zip(Xs: Dict[str, np.ndarray], ys: np.ndarray, folder_name: str, symbol: str, ) -> None:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    zip_file_name = f"dataset-{symbol}-{timestamp}.zip"
    zip_file_path = os.path.join(app_config.path_of_data, folder_name, zip_file_name)

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for key in Xs:
            zipf.writestr(f'Xs-{key}.npy', Xs[key].tobytes())
        zipf.writestr('ys.npy', ys.tobytes())


def read_batch_zip(x_shape: Dict[str, Tuple[int, int]], batch_size: int) \
        -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    folder_name = dataset_folder(x_shape, batch_size)
    folder_path: str = os.path.join(app_config.path_of_data, folder_name)

    files = [f for f in os.listdir(folder_path) if f.startswith('dataset-')]
    if not files:
        raise ValueError("No dataset files found!")

    picked_file = random.choice(files)
    file_path = os.path.join(app_config.path_of_data, folder_name, picked_file)
    Xs: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(file_path, 'r') as zipf:
        for name in zipf.namelist():
            if name.startswith('Xs-') and name.endswith('.npy'):
                key: str = name[3:-4]
                with zipf.open(name) as f:
                    arr: np.ndarray = np.frombuffer(f.read(), dtype=np.float32)
                    shape_key = 'indicators' if 'indicators' in key else key
                    Xs[key] = arr.reshape(-1, *x_shape[shape_key])
        with zipf.open('ys.npy') as f:
            ys: np.ndarray = np.frombuffer(f.read(), dtype=np.float32)
            # ys = ys.reshape(-1, *y_shape)

    log_d(f"Xs dataset size: {str(get_size(Xs))}")
    return Xs, ys

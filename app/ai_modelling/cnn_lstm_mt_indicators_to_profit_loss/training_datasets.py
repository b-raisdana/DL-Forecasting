import logging
import random
import sys
import textwrap
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
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.base import overlapped_quarters, master_x_shape, dataset_folder, \
    save_batch_zip, save_validators_zip
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.classic_indicators import add_classic_indicators, \
    classic_indicator_columns, scaleless_indicators
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.profit_loss.profit_loss_adder import \
    add_long_n_short_profit
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.base import sync_br_lib_init
from helper.br_py.br_py.do_log import log_d
from helper.data_preparation import pattern_timeframe, trigger_timeframe, single_timeframe
from helper.functions import date_range, date_range_to_string
from helper.importer import pt


# (tf) brais@Behrooz:/mnt/c/Code/DL-Forecasting$ PYTHONPATH=/mnt/c/Code/DL-Forecasting/app/ python /mnt/c/Code/DL-Forecasting/app/ai_modelling/cnn_lstm_mt_indicators_to_profit_loss/training_datasets.py

# def slice_indicators(timeframes_df_dict: Dict[str, pd.DataFrame], end_time: datetime, length: int) \
#         -> Dict[str, pd.DataFrame]:
#     try:
#         t_slice = {
#             df_name: pd.DataFrame({
#                 indicator_column: timeframe_df.loc[pd.IndexSlice[:end_time], indicator_column].iloc[-length:]
#                 for indicator_column in classic_indicator_columns()
#             })
#             for df_name, timeframe_df in timeframes_df_dict.items()
#         }
#     except Exception as e:
#         nop = 1
#         raise e
#
#     return t_slice


def single_timeframe_n_indicators(mt_ohlcv: pt.DataFrame[MultiTimeframe], timeframe: str) -> pd.DataFrame:
    ohlcv = single_timeframe(mt_ohlcv, timeframe)
    ohlcv = add_classic_indicators(ohlcv)
    return ohlcv


# @profile_it
def train_data_of_mt_n_profit(structure_tf: str, mt_ohlcv: pt.DataFrame[MultiTimeframe],
                              x_shape: Dict[str, Tuple[int, int]], batch_size: int, dataset_batches: int = 100,
                              forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1,
                              actionable_rate=0.2  # try to generate 20% actionable and 80% not-actionable batches
                              # only_actionable: bool = True
                              ) \
        -> Tuple[
            Dict[str, np.ndarray], np.ndarray, Dict[str, pd.DataFrame], List[pd.DataFrame], str, List[pd.DataFrame]]:
    training_x_columns = ['open', 'high', 'low', 'close', 'volume', ] + classic_indicator_columns()
    training_y_columns = ['long_signal', 'short_signal', 'min_low', 'max_high', 'long_profit', 'short_profit',
                          'long_risk', 'short_risk', 'long_drawdown', 'short_drawdown',
                          'long_drawdown', 'short_drawdown', ]
    pattern_tf = pattern_timeframe(structure_tf)
    trigger_tf = trigger_timeframe(structure_tf)
    double_tf = pattern_timeframe(trigger_timeframe(structure_tf))
    dfs: Dict[str, pd.DataFrame] = {}
    for df_name, timeframe in [('structure', structure_tf), ('pattern', pattern_tf),
                               ('trigger', trigger_tf), ('double', double_tf)]:
        dfs[df_name] = single_timeframe_n_indicators(mt_ohlcv, timeframe)
    dfs['trigger']['atr'] = ta.atr(high=dfs['trigger']['high'], low=dfs['trigger']['low'],
                                   close=dfs['trigger']['close'], length=256)
    dfs = dfs.copy()
    dfs['future'] = add_long_n_short_profit(ohlc=dfs['trigger'], position_max_bars=forecast_trigger_bars,
                                            trigger_tf=trigger_tf)

    train_safe_end, train_safe_start, dfs = not_na_range(dfs)
    train_safe_start += pd.to_timedelta(structure_tf)
    train_safe_end -= forecast_trigger_bars * pd.to_timedelta(trigger_tf)
    duration_seconds = int((train_safe_end - train_safe_start) / timedelta(seconds=1))
    if duration_seconds <= 0:
        start, end = date_range(app_config.processing_date_range)
        raise RuntimeError(
            f"Extend date boundary +{-duration_seconds}s({duration_seconds / (60 * 60 * 24)}days, "
            f"start:{start}<{start + duration_seconds * timedelta(seconds=1)} or "
            f"end:{end}>{end - duration_seconds * timedelta(seconds=1)}) to make possible range of end dates positive!")
    x_dfs, y_dfs, y_tester_dfs = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, [], []
    Xs: Dict[str, List[np.ndarray]]
    Xs, ys = {'double': [], 'trigger': [], 'pattern': [], 'structure': [], }, []
    # for timeframe in ['structure', 'pattern', 'trigger', 'double']:
    #     x_dfs[f'{timeframe}-indicators'] = []
    #     Xs[f'{timeframe}-indicators'] = []  # np.ndarray([])
    remained_samples = batch_size * dataset_batches
    actionable_batches = 0
    not_actionable_batches = 0
    while remained_samples > 0:
        # for relative_double_end in np.random.randint(0, duration_seconds, size=batch_size):
        double_end, trigger_end, pattern_end, structure_end = \
            batch_ends(duration_seconds, double_tf, trigger_tf, pattern_tf, x_shape, train_safe_end)
        # future = dfs['future'].loc[pd.IndexSlice[:double_end], training_y_columns].iloc[-1]
        future_slice = dfs['future'].loc[pd.IndexSlice[double_end:], training_y_columns].iloc[
                       :forecast_trigger_bars]
        if future_slice.shape[0] != forecast_trigger_bars:
            raise AssertionError(future_slice.shape[0] != forecast_trigger_bars)
        if future_slice['long_signal'][0] == 0 and future_slice['short_signal'][0] == 0:
            is_actionable = False
            if actionable_batches == 0 or (not_actionable_batches / actionable_batches > (1 / actionable_rate - 1)):
                continue
        else:
            is_actionable = True
        double_slice, pattern_slice, structure_slice, trigger_slice = \
            slicing(dfs, structure_end, pattern_end, trigger_end, double_end, training_x_columns, x_shape)
        # future_slice = (
        #     dfs['trigger'].loc[
        #         pd.IndexSlice[double_end: double_end + forecast_trigger_bars * pd.to_timedelta(trigger_tf)],
        #         training_x_columns])
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
            log_d(str(e))
            continue
        (sc_double_slice, sc_pattern_slice, sc_trigger_slice, sc_structure_slice, sc_future,
         ) = \
            normalize(structure_slice, pattern_slice, trigger_slice, double_slice, future_slice,  # indicators_slice,
                      training_x_columns)
        if (
                len(np.array(sc_double_slice[training_x_columns])) != x_shape['double'][0]
                or len(np.array(sc_trigger_slice[training_x_columns])) != x_shape['trigger'][0]
                or len(np.array(sc_pattern_slice[training_x_columns])) != x_shape['pattern'][0]
                or len(np.array(sc_structure_slice[training_x_columns])) != x_shape['structure'][0]
                or get_shape(sc_prediction_testing_slice) != (forecast_trigger_bars, 12)
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
                  + ("get_shape(sc_prediction_testing_slice) != (forecast_trigger_bars, 12)"
                     if get_shape(sc_prediction_testing_slice) != (forecast_trigger_bars, 12) else "")
                  )
            raise AssertionError
        x_dfs['double'].append(sc_double_slice[training_x_columns])
        x_dfs['trigger'].append(sc_trigger_slice[training_x_columns])
        x_dfs['pattern'].append(sc_pattern_slice[training_x_columns])
        x_dfs['structure'].append(sc_structure_slice[training_x_columns])
        # for timeframe in ['structure', 'pattern', 'trigger', 'double']:
        #     x_dfs[f'{timeframe}-indicators'].append(sc_indicators_slice[timeframe])
        Xs['double'].append(np.array(x_dfs['double'][-1]))
        Xs['trigger'].append(np.array(x_dfs['trigger'][-1]))
        Xs['pattern'].append(np.array(x_dfs['pattern'][-1]))
        Xs['structure'].append(np.array(x_dfs['structure'][-1]))
        # for timeframe in ['structure', 'pattern', 'trigger', 'double']:
        #     Xs[f'{timeframe}-indicators'].append(np.array(x_dfs[f'{timeframe}-indicators'][-1]))
        y_dfs.append(sc_future)
        y_tester_dfs.append(sc_prediction_testing_slice)
        ys.append(np.array(y_dfs[-1].iloc[0][['short_signal', 'long_signal']]))
        remained_samples -= 1
        if is_actionable:
            actionable_batches += 1
        else:
            not_actionable_batches += 1
        if (remained_samples % 10) == 0 and remained_samples > 0:
            log_d(f'Remained Samples {remained_samples}/{batch_size}')
    # converting list of batches to a combined ndarray
    try:
        # for key in Xs:
        #     Xs[key] = np.array(Xs[key])
        for key in x_dfs:
            x_dfs[key] = pd.concat(x_dfs[key])
        Xs['double'] = np.array(Xs['double'])
        Xs['trigger'] = np.array(Xs['trigger'])
        Xs['pattern'] = np.array(Xs['pattern'])
        Xs['structure'] = np.array(Xs['structure'])
        # for timeframe in sc_indicators_slice:
        #     Xs[f'{timeframe}-indicators'] = np.array(Xs[f'{timeframe}-indicators'])
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
                    batch_size: int = 120, dataset_batched: int = 100, forecast_trigger_bars: int = 192,
                    y_parameters=2, y_df_parameters=12) -> None:
    """
    x_shape = {'double': (255, 5), 'indicators': (129,), 'pattern': (253, 5), 'structure': (127, 5), 'trigger': (254, 5)}
    """
    b_l = batch_size * dataset_batched
    # i_l = x_shape['indicators']
    x_shape_assertion(Xs, b_l, x_shape)
    if get_shape(ys) != (b_l, y_parameters):
        raise AssertionError(f"get_shape(ys) != (b_l, {y_parameters})")
    from deepdiff import DeepDiff
    if DeepDiff(get_shape(x_dfs), {
        'double': (b_l * x_shape['double'][0], x_shape['double'][1]),
        'trigger': (b_l * x_shape['trigger'][0], x_shape['trigger'][1]),
        'pattern': (b_l * x_shape['pattern'][0], x_shape['pattern'][1]),
        'structure': (b_l * x_shape['structure'][0], x_shape['structure'][1]),
    }) != {}:
        raise AssertionError("DeepDiff(get_shape(x_dfs), {")
    if get_shape(y_dfs) != [b_l, (forecast_trigger_bars, y_df_parameters,)]:
        raise AssertionError(f"get_shape(y_dfs) != [b_l, ({y_parameters},)]")
    if get_shape(y_tester_dfs) != [b_l, (forecast_trigger_bars, 5)]:
        raise AssertionError("get_shape(y_tester_dfs) != [b_l, (forecast_trigger_bars, 5)]")  # todo: this happens!


def x_shape_assertion(Xs: Dict[str, np.ndarray], batch_size: int, x_shape: Dict[str, Tuple[int, int]],
                      num_of_indicators: int = 12) -> None:
    # i_l = x_shape['indicators'][0]
    b_l = batch_size
    if get_shape(Xs) != {
        'double': (b_l, x_shape['double'][0], 5 + num_of_indicators),
        # 'double-indicators': (b_l, i_l, num_of_indicators),
        'pattern': (b_l, x_shape['pattern'][0], 5 + num_of_indicators),
        # 'pattern-indicators': (b_l, i_l, num_of_indicators),
        'structure': (b_l, x_shape['structure'][0], 5 + num_of_indicators),
        # 'structure-indicators': (b_l, i_l, num_of_indicators),
        'trigger': (b_l, x_shape['trigger'][0], 5 + num_of_indicators),
        # 'trigger-indicators': (b_l, i_l, num_of_indicators)
    }:
        raise AssertionError("get_shape(Xs) != {")


def not_na_range(dfs: Dict[str, pd.DataFrame]) -> Tuple[datetime, datetime, pd.DataFrame]:
    future_end = dfs['future'].dropna(how='any', axis=0).index[-1]
    double_end = dfs['double'].dropna(how='any', axis=0).index[-1]
    train_safe_start = dfs['structure'].dropna(how='any', axis=0).index[1]
    train_safe_end = min(future_end, double_end)
    # train_safe_start, train_safe_end = (None, None)
    # for df_name in dfs:
    #     df = dfs[df_name]
    #     not_na_df = df.dropna(how='any')
    #     not_na_start = not_na_df.index.get_level_values(level='date').min()
    #     not_na_end = not_na_df.index.get_level_values(level='date').max()
    #     if train_safe_start is None or train_safe_start < not_na_start:
    #         train_safe_start = not_na_start
    #     if train_safe_end is None or train_safe_end > not_na_end:
    #         train_safe_end = not_na_end
    #     nop = 1
    for df_name in dfs:
        dfs[df_name] = dfs[df_name].loc[pd.IndexSlice[train_safe_start:train_safe_end, :]].dropna(how='any', axis=0)
        if dfs[df_name].isna().any().any():
            raise AssertionError(f"Found NA in dfs[{df_name}]")
    return train_safe_end, train_safe_start, dfs


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


def scale_slice(slc: pd.DataFrame, price_shift, price_scale, volume_scale,
                # obv_shift, obv_scale, cci_scale, cci_shift,
                ) -> pd.DataFrame:
    t = slc.copy()
    for column in ['open', 'high', 'low', 'close']:
        if column in slc.columns:
            t[column] = (t[column] + price_shift) * price_scale
    if 'volume' in slc.columns:
        t['volume'] = t['volume'] * volume_scale
        t['rsi'] = t['rsi'] - 50
        t['mfi'] = t['mfi'] - 50

        # t['obv'] = (t['obv'] + obv_shift) * obv_scale
        # t['cci'] = (t['cci'] + cci_shift) * cci_scale
        t['obv'] = 10 * np.tanh(t['obv'])  # + obv_shift) * obv_scale
        t['cci'] = 10 * np.tanh(t['cci'])  # + cci_shift) * cci_scale
        columns_to_scale = set(classic_indicator_columns()) - set(scaleless_indicators())
        for column in columns_to_scale:
            if column in slc.columns:
                t[column] = (t[column] + price_shift) * price_scale
    return t


def normalize(structure_slice: pd.DataFrame, pattern_slice: pd.DataFrame, trigger_slice: pd.DataFrame,
              double_slice: pd.DataFrame, future_slice: pd.DataFrame,
              # indicators_slice: Dict[str, pd.DataFrame],future_slice: pd.DataFrame, training_x_columns: List[str]
              ) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,]:
    (price_scale, price_shift, volume_scale,
     # obv_scale,obv_shift, cci_scale, cci_shift
     ) = scaler_trainer(
        {'double': double_slice, 'pattern': pattern_slice, 'structure': structure_slice, 'trigger': trigger_slice, },
        mean_atr=trigger_slice['atr'].mean(),
        close=double_slice.iloc[-1]['close'], )
    (sc_double_slice, sc_trigger_slice, sc_pattern_slice, sc_structure_slice, sc_future_slice) = \
        (scale_slice(t, price_shift, price_scale, volume_scale,
                     # obv_shift, obv_scale, cci_scale, cci_shift,
                     )
         for t in [double_slice, trigger_slice, pattern_slice, structure_slice, future_slice])
    # sc_indicators_slice = scale_indicators(indicators_slice, price_shift, price_scale, obv_scale, obv_shift)
    sc_prediction = scale_future(future_slice, price_shift, price_scale, )
    return sc_double_slice, sc_pattern_slice, sc_trigger_slice, sc_structure_slice, sc_future_slice  # , sc_prediction_testing_slice  # sc_indicators_slice,


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
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,]:
    double_slice = dfs['double'].loc[pd.IndexSlice[: double_end], training_x_columns].iloc[
                   -x_shape['double'][0]:]
    trigger_slice = dfs['trigger'].loc[
                        pd.IndexSlice[: trigger_end], training_x_columns + ['atr']].iloc[
                    -x_shape['trigger'][0]:]
    pattern_slice = dfs['pattern'].loc[pd.IndexSlice[: pattern_end], training_x_columns].iloc[
                    -x_shape['pattern'][0]:]
    structure_slice = dfs['structure'].loc[pd.IndexSlice[: structure_end], training_x_columns].iloc[
                      -x_shape['structure'][0]:]
    # indicators_slice = slice_indicators(timeframes_df_dict=dfs, end_time=double_end, length=x_shape['indicators'][0])
    if double_slice.isna().any().any():
        raise AssertionError(f"double_slice.isna().any().any()")
    if trigger_slice.isna().any().any():
        raise AssertionError(f"rigger_slice.isna().any().any()")
    if pattern_slice.isna().any().any():
        raise AssertionError("pattern_slice.isna().any().any()")
    if structure_slice.isna().any().any():
        raise AssertionError("structure_slice.isna().any().any()")
    # assert all([level_indicators.notna().any().any()
    #             for level, level_indicators in indicators_slice.items()])
    return double_slice, pattern_slice, structure_slice, trigger_slice  # , indicators_slice


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


def scale_future(future: pd.DataFrame, price_scaler_shift: float, price_scaler_size: float) -> pd.DataFrame:
    future['min_low'] = (future['min_low'] + price_scaler_shift) * price_scaler_size
    future['max_high'] = (future['max_high'] + price_scaler_shift) * price_scaler_size
    future['long_profit'] = (future['long_profit']) * price_scaler_size
    future['short_profit'] = (future['short_profit']) * price_scaler_size
    return future


def scaler_trainer(slices: Dict[str, pd.DataFrame], mean_atr: float, close: float) -> Tuple[
    float, float, float,]:
    price_scale = (1 / mean_atr)
    price_shift = - close
    t_slice = pd.concat(slices)
    volume_scale = 1 / t_slice['volume'].mean()
    # obv_shift = - slices['trigger']['obv'].iloc[-1]
    # obv_scale = 50 / ((slices['trigger']['obv'].max() - slices['trigger']['obv'].min()) / 2)
    # cci_shift = - slices['trigger']['cci'].iloc[-1]
    # cci_scale = 50 / ((slices['trigger']['cci'].max() - slices['trigger']['cci'].min()) / 2)
    return price_scale, price_shift, volume_scale,  # obv_scale, obv_shift, cci_scale, cci_shift


def generate_batch(batch_size: int, mt_ohlcv: pt.DataFrame[MultiTimeframe],
                   x_shape: Dict[str, Tuple[int, int]], save=False):
    Xs, ys, X_dfs, y_dfs, y_timeframe, y_tester_dfs = (
        train_data_of_mt_n_profit(
            structure_tf='4h', mt_ohlcv=mt_ohlcv, x_shape=x_shape, batch_size=batch_size, dataset_batches=1,
            forecast_trigger_bars=3 * 4 * 4 * 4 * 1, ))
    folder_name = dataset_folder(x_shape, batch_size, create=True)
    if save:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_batch_zip(Xs, ys, folder_name, app_config.under_process_symbol, timestamp)
        save_validators_zip(X_dfs, y_dfs, y_timeframe, y_tester_dfs, folder_name,
                            app_config.under_process_symbol, timestamp)
    return Xs, ys, X_dfs, y_dfs, y_timeframe, y_tester_dfs
    #     plot_train_data_of_mt_n_profit(X_dfs, y_dfs, y_tester_dfs, i)


# def dataset_scale(batch_size: int,
#                   # mt_ohlcv: pt.DataFrame[MultiTimeframe],
#                   x_shape: Dict[str, Tuple[int, int]], number_of_batches=100):
#     all_ys = None  # We'll accumulate all ys arrays here
#
#     for i in range(number_of_batches):
#         Xs, ys = load_batch_zip(x_shape, batch_size, n=i)
#         if i == 0:
#             X_dfs, y_dfs, y_timeframe, y_tester_dfs = load_validators_zip(x_shape, batch_size, n=i)
#
#         if i == 0:
#             all_ys = ys
#         else:
#             all_ys = np.concatenate([all_ys, ys], axis=0)
#     names = y_dfs[0].axes[0].to_list()
#     ys_mean, ys_std = ndarray_stats(all_ys, names)
#     return (
#         # np.mean(Xs_batches, axis=0), np.std(Xs_batches, axis=0),
#         ys_mean, ys_std
#     )


def ndarray_stats(input_array: np.ndarray, names):
    ys_mean = np.mean(input_array, axis=0)
    ys_std = np.std(input_array, axis=0)
    ys_min = np.min(input_array, axis=0)
    ys_max = np.max(input_array, axis=0)
    df_stats = pd.DataFrame({
        'min': ys_min,
        'max': ys_max,
        'mean': ys_mean,
        'std': ys_std
    }, index=names)
    return df_stats


def training_dataset_main():
    log_d("Starting")
    sync_br_lib_init(path_of_logs='logs', root_path=app_config.root_path, log_to_file_level=logging.DEBUG,
                     log_to_std_out_level=logging.DEBUG)
    # parser = argparse.ArgumentParser(description="Script for processing OHLCV data.")
    # args = parser.parse_args()
    app_config.processing_date_range = date_range_to_string(start=pd.to_datetime('03-01-24'),
                                                            end=pd.to_datetime('09-01-24'))
    quarters = overlapped_quarters(app_config.processing_date_range)
    mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)
    batch_size = 100 * 4

    # parser.add_argument("--do_not_fetch_prices", action="store_true", default=False,
    #                     help="Flag to indicate if prices should not be fetched (default: False).")
    print("Python:" + sys.version)

    # Apply config from arguments
    app_config.processing_date_range = "22-08-15.00-00T24-10-30.00-00"
    # config.do_not_fetch_prices = args.do_not_fetch_prices
    # seed(42)
    # np.random.seed(42)

    while True:
        random.shuffle(quarters)
        for start, end in quarters:
            log_d(f'quarter start:{start} end:{end}##########################################')
            app_config.processing_date_range = date_range_to_string(start=start, end=end)
            for symbol in [
                'BTCUSDT',
                # # # 'ETHUSDT',
                # 'BNBUSDT',
                # 'EOSUSDT',
                # # 'TRXUSDT',
                # 'TONUSDT',
                # # 'SOLUSDT',
            ]:
                log_d(f'Symbol:{symbol}##########################################')
                app_config.under_process_symbol = symbol
                generate_batch(batch_size, mt_ohlcv, master_x_shape)


if __name__ == "__main__":
    training_dataset_main()

# todo: check the dataset files to check if input_y is reperesting good results?

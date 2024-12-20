import numpy as np
from plotly.subplots import make_subplots
from app.helper.importer import go


def max_profit_n_loss(ohlc, position_max_days, action_delay, rolling_window):
    ohlc['worst_long_open'] = (
        ohlc['High'].rolling(window=action_delay, min_periods=1).max().shift(1 - action_delay))
    ohlc['worst_short_open'] = (
        ohlc['Low'].rolling(window=action_delay, min_periods=1).min().shift(1 - action_delay))
    ohlc['max_high'] = ohlc['High'].rolling(rolling_window, min_periods=1).max().shift(
        -position_max_days)
    ohlc['min_low'] = ohlc['Low'].rolling(rolling_window, min_periods=1).min().shift(
        -position_max_days)

    ohlc['max_high_distance'] = ohlc['High'].rolling(rolling_window).apply(lambda x: x.argmax(), raw=True).shift(
        -position_max_days) + action_delay
    ohlc['min_low_distance'] = ohlc['Low'].rolling(rolling_window).apply(lambda x: x.argmin(), raw=True).shift(
        -position_max_days) + action_delay
    return ohlc


def quantile_maxes(ohlc, rolling_window, quantiles=10):
    for i in range(1, quantiles + 1):
        q_rolling_window = int(i * rolling_window / quantiles)
        ohlc[f'q{i}_max_high'] = \
            ohlc['High'].rolling(q_rolling_window, min_periods=q_rolling_window).max().shift(-q_rolling_window)
        ohlc[f'q{i}_min_low'] = \
            ohlc['Low'].rolling(q_rolling_window, min_periods=q_rolling_window).min().shift(-q_rolling_window)

        ohlc[f'q{i}_max_high_distance'] = (ohlc['High'].rolling(q_rolling_window)
                                           .apply(lambda x: x.argmax(), raw=True).shift(-q_rolling_window))
        ohlc[f'q{i}_min_low_distance'] = (ohlc['Low'].rolling(q_rolling_window)
                                          .apply(lambda x: x.argmin(), raw=True).shift(-q_rolling_window))
    return ohlc


def long_n_short_drawdown(ohlc, position_max_days, quantiles):
    not_na_indexes = ohlc[~ohlc.isna().any(axis='columns')].index
    ohlc.loc[not_na_indexes, 'max_high_quantile'] = (
            ohlc.loc[not_na_indexes, 'max_high_distance'] / (position_max_days / quantiles))
    ohlc.loc[not_na_indexes, 'min_low_quantile'] = (
            ohlc.loc[not_na_indexes, 'min_low_distance'] / (position_max_days / quantiles))
    min_low_np_a = ohlc.loc[not_na_indexes, [f'q{i}_min_low' for i in range(1, quantiles + 1)]].to_numpy()
    quantile_max_high_consistency = ohlc.loc[not_na_indexes, 'max_high_quantile'].astype(int).eq(0)
    consistent_quantile_max_high = quantile_max_high_consistency[~quantile_max_high_consistency].index
    ohlc.loc[not_na_indexes, 'quantile_long_min_low'] = min_low_np_a[
        np.arange(len(ohlc.loc[not_na_indexes])), ohlc.loc[not_na_indexes, 'max_high_quantile'].astype(int)]
    # ohlc.loc[consistent_quantile_max_high, 'pre_quantile_long_min_low'] = min_low_np_a[
    #     np.arange(len(consistent_quantile_max_high)), ohlc.loc[consistent_quantile_max_high, 'max_high_quantile'].astype(
    #         int) - 1]
    # ohlc.loc[consistent_quantile_max_high, 'long_drawdown_consistency'] = (
    #         ohlc.loc[consistent_quantile_max_high, 'pre_quantile_long_min_low']
    #         == ohlc.loc[consistent_quantile_max_high, 'quantile_long_min_low'])
    ohlc['long_drawdown'] = \
        (ohlc['worst_long_open'] - ohlc['quantile_long_min_low']) / ohlc['worst_long_open']
    max_high_np_a = ohlc.loc[not_na_indexes, [f'q{i}_max_high' for i in range(1, quantiles + 1)]].to_numpy()
    quantile_min_low_consistency = ohlc.loc[not_na_indexes, 'min_low_quantile'].astype(int).eq(0)
    consistent_quantile_min_low = quantile_min_low_consistency[~quantile_min_low_consistency].index
    ohlc.loc[not_na_indexes, 'quantile_short_max_high'] = max_high_np_a[
        np.arange(len(ohlc.loc[not_na_indexes])), ohlc.loc[not_na_indexes, 'min_low_quantile'].astype(int)]
    # ohlc.loc[consistent_quantile_min_low, 'pre_quantile_short_max_high'] = max_high_np_a[
    #     np.arange(len(consistent_quantile_min_low)), ohlc.loc[consistent_quantile_min_low, 'min_low_quantile'].astype(
    #         int) - 1]
    # ohlc.loc[consistent_quantile_min_low, 'short_drawdown_consistency'] = (
    #         ohlc.loc[consistent_quantile_min_low, 'pre_quantile_short_max_high']
    #         == ohlc.loc[consistent_quantile_min_low, 'quantile_short_max_high'])
    ohlc['short_drawdown'] = \
        (ohlc['quantile_short_max_high'] - ohlc['worst_short_open']) / ohlc['worst_short_open']
    return ohlc


def add_profit_n_loss(ohlc, risk_free_daily_rate, order_fee, max_risk):
    ohlc['long_profit'] = ohlc['max_high'] - ohlc['worst_long_open']
    ohlc['short_profit'] = ohlc['worst_short_open'] - ohlc['min_low']
    ohlc['weighted_long_profit'] = (
        ohlc['long_profit'] / ohlc['Close'] - ohlc['max_high_distance'] * risk_free_daily_rate - order_fee)
    ohlc['weighted_short_profit'] = (
        ohlc['short_profit'] / ohlc['Close'] - ohlc['min_low_distance'] * risk_free_daily_rate - order_fee)
    ohlc['long_risk'] = (ohlc['long_drawdown'] / ohlc['weighted_long_profit'])
    ohlc['short_risk'] = (ohlc['short_drawdown'] / ohlc['weighted_short_profit'])

    loser_shorts = ohlc[(ohlc['weighted_short_profit'] <= 0) | (ohlc['short_risk'] > max_risk)].index
    loser_longs = ohlc[(ohlc['weighted_long_profit'] <= 0) | (ohlc['long_risk'] > max_risk)].index
    ohlc.loc[loser_shorts, 'short_risk'] = max_risk
    ohlc.loc[loser_longs, 'long_risk'] = max_risk
    ohlc['long_signal'] = (1 - ohlc['long_risk'].fillna(1)) * ohlc['weighted_long_profit'].fillna(0)
    # ohlc['long_signal'] =  (ohlc['long_signal']) / ta.ema(ohlc['long_signal'], length=position_max_days)
    ohlc['short_signal'] = (1 - ohlc['short_risk'].fillna(1)) * ohlc['weighted_short_profit'].fillna(0)
    return ohlc


def add_long_n_short_profit(ohlc,
                            position_max_days=3 * 4 * 4 * 4 * 4,  # 768 5mins = 64h = 2.66D
                            action_delay=2,
                            risk_free_daily_rate=0, #(0.1 / 365),
                            order_fee=0.005,
                            quantiles=10,
                            max_risk=1,
                            ):
    rolling_window = position_max_days - action_delay

    ohlc = max_profit_n_loss(ohlc, position_max_days, action_delay, rolling_window)
    ohlc = quantile_maxes(ohlc, rolling_window, quantiles)
    ohlc = long_n_short_drawdown(ohlc, position_max_days, quantiles)
    ohlc = add_profit_n_loss(ohlc, risk_free_daily_rate, order_fee, max_risk)
    return ohlc


def plot_short_profit(t):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.5, 0.5, 0.1])
    fig.add_trace(go.Candlestick(x=t.index, open=t['Open'], close=t['Close'], high=t['High'], low=t['Low']), row=2,
                  col=1)
    fig.add_scatter(x=t.index, y=t['max_high'], mode='lines', line=dict(color='red', width=1), name='max_high', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['min_low'], mode='lines', line=dict(color='blue', width=1), name='min_low', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['quantile_short_max_high'], mode='lines', line=dict(color='magenta', width=1),
                    name='quantile_short_max_high', row=2, col=1)
    fig.add_scatter(x=t.index, y=t['short_risk'], mode='lines', line=dict(color='yellow', width=1),
                    name='short_risk')
    fig.add_scatter(x=t.index, y=t['short_drawdown'], mode='lines', line=dict(color='pink', width=1),
                    name='short_drawdown')
    fig.add_scatter(x=t.index, y=t['weighted_short_profit'], mode='lines', line=dict(color='blue', width=1),
                    name='weighted_short_profit')
    fig.add_scatter(x=t.index, y=t['short_signal'], mode='lines', line=dict(color='green', width=1),
                    name='short_signal')
    fig.update_layout(dragmode="zoom", yaxis2=dict(fixedrange=False), margin=dict(l=0, r=0, t=10, b=10), ).show()


def plot_long_profit(t):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.6])
    fig.add_trace(go.Candlestick(x=t.index, open=t['Open'], close=t['Close'], high=t['High'], low=t['Low']), row=2,
                  col=1)
    fig.add_scatter(x=t.index, y=t['max_high'], mode='lines', line=dict(color='blue', width=1), name='max_high', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['min_low'], mode='lines', line=dict(color='red', width=1), name='min_low', row=2,
                    col=1)
    fig.add_scatter(x=t.index, y=t['quantile_long_min_low'], mode='lines', line=dict(color='magenta', width=1),
                    name='quantile_long_min_low', row=2, col=1)
    fig.add_scatter(x=t.index, y=t['long_risk'], mode='lines', line=dict(color='yellow', width=1),
                    name='long_risk')
    fig.add_scatter(x=t.index, y=t['long_drawdown'], mode='lines', line=dict(color='pink', width=1),
                    name='long_drawdown')
    fig.add_scatter(x=t.index, y=t['weighted_long_profit'], mode='lines', line=dict(color='blue', width=1),
                    name='weighted_long_profit')
    fig.add_scatter(x=t.index, y=t['long_signal'], mode='lines', line=dict(color='green', width=1),
                    name='long_signal')
    fig.update_layout(dragmode="zoom", yaxis2=dict(fixedrange=False), margin=dict(l=0, r=0, t=10, b=10), ).show()


def plot_long_n_short_profit(t):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.3, 0.7, 0.3], )
    fig.add_scatter(x=t.index, y=t['long_profit'], mode='lines', line=dict(color='blue', width=1), name='long_profit')
    fig.add_scatter(x=t.index, y=t['short_profit'], mode='lines', line=dict(color='red', width=1), name='short_profit')
    fig.add_scatter(x=t.index, y=1000 * t['long_risk'], mode='lines', line=dict(color='LightSkyBlue', width=1),
                    name='long_risk')
    fig.add_scatter(x=t.index, y=1000 * t['short_risk'], mode='lines', line=dict(color='yellow', width=1),
                    name='short_risk')
    fig.add_scatter(x=t.index, y=1000 * t['long_signal'], mode='lines', line=dict(color='green', width=1),
                    name='long_signal')
    fig.add_scatter(x=t.index, y=1000 * t['short_signal'], mode='lines', line=dict(color='orange', width=1),
                    name='short_signal')
    fig.add_scatter(x=t.index, y=t['min_low'], mode='lines', line=dict(color='red', width=1), name='min_low', row=2,
                    col=1)
    fig.add_trace(
        go.Candlestick(x=t.index, open=t['Open'], close=t['Close'], high=t['High'], low=t['Low']), row=2, col=1)
    fig.add_scatter(x=t.index, y=t['max_high'], mode='lines', line=dict(color='blue', width=1), name='max_high', row=2,
                    col=1)

    showlegend = True
    # for sample in samples:
    #     # x = [sample.replace(tzinfo=None), sample.replace(tzinfo=None) + position_max_days * timedelta(days=1)]
    #     # y = [float(t.loc[sample, 'Close']), float(t.loc[sample, 'Close'])]
    #     # fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='gray', width=1), name='position_max_days', row=2, col=1,
    #     #                 legendgroup='position_max_days', showlegend=showlegend)
    #     x = [sample.replace(tzinfo=None),
    #          sample.replace(tzinfo=None) + t.loc[sample, 'max_high_distance'] * timedelta(days=1)]
    #     y = [float(t.loc[sample, 'High']), float(t.loc[sample, 'max_high'])]
    #     fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='cyan', width=1), name='max_high', row=2, col=1,
    #                     legendgroup='max_high', showlegend=showlegend)
    #     x = [sample.replace(tzinfo=None),
    #          sample.replace(tzinfo=None) + t.loc[sample, 'min_low_distance'] * timedelta(days=1)]
    #     y = [float(t.loc[sample, 'Low']), float(t.loc[sample, 'min_low'])]
    #     fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='pink', width=1), name='min_low', row=2, col=1,
    #                     legendgroup='min_low', showlegend=showlegend)
    #     # x = [sample.replace(tzinfo=None),
    #     #      sample.replace(tzinfo=None) + t.loc[sample, 'min_low_distance'] * timedelta(days=1)]
    #     # y = [float(t.loc[sample, 'worst_short_open']),
    #     #      float(t.loc[sample, 'worst_short_open'] + t.loc[sample, 'short_drawdown'])]
    #     # fig.add_scatter(x=x, y=y, mode='lines', line=dict(color='orange', width=1), name='short_drawdown', row=2, col=1,
    #     #                 legendgroup='short_drawdown', showlegend=showlegend)
    fig.update_layout(dragmode="zoom", yaxis2=dict(title="Price", fixedrange=False),
                      margin=dict(l=0, r=0, t=10, b=10), ).show()

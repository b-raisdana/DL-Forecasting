import backtrader as bt
import pandas as pd
import pytz
from pandera import typing as pt

from app.Model.Order import BracketOrderType
from app.PanderaDFM.SignalDf import SignalDf, SignalDFM


def order_name(order: bt.Order):
    # TRX<13USDT@0.02
    """
    tojoin.append('CommInfo: {}'.format(self.comminfo))
    tojoin.append('End of Session: {}'.format(self.dteos))
    tojoin.append('Alive: {}'.format(self.alive()))

    :param order:
    :return:
    """
    name = (
        f"{order.ordtypename()}"
        # f"Order"  
        # f"{('Buy' if order.isbuy() else 'Sell')}"
        f"{order.size:.4f}"
        # if order.params.pricelimit is not None:
        f"@{order.price:.2f}"
        f"Ref{order.ref}"
        # f"{'Alive' if order.alive() else 'Dead'}/{order.getstatusname()}"
        f"{order.ExecTypes[order.exectype]}"
    )
    if order.plimit is not None:
        name += f"PL{order.pricelimit}"
    if order.trailamount is not None:
        name += f"TR{order.trailamount}/{order.trailpercent}"
    return name


def dict_of_order(order: bt.Order):
    """
    order elements:
    'date', 'ref', 'triggered', 'status', 'valid', 'date_eos', 'plen', 'created_pclose', 'created_date', 'created_size',
     'created_price', 'created_pricelimit', 'executed_size', 'executed_remsize', 'executed_price', 'executed_value',
     'executed_pnl', 'executed_psize', 'executed_pprice', 'info_pprice', 'info_order_group_id', 'signal',
     'reference_date', 'reference_timeframe', 'info_custom_type', 'hidden', 'size', 'pannotated', 'executed_date'
    :param order:
    :return:
    """
    t = {}
    for key, value in order.__dict__.items():
        if not key.startswith("_") and not type(value) == bt.OrderData:
            if key == 'dt':
                t['date'] = bt.num2date(value).replace(tzinfo=pytz.UTC) if value is not None else None
            elif key == 'dteos':
                t['date_eos'] = bt.num2date(value).replace(tzinfo=pytz.UTC) if value is not None else None
            elif key == 'valid':
                t[key] = bt.num2date(value).replace(tzinfo=pytz.UTC)
            elif key == 'status':
                t[key] = order.getstatusname()
            else:
                t[key] = str(value)
    for key, value in order.created.__dict__.items():
        if not key.startswith("_"):
            if key == 'dt':
                t['created_date'] = bt.num2date(value).replace(tzinfo=pytz.UTC) if value is not None else None
            else:
                t[f"created_{key}"] = str(value)
    for key, value in order.executed.__dict__.items():
        if not key.startswith("_"):
            if key == 'dt':
                t['executed_date'] = bt.num2date(value).replace(tzinfo=pytz.UTC) if value is not None else None
            else:
                t[f"executed_{key}"] = str(value)
    for key, value in order.info.items():
        if not key.startswith("_"):
            if key == 'signal':
                value: pt.Series[SignalDFM]
                signal_index = order.info['signal_index']
                t['signal'] = SignalDf.to_str(order.info['signal_index'], value)
                t['reference_date'] = signal_index[1]
                t['reference_timeframe'] = signal_index[2]
            else:
                t[f"info_{key}"] = str(value)
    if order.comminfo is not None:
        t['comminfo'] = str(order.comminfo.params.__dict__)
    if len(order.executed.exbits) > 0:
        t['comminfo'] = ",".join(
            [f"[{','.join([f'{k}:{v}' for k, v in row.__dict__.items()])}]" for row in order.executed.exbits])

    hidden_keys = [k for k in t.keys() if
                   k in ['p', 'params', 'broker', 'created_exbits', 'created_comm', 'created_margin', 'created_pnl',
                         'created_psize', 'created_pprice', 'executed_pclose', 'position', 'created_p1',
                         'created_p2', 'created_remsize', 'created_trailamount', 'created_trailpercent',
                         'created_value', 'executed_p1', 'executed_p2', 'executed_pricelimit',
                         'executed_trailamount', 'executed_trailpercent', 'info',
                         'executed_margin', 'comminfo', 'executed_exbits', 'info_signal_index',
                         ]]
    hidden_values = []
    for k in hidden_keys:
        v = t.pop(k)
        hidden_values += [f"{k}:{v}"]
    t['hidden'] = ",\r\n".join(hidden_values)
    return t


def add_order_info(order: bt.Order, signal: pd.Series, signal_index: pd.MultiIndex,
                   order_type: BracketOrderType, order_group_id):
    order.addinfo(order_group_id=order_group_id)
    order.addinfo(signal=signal)
    order.addinfo(signal_index=signal_index)
    order.addinfo(custom_type=order_type.value)
    return order


# def order_prices(order: bt.Order):
#     return order.info['limit_price'], order.info['stop_loss'], order.info['take_profit'],


# class ExtendedBuyOrder(ExtendedOrder):
#     ordtype = bt.BuyOrder
#
#
# class ExtendedSellOrder(ExtendedOrder):
#     ordtype = bt.SellOrder


def order_is_open(order):
    return order.status in [bt.Order.Created, bt.Order.Accepted, bt.Order.Submitted, bt.Order.Partial]


def order_is_closed(order):
    return order.status in [bt.Order.Canceled, bt.Order.Expired, bt.Order.Rejected]

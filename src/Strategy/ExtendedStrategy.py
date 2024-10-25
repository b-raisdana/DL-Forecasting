import os
from typing import Optional, Tuple, Dict

import backtrader as bt
import pandas as pd
import pytz
from pandera import typing as pt

from Config import config
from Model.Order import OrderSide, BracketOrderType
from PanderaDFM.OHLCVA import MultiTimeframeOHLCVA
from PanderaDFM.SignalDf import SignalDFM, SignalDf
from Strategy.BaseTickStructure import BaseTickStructure
from Strategy.order_helper import order_name, add_order_info, order_is_open, \
    order_is_closed, dict_of_order
from data_processing.fragmented_data import symbol_data_path
from helper.data_preparation import concat, dict_of_list
from helper.helper import log_d, profile_it, log_w


class ExtendedStrategy(bt.Strategy):
    signal_df: pt.DataFrame[SignalDFM] = None
    orders_df: pd.DataFrame = None
    vault_df: pd.DataFrame = None

    multi_timeframe_ohlcva: pt.DataFrame[MultiTimeframeOHLCVA] = None

    original_orders: Dict[int, bt.Order] = {}
    sl_orders: Dict[int, bt.Order] = {}
    tp_orders: Dict[int, bt.Order] = {}
    date_range_str: str = None
    true_risked_money = 0.0
    order_group_counter = None
    order_groups_info = {}
    broker_initial_cash = None

    def get_order_groups(self):
        result = {order_group_id: (
            self.original_orders[order_group_id],
            self.sl_orders[order_group_id],
            self.tp_orders[order_group_id]) for order_group_id in self.original_orders.keys()}
        return result

    @profile_it
    def __init__(self):
        self.signal_df = SignalDf.new()
        self.orders_df = pd.DataFrame()
        self.vault_df = pd.DataFrame()

        self.set_date_range()
        self.bracket_executors = {
            'buy': self.buy_bracket,
            'sell': self.sell_bracket,
        }
        self.next_runs = 0

    def start(self):
        self.broker_initial_cash = self.broker.get_cash()

    def new_order_group_id(self):
        return len(self.order_groups_info)

    def add_signal_source_data(self):
        raise NotImplementedError

    @staticmethod
    def risk_size_sizer(limit_price: float, sl_price: float, size: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate position size based on limit and stop prices.

        Parameters:
        - limit_price (float): The limit price of the order.
        - sl_price (float): The stop-loss price of the order.
        - size (float, optional): The desired position size. If not provided, it will be calculated based on risk percentage.

        Returns:
        Tuple[float, float]: A tuple containing the calculated position size and the true risked money.

        Raises:
        AssertionError: If the true risked money exceeds the configured risk percentage.
        """
        sl_size = abs(sl_price - limit_price)

        if size is None:
            risk_per_order_size = config.initial_cash * config.risk_per_order_percent  # 10$
            size = risk_per_order_size / sl_size

        true_risked_money = size * sl_size

        if not true_risked_money <= config.initial_cash * config.risk_per_order_percent:
            log_w("True risked money exceeds configured risk percentage", stack_trace=False)

        return size, true_risked_money

    def post_bracket_order(self, original_order: bt.Order, sl_order: bt.Order, tp_order: bt.Order,
                           signal: pt.Series[SignalDFM], signal_index: pd.MultiIndex):
        price = original_order.created.price
        stop_loss_price = sl_order.created.price
        true_risked_money = abs((price - stop_loss_price) * original_order.created.size)
        self.true_risked_money += true_risked_money

        order_group_id = self.new_order_group_id()
        self.order_groups_info[order_group_id] = {
            'price': original_order.created.price,
        }
        if config.check_assertions and (
                order_group_id in self.original_orders.keys() or
                order_group_id in self.sl_orders.keys() or
                order_group_id in self.tp_orders.keys()
        ):
            raise AssertionError(f"order_group_id is not unique!")
        original_order = add_order_info(original_order, signal, signal_index, BracketOrderType.Original, order_group_id)
        sl_order = add_order_info(sl_order, signal, signal_index, BracketOrderType.StopLoss, order_group_id)
        tp_order = add_order_info(tp_order, signal, signal_index, BracketOrderType.TakeProfit, order_group_id)

        self.original_orders[order_group_id] = original_order
        self.sl_orders[order_group_id] = sl_order
        self.tp_orders[order_group_id] = tp_order

        self.signal_df.loc[signal_index, 'original_order_id'] = order_name(original_order)
        self.signal_df.loc[signal_index, 'stop_loss_order_id'] = order_name(sl_order)
        self.signal_df.loc[signal_index, 'take_profit_order_id'] = order_name(tp_order)
        self.signal_df.loc[signal_index, 'order_is_active'] = True

        return original_order, sl_order, tp_order

    def spent_money(self) -> float:
        if self.broker_initial_cash is None:
            raise ProcessLookupError("Do not run before start()")
        return self.broker_initial_cash - self.broker.get_cash()

    def allocate_order_cash(self, limit_price: float, sl_price: float, size: float = None) -> float:
        # Allocate 10% of the initial cash
        max_allowed_total_capital_at_risk = config.initial_cash * config.capital_max_total_risk_percentage
        remained_risk_able = max_allowed_total_capital_at_risk - self.true_risked_money  # self.spent_money()
        if size is None:
            size, true_risked_money = self.risk_size_sizer(limit_price, sl_price, size)
        else:
            raise NotImplementedError  # true_risked_money = size * self.candle().close
        if true_risked_money < remained_risk_able:
            return size
        elif remained_risk_able > 0:
            return remained_risk_able / self.candle().close
        else:
            return 0

    def candle(self, backward_index: int = 0) -> BaseTickStructure:
        return BaseTickStructure(
            date=bt.num2date(self.datas[0].datetime[backward_index]).replace(tzinfo=pytz.UTC),
            close=self.datas[0].close[backward_index],
            open=self.datas[0].open[backward_index],
            high=self.datas[0].high[backward_index],
            low=self.datas[0].low[backward_index],
            volume=self.datas[0].volume[backward_index],
        )

    def set_date_range(self, date_range_str: str = None):
        if date_range_str is None:
            date_range_str = config.processing_date_range
        self.date_range_str = date_range_str
        log_d(f"date_range_str: {self.date_range_str}")
        # self.start_datime, self.end = date_range(date_range_str)

    def get_order_group(self, order: bt.Order) -> (bt.Order, bt.Order, bt.Order,):
        order_id = order.info['order_group_id']
        original_order = self.original_orders[order_id]
        stop_order = self.sl_orders[order_id]
        profit_order = self.tp_orders[order_id]
        if config.check_assertions and order not in [original_order, stop_order, profit_order]:
            raise AssertionError(f"The order: should be one of group members "
                                 f"O:{original_order} S:{stop_order} P:{profit_order}")
        return original_order, stop_order, profit_order

    def next_log(self, force=False):
        if force or self.next_runs % 100 == 0:
            log_d(f"Ran {self.next_runs} Next()s reached {self.candle().date}")
        self.next_runs += 1

    def movement_intersect(self, target_low: float, target_high: float):
        movement_low = min(self.candle().low, self.candle(1).low)
        movement_high = max(self.candle().high, self.candle(1).high)
        intersect_high = min(movement_high, target_high)
        intersect_low = max(movement_low, target_low)
        if intersect_high > intersect_low:
            return intersect_high - intersect_low
        else:
            return 0

    def next(self):
        self.next_log()
        if len(self.original_orders) > 0:
            self.verify_triple_oder_status()
        self.extract_signals()
        self.execute_active_signals()

    def verify_triple_oder_status(self):
        '''
        check:
         - length of self.original_orders and alef.stop_orders and self.profit_orders are equal
         - if the self.original_orders.status is open the same index in both other lists
         have the same status
         - if the self.original_orders.status in closed (by any way) the same index in both other lists are closed
         (canceled) also
        :return:
        '''
        if config.check_assertions and len(self.original_orders) != len(self.sl_orders):
            raise AssertionError("len(self.original_orders) != len(self.stop_orders)")
        if config.check_assertions and len(self.original_orders) != len(self.tp_orders):
            raise AssertionError("len(self.original_orders) != len(self.profit_orders)")
        if config.check_assertions:
            for i in self.original_orders.keys():
                if order_is_open(self.original_orders[i]):
                    if not order_is_open(self.sl_orders[i]):
                        raise AssertionError("!order_is_open(self.stop_orders[i])")
                    if not order_is_open(self.tp_orders[i]):
                        raise AssertionError("!order_is_open(self.profit_orders[i])")
                elif order_is_closed(self.original_orders[i]):
                    if not order_is_closed(self.sl_orders[i]):
                        raise AssertionError("!order_is_closed(self.stop_orders[i])")
                    if not order_is_closed(self.tp_orders[i]):
                        raise AssertionError("!order_is_closed(self.profit_orders[i])")

    def notify_trade(self, trade):
        '''
        Receives a trade whenever there has been a change in one
        '''
        log_d("notify_trade trade:" + ", ".join([f"{k}:{v}" for k, v in trade.__dict__.items()]))

    def stop(self):
        self.next_log(force=True)
        log_d("Stopping!")
        if self.vault_df is not None:
            self.vault_df.to_csv(os.path.join(symbol_data_path(),
                                              f"{self.__class__.__name__}.vault.{config.id}.{self.date_range_str}.csv"))
        if self.orders_df is not None:
            self.orders_df.to_csv(os.path.join(symbol_data_path(),
                                               f"{self.__class__.__name__}.orders.{config.id}.{self.date_range_str}.csv"))
        if self.signal_df is not None:
            self.signal_df.to_csv(os.path.join(symbol_data_path(),
                                               f"{self.__class__.__name__}.signals.{config.id}.{self.date_range_str}.csv"))
        if self.original_orders is not None and len(self.original_orders) > 0:
            df = pd.DataFrame()
            for index in self.original_orders.keys():
                t = dict_of_list(dict_of_order(self.original_orders[index]))
                df = concat(self.orders_df, pd.DataFrame(t, index=[index]))
            df.index.name = 'ref_id'
            if not df.empty:
                df.to_csv(
                    os.path.join(symbol_data_path(),
                                 f"{self.__class__.__name__}.original_orders.{config.id}.{self.date_range_str}.csv"))
        if self.sl_orders is not None and len(self.sl_orders) > 0:
            for index in self.sl_orders.keys():
                t = dict_of_list(dict_of_order(self.sl_orders[index]))
                df = concat(self.orders_df, pd.DataFrame(t, index=[index]))
            df.index.name = 'ref_id'
            if not df.empty:
                df.to_csv(
                    os.path.join(symbol_data_path(),
                                 f"{self.__class__.__name__}.stop_orders.{config.id}.{self.date_range_str}.csv"))
        if self.tp_orders is not None and len(self.tp_orders) > 0:
            for index in self.tp_orders.keys():
                t = dict_of_list(dict_of_order(self.tp_orders[index]))
                df = concat(self.orders_df, pd.DataFrame(t, index=[index]))
            df.index.name = 'ref_id'
            if not df.empty:
                df.to_csv(
                    os.path.join(symbol_data_path(),
                                 f"{self.__class__.__name__}.profit_orders.{config.id}.{self.date_range_str}.csv"))

    def log_order(self, order: bt.Order, index=None):
        if index is None:
            index = order.ref
        order_dict = dict_of_list(dict_of_order(order))
        vault_dict = self.get_vault()
        order_dict = order_dict | vault_dict
        self.orders_df = concat(self.orders_df, pd.DataFrame(order_dict, index=[self.candle().date]))
        if self.orders_df.index.name != 'date':
            self.orders_df.index.name = 'date'

    def log_vault(self, save: bool = True):
        _dict = self.get_vault()
        self.vault_df = concat(self.vault_df,
                               pd.DataFrame(_dict,
                                            index=[self.candle().date]))  # pd.DataFrame({**_dict}, index=order.ref)
        if self.vault_df.index.name != 'date':
            self.vault_df.index.name = 'date'

    def get_vault(self):
        _dict = {
            'cash': self.broker.get_cash(),
            'fund_shares': self.broker.get_fundshares(),
            'fund_value': self.broker.get_fundvalue(),
        }
        if len(self.getdatanames()) > 1:
            raise NotImplementedError
        self.broker: bt.BackBroker
        # for index in len(self.datas):
        #     asset = list_of_assets[index]
        asset = self.getdatanames()[0]
        _dict[f"{asset}_value"] = self.broker.get_value()
        position = self.broker.getposition(self.datas[0])
        _dict[f"{asset}_position_size"] = position.size
        _dict[f"{asset}_position_price"] = position.price
        _dict[f"{asset}_position_adjbase"] = position.adjbase
        _dict[f"{asset}_position_price_orig"] = position.price_orig
        _dict[f"{asset}_position_upclosed"] = position.upclosed
        _dict[f"{asset}_position_updt"] = position.updt
        _dict[f"{asset}_position_upopened"] = position.upopened
        # _dict = _dict | self.broker.positions.__dict__
        _dict = dict_of_list(_dict)
        return _dict

    # @measure_time
    def notify_order(self, order: bt.Order):
        self.log_order(order)
        self.log_vault()
        if config.check_assertions and not (
                order_name(order) in self.signal_df['original_order_id'].dropna().tolist() or
                order_name(order) in self.signal_df['stop_loss_order_id'].dropna().tolist() or
                order_name(order) in self.signal_df['take_profit_order_id'].dropna().tolist()
        ):
            raise AssertionError(f"{order_name(order)} not found in tracked orders!")
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted or accepted
            log_d(f"Order:{order_name(order)} Submitted")
        elif order.status in [order.Completed]:
            # Order has been completed (executed)
            log_d(f"Order:{order_name(order)} Completed")
            vault_dict = self.get_vault()
            log_d(f"Vault({self.candle().date}): " + ", ".join([f"{k}:{v}" for k, v in vault_dict.items()]))
        elif order.status in [order.Partial]:
            log_w(f"Order:{order_name(order)} Partially executed", stack_trace=True)
            # raise NotImplementedError
        elif order.status in [order.Expired]:
            log_d(f"Order:{order_name(order)} Expired")
        elif order.status in [order.Canceled]:
            log_d(f"Order:{order_name(order)} Canceled")
        elif order.status in [order.Rejected]:
            raise Exception(f"Order:{order_name(order)} Rejected")
        else:
            raise Exception(f"Order:{order_name(order)} Unexpected status {order.status}")

        if order.info['custom_type'] in [BracketOrderType.StopLoss.value, BracketOrderType.TakeProfit.value]:
            if order.status in [order.Completed, order.Partial]:
                # the stop-loss order or took profit executed (completed or partial)
                # risked money is reflected in the cash
                order_group_id = order.info['order_group_id']
                price = self.original_orders[order_group_id].created.price
                stop_loss_price = self.sl_orders[order_group_id].created.price
                true_risked_money = abs((price - stop_loss_price) * order.executed.size)
                self.true_risked_money -= true_risked_money

    def ordered_signals(self, signal_df: SignalDf = None) -> pt.DataFrame[SignalDFM]:
        if signal_df is None:
            signal_df = self.signal_df
        return signal_df[signal_df['order_is_active'].notna() & signal_df['order_is_active']]

    def active_signals(self) -> pt.DataFrame[SignalDFM]:
        """
        return all signals which the self.candle()date is between signal start (signal.index) and signal end.
        :return:
        """
        result = self.signal_df[
            (self.signal_df.index.get_level_values(level='date') <= self.candle().date) &
            (
                # ('end' not in self.signal_df.columns) |
                    (self.signal_df['end'].isna()) |
                    (self.signal_df['end'] > self.candle().date)) &
            (self.signal_df['order_is_active'].isna() | ~self.signal_df['order_is_active'])
            ]
        return result

    def execute_active_signals(self):
        _executable_signals = self.active_signals()
        # for signal_index, signal in self.executable_signals().iterrows():
        for idx in range(len(_executable_signals)):
            signal = _executable_signals.iloc[idx]
            signal_side = _executable_signals.index.get_level_values(level='side')[idx]
            signal_index = _executable_signals.index[idx]
            if pd.notna(_executable_signals.iloc[idx]['take_profit']):
                original_order: bt.Order
                # try:
                size = self.allocate_order_cash(_executable_signals.iloc[idx]['limit_price'],
                                                signal['stop_loss'])  # signal['base_asset_amount'])
                if size <= 0:
                    log_w(f"Size is {size} @ {self.candle().date}")
                    continue
                if pd.notna(signal['limit_price']) and pd.notna(signal['stop_loss']):
                    if (
                            pd.isna(signal['trigger_price']) or
                            ((signal_side == OrderSide.Buy.value) and (self.candle().high > signal['trigger_price']))
                            or
                            ((signal_side == OrderSide.Sell.value) and (self.candle().low < signal['trigger_price']))
                    ):
                        original_order, stop_order, profit_order = \
                            self.bracket_executors[signal_side](size=size,
                                                                exectype=bt.Order.Limit,
                                                                limitprice=signal['take_profit'],
                                                                price=signal['limit_price'],
                                                                stopprice=signal['stop_loss'],
                                                                valid=signal['end'],
                                                                )
                    else:  # the trigger_price is not satisfied
                        original_order, stop_order, profit_order = \
                            self.bracket_executors[signal_side](size=size,
                                                                exectype=bt.Order.StopLimit,
                                                                price=signal['trigger_price'],
                                                                pprice=signal['limit_price'],
                                                                limitprice=signal['take_profit'],
                                                                stopprice=signal['stop_loss'],
                                                                valid=signal['end'],
                                                                )
                else:
                    raise NotImplementedError
                self.post_bracket_order(original_order, stop_order, profit_order, signal, signal_index)
                log_d(f"Signal:{SignalDf.to_str(signal_index, signal)} Ordered: "
                      f"O:{order_name(original_order)} S:{order_name(stop_order)} P:{order_name(profit_order)}")
            else:
                raise NotImplementedError

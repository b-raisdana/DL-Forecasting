import backtrader as bt

from app.Config import config
from app.helper.helper import date_range_to_string
from app.data_processing.ohlcv import read_base_timeframe_ohlcv


# Create a subclass of SignaStrategy to define the indicators and signals

class SmaCross(bt.SignalStrategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30  # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal
        self.signal_add(bt.SIGNAL_LONG, crossover)  # use it as LONG signal


cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

# Create a data feed
config.processing_date_range = date_range_to_string(days=2)
raw_data = read_base_timeframe_ohlcv(config.processing_date_range)
data = bt.feeds.PandasData(dataname=raw_data, datetime=None, open=0, close=1, high=2, low=3, volume=4,
                           openinterest=-1)
cerebro.adddata(data)  # Add the data feed
cerebro.addstrategy(SmaCross)  # Add the trading strategy
cerebro.run()  # run it all
cerebro.plot()  # and plot it with a single command

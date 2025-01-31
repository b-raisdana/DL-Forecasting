import pandas_ta as ta


def bollinger_width(ohlc, position_max_bars):
    bollinger = ta.bbands(ohlc['close'], length=position_max_bars, std=2)
    ohlc['bollinger_width'] = bollinger[f'BBU_{position_max_bars}_2.0'] - bollinger[f'BBL_{position_max_bars}_2.0']
    return ohlc

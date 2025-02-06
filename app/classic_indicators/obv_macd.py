import numpy as np

from PanderaDFM.OHLCV import OHLCV
from data_processing import ohlcv
from helper.importer import ta, pt


def obv_macd_with_pivots(ohlcv: pt.DataFrame[OHLCV], sma_lenght = 14, std_lenght = 28, ema_lneght = 1):
    t_ohlcv = ohlcv.copy()
    t_ohlcv['OBV'] = ta.obv(t_ohlcv['close'], t_ohlcv['volume'])

    t_ohlcv['OBV_Smooth'] = ta.sma(t_ohlcv['OBV'], length=sma_lenght)

    # 3. Spread Adjustment
    price_spread = t_ohlcv['high'] - t_ohlcv['low']
    price_spread_std = price_spread.rolling(std_lenght).std()

    v_spread = (t_ohlcv['OBV'] - t_ohlcv['OBV_Smooth']).rolling(std_lenght).std()
    shadow = (t_ohlcv['OBV'] - t_ohlcv['OBV_Smooth']) / v_spread * price_spread_std
    t_ohlcv['Shadow_Adjusted_OBV'] = np.where(shadow > 0, t_ohlcv['high'] + shadow, t_ohlcv['low'] + shadow)

    # 4. Apply EMA to Adjusted OBV
    len10 = 1
    t_ohlcv['OBV_EMA'] = ta.ema(t_ohlcv['Shadow_Adjusted_OBV'], length=len10)

    # 5. MACD on Adjusted OBV
    macd = ta.macd(t_ohlcv['OBV_EMA'])
    t_ohlcv['MACD'] = macd['MACD_12_26_9']
    t_ohlcv['Signal'] = macd['MACDs_12_26_9']
    t_ohlcv['Histogram'] = macd['MACDh_12_26_9']

    # 6. Slope Calculation
    len5 = 2
    t_ohlcv['Slope'] = t_ohlcv['MACD'].rolling(len5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    # 7. Pivot Detection
    xbars = 50
    t_ohlcv['Max_Upper'] = t_ohlcv['MACD'].rolling(xbars).max()
    t_ohlcv['Min_Lower'] = t_ohlcv['MACD'].rolling(xbars).min()

    t_ohlcv['PivotH'] = (t_ohlcv['Max_Upper'] == t_ohlcv['Max_Upper'].shift(2)) & (
            t_ohlcv['Max_Upper'].shift(2) != t_ohlcv['Max_Upper'].shift(3))
    t_ohlcv['PivotL'] = (t_ohlcv['Min_Lower'] == t_ohlcv['Min_Lower'].shift(2)) & (
            t_ohlcv['Min_Lower'].shift(2) != t_ohlcv['Min_Lower'].shift(3))

    # 8. Signal and Color Variable
    t_ohlcv['Signal_Color'] = np.where(t_ohlcv['Slope'] > 0, 'blue', 'red')
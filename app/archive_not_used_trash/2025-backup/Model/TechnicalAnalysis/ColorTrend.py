from helper.functions import profile_it


def read_multi_timeframe_color_trend_pivots(date_range_str: str = None):
    raise Exception('Not implemented!')


@profile_it
def generate_multi_timeframe_color_trend_pivots():
    """
        trend pivots:
            find any sequence of same color candles
            marge same color boundaries separated with a reverse color trend < 1 atr
                each one with movement >= 3 atr
                    follwed with a reverse color trend >= 1 atr
    :return:
    """
    # todo: implement  generate_multi_timeframe_same_color_trend_pivots
    raise Exception('Not implemented')

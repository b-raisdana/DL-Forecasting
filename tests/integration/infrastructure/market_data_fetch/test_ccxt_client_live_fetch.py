from datetime import UTC, datetime, timedelta

import pytest
from infrastructure.market_data_fetch.binance import fetch_ohlcv as binance_fetcher
from infrastructure.market_data_fetch.kucoin import fetch_ohlcv as kucoin_fetcher

NUMBER_OF_TICKS = 5


def _assert_valid_candle(row):
    timestamp, open_, high, low, close, volume = row
    assert isinstance(timestamp, int)
    assert high >= open_
    assert high >= close
    assert high >= low
    assert low <= open_
    assert low <= close
    assert volume >= 0
    return timestamp


@pytest.mark.integration
@pytest.mark.parametrize(
    ("fetcher", "broker"),
    [(kucoin_fetcher, "kucoin"), (binance_fetcher, "binance")],
)
def test_fetch_ohlcv_returns_recent_ohlc_and_volume_from_live_broker(fetcher, broker):
    start = datetime.now(UTC) - timedelta(hours=NUMBER_OF_TICKS + 1)

    response = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1H", start=start, number_of_ticks=NUMBER_OF_TICKS)

    assert len(response) > 0, f"{broker} returned no candles"
    timestamps = [_assert_valid_candle(row) for row in response]

    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)

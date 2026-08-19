from datetime import UTC, datetime

import pytest
from archive_not_used_trash.infrastructure.market_data_fetch.binance import fetch_ohlcv as binance_fetcher
from archive_not_used_trash.infrastructure.market_data_fetch.kucoin import fetch_ohlcv as kucoin_fetcher


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fetcher", "broker"),
    [(kucoin_fetcher, "kucoin"), (binance_fetcher, "binance")],
)
def test_fetch_ohlcv_by_range_delegates_to_the_bound_broker(monkeypatch, fetcher, broker):
    captured = {}
    expected = [[1, 2, 3, 4, 5, 6]]

    def fake_fetch(*args):
        captured["args"] = args
        return expected

    monkeypatch.setattr(fetcher.ccxt_client, "fetch_ohlcv_by_range", fake_fetch)

    result = fetcher.fetch_ohlcv_by_range("2024-01-01_2024-01-02", "BTC/USDT", "1H", True)

    assert result == expected
    assert captured["args"] == (broker, "2024-01-01_2024-01-02", "BTC/USDT", "1H", True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fetcher", "broker"),
    [(kucoin_fetcher, "kucoin"), (binance_fetcher, "binance")],
)
def test_fetch_ohlcv_delegates_to_the_bound_broker(monkeypatch, fetcher, broker):
    captured = {}
    start = datetime(2024, 1, 1, tzinfo=UTC)
    params = {"paginate": True}
    expected = [[1, 2, 3, 4, 5, 6]]

    def fake_fetch(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(fetcher.ccxt_client, "fetch_ohlcv", fake_fetch)

    result = fetcher.fetch_ohlcv("BTC/USDT", "1H", start, 100, params)

    assert result == expected
    assert captured["args"] == (broker, "BTC/USDT")
    assert captured["kwargs"] == {
        "timeframe": "1H",
        "start": start,
        "number_of_ticks": 100,
        "params": params,
    }

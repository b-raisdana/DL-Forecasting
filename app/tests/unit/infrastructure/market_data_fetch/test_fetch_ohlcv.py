from datetime import UTC, datetime

import pytest
from ccxt import RequestTimeout
from infrastructure.market_data_fetch import ccxt_client
from infrastructure.market_data_fetch.binance import fetch_ohlcv as binance_fetcher
from infrastructure.market_data_fetch.kucoin import fetch_ohlcv as kucoin_fetcher


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


@pytest.mark.unit
def test_fetch_ohlcv_uses_distinct_daily_batch_timestamps(monkeypatch):
    requests = []

    class FakeExchange:
        def fetch_ohlcv(self, _symbol, **kwargs):
            requests.append(kwargs)
            return []

    monkeypatch.setattr(ccxt_client, "broker_exchange", lambda _broker: FakeExchange())

    ccxt_client.fetch_ohlcv(
        "kucoin",
        "BTC/USDT",
        timeframe="1D",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        number_of_ticks=1001,
    )

    assert [request["limit"] for request in requests] == [1000, 1]
    assert requests[1]["since"] - requests[0]["since"] == 1000 * 24 * 60 * 60 * 1000


@pytest.mark.unit
def test_fetch_ohlcv_raises_the_final_network_error_after_retries(monkeypatch):
    class TimeoutExchange:
        def fetch_ohlcv(self, _symbol, **_kwargs):
            raise RequestTimeout("timed out")

    monkeypatch.setattr(ccxt_client, "broker_exchange", lambda _broker: TimeoutExchange())

    with pytest.raises(RequestTimeout, match="timed out"):
        ccxt_client.fetch_ohlcv(
            "binance",
            "BTC/USDT",
            timeframe="1D",
            start=datetime(2020, 1, 1, tzinfo=UTC),
            number_of_ticks=1,
        )

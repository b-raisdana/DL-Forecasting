from datetime import UTC, datetime

import pytest
from ccxt import RequestTimeout
from infrastructure.market_data_fetch import ccxt_client


@pytest.mark.unit
def test_fetch_ohlcv_delegates_pagination_to_ccxt(monkeypatch):
    captured = {}
    expected = [[1, 2, 3, 4, 5, 6]]

    class FakeExchange:
        def fetch_paginated_call_deterministic(self, method, symbol, since, limit, timeframe, params, max_entries):
            captured["args"] = (method, symbol, since, limit, timeframe, params, max_entries)
            return expected

    monkeypatch.setattr(ccxt_client, "broker_exchange", lambda _broker: FakeExchange())

    start = datetime(2020, 1, 1, tzinfo=UTC)
    result = ccxt_client.fetch_ohlcv(
        "kucoin",
        "BTC/USDT",
        timeframe="1D",
        start=start,
        number_of_ticks=1001,
    )

    assert result == expected
    method, symbol, since_arg, limit_arg, timeframe_arg, params, max_entries = captured["args"]
    since = int(start.timestamp() * 1000)
    assert method == "fetchOHLCV"
    assert symbol == "BTC/USDT"
    assert since_arg == since
    assert limit_arg == 1001
    assert timeframe_arg == "1d"
    assert max_entries is None
    assert params["until"] == since + 1001 * 24 * 60 * 60 * 1000
    assert params["paginationCalls"] >= 2


@pytest.mark.unit
def test_fetch_ohlcv_skips_the_broker_call_for_zero_ticks(monkeypatch):
    calls = []

    class FakeExchange:
        def fetch_paginated_call_deterministic(self, *args, **kwargs):
            calls.append((args, kwargs))
            return [[1, 2, 3, 4, 5, 6]]

    monkeypatch.setattr(ccxt_client, "broker_exchange", lambda _broker: FakeExchange())

    result = ccxt_client.fetch_ohlcv(
        "kucoin", "BTC/USDT", timeframe="1D", start=datetime(2020, 1, 1, tzinfo=UTC), number_of_ticks=0
    )

    assert result == []
    assert calls == []


@pytest.mark.unit
def test_fetch_ohlcv_propagates_broker_errors(monkeypatch):
    class TimeoutExchange:
        def fetch_paginated_call_deterministic(self, *_args, **_kwargs):
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

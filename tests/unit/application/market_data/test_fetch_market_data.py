import pytest
from application.market_data import fetch_market_data
from config import app_config


@pytest.mark.unit
def test_fetch_and_cache_ohlcv_binds_broker_symbol_and_market_then_delegates(monkeypatch):
    captured = {}
    expected = object()

    def fake_get_base_timeframe_ohlcv(date_range_str, base_timeframe=None):
        captured["date_range_str"] = date_range_str
        captured["base_timeframe"] = base_timeframe
        captured["under_process_exchange"] = app_config.under_process_exchange
        captured["under_process_symbol"] = app_config.under_process_symbol
        captured["under_process_market"] = app_config.under_process_market
        return expected

    monkeypatch.setattr(fetch_market_data, "get_base_timeframe_ohlcv", fake_get_base_timeframe_ohlcv)
    monkeypatch.setattr(app_config, "under_process_exchange", app_config.under_process_exchange)
    monkeypatch.setattr(app_config, "under_process_symbol", app_config.under_process_symbol)
    monkeypatch.setattr(app_config, "under_process_market", app_config.under_process_market)

    result = fetch_market_data.fetch_and_cache_ohlcv(
        broker="binance",
        trading_pair="ETHUSDT",
        date_range_str="24-01-01.00-00T24-01-31.23-59",
        market="Futures",
        base_timeframe="1h",
    )

    assert result is expected
    assert captured == {
        "date_range_str": "24-01-01.00-00T24-01-31.23-59",
        "base_timeframe": "1h",
        "under_process_exchange": "Binance",
        "under_process_symbol": "ETHUSDT",
        "under_process_market": "Futures",
    }


@pytest.mark.unit
def test_fetch_and_cache_ohlcv_rejects_unsupported_broker_without_touching_app_config(monkeypatch):
    called = False

    def fake_get_base_timeframe_ohlcv(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(fetch_market_data, "get_base_timeframe_ohlcv", fake_get_base_timeframe_ohlcv)
    original_symbol = app_config.under_process_symbol

    with pytest.raises(ValueError, match="Unsupported broker"):
        fetch_market_data.fetch_and_cache_ohlcv(
            broker="not-a-broker", trading_pair="BTCUSDT", date_range_str="24-01-01.00-00T24-01-31.23-59"
        )

    assert called is False
    assert app_config.under_process_symbol == original_symbol


@pytest.mark.unit
def test_fill_ohlcv_gaps_logs_and_returns_empty_when_no_gaps(monkeypatch):
    monkeypatch.setattr(fetch_market_data, "find_cache_gaps", lambda *_a, **_k: [])

    def fail(*_a, **_k):
        raise AssertionError("should not be called when there are no gaps")

    monkeypatch.setattr(fetch_market_data, "fetch_ohlcv_by_range", fail)
    monkeypatch.setattr(fetch_market_data, "get_base_timeframe_ohlcv", fail)
    monkeypatch.setattr(fetch_market_data, "cleanup_redundant_cache_files", fail)

    result = fetch_market_data.fill_ohlcv_gaps(
        broker="binance", trading_pair="ETHUSDT", date_range_str="23-01-01.00-00T23-12-31.23-59"
    )

    assert result == []


@pytest.mark.unit
def test_fill_ohlcv_gaps_fetches_newest_gap_first_and_cleans_up(monkeypatch):
    january_gap = "23-01-01.00-00T23-01-02.23-59"
    march_gap = "23-03-01.00-00T23-03-02.23-59"
    monkeypatch.setattr(fetch_market_data, "find_cache_gaps", lambda *_a, **_k: [january_gap, march_gap])

    raw_fetch_calls = []
    monkeypatch.setattr(fetch_market_data, "fetch_ohlcv_by_range", lambda *a, **k: raw_fetch_calls.append(a) or [1])

    generated_calls = []
    monkeypatch.setattr(fetch_market_data, "get_base_timeframe_ohlcv", lambda gap, **k: generated_calls.append(gap))

    cleanup_calls = []
    monkeypatch.setattr(
        fetch_market_data, "cleanup_redundant_cache_files", lambda *a, **k: cleanup_calls.append((a, k))
    )

    result = fetch_market_data.fill_ohlcv_gaps(
        broker="binance", trading_pair="ETHUSDT", date_range_str="23-01-01.00-00T23-12-31.23-59"
    )

    assert result == [march_gap, january_gap]
    assert generated_calls == [march_gap, january_gap]
    assert len(cleanup_calls) == 1


@pytest.mark.unit
def test_fill_ohlcv_gaps_marks_a_confirmed_unavailable_gap_as_empty_without_generating(monkeypatch):
    gap = "23-01-01.00-00T23-01-02.23-59"
    monkeypatch.setattr(fetch_market_data, "find_cache_gaps", lambda *_a, **_k: [gap])

    raw_fetch_calls = []
    monkeypatch.setattr(fetch_market_data, "fetch_ohlcv_by_range", lambda *a, **k: raw_fetch_calls.append(a) or [])

    def fail(*_a, **_k):
        raise AssertionError("should not go through the validating cache-or-generate path")

    monkeypatch.setattr(fetch_market_data, "get_base_timeframe_ohlcv", fail)

    written = {}

    def fake_write(df, data_frame_type, date_range_str, file_path):
        written["df"] = df
        written["data_frame_type"] = data_frame_type
        written["date_range_str"] = date_range_str

    monkeypatch.setattr(fetch_market_data, "write_data_file", fake_write)
    monkeypatch.setattr(fetch_market_data, "cleanup_redundant_cache_files", lambda *_a, **_k: None)

    result = fetch_market_data.fill_ohlcv_gaps(
        broker="binance", trading_pair="ETHUSDT", date_range_str="23-01-01.00-00T23-12-31.23-59"
    )

    assert result == []
    assert len(raw_fetch_calls) == 2  # asked twice before accepting "no data"
    assert written["data_frame_type"] == "ohlcv"
    assert written["date_range_str"] == gap
    assert len(written["df"]) == 0


@pytest.mark.unit
def test_fill_ohlcv_gaps_propagates_a_genuine_broker_error(monkeypatch):
    gap = "23-01-01.00-00T23-01-02.23-59"
    monkeypatch.setattr(fetch_market_data, "find_cache_gaps", lambda *_a, **_k: [gap])

    def raising_fetch(*_a, **_k):
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(fetch_market_data, "fetch_ohlcv_by_range", raising_fetch)

    with pytest.raises(RuntimeError, match="broker unreachable"):
        fetch_market_data.fill_ohlcv_gaps(
            broker="binance", trading_pair="ETHUSDT", date_range_str="23-01-01.00-00T23-12-31.23-59"
        )

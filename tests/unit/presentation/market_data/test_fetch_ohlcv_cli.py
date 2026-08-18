import argparse

import pytest
from config import app_config
from presentation.market_data import fetch_ohlcv_cli


def _args(**overrides):
    defaults = {
        "broker": "binance",
        "trading_pair": "ETHUSDT",
        "date_range": None,
        "days": None,
        "market": "Spot",
        "timeframe": None,
        "list_gaps": False,
        "list_overlaps": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.unit
def test_main_with_no_range_and_no_days_backfills_the_full_configured_span(monkeypatch, capsys):
    captured = {}

    def fake_fill_ohlcv_gaps(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(fetch_ohlcv_cli, "fill_ohlcv_gaps", fake_fill_ohlcv_gaps)

    fetch_ohlcv_cli.main(_args())

    assert captured["broker"] == "binance"
    assert captured["trading_pair"] == "ETHUSDT"
    start, _sep, _end = captured["date_range_str"].partition("T")
    assert start == app_config.ohlcv_oldest_fetch_date
    assert "all up to date" in capsys.readouterr().out


@pytest.mark.unit
def test_main_with_no_range_and_no_days_reports_filled_gaps(monkeypatch, capsys):
    monkeypatch.setattr(fetch_ohlcv_cli, "fill_ohlcv_gaps", lambda **_k: ["23-01-01.00-00T23-01-02.23-59"])

    fetch_ohlcv_cli.main(_args())

    out = capsys.readouterr().out
    assert "filled 1 gap(s)" in out


@pytest.mark.unit
def test_main_with_explicit_date_range_delegates_to_fetch_and_cache(monkeypatch):
    captured = {}

    def fail(**_k):
        raise AssertionError("fill_ohlcv_gaps should not run when --date-range is explicit")

    monkeypatch.setattr(fetch_ohlcv_cli, "fill_ohlcv_gaps", fail)

    def fake_fetch_and_cache(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(fetch_ohlcv_cli, "fetch_and_cache_ohlcv", fake_fetch_and_cache)

    fetch_ohlcv_cli.main(_args(date_range="23-01-01.00-00T23-01-02.23-59"))

    assert captured["date_range_str"] == "23-01-01.00-00T23-01-02.23-59"


@pytest.mark.unit
def test_main_list_gaps_prints_and_does_not_fetch(monkeypatch, capsys):
    def fail(**_k):
        raise AssertionError("--list-gaps must not fetch")

    monkeypatch.setattr(fetch_ohlcv_cli, "fill_ohlcv_gaps", fail)
    monkeypatch.setattr(fetch_ohlcv_cli, "fetch_and_cache_ohlcv", fail)
    monkeypatch.setattr(fetch_ohlcv_cli, "find_cache_gaps", lambda *_a, **_k: ["23-01-01.00-00T23-01-02.23-59"])

    fetch_ohlcv_cli.main(_args(list_gaps=True, date_range="23-01-01.00-00T23-01-31.23-59"))

    assert "1 gap(s)" in capsys.readouterr().out


@pytest.mark.unit
def test_main_list_overlaps_prints_and_does_not_fetch(monkeypatch, capsys):
    def fail(**_k):
        raise AssertionError("--list-overlaps must not fetch")

    monkeypatch.setattr(fetch_ohlcv_cli, "fill_ohlcv_gaps", fail)
    monkeypatch.setattr(fetch_ohlcv_cli, "fetch_and_cache_ohlcv", fail)
    monkeypatch.setattr(
        fetch_ohlcv_cli,
        "find_overlapping_cache_files",
        lambda *_a, **_k: [("23-01-01.00-00T23-01-31.23-59", "feather")],
    )

    fetch_ohlcv_cli.main(_args(list_overlaps=True, date_range="23-01-01.00-00T23-01-31.23-59"))

    assert "1 overlapping cache file(s)" in capsys.readouterr().out

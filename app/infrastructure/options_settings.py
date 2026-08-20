from datetime import datetime
from pathlib import Path
from typing import cast

from diskcache import Cache
from helper.logging.do_log.log_it import log_e
from infrastructure.market_data_fetch.ccxt_client import fetch_oldest_available_timestamp

# Initialize the options_settings cache in the data directory
_options_settings_cache_dir = Path("./data/options_settings")
_options_settings_cache_dir.mkdir(parents=True, exist_ok=True)

options_settings = Cache(str(_options_settings_cache_dir))


def get_oldest_available_timestamp(broker: str, symbol: str) -> datetime:
    """
    Get the oldest available timestamp for a trading pair from the broker.

    First checks the options_settings cache. If not found, fetches from the broker
    and stores the result in the cache for future use.

    Args:
        broker: The broker name (e.g., "binance", "kucoin")
        symbol: The trading pair symbol in app format (e.g., "BTCUSDT")

    Returns:
        The oldest available timestamp as a timezone-aware UTC datetime, or None if unavailable.
    """
    cache_key = f"oldest_timestamp_{broker.lower()}_{symbol}"

    # Check if already cached
    cached_timestamp = options_settings.get(cache_key)
    if cached_timestamp is not None:
        return cast(datetime, cached_timestamp)

    # Fetch from broker
    oldest_timestamp = fetch_oldest_available_timestamp(broker.lower(), symbol)
    if not oldest_timestamp:
        msg = f"Unable to fetch oldest_available_timestamp for {symbol} from {broker}."
        log_e(msg)
        raise RuntimeError(msg)
    # Cache the result (even if None, to avoid repeated failed fetches)
    options_settings.set(cache_key, oldest_timestamp)

    return oldest_timestamp

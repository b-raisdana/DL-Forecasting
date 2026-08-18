from typing import Protocol

import pandas as pd
import pytest


class ZigzagOhlcFactory(Protocol):
    def __call__(self, n: int = ..., atr: float = ...) -> pd.DataFrame: ...


@pytest.fixture
def zigzag_ohlc() -> ZigzagOhlcFactory:
    """Factory for a small, deterministic, non-flat synthetic OHLC frame.

    Each candle's high/low widen slightly as the index advances so that rolling
    max/min/argmax/argmin calculations produce distinct, non-degenerate values.
    """

    def _build(n: int = 12, atr: float = 1.0) -> pd.DataFrame:
        high = [105 + 3 * i + (3 if i % 2 else 0) for i in range(n)]
        low = [95 + 3 * i - (3 if i % 2 else 0) for i in range(n)]
        open_ = [lo + (hi - lo) * 0.3 for hi, lo in zip(high, low, strict=True)]
        close = [lo + (hi - lo) * 0.7 for hi, lo in zip(high, low, strict=True)]
        ohlc = pd.DataFrame(
            {
                "open": open_,
                "high": [float(h) for h in high],
                "low": [float(v) for v in low],
                "close": close,
            }
        )
        ohlc["atr"] = atr
        return ohlc

    return _build

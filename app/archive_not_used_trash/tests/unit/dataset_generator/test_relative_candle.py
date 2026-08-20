"""Unit tests for the archived relative_candle_columns() (relative_candle.py) — see
app/archive_not_used_trash/README.md.
"""

import pytest
from archive_not_used_trash.application.dataset_generation.relative_candle import relative_candle_columns

pytestmark = pytest.mark.unit


def test_relative_candle_columns_lists_the_five_derived_fields() -> None:
    assert relative_candle_columns() == [
        "rel_close",
        "rel_high_close",
        "rel_close_low",
        "gap",
        "rel_candle_height",
    ]

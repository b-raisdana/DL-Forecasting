"""Unit tests for the archived volume_feature_columns()/log_sma_volume_feature_columns()
(volume_feature.py) — see app/archive_not_used_trash/README.md.
"""

import pytest
from archive_not_used_trash.application.dataset_generation.volume_feature import (
    log_sma_volume_feature_columns,
    volume_feature_columns,
)

pytestmark = pytest.mark.unit


def test_volume_feature_columns_lists_the_one_derived_field() -> None:
    assert volume_feature_columns() == ["volume_atr"]


def test_log_sma_volume_feature_columns_lists_the_one_derived_field() -> None:
    assert log_sma_volume_feature_columns() == ["log_volume_sma_ratio"]

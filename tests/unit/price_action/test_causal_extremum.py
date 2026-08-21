"""Unit tests for domain/price_action/CausalExtremum.py's Step A (compute_true_extremum, the
full-hindsight monotonic-stack reach). Step B (observed_extremum_tf_minutes, the causal cap) is
unreachable from any presentation entrypoint and archived — see
tests/archive_not_used_trash/unit/price_action/test_causal_extremum.py.

Values are hand-derived from small, fully-controlled synthetic OHLC fixtures, not captured from
running the function — new spec-conformance code, not legacy behavior to pin. See
docs/ML_Forecasting_System_Design/todo/01-input-data-channels.md for why this causal-cap logic exists
(the part of the schema flagged as "most likely to silently leak future information if implemented naively").
"""

import numpy as np
import pandas as pd
import pandera
import pytest
from domain.price_action.CausalExtremum import (
    TF_MINUTES,
    compute_true_extremum,
    floor_to_tf_ladder,
    plus2tf,
    plus3tf,
)
from domain.schemas.price_action.CausalExtremum import CausalExtremumResult

pytestmark = pytest.mark.unit


def _make_ohlc(high: list[float], low: list[float], freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(high), freq=freq, tz="UTC")
    return pd.DataFrame({"high": high, "low": low}, index=idx)


# --- floor_to_tf_ladder / plus2tf / plus3tf -----------------------------------------------------


def test_floor_to_tf_ladder_snaps_down_to_the_largest_rung_not_exceeding_the_value() -> None:
    assert floor_to_tf_ladder(4.9) == 0.0
    assert floor_to_tf_ladder(5.0) == 5.0
    assert floor_to_tf_ladder(59.9) == 15.0
    assert floor_to_tf_ladder(60.0) == 60.0
    assert floor_to_tf_ladder(np.inf) == TF_MINUTES["1Y"]


def test_floor_to_tf_ladder_is_vectorized_over_an_array() -> None:
    result = floor_to_tf_ladder(np.array([0.0, 4.9, 5.0, 100_000.0]))
    np.testing.assert_array_equal(result, [0.0, 0.0, 5.0, TF_MINUTES["1M"]])


def test_plus2tf_plus3tf_match_the_spec_jsonc_examples() -> None:
    assert plus2tf("5min") == "1h"
    assert plus3tf("5min") == "4h"
    assert plus2tf("1h") == "1D"
    assert plus3tf("1h") == "1W"
    # extends past the 6 real cached branches, per model.py's documented judgment call
    assert plus2tf("1W") == "4M"
    assert plus3tf("1W") == "1Y"


# --- compute_true_extremum (Step A) -------------------------------------------------------------


def test_isolated_spike_and_dip_have_true_extremum_capped_at_top_ladder_rung() -> None:
    """A candle never beaten in either direction across the whole available series (peak at idx5,
    valley at idx2) reaches np.inf, which floors to the ladder's top rung (1Y)."""
    high = [5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5]
    low = [1, 1, -5, 1, 1, 1, 1, 1, 1, 1, 1]
    ohlc = _make_ohlc(high, low)

    result = compute_true_extremum(ohlc)

    assert result["extremum_sign"].iloc[5] == 1
    assert result["true_extremum_tf_minutes"].iloc[5] == TF_MINUTES["1Y"]
    assert result["extremum_sign"].iloc[2] == -1
    assert result["true_extremum_tf_minutes"].iloc[2] == TF_MINUTES["1Y"]


def test_tied_peak_and_valley_reach_breaks_to_peak() -> None:
    """Candle idx4 (same fixture as above): both its peak-reach and valley-reach are exactly one
    native step (5min, immediately matched by a flat neighbor) — a genuine tie. Peak wins ties, a
    documented judgment call (not specified either way in the input spec jsonc)."""
    high = [5, 5, 5, 5, 5, 20, 5, 5, 5, 5, 5]
    low = [1, 1, -5, 1, 1, 1, 1, 1, 1, 1, 1]
    ohlc = _make_ohlc(high, low)

    result = compute_true_extremum(ohlc)

    assert result["extremum_sign"].iloc[4] == 1
    assert result["true_extremum_tf_minutes"].iloc[4] == 5.0


def test_flat_run_of_equal_highs_yields_only_native_step_reach() -> None:
    """Mirrors PeakValley.py's own same-value-run convention (a run of identical highs shouldn't
    register as an isolated strong peak): every candle inside a flat run of equal highs is
    immediately "matched" by its neighbor (>=, not strict >), capping reach at exactly one native
    step, never something larger — no dedup preprocessing needed, the >=/<= comparison already
    handles it."""
    high = [5.0, 10.0, 10.0, 10.0, 5.0]
    low = [1.0, 1.0, 1.0, 1.0, 1.0]
    ohlc = _make_ohlc(high, low)

    result = compute_true_extremum(ohlc)

    for i in (1, 2, 3):
        assert result["true_peak_reach_minutes"].iloc[i] == 5.0


def test_reach_uses_real_elapsed_minutes_not_a_fixed_step_assumption() -> None:
    """A gap in the DatetimeIndex (e.g. missing candles) must reflect in the elapsed-minutes reach,
    not an assumed uniform step — compute_true_extremum reads real timestamp deltas."""
    idx = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 01:05"],
        tz="UTC",  # 60min gap before the 3rd candle
    )
    ohlc = pd.DataFrame({"high": [5.0, 20.0, 5.0], "low": [1.0, 1.0, 1.0]}, index=idx)

    result = compute_true_extremum(ohlc)

    # candle 0's nearest right-beating neighbor (high>=5) is candle 1 (20>=5), 5 minutes away
    assert result["true_peak_reach_minutes"].iloc[0] == 5.0


# --- compute_true_extremum validation guards (schema regression) ---------------------------------


def test_missing_high_column_raises_schema_errors() -> None:
    """If the input is missing the required 'high' column, compute_true_extremum must raise
    pandera.errors.SchemaErrors — never silently produce garbage."""
    ohlc = pd.DataFrame({"low": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"))

    with pytest.raises(pandera.errors.SchemaErrors):
        compute_true_extremum(ohlc)


def test_missing_low_column_raises_schema_errors() -> None:
    """If the input is missing the required 'low' column, compute_true_extremum must raise
    pandera.errors.SchemaErrors — never silently produce garbage."""
    ohlc = pd.DataFrame({"high": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"))

    with pytest.raises(pandera.errors.SchemaErrors):
        compute_true_extremum(ohlc)


def test_wrong_dtype_high_raises_schema_errors() -> None:
    """If 'high' is string instead of float, compute_true_extremum must raise SchemaErrors."""
    ohlc = pd.DataFrame(
        {"high": ["a", "b", "c"], "low": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        compute_true_extremum(ohlc)


def test_wrong_dtype_low_raises_schema_errors() -> None:
    """If 'low' is string instead of float, compute_true_extremum must raise SchemaErrors."""
    ohlc = pd.DataFrame(
        {"high": [1.0, 2.0, 3.0], "low": ["a", "b", "c"]},
        index=pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        compute_true_extremum(ohlc)


def test_result_missing_required_column_raises_schema_errors() -> None:
    """If a post-processing step drops a required column from the result, the output validation
    must catch it — guarantees the returned DataFrame always has the documented 4 columns."""
    ohlc = _make_ohlc([5, 20, 5], [1, 1, 1])
    result = compute_true_extremum(ohlc)

    # Simulate a broken post-processing step that drops a column
    broken = result.drop(columns=["extremum_sign"])

    with pytest.raises(pandera.errors.SchemaErrors):
        CausalExtremumResult.validate(broken, lazy=True)


def test_result_wrong_index_dtype_raises_schema_errors() -> None:
    """If the result's DatetimeIndex is microsecond-precision instead of nanosecond, the output
    validation must catch it — guards against pandas 3.x us-default regressions."""
    ohlc = _make_ohlc([5, 20, 5], [1, 1, 1])
    result = compute_true_extremum(ohlc)
    # Force the index to microsecond precision (pandas 3.x default)
    broken = result.copy()
    broken.index = broken.index.astype("datetime64[us, UTC]")

    with pytest.raises(pandera.errors.SchemaErrors):
        CausalExtremumResult.validate(broken, lazy=True)

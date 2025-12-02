"""
Property-based tests for position sizing utilities.
TODO: Add hypothesis to the test dependencies if it is not already installed.
"""

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st  # type: ignore  # noqa: E402

from finantradealgo.risk.position_sizing import PositionSizingInput, calc_size_atr_stop

pytestmark = pytest.mark.property


@given(
    equity=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=1e-6, max_value=1e4, allow_nan=False, allow_infinity=False),
    atr=st.one_of(
        st.none(),
        st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False),
    ),
    risk_pct=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    max_notional=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)
def test_calc_size_atr_stop_is_non_negative(equity, price, atr, risk_pct, max_notional):
    inp = PositionSizingInput(
        equity=equity,
        price=price,
        atr=atr,
        capital_risk_pct_per_trade=risk_pct,
        max_notional_per_symbol=max_notional,
    )

    size = calc_size_atr_stop(inp)

    assert size >= 0.0
    if max_notional:
        assert size * price <= max_notional + 1e-6


@given(
    equity=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
    atr=st.one_of(
        st.none(),
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    ),
    risk_pct=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_position_size_monotonic_with_equity(equity, delta, price, atr, risk_pct):
    inp_low = PositionSizingInput(
        equity=equity,
        price=price,
        atr=atr,
        capital_risk_pct_per_trade=risk_pct,
    )
    inp_high = PositionSizingInput(
        equity=equity + delta,
        price=price,
        atr=atr,
        capital_risk_pct_per_trade=risk_pct,
    )

    size_low = calc_size_atr_stop(inp_low)
    size_high = calc_size_atr_stop(inp_high)

    assert size_high >= size_low - 1e-9

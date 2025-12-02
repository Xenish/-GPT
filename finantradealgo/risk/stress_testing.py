from __future__ import annotations

"""
Stress testing utilities for historical and hypothetical scenarios.

Design notes:
- Single-asset aggregate return stream only (no covariance / multi-asset risk yet).
- Hypothetical shocks are intentionally simple: users can specify a crash, a shift,
  and/or a volatility multiplier via `scenario.shocks`, e.g.:
    {"shock_pct": -0.1}            -> apply a one-period -10% return
    {"shift": -0.01}               -> subtract 1% from every return
    {"vol_mult": 2.0}              -> double deviations from the mean (volatility amp)
    {"scale_all": 1.2}             -> scale all returns by 1.2
  These can be combined; they are applied in the order above for transparency.
"""

from dataclasses import dataclass
import numpy as np

from finantradealgo.risk import (
    ReturnSeriesLike,
    StressScenario,
    StressScenarioType,
    StressTestResult,
)


def max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum peak-to-trough drawdown of an equity curve."""
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return float(-drawdowns.min()) if drawdowns.size else 0.0


def apply_returns_to_equity(initial_equity: float, returns: np.ndarray) -> np.ndarray:
    """
    Generate an equity curve from an array of returns.

    Uses cumulative compounding; returns are assumed to be simple (not log) returns.
    """
    if returns.size == 0:
        return np.array([initial_equity], dtype=float)
    growth = np.cumprod(1.0 + returns)
    return initial_equity * growth


def _returns_from_equity_curve(equity: np.ndarray) -> np.ndarray:
    """Infer period returns from an equity curve."""
    if equity.size < 2:
        return np.array([], dtype=float)
    prev = equity[:-1]
    curr = equity[1:]
    valid = prev > 0
    returns = np.zeros_like(prev, dtype=float)
    returns[valid] = (curr[valid] / prev[valid]) - 1.0
    return returns


def _to_np_array(series: ReturnSeriesLike | None) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    if hasattr(series, "to_numpy"):
        return np.asarray(series.to_numpy(), dtype=float).reshape(-1)
    return np.asarray(series, dtype=float).reshape(-1)


def _apply_hypothetical_shocks(base_returns: np.ndarray, shocks: dict[str, float]) -> np.ndarray:
    """
    Apply simple hypothetical shocks to a return path.

    Supported keys (all optional):
    - shock_pct: single-period shock applied to the first period (e.g., -0.1 for -10%).
    - shift: additive shift applied to all returns.
    - vol_mult: multiplies deviations from the mean to mimic volatility change.
    - scale_all: scales all returns by a constant factor.
    """
    if base_returns.size == 0:
        return base_returns

    stressed = base_returns.astype(float).copy()
    shock_pct = shocks.get("shock_pct")
    if shock_pct is not None:
        stressed[0] = shock_pct

    shift = shocks.get("shift")
    if shift is not None:
        stressed = stressed + float(shift)

    vol_mult = shocks.get("vol_mult")
    if vol_mult is not None:
        mean = stressed.mean()
        stressed = mean + (stressed - mean) * float(vol_mult)

    scale_all = shocks.get("scale_all")
    if scale_all is not None:
        stressed = stressed * float(scale_all)

    return stressed


@dataclass
class StressTester:
    """
    Stress testing engine for historical and hypothetical scenarios on a single return stream.
    """

    initial_equity: float = 1_000_000.0
    base_returns: np.ndarray | None = None

    def run_scenario(
        self,
        scenario: StressScenario,
        *,
        base_returns: np.ndarray | None = None,
        base_equity: np.ndarray | None = None,
    ) -> StressTestResult:
        """
        Apply a stress scenario to the provided returns/equity stream and compute summary metrics.

        Priority for baseline data:
        1) base_returns param
        2) base_equity param (converted to returns)
        3) self.base_returns
        """
        base_ret = _to_np_array(base_returns)
        if base_ret.size == 0 and base_equity is not None:
            equity_arr = _to_np_array(base_equity)
            base_ret = _returns_from_equity_curve(equity_arr)
        if base_ret.size == 0 and self.base_returns is not None:
            base_ret = _to_np_array(self.base_returns)

        stressed_returns: np.ndarray
        if scenario.scenario_type == StressScenarioType.HISTORICAL and scenario.shock_returns is not None:
            stressed_returns = np.asarray(scenario.shock_returns, dtype=float).reshape(-1)
        elif scenario.scenario_type == StressScenarioType.HYPOTHETICAL and scenario.shocks is not None:
            stressed_returns = _apply_hypothetical_shocks(base_ret, scenario.shocks)
        else:
            stressed_returns = base_ret

        equity_curve = apply_returns_to_equity(self.initial_equity, stressed_returns)
        pnl_series = list((equity_curve - self.initial_equity).tolist())
        mdd = max_drawdown(equity_curve)
        worst_return = float(stressed_returns.min()) if stressed_returns.size else 0.0
        var_like_loss = max(0.0, -worst_return) * self.initial_equity

        summary_metrics: dict[str, float] = {
            "final_equity": float(equity_curve[-1]) if equity_curve.size else self.initial_equity,
            "total_pnl": float(equity_curve[-1] - self.initial_equity) if equity_curve.size else 0.0,
            "max_drawdown": mdd,
            "var_like_loss": var_like_loss,
            "min_equity": float(equity_curve.min()) if equity_curve.size else self.initial_equity,
            "worst_return": worst_return,
        }

        return StressTestResult(
            scenario=scenario,
            pnl_series=pnl_series,
            equity_curve=equity_curve.tolist(),
            max_drawdown=mdd,
            var_like_loss=var_like_loss,
            summary_metrics=summary_metrics,
            metadata={"base_returns_len": int(base_ret.size)},
        )

    def run_scenarios(
        self,
        scenarios: list[StressScenario],
        *,
        base_returns: np.ndarray | None = None,
        base_equity: np.ndarray | None = None,
    ) -> list[StressTestResult]:
        """Run multiple scenarios sequentially."""
        results: list[StressTestResult] = []
        for scenario in scenarios:
            results.append(
                self.run_scenario(
                    scenario,
                    base_returns=base_returns,
                    base_equity=base_equity,
                )
            )
        return results

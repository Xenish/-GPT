from __future__ import annotations

"""
Tail risk analysis helpers (Expected Shortfall, drawdowns, EVT, ratios).
"""

import math

import numpy as np

from finantradealgo.risk import (
    ReturnSeriesLike,
    RiskMetricConfig,
    TailRiskMetrics,
    VaRMethod,
)
from finantradealgo.risk.var_calculator import VaRCalculator


def _to_np_array(data: ReturnSeriesLike | None) -> np.ndarray:
    if data is None:
        return np.array([], dtype=float)
    if hasattr(data, "to_numpy"):
        return np.asarray(data.to_numpy(), dtype=float).reshape(-1)
    return np.asarray(data, dtype=float).reshape(-1)


def _equity_from_returns(returns: np.ndarray, initial_equity: float = 1.0) -> np.ndarray:
    if returns.size == 0:
        return np.array([initial_equity], dtype=float)
    growth = np.cumprod(1.0 + returns)
    return initial_equity * growth


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    return float(-dd.min()) if dd.size else 0.0


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, *, annualization_factor: float = 252.0) -> float:
    """Annualized Sharpe ratio using excess returns over risk-free."""
    if returns.size == 0:
        return 0.0
    excess = returns - risk_free / annualization_factor
    vol = excess.std(ddof=1)
    if vol <= 0:
        return 0.0
    return float(excess.mean() * math.sqrt(annualization_factor) / vol)


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0, *, annualization_factor: float = 252.0) -> float:
    """Annualized Sortino ratio using downside deviation only."""
    if returns.size == 0:
        return 0.0
    excess = returns - risk_free / annualization_factor
    downside = excess[excess < 0.0]
    dd = downside.std(ddof=1) if downside.size else 0.0
    if dd <= 0:
        return 0.0
    return float(excess.mean() * math.sqrt(annualization_factor) / dd)


def calmar_ratio(
    returns: np.ndarray,
    equity: np.ndarray | None = None,
    *,
    annualization_factor: float = 252.0,
) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    if returns.size == 0:
        return 0.0
    eq = _equity_from_returns(returns) if equity is None else np.asarray(equity, dtype=float)
    mdd = _max_drawdown(eq)
    total_return = float(eq[-1] / eq[0] - 1.0) if eq.size else 0.0
    periods = len(returns)
    if periods <= 0:
        return 0.0
    ann_return = (1.0 + total_return) ** (annualization_factor / periods) - 1.0
    if mdd <= 0:
        return float("inf") if ann_return > 0 else 0.0
    return float(ann_return / mdd)


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Empirical Omega ratio: gains above threshold / losses below threshold.
    """
    if returns.size == 0:
        return 0.0
    gains = np.clip(returns - threshold, a_min=0.0, a_max=None)
    losses = np.clip(threshold - returns, a_min=0.0, a_max=None)
    num = gains.sum()
    den = losses.sum()
    if den <= 0:
        return float("inf") if num > 0 else 0.0
    return float(num / den)


def information_ratio(
    active_returns: np.ndarray,
    *,
    tracking_error_annualized: bool = True,
    annualization_factor: float = 252.0,
) -> float:
    """
    Information ratio = annualized mean active return / annualized tracking error.
    active_returns are assumed daily unless already annualized.
    """
    if active_returns.size == 0:
        return 0.0
    mean_active = active_returns.mean()
    te = active_returns.std(ddof=1)
    if tracking_error_annualized:
        te *= math.sqrt(annualization_factor)
    ann_mean = mean_active * annualization_factor
    if te <= 0:
        return 0.0
    return float(ann_mean / te)


def _hill_tail_index(losses: np.ndarray, tail_fraction: float = 0.1) -> float | None:
    """
    Very simple Hill estimator for left-tail heaviness on loss magnitudes.
    """
    losses = losses[losses > 0]
    if losses.size < 5:
        return None
    losses_sorted = np.sort(losses)[::-1]
    k = max(5, int(losses_sorted.size * tail_fraction))
    k = min(k, losses_sorted.size)
    if k <= 1 or losses_sorted[k - 1] <= 0:
        return None
    top = losses_sorted[:k]
    denom = losses_sorted[k - 1]
    hill = np.log(top / denom).mean()
    return float(hill) if hill > 0 else None


def compute_tail_risk_metrics(
    returns: ReturnSeriesLike,
    config: RiskMetricConfig,
    *,
    use_parametric: bool = False,
) -> TailRiskMetrics:
    """
    Aggregate tail risk metrics based on VaR/CVaR and auxiliary ratios.

    - Historical VaR/CVaR by default; parametric can be included via `use_parametric`.
    - Tail index is an optional Hill estimator on loss magnitudes (left tail).
    """
    arr = _to_np_array(returns)
    var_calc = VaRCalculator(config)

    hist_res = var_calc.compute(arr, VaRMethod.HISTORICAL)
    var_value = hist_res.var_value
    cvar_value = hist_res.cvar_value or var_value
    extra: dict[str, float] | None = None

    if use_parametric:
        param_res = var_calc.compute(arr, VaRMethod.PARAMETRIC)
        extra = {
            "parametric_var": param_res.var_value,
            "parametric_cvar": param_res.cvar_value or param_res.var_value,
        }

    equity_curve = _equity_from_returns(arr)
    mdd = _max_drawdown(equity_curve)
    tail_losses = np.clip(-arr, a_min=0.0, a_max=None)
    tail_index = _hill_tail_index(tail_losses)

    omega = omega_ratio(arr)
    ir = None  # Active returns not provided; placeholder for future extension.

    return TailRiskMetrics(
        confidence_level=config.confidence_level,
        var=var_value,
        cvar=cvar_value,
        expected_max_drawdown=mdd,
        tail_index=tail_index,
        omega_ratio=omega,
        information_ratio=ir,
        extra=extra,
    )


def risk_adjusted_position_size(
    capital: float,
    returns: ReturnSeriesLike,
    *,
    config: RiskMetricConfig,
    target_var_fraction: float = 0.01,
    method: VaRMethod = VaRMethod.HISTORICAL,
) -> float:
    """
    Compute position size such that 1-day VaR is approximately capital * target_var_fraction.

    Assumes linear scaling of P&L with position size and single-asset exposure.
    """
    if capital <= 0 or target_var_fraction <= 0:
        return 0.0
    var_res = VaRCalculator(config).compute(returns, method)
    var_per_unit = float(var_res.var_value)
    if var_per_unit <= 0:
        return 0.0
    return (capital * target_var_fraction) / var_per_unit

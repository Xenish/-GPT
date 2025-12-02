from __future__ import annotations

"""
Value at Risk calculations (historical, parametric, Monte Carlo) and Expected Shortfall helpers.

Design assumptions:
- Single-asset daily returns as input.
- Parametric VaR uses a Gaussian assumption with sqrt-time scaling for horizon (approximation).
- Monte Carlo currently simulates from a normal distribution; distribution can be swapped later.
"""

import math
from statistics import NormalDist
from typing import Callable

import numpy as np

try:  # Optional dependency; falls back to statistics.NormalDist if missing.
    from scipy.stats import norm as _scipy_norm  # type: ignore
except Exception:  # pragma: no cover - absence is fine
    _scipy_norm = None

from finantradealgo.risk import ReturnSeriesLike, RiskMetricConfig, VaRMethod, VaRResult


def to_return_array(returns: ReturnSeriesLike, use_log_returns: bool) -> np.ndarray:
    """
    Convert a return-like input into a clean 1D numpy array.

    Optionally converts simple returns to log-returns via log1p.
    Filters out non-finite values.
    """
    if returns is None:
        return np.array([], dtype=float)

    if hasattr(returns, "to_numpy"):
        arr = np.asarray(returns.to_numpy(), dtype=float).reshape(-1)
    else:
        arr = np.asarray(returns, dtype=float).reshape(-1)

    if use_log_returns:
        arr = np.log1p(arr)
    arr = arr[np.isfinite(arr)]
    return arr


def scale_horizon(var_1d: float, horizon_days: int) -> float:
    """
    Scale 1-day VaR to a multi-day horizon using the square-root-of-time rule (approximation).
    """
    horizon = max(1, int(horizon_days))
    return var_1d * math.sqrt(horizon)


def _normal_quantile(p: float) -> float:
    """Return the standard normal quantile for probability p."""
    if _scipy_norm is not None:
        return float(_scipy_norm.ppf(p))
    return float(NormalDist().inv_cdf(p))


def _normal_pdf(z: float) -> float:
    """Standard normal PDF at z."""
    if _scipy_norm is not None:
        return float(_scipy_norm.pdf(z))
    return float(NormalDist().pdf(z))


class VaRCalculator:
    """Compute VaR and CVaR using historical, parametric (Gaussian), and Monte Carlo approaches."""

    def __init__(self, config: RiskMetricConfig):
        self.config = config

    def _alpha(self) -> float:
        cl = float(self.config.confidence_level)
        if not 0.0 < cl < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")
        return 1.0 - cl

    def historical_var(self, returns: ReturnSeriesLike) -> VaRResult:
        """
        Empirical (historical) VaR/CVaR using observed return distribution.
        VaR is reported as a positive loss threshold.
        """
        arr = to_return_array(returns, self.config.use_log_returns)
        sample_size = len(arr)
        if sample_size == 0:
            raise ValueError("No returns provided for historical VaR.")

        cl = float(self.config.confidence_level)
        alpha = self._alpha()
        cutoff = np.quantile(arr, alpha)
        var_value = max(0.0, -float(cutoff))

        tail = arr[arr <= cutoff]
        if tail.size == 0:
            cvar_value = var_value
        else:
            cvar_value = max(0.0, -float(tail.mean()))

        return VaRResult(
            method=VaRMethod.HISTORICAL,
            confidence_level=cl,
            horizon_days=self.config.horizon_days,
            var_value=scale_horizon(var_value, self.config.horizon_days),
            cvar_value=scale_horizon(cvar_value, self.config.horizon_days),
            sample_size=sample_size,
            metadata={"cutoff_return": float(cutoff)},
        )

    def parametric_var(self, returns: ReturnSeriesLike) -> VaRResult:
        """
        Parametric Gaussian VaR/CVaR.

        Assumes returns ~ N(mu, sigma^2). Uses sqrt-time scaling for horizon (approximation).
        CVaR uses the closed-form for a normal left-tail expected shortfall.
        """
        arr = to_return_array(returns, self.config.use_log_returns)
        sample_size = len(arr)
        if sample_size == 0:
            raise ValueError("No returns provided for parametric VaR.")

        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if sample_size > 1 else 0.0
        cl = float(self.config.confidence_level)
        alpha = self._alpha()
        z = _normal_quantile(alpha)

        horizon = max(1, int(self.config.horizon_days))
        mu_h = mu * horizon
        sigma_h = sigma * math.sqrt(horizon)

        var_return = mu_h + sigma_h * z
        var_value = max(0.0, -var_return)

        if sigma_h > 0:
            pdf_z = _normal_pdf(z)
            cvar_return = mu_h - sigma_h * (pdf_z / alpha)
        else:
            cvar_return = var_return
        cvar_value = max(0.0, -cvar_return)

        return VaRResult(
            method=VaRMethod.PARAMETRIC,
            confidence_level=cl,
            horizon_days=horizon,
            var_value=var_value,
            cvar_value=cvar_value,
            sample_size=sample_size,
            metadata={"mu": mu, "sigma": sigma, "z_score": z},
        )

    def monte_carlo_var(self, returns: ReturnSeriesLike, n_sims: int = 10000) -> VaRResult:
        """
        Monte Carlo VaR/CVaR by simulating returns from a Gaussian fit.

        Keeps the distribution assumption simple for now (normal with sample mu/sigma),
        making it easy to swap in heavier-tailed distributions later.
        """
        arr = to_return_array(returns, self.config.use_log_returns)
        sample_size = len(arr)
        if sample_size == 0:
            raise ValueError("No returns provided for Monte Carlo VaR.")

        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if sample_size > 1 else 0.0
        cl = float(self.config.confidence_level)
        alpha = self._alpha()

        horizon = max(1, int(self.config.horizon_days))
        rng = np.random.default_rng()
        sims = rng.normal(mu, sigma, size=(n_sims, horizon))
        aggregated = sims.sum(axis=1)

        cutoff = np.quantile(aggregated, alpha)
        var_value = max(0.0, -float(cutoff))
        tail = aggregated[aggregated <= cutoff]
        if tail.size == 0:
            cvar_value = var_value
        else:
            cvar_value = max(0.0, -float(tail.mean()))

        return VaRResult(
            method=VaRMethod.MONTE_CARLO,
            confidence_level=cl,
            horizon_days=horizon,
            var_value=var_value,
            cvar_value=cvar_value,
            sample_size=sample_size,
            metadata={"mu": mu, "sigma": sigma, "n_sims": n_sims},
        )

    def compute(self, returns: ReturnSeriesLike, method: VaRMethod, **kwargs) -> VaRResult:
        """Dispatch to the requested VaR/CVaR computation method."""
        dispatch: dict[VaRMethod, Callable[..., VaRResult]] = {
            VaRMethod.HISTORICAL: self.historical_var,
            VaRMethod.PARAMETRIC: self.parametric_var,
            VaRMethod.MONTE_CARLO: self.monte_carlo_var,
        }
        fn = dispatch.get(method)
        if fn is None:
            raise ValueError(f"Unsupported VaR method: {method}")
        return fn(returns, **kwargs)

"""
Production Readiness Validator.

Validates if strategy is ready for production deployment.
"""

from __future__ import annotations

from typing import List

from finantradealgo.research.livetesting.models import (
    LiveTestResult,
    ProductionReadiness,
)


class ProductionReadinessValidator:
    """Validate production readiness of strategy."""

    def __init__(
        self,
        min_test_duration_hours: float = 24.0,
        min_trades: int = 50,
        min_sharpe: float = 0.5,
        max_drawdown_threshold: float = 20.0,
        max_slippage_bps: float = 10.0,
    ):
        """Initialize validator."""
        self.min_test_duration_hours = min_test_duration_hours
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_drawdown_threshold = max_drawdown_threshold
        self.max_slippage_bps = max_slippage_bps

    def validate(self, result: LiveTestResult) -> ProductionReadiness:
        """Validate live test result for production readiness."""
        passed = []
        failed = []
        warnings = []

        # 1. Test duration
        if result.duration_hours >= self.min_test_duration_hours:
            passed.append(f"Test duration: {result.duration_hours:.1f}h >= {self.min_test_duration_hours}h")
        else:
            failed.append(f"Insufficient test duration: {result.duration_hours:.1f}h < {self.min_test_duration_hours}h")

        # 2. Trade count
        if result.total_trades >= self.min_trades:
            passed.append(f"Trade count: {result.total_trades} >= {self.min_trades}")
        else:
            failed.append(f"Insufficient trades: {result.total_trades} < {self.min_trades}")

        # 3. Sharpe ratio
        if result.sharpe_ratio >= self.min_sharpe:
            passed.append(f"Sharpe ratio: {result.sharpe_ratio:.2f} >= {self.min_sharpe}")
        else:
            failed.append(f"Low Sharpe ratio: {result.sharpe_ratio:.2f} < {self.min_sharpe}")

        # 4. Max drawdown
        if abs(result.max_drawdown_pct) <= self.max_drawdown_threshold:
            passed.append(f"Max drawdown: {result.max_drawdown_pct:.1f}% <= {self.max_drawdown_threshold}%")
        else:
            failed.append(f"Excessive drawdown: {result.max_drawdown_pct:.1f}% > {self.max_drawdown_threshold}%")

        # 5. Slippage
        if result.avg_slippage_bps <= self.max_slippage_bps:
            passed.append(f"Avg slippage: {result.avg_slippage_bps:.1f} bps <= {self.max_slippage_bps} bps")
        else:
            warnings.append(f"High slippage: {result.avg_slippage_bps:.1f} bps > {self.max_slippage_bps} bps")

        # 6. Profitability
        if result.total_pnl > 0:
            passed.append(f"Profitable: {result.total_pnl:.2f}")
        else:
            failed.append(f"Unprofitable: {result.total_pnl:.2f}")

        # Calculate overall score
        total_checks = 6
        passed_count = len(passed)
        overall_score = (passed_count / total_checks) * 100

        is_ready = len(failed) == 0 and overall_score >= 80

        return ProductionReadiness(
            is_ready=is_ready,
            overall_score=overall_score,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
        )

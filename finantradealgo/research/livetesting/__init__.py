"""
Live Testing Framework.

Paper trading and production readiness validation.
"""

from finantradealgo.research.livetesting.models import (
    OrderSide,
    OrderType,
    OrderStatus,
    SlippageModel,
    LiveTestConfig,
    PaperOrder,
    Position,
    LiveTestResult,
    ProductionReadiness,
)
from finantradealgo.research.livetesting.paper_trader import PaperTradingEngine
from finantradealgo.research.livetesting.readiness import ProductionReadinessValidator

__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "SlippageModel",
    # Models
    "LiveTestConfig",
    "PaperOrder",
    "Position",
    "LiveTestResult",
    "ProductionReadiness",
    # Engine
    "PaperTradingEngine",
    # Validator
    "ProductionReadinessValidator",
]

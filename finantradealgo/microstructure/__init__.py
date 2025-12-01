"""
Microstructure module for market microstructure feature computation.

Task S2.2: Single entry-point exports.
"""
from finantradealgo.microstructure.config import MicrostructureConfig
from finantradealgo.microstructure.microstructure_engine import (
    compute_microstructure_df,
    compute_microstructure_features,
)
from finantradealgo.microstructure.types import MicrostructureSignals

__all__ = [
    "MicrostructureConfig",
    "compute_microstructure_df",
    "compute_microstructure_features",
    "MicrostructureSignals",
]

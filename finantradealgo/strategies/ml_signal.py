from __future__ import annotations

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext
from finantradealgo.ml.model import SklearnLongModel


class MLSignalStrategy(BaseStrategy):
    def __init__(
        self,
        model: SklearnLongModel,
        feature_cols: list[str],
        proba_entry: float = 0.55,
        proba_exit: float = 0.50,
        warmup_bars: int = 100,
    ):
        if proba_exit > proba_entry:
            raise ValueError("proba_exit must be <= proba_entry.")

        self.model = model
        self.feature_cols = feature_cols
        self.proba_entry = proba_entry
        self.proba_exit = proba_exit
        self.warmup_bars = warmup_bars
        self.df: pd.DataFrame | None = None

    def init(self, df: pd.DataFrame) -> None:
        self.df = df
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in df: {missing}")

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        if self.df is None:
            return None

        i = ctx.index
        if i < self.warmup_bars:
            return None

        row_df = row[self.feature_cols].astype(float).to_frame().T
        proba = float(self.model.predict_proba(row_df)[0, 1])

        if ctx.position is None:
            if proba >= self.proba_entry:
                return "LONG"
        else:
            if proba <= self.proba_exit:
                return "CLOSE"

        return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience helper to score a full DataFrame and emit proba/signal columns.

        Returns a copy with:
          - ml_long_proba: model probabilities
          - ml_long_signal: 1.0 when proba >= entry_threshold (post-warmup), 0.0 when proba <= exit_threshold, NaN otherwise
        """
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in df: {missing}")

        out = df.copy()
        proba = self.model.predict_proba(out[self.feature_cols])[:, 1].astype(float)
        out["ml_long_proba"] = proba

        signals = pd.Series(float("nan"), index=out.index, dtype=float)
        for idx, p in enumerate(proba):
            if idx < self.warmup_bars:
                continue
            if p >= self.proba_entry:
                signals.iloc[idx] = 1.0
            elif p <= self.proba_exit:
                signals.iloc[idx] = 0.0

        out["ml_long_signal"] = signals
        return out

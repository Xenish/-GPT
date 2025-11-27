from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import pandas as pd

from finantradealgo.backtester.backtest_engine import BacktestEngine


@dataclass
class PortfolioBacktestResult:
    portfolio_equity: pd.Series
    symbol_equity: Dict[str, pd.Series]
    trades: pd.DataFrame
    metrics: Dict[str, float]


class PortfolioBacktestEngine:
    def __init__(
        self,
        engines: Dict[Tuple[str, str], BacktestEngine],
        data: Dict[Tuple[str, str], pd.DataFrame],
        capital_allocation: Dict[str, float],
        initial_capital: float = 1_000.0,
    ):
        self.engines = engines
        self.data = data
        self.capital_allocation = capital_allocation
        self.initial_capital = initial_capital

    def run(self) -> PortfolioBacktestResult:
        symbol_equity: Dict[str, pd.Series] = {}
        trades_frames = []
        all_times: set[pd.Timestamp] = set()

        # Run each engine and collect equity/trades
        for (symbol, strategy_name), engine in self.engines.items():
            df = self.data[(symbol, strategy_name)]
            result = engine.run(df)
            eq = result.get("equity_curve")
            if eq is None or eq.empty:
                continue
            eq.name = symbol
            all_times.update(eq.index)
            trades_df = result.get("trades", pd.DataFrame())
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                trades_df = trades_df.copy()
                trades_df["symbol"] = symbol
                trades_df["strategy"] = strategy_name
                trades_frames.append(trades_df)
            symbol_equity[symbol] = eq

        if not all_times:
            return PortfolioBacktestResult(
                portfolio_equity=pd.Series(dtype=float),
                symbol_equity=symbol_equity,
                trades=pd.DataFrame(),
                metrics={},
            )

        all_times_sorted = sorted(all_times)
        # Align and forward-fill equities
        aligned = {}
        for sym, eq in symbol_equity.items():
            aligned_eq = eq.reindex(all_times_sorted).ffill()
            aligned[sym] = aligned_eq

        # Build portfolio equity from weighted relative returns
        portfolio_series = pd.Series(index=all_times_sorted, dtype=float)
        portfolio_series.iloc[:] = 0.0
        for sym, eq in aligned.items():
            weight = self.capital_allocation.get(sym, 0.0)
            if eq.iloc[0] == 0:
                rel = eq * 0.0 + 1.0
            else:
                rel = eq / eq.iloc[0]
            contribution = rel * (self.initial_capital * weight)
            portfolio_series = portfolio_series.add(contribution, fill_value=0.0)
            aligned[sym] = contribution.rename(sym)

        portfolio_series.name = "portfolio_equity"

        all_trades = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()

        metrics: Dict[str, float] = {}
        if not portfolio_series.empty:
            eq = portfolio_series
            returns = eq.pct_change().dropna()
            metrics["final_equity"] = float(eq.iloc[-1])
            start = float(eq.iloc[0])
            metrics["cum_return"] = float(eq.iloc[-1] / start - 1) if start else 0.0
            max_equity = eq.cummax()
            dd = (eq - max_equity) / max_equity
            metrics["max_drawdown"] = float(dd.min())
            metrics["sharpe"] = float((returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() != 0 else 0.0)

        return PortfolioBacktestResult(
            portfolio_equity=portfolio_series,
            symbol_equity=aligned,
            trades=all_trades,
            metrics=metrics,
        )


__all__ = ["PortfolioBacktestEngine", "PortfolioBacktestResult"]

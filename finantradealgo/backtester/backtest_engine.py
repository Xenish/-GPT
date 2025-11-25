from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from finantradealgo.core.portfolio import Portfolio, Position
from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_pct: float = 0.0004
    slippage_pct: float = 0.0005
    use_bar_extremes_for_stop: bool = True
    flip_on_opposite_signal: bool = True


class Backtester:
    def __init__(
        self,
        strategy: BaseStrategy,
        risk_engine: Optional[RiskEngine] = None,
        config: Optional[BacktestConfig] = None,
    ):
        self.strategy = strategy
        self.risk_engine = risk_engine or RiskEngine(RiskConfig())
        self.config = config or BacktestConfig()

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            raise ValueError("DataFrame is empty")

        portfolio = Portfolio(
            initial_cash=self.config.initial_cash,
            cash=self.config.initial_cash,
            equity=self.config.initial_cash,
        )

        self.strategy.init(df)
        trades = []

        for i, row in df.iterrows():
            price_close = float(row["close"])
            ts = row["timestamp"]

            if portfolio.position is not None:
                position = portfolio.position
                exit_reason = None
                exit_price = None

                if self.config.use_bar_extremes_for_stop:
                    bar_low = float(row["low"])
                    bar_high = float(row["high"])
                else:
                    bar_low = bar_high = price_close

                if position.side == "LONG":
                    if (
                        position.stop_loss_price is not None
                        and bar_low <= position.stop_loss_price
                    ):
                        exit_price = position.stop_loss_price
                        exit_reason = "STOP"

                    if (
                        exit_price is None
                        and position.take_profit_price is not None
                        and bar_high >= position.take_profit_price
                    ):
                        exit_price = position.take_profit_price
                        exit_reason = "TAKE_PROFIT"

                elif position.side == "SHORT":
                    if (
                        position.stop_loss_price is not None
                        and bar_high >= position.stop_loss_price
                    ):
                        exit_price = position.stop_loss_price
                        exit_reason = "STOP"

                    if (
                        exit_price is None
                        and position.take_profit_price is not None
                        and bar_low <= position.take_profit_price
                    ):
                        exit_price = position.take_profit_price
                        exit_reason = "TAKE_PROFIT"

                if exit_price is not None:
                    self._close_position(
                        portfolio=portfolio,
                        exit_price=exit_price,
                        timestamp=ts,
                        reason=exit_reason,
                        trades=trades,
                    )

            portfolio.update_equity(price_close)
            ctx = StrategyContext(
                equity=portfolio.equity,
                position=portfolio.position,
                index=i,
            )

            signal: SignalType = self.strategy.on_bar(row, ctx)

            if portfolio.position is None:
                if signal == "LONG":
                    self._open_position(
                        side="LONG",
                        portfolio=portfolio,
                        row=row,
                        trades=trades,
                    )
                elif signal == "SHORT":
                    self._open_position(
                        side="SHORT",
                        portfolio=portfolio,
                        row=row,
                        trades=trades,
                    )
            else:
                if signal == "CLOSE":
                    self._close_position(
                        portfolio=portfolio,
                        exit_price=price_close,
                        timestamp=ts,
                        reason="SIGNAL_CLOSE",
                        trades=trades,
                    )
                else:
                    current_side = portfolio.position.side
                    if (
                        self.config.flip_on_opposite_signal
                        and signal in ("LONG", "SHORT")
                        and signal is not None
                        and signal != current_side
                    ):
                        self._close_position(
                            portfolio=portfolio,
                            exit_price=price_close,
                            timestamp=ts,
                            reason="FLIP",
                            trades=trades,
                        )
                        self._open_position(
                            side=signal,
                            portfolio=portfolio,
                            row=row,
                            trades=trades,
                        )

            portfolio.record(ts, price_close)

        if portfolio.position is not None:
            last_row = df.iloc[-1]
            self._close_position(
                portfolio=portfolio,
                exit_price=float(last_row["close"]),
                timestamp=last_row["timestamp"],
                reason="END_OF_DATA",
                trades=trades,
            )
            portfolio.record(last_row["timestamp"], float(last_row["close"]))

        result = self._build_result(portfolio, trades)
        return result

    def _open_position(
        self,
        side: str,
        portfolio: Portfolio,
        row: pd.Series,
        trades: list,
    ) -> None:
        price = float(row["close"])
        ts = row["timestamp"]

        if side == "LONG":
            entry_price = price * (1 + self.config.slippage_pct)
        elif side == "SHORT":
            entry_price = price * (1 - self.config.slippage_pct)
        else:
            return

        qty = self.risk_engine.get_position_size(entry_price, portfolio.equity)
        if qty <= 0:
            return

        max_notional = portfolio.equity * self.risk_engine.config.max_leverage
        notional = entry_price * qty
        if notional > max_notional:
            qty = max_notional / entry_price
            notional = entry_price * qty

        fee_pct = self.config.fee_pct
        commission_open = notional * fee_pct
        total_cost = notional + commission_open

        if total_cost > portfolio.cash:
            affordable_notional = portfolio.cash / (1 + fee_pct)
            if affordable_notional <= 0:
                return
            qty = affordable_notional / entry_price
            if qty <= 0:
                return
            notional = entry_price * qty
            commission_open = notional * fee_pct
            total_cost = notional + commission_open
            if total_cost > portfolio.cash:
                return

        portfolio.cash -= total_cost

        stop_loss_price = None
        take_profit_price = None

        stop_pct = self.risk_engine.config.stop_loss_pct
        rr = 2.0

        if stop_pct > 0:
            if side == "LONG":
                stop_loss_price = entry_price * (1 - stop_pct)
                take_profit_price = entry_price * (1 + rr * stop_pct)
            elif side == "SHORT":
                stop_loss_price = entry_price * (1 + stop_pct)
                take_profit_price = entry_price * (1 - rr * stop_pct)

        portfolio.position = Position(
            side=side,
            qty=qty,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

        trades.append(
            {
                "timestamp": ts,
                "side": side,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": None,
                "pnl": None,
                "reason": "OPEN",
            }
        )

    def _close_position(
        self,
        portfolio: Portfolio,
        exit_price: float,
        timestamp,
        reason: str,
        trades: list,
    ) -> None:
        if portfolio.position is None:
            return

        position = portfolio.position
        qty = position.qty
        side = position.side

        if side == "LONG":
            exit_price_adj = exit_price * (1 - self.config.slippage_pct)
        elif side == "SHORT":
            exit_price_adj = exit_price * (1 + self.config.slippage_pct)
        else:
            exit_price_adj = exit_price

        notional = exit_price_adj * qty
        commission_close = notional * self.config.fee_pct

        portfolio.cash += notional
        portfolio.cash -= commission_close

        if side == "LONG":
            gross_pnl = (exit_price_adj - position.entry_price) * qty
        elif side == "SHORT":
            gross_pnl = (position.entry_price - exit_price_adj) * qty
        else:
            gross_pnl = 0.0

        commission_open = position.entry_price * qty * self.config.fee_pct
        total_commission = commission_open + commission_close
        net_pnl = gross_pnl - total_commission

        for t in reversed(trades):
            if t["reason"] == "OPEN" and t.get("exit_price") is None:
                t["exit_price"] = exit_price_adj
                t["pnl"] = net_pnl
                t["reason"] = reason
                t["timestamp_exit"] = timestamp
                break

        portfolio.position = None

    def _build_result(self, portfolio: Portfolio, trades: list) -> Dict[str, Any]:
        import numpy as np

        equity_series = pd.Series(
            data=portfolio.equity_curve,
            index=portfolio.timestamps,
            name="equity",
        )

        result = {
            "initial_cash": portfolio.initial_cash,
            "final_equity": portfolio.equity,
            "equity_curve": equity_series,
            "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        }

        if len(equity_series) > 1:
            returns = equity_series.pct_change().fillna(0.0)
            cum_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
            max_equity = equity_series.cummax()
            drawdown = (equity_series - max_equity) / max_equity
            max_dd = drawdown.min()

            result["cum_return"] = float(cum_return)
            result["max_drawdown"] = float(max_dd)

            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe = 0.0
            result["sharpe"] = float(sharpe)
        else:
            result["cum_return"] = 0.0
            result["max_drawdown"] = 0.0
            result["sharpe"] = 0.0

        return result

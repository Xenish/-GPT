from __future__ import annotations

from collections import defaultdict
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
        daily_realized_pnl: Dict[pd.Timestamp, float] = defaultdict(float)
        daily_blocked_entries: Dict[pd.Timestamp, int] = defaultdict(int)
        equity_start_of_day: Dict[pd.Timestamp, float] = {}

        for i, row in df.iterrows():
            price_close = float(row["close"])
            ts = pd.to_datetime(row["timestamp"])
            day_key = ts.normalize()

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
                    pnl = self._close_position(
                        portfolio=portfolio,
                        exit_price=exit_price,
                        timestamp=ts,
                        reason=exit_reason,
                        trades=trades,
                    )
                    if pnl is not None:
                        exit_day = ts.normalize()
                        daily_realized_pnl[exit_day] += float(pnl)

            portfolio.update_equity(price_close)
            if day_key not in equity_start_of_day:
                equity_start_of_day[day_key] = portfolio.equity
            ctx = StrategyContext(
                equity=portfolio.equity,
                position=portfolio.position,
                index=i,
            )

            signal: SignalType = self.strategy.on_bar(row, ctx)

            if portfolio.position is None:
                if signal == "LONG":
                    blocked = self._attempt_open(
                        side="LONG",
                        portfolio=portfolio,
                        row=row,
                        trades=trades,
                        ts=ts,
                        day_key=day_key,
                        price=price_close,
                        atr=row.get("atr_14"),
                        equity_start_of_day=equity_start_of_day[day_key],
                        daily_realized_pnl=daily_realized_pnl,
                        daily_blocked_entries=daily_blocked_entries,
                    )
                    if blocked:
                        continue
                elif signal == "SHORT":
                    blocked = self._attempt_open(
                        side="SHORT",
                        portfolio=portfolio,
                        row=row,
                        trades=trades,
                        ts=ts,
                        day_key=day_key,
                        price=price_close,
                        atr=row.get("atr_14"),
                        equity_start_of_day=equity_start_of_day[day_key],
                        daily_realized_pnl=daily_realized_pnl,
                        daily_blocked_entries=daily_blocked_entries,
                    )
                    if blocked:
                        continue
            else:
                if signal == "CLOSE":
                    pnl = self._close_position(
                        portfolio=portfolio,
                        exit_price=price_close,
                        timestamp=ts,
                        reason="SIGNAL_CLOSE",
                        trades=trades,
                    )
                    if pnl is not None:
                        exit_day = ts.normalize()
                        daily_realized_pnl[exit_day] += float(pnl)
                else:
                    current_side = portfolio.position.side
                    if (
                        self.config.flip_on_opposite_signal
                        and signal in ("LONG", "SHORT")
                        and signal is not None
                        and signal != current_side
                        ):
                            pnl = self._close_position(
                                portfolio=portfolio,
                                exit_price=price_close,
                                timestamp=ts,
                                reason="FLIP",
                                trades=trades,
                            )
                            if pnl is not None:
                                exit_day = ts.normalize()
                                daily_realized_pnl[exit_day] += float(pnl)
                            blocked = self._attempt_open(
                                side=signal,
                                portfolio=portfolio,
                                row=row,
                                trades=trades,
                                ts=ts,
                                day_key=day_key,
                                price=price_close,
                                atr=row.get("atr_14"),
                                equity_start_of_day=equity_start_of_day[day_key],
                                daily_realized_pnl=daily_realized_pnl,
                                daily_blocked_entries=daily_blocked_entries,
                            )
                            if blocked:
                                continue

            portfolio.record(ts, price_close)

        if portfolio.position is not None:
            last_row = df.iloc[-1]
            last_ts = pd.to_datetime(last_row["timestamp"])
            pnl = self._close_position(
                portfolio=portfolio,
                exit_price=float(last_row["close"]),
                timestamp=last_ts,
                reason="END_OF_DATA",
                trades=trades,
            )
            if pnl is not None:
                exit_day = last_ts.normalize()
                daily_realized_pnl[exit_day] += float(pnl)
            portfolio.record(last_row["timestamp"], float(last_row["close"]))

        risk_stats = {
            "daily_realized_pnl": {str(k.date()): v for k, v in daily_realized_pnl.items()},
            "blocked_entries": {str(k.date()): v for k, v in daily_blocked_entries.items()},
        }

        result = self._build_result(portfolio, trades, risk_stats)
        return result

    def _attempt_open(
        self,
        *,
        side: str,
        portfolio: Portfolio,
        row: pd.Series,
        trades: list,
        ts: pd.Timestamp,
        day_key: pd.Timestamp,
        price: float,
        atr,
        equity_start_of_day: float,
        daily_realized_pnl: Dict[pd.Timestamp, float],
        daily_blocked_entries: Dict[pd.Timestamp, int],
    ) -> bool:
        realized_today = daily_realized_pnl.get(day_key, 0.0)
        if not self.risk_engine.can_open_new_trade(
            current_date=day_key,
            equity_start_of_day=equity_start_of_day,
            realized_pnl_today=realized_today,
            row=row,
        ):
            daily_blocked_entries[day_key] += 1
            print(f"[RISK] Blocked entry at {ts} due to guard or loss limit.")
            return True

        qty = self.risk_engine.calc_position_size(
            equity=portfolio.equity,
            price=price,
            atr=atr,
            row=row,
        )
        if qty <= 0:
            daily_blocked_entries[day_key] += 1
            print(f"[RISK] Computed size <= 0 at {ts}; entry skipped.")
            return True

        self._open_position(
            side=side,
            portfolio=portfolio,
            row=row,
            trades=trades,
            qty=qty,
            timestamp=ts,
        )
        return False

    def _open_position(
        self,
        side: str,
        portfolio: Portfolio,
        row: pd.Series,
        trades: list,
        qty: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        price = float(row["close"])
        ts = timestamp or row["timestamp"]

        if side == "LONG":
            entry_price = price * (1 + self.config.slippage_pct)
        elif side == "SHORT":
            entry_price = price * (1 - self.config.slippage_pct)
        else:
            return

        if qty is None:
            qty = self.risk_engine.calc_position_size(
                equity=portfolio.equity,
                price=price,
                atr=row.get("atr_14"),
                row=row,
            )
        if qty <= 0:
            return

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
    ) -> Optional[float]:
        if portfolio.position is None:
            return None

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
        return net_pnl

    def _build_result(
        self,
        portfolio: Portfolio,
        trades: list,
        risk_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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

        result["risk_stats"] = risk_stats or {}
        return result


class BacktestEngine:
    """
    Thin orchestration wrapper that normalizes the output of Backtester
    so downstream callers and tests can rely on a consistent interface.
    """

    def __init__(
        self,
        *,
        strategy: BaseStrategy,
        risk_engine: Optional[RiskEngine] = None,
        config: Optional[BacktestConfig] = None,
        price_col: str = "close",
        timestamp_col: str = "timestamp",
    ) -> None:
        self.price_col = price_col
        self.timestamp_col = timestamp_col
        self._backtester = Backtester(
            strategy=strategy,
            risk_engine=risk_engine,
            config=config,
        )

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [col for col in (self.price_col, self.timestamp_col) if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns for backtest: {missing}")

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        self._validate_input(df)
        raw = self._backtester.run(df)
        trades = raw.get("trades", pd.DataFrame())
        if trades is None:
            trades = pd.DataFrame()

        metrics = {
            "initial_cash": raw.get("initial_cash"),
            "final_equity": raw.get("final_equity"),
            "cum_return": raw.get("cum_return"),
            "max_drawdown": raw.get("max_drawdown"),
            "sharpe": raw.get("sharpe"),
            "trade_count": int(len(trades)) if isinstance(trades, pd.DataFrame) else 0,
        }

        return {
            "equity_curve": raw.get("equity_curve"),
            "trades": trades,
            "metrics": metrics,
            "risk_stats": raw.get("risk_stats", {}),
            "raw_result": raw,
        }

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, StrategyContext
from finantradealgo.data_engine.live_data_source import LiveDataSource
from finantradealgo.risk.risk_engine import RiskEngine
from finantradealgo.system.config_loader import LiveConfig


class LiveEngine:
    """
    Orchestrates the live trading workflow (data -> strategy -> risk -> execution).
    The engine is intentionally lightweight so we can swap data sources and
    execution adapters during development.
    """

    def __init__(
        self,
        *,
        config: LiveConfig,
        data_source: LiveDataSource,
        strategy: BaseStrategy,
        risk_engine: RiskEngine,
        execution_client,
        logger: Optional[logging.Logger] = None,
        run_id: Optional[str] = None,
        pipeline_meta: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.data_source = data_source
        self.strategy = strategy
        self.risk_engine = risk_engine
        self.execution_client = execution_client
        self.logger = logger or logging.getLogger(__name__)
        self.run_id = run_id or "live_engine"
        self.pipeline_meta = pipeline_meta or {}

        self.price_col = "close"
        self.timestamp_col = "timestamp"

        self.daily_realized_pnl: Dict[pd.Timestamp, float] = defaultdict(float)
        self.equity_start_of_day: Dict[pd.Timestamp, float] = {}
        self.iteration = 0
        self.blocked_entries = 0
        self.executed_trades = 0

        self.state_path = Path(self.config.paper.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_state_every = max(int(self.config.paper.save_state_every_n_bars), 0)

    # ------------
    # Main runner
    # ------------
    def run_loop(self, max_iterations: Optional[int] = None) -> None:
        bars_limit = max_iterations if max_iterations is not None else self.config.replay.bars_limit
        self.logger.info(
            (
                "Starting live engine run_id=%s mode=%s exchange=%s symbol=%s timeframe=%s "
                "bars_limit=%s initial_cash=%.2f pipeline_version=%s"
            ),
            self.run_id,
            self.config.mode,
            self.config.exchange,
            self.config.symbol,
            self.config.timeframe,
            bars_limit if bars_limit is not None else "unbounded",
            self.execution_client.portfolio.initial_cash,
            self.pipeline_meta.get("pipeline_version", "unknown"),
        )

        self.data_source.connect()
        last_timestamp = pd.Timestamp.utcnow()

        try:
            while True:
                if bars_limit is not None and self.iteration >= bars_limit:
                    self.logger.info("Reached bars_limit=%s. Stopping loop.", bars_limit)
                    break

                bar = self.data_source.next_bar()
                if bar is None:
                    self.logger.info("Data source exhausted; stopping live loop.")
                    break

                if not isinstance(bar, pd.Series):
                    bar = pd.Series(bar)

                price = bar.get(self.price_col)
                ts_val = bar.get(self.timestamp_col)
                if price is None or pd.isna(price) or ts_val is None:
                    self.logger.warning("Skipping bar without price/timestamp.")
                    self.iteration += 1
                    continue

                ts = pd.to_datetime(ts_val)
                price = float(price)
                last_timestamp = ts
                self.execution_client.mark_to_market(price, ts)

                day_key = ts.normalize()
                portfolio_snapshot = self.execution_client.get_portfolio()
                if day_key not in self.equity_start_of_day:
                    self.equity_start_of_day[day_key] = portfolio_snapshot["equity"]

                ctx = StrategyContext(
                    equity=portfolio_snapshot["equity"],
                    position=self.execution_client.get_position(),
                    index=self.iteration,
                )
                signal = self.strategy.on_bar(bar, ctx)
                self._process_signal(signal, bar, price, ts, day_key)

                self.iteration += 1
                if self.save_state_every > 0 and self.iteration % self.save_state_every == 0:
                    self._write_state(ts)

        except KeyboardInterrupt:
            self.logger.info("Live loop interrupted by user.")
        except Exception:
            self.logger.exception("Unhandled exception inside live loop.")
            raise
        finally:
            self.data_source.close()
            self._write_state(last_timestamp)

        self.logger.info(
            "Live engine finished iterations=%s trades=%s blocked_entries=%s",
            self.iteration,
            self.executed_trades,
            self.blocked_entries,
        )

    def shutdown(self) -> None:
        self.logger.info("Shutting down live engine run_id=%s.", self.run_id)
        self._write_state(pd.Timestamp.utcnow())

    def export_results(self) -> Dict[str, Path]:
        return self.execution_client.export_logs(timeframe=self.config.timeframe)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "iterations": self.iteration,
            "blocked_entries": self.blocked_entries,
            "executed_trades": self.executed_trades,
        }

    # -------------------
    # Internal utilities
    # -------------------
    def _process_signal(
        self,
        signal,
        row: pd.Series,
        price: float,
        timestamp: pd.Timestamp,
        day_key: pd.Timestamp,
    ) -> None:
        if signal == "LONG" and not self.execution_client.has_position():
            self._open_position(row, price, timestamp, day_key, side="LONG")
        elif signal == "SHORT" and not self.execution_client.has_position():
            self._open_position(row, price, timestamp, day_key, side="SHORT")
        elif signal == "CLOSE" and self.execution_client.has_position():
            self._close_position(price, timestamp, day_key)

    def _open_position(
        self,
        row: pd.Series,
        price: float,
        timestamp: pd.Timestamp,
        day_key: pd.Timestamp,
        *,
        side: str,
    ) -> None:
        equity_today = self.equity_start_of_day.get(day_key)
        current_equity = self.execution_client.get_portfolio()["equity"]
        realized_today = self.daily_realized_pnl.get(day_key, 0.0)

        if equity_today is None:
            equity_today = current_equity

        if not self.risk_engine.can_open_new_trade(
            current_date=day_key,
            equity_start_of_day=equity_today,
            realized_pnl_today=realized_today,
            row=row,
        ):
            self.blocked_entries += 1
            self.logger.warning(
                "[RISK][BLOCK] side=%s timestamp=%s realized_today=%.2f equity_start=%.2f",
                side,
                timestamp,
                realized_today,
                equity_today,
            )
            return

        qty = self.risk_engine.calc_position_size(
            equity=current_equity,
            price=price,
            atr=row.get("atr_14"),
            row=row,
        )
        if qty <= 0:
            self.logger.debug("Risk engine produced non-positive size; skipping.")
            return

        trade = self.execution_client.submit_order(
            side,
            price=price,
            size=qty,
            timestamp=timestamp,
        )
        if trade:
            self.logger.info(
                "[ENTRY] side=%s qty=%.4f price=%.2f equity=%.2f",
                side,
                qty,
                price,
                current_equity,
            )

    def _close_position(
        self,
        price: float,
        timestamp: pd.Timestamp,
        day_key: pd.Timestamp,
    ) -> None:
        trade = self.execution_client.submit_order(
            "CLOSE",
            price=price,
            size=None,
            timestamp=timestamp,
        )
        if trade and trade.get("pnl") is not None:
            self.daily_realized_pnl[day_key] += float(trade["pnl"])
            self.executed_trades += 1
            self.logger.info(
                "[EXIT] side=%s qty=%.4f price=%.2f pnl=%.2f",
                trade.get("side"),
                trade.get("qty"),
                price,
                trade["pnl"],
            )

    def _write_state(self, timestamp: pd.Timestamp) -> None:
        snapshot = {
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "mode": self.config.mode,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "iteration": self.iteration,
            "stats": self.get_stats(),
            "pipeline": self.pipeline_meta,
            "execution_state": self.execution_client.to_state_dict(),
        }
        with self.state_path.open("w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, ensure_ascii=False, indent=2)

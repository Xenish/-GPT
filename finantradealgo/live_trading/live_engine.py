from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
import datetime as dt
from collections import deque
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, StrategyContext
from finantradealgo.data_engine.live_data_source import AbstractLiveDataSource, Bar
from finantradealgo.execution import ExecutionClient
from finantradealgo.execution.client_base import ExecutionClientBase
from finantradealgo.execution.execution_client import ExchangeRiskLimitError
from finantradealgo.live_trading.routing import (
    ExchangeRoutingEngine,
    RoutingConfig,
    RoutingDecision,
)
from finantradealgo.exchanges import ExchangeId
from finantradealgo.risk.risk_engine import RiskEngine
from finantradealgo.system.config_loader import LiveConfig
from finantradealgo.core.portfolio import Position
from finantradealgo.system.kill_switch import KillSwitch, KillSwitchReason, KillSwitchState
from finantradealgo.validation.live_validator import detect_suspect_bars


class LiveEngine:
    """
    Orchestrates the live trading workflow (data -> strategy -> risk -> execution).
    Supports multi-exchange order routing via ExchangeRoutingEngine and per-exchange
    execution clients. To wire up, construct SymbolRegistry + MultiExchangeAggregator +
    ExchangeHealthMonitor, build a RoutingConfig, instantiate ExchangeRoutingEngine,
    create per-exchange ExecutionClient implementations, then pass them to LiveEngine.
    """

    def __init__(
        self,
        *,
        system_cfg: Dict[str, Any],
        strategy: BaseStrategy,
        risk_engine: RiskEngine,
        execution_client: ExecutionClientBase | None,
        routing_engine: Optional[ExchangeRoutingEngine] = None,
        execution_clients: Optional[Dict[ExchangeId, ExecutionClient]] = None,
        data_source: AbstractLiveDataSource,
        run_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        pipeline_meta: Optional[Dict[str, Any]] = None,
        strategy_name: str = "unknown",
        kill_switch: Optional[KillSwitch] = None,
        notifier: Optional["NotificationManager"] = None,
    ):
        live_cfg = system_cfg.get("live_cfg")
        if not isinstance(live_cfg, LiveConfig):
            live_cfg = LiveConfig.from_dict(
                system_cfg.get("live"),
                default_symbol=system_cfg.get("symbol"),
                default_timeframe=system_cfg.get("timeframe"),
            )
        self.system_cfg = system_cfg
        self.config = live_cfg
        self.data_source = data_source
        self.strategy = strategy
        self.risk_engine = risk_engine
        self.routing_engine = routing_engine
        self.execution_clients = execution_clients or {}
        self.execution_client = execution_client
        if self.execution_client is None:
            self.execution_client = self._select_default_execution_client()
        if self.execution_client is None:
            raise ValueError("An execution_client or execution_clients mapping is required.")
        self.logger = logger or logging.getLogger(__name__)
        self.run_id = run_id or "live_engine"
        self.strategy_name = strategy_name
        self.pipeline_meta = pipeline_meta or {}
        self.kill_switch = kill_switch
        self.notifier = notifier
        self._kill_switch_eval_interval = (
            max(
                1,
                int(
                    getattr(
                        getattr(self.kill_switch, "cfg", None),
                        "evaluation_interval_bars",
                        1,
                    )
                ),
            )
            if self.kill_switch
            else None
        )
        self._kill_switch_last_eval_iter: int = -1
        self.kill_switch_triggered_flag: bool = False
        self.kill_switch_triggered_reason: Optional[str] = None
        self.kill_switch_triggered_ts: Optional[pd.Timestamp] = None
        self._recent_bars = deque(maxlen=500)
        self.validation_issues: int = 0
        self.status: str = "RUNNING"
        self.is_running: bool = True
        self.is_paused: bool = False
        self.requested_action: Optional[str] = None  # "pause", "resume", "stop", "flatten"
        self.data_source_name = getattr(live_cfg, "data_source", "replay").lower()
        self.stale_data_seconds: Optional[float] = None
        self.ws_reconnect_count: int = 0
        self.last_bar_time: Optional[float] = None
        self._last_bar_timestamp: Optional[pd.Timestamp] = None
        self.ws_stale_alarm: bool = False

        self.price_col = "close"
        self.timestamp_col = "timestamp"

        self.daily_realized_pnl: Dict[pd.Timestamp, float] = defaultdict(float)
        self.equity_start_of_day: Dict[pd.Timestamp, float] = {}
        self.iteration = 0
        self.blocked_entries = 0
        self.executed_trades = 0
        self.open_positions: List[Dict[str, Any]] = []
        self._last_position_sync_iter: int = -1

        self.start_time: Optional[pd.Timestamp] = None

        live_dir = Path(getattr(self.config, "state_dir", "outputs/live"))
        state_path_override = getattr(self.config, "state_path", None)
        latest_state_override = getattr(self.config, "latest_state_path", None)
        live_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = (
            Path(state_path_override)
            if state_path_override
            else live_dir / f"live_state_{self.run_id}.json"
        )
        self.latest_state_path = (
            Path(latest_state_override) if latest_state_override else live_dir / "live_state.json"
        )
        self.save_state_every = max(int(self.config.paper.save_state_every_n_bars), 0)
        heartbeat_template = getattr(self.config, "heartbeat_path", None) or "outputs/live/heartbeat_{run_id}.json"
        try:
            heartbeat_value = heartbeat_template.format(run_id=self.run_id)
        except (KeyError, IndexError):
            heartbeat_value = heartbeat_template
        self.heartbeat_path = Path(heartbeat_value)

    def run(self, max_iterations: Optional[int] = None) -> None:
        bars_limit = (
            max_iterations
            if max_iterations is not None
            else getattr(self.config.replay, "bars_limit", None)
        )
        initial_cash = getattr(
            getattr(self.execution_client, "portfolio", None), "initial_cash", None
        )
        if initial_cash is None:
            initial_cash = getattr(self.config.paper, "initial_cash", 0.0)
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
            initial_cash,
            self.pipeline_meta.get("pipeline_version", "unknown"),
        )
        self.start_time = pd.Timestamp.now(tz="UTC")
        self.data_source.connect()
        self._refresh_positions()
        last_timestamp = self.start_time
        try:
            while self.is_running:
                if bars_limit is not None and self.iteration >= bars_limit:
                    self.logger.info("Reached bars_limit=%s. Stopping loop.", bars_limit)
                    break
                bar = self.data_source.next_bar()
                if bar is None:
                    if not self._handle_no_bar():
                        break
                    continue
                try:
                    last_timestamp = self._on_bar(bar)
                except Exception:
                    self.logger.exception("Error while processing bar.")
                    if self.kill_switch:
                        self.kill_switch.register_exception(dt.datetime.now(dt.timezone.utc))
                        self._evaluate_kill_switch(
                            equity=float(
                                self.execution_client.get_portfolio().get("equity", 0.0)
                            ),
                            day_key=self._last_bar_timestamp.normalize()
                            if self._last_bar_timestamp
                            else None,
                            timestamp=self._last_bar_timestamp or pd.Timestamp.now(tz="UTC"),
                            reason_source="exception",
                            force=True,
                        )
                    continue
        except KeyboardInterrupt:
            self.logger.info("Live loop interrupted by user.")
        except Exception:
            self.logger.exception("Unhandled exception inside live loop.")
            raise
        finally:
            try:
                self.data_source.close()
            except Exception:  # pragma: no cover - defensive
                self.logger.exception("Failed to close data source cleanly.")
            close_fn = getattr(self.execution_client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    self.logger.exception("Execution client close failed.")
            self._write_snapshot(last_timestamp)
        self.logger.info(
            "Live engine finished iterations=%s trades=%s blocked_entries=%s",
            self.iteration,
            self.executed_trades,
            self.blocked_entries,
        )

    # ------------
    # Main runner
    # ------------
    def run_loop(self, max_iterations: Optional[int] = None) -> None:
        self.run(max_iterations)

    def _handle_no_bar(self) -> bool:
        """Handle missing bar from data source. Returns False if loop should stop."""
        if self.data_source_name == "binance_ws":
            now_ts = time.time()
            if self.last_bar_time is not None:
                self.stale_data_seconds = now_ts - self.last_bar_time
            else:
                self.stale_data_seconds = None
            threshold = getattr(self.config, "ws_max_stale_seconds", 30)
            is_stale = (
                self.stale_data_seconds is not None and self.stale_data_seconds > threshold
            )
            self.ws_stale_alarm = bool(is_stale)
            if is_stale:
                self.logger.warning(
                    "WebSocket data stale for %.2f seconds",
                    self.stale_data_seconds,
                )
            self.ws_reconnect_count = getattr(
                self.data_source,
                "get_reconnect_count",
                lambda: 0,
            )()
            return True
        self.logger.info("Data source exhausted; stopping live loop.")
        return False

    def _prepare_row(self, bar: Bar | pd.Series) -> tuple[pd.Series, float, pd.Timestamp]:
        if isinstance(bar, Bar):
            price = float(bar.close)
            ts = pd.Timestamp(bar.close_time)
            row_series = pd.Series(bar.extras or {})
            if self.timestamp_col not in row_series:
                row_series[self.timestamp_col] = ts
            row_series[self.price_col] = price
            return row_series, price, ts
        if isinstance(bar, pd.Series):
            row_series = bar.copy()
        else:
            row_series = pd.Series(bar)
        price = row_series.get(self.price_col)
        ts_val = row_series.get(self.timestamp_col)
        if price is None or pd.isna(price) or ts_val is None:
            raise ValueError("Bar missing price or timestamp.")
        ts = pd.to_datetime(ts_val)
        # Validation buffer for live suspect bar detection
        self._recent_bars.append(row_series)
        return row_series, float(price), ts

    def _on_bar(self, bar: Bar | pd.Series) -> pd.Timestamp:
        try:
            row_series, price, ts = self._prepare_row(bar)
        except ValueError:
            self.logger.warning("Skipping bar without price/timestamp.")
            self.iteration += 1
            return self._last_bar_timestamp or pd.Timestamp.now(tz="UTC")

        self._last_bar_timestamp = ts
        ts_float = pd.Timestamp(ts).timestamp()
        self.last_bar_time = ts_float
        self.stale_data_seconds = max(time.time() - ts_float, 0.0)
        self.ws_stale_alarm = False
        if self.data_source_name == "binance_ws":
            self.ws_reconnect_count = getattr(self.data_source, "get_reconnect_count", lambda: 0)()

        self.execution_client.mark_to_market(price, ts)
        self._load_requested_action()
        self._apply_pending_action()
        if not self.is_running:
            return ts
        if self.is_paused:
            self.iteration += 1
            if self.save_state_every > 0 and self.iteration % self.save_state_every == 0:
                self._write_snapshot(ts)
            return ts

        day_key = ts.normalize()
        portfolio_snapshot = self.execution_client.get_portfolio() or {}
        equity = float(portfolio_snapshot.get("equity", 0.0) or 0.0)
        if day_key not in self.equity_start_of_day:
            self.equity_start_of_day[day_key] = equity
        if self._evaluate_kill_switch(equity, day_key, ts):
            return ts

        # Live validation of recent bars (lightweight)
        if len(self._recent_bars) >= 10:
            try:
                df_recent = pd.DataFrame(list(self._recent_bars))
                vr = detect_suspect_bars(df_recent.tail(100))
                if vr.warnings_count > 0:
                    self.validation_issues += vr.warnings_count
                    self.logger.warning("Suspect bars detected: %s", vr.summary())
            except Exception:
                # Validation should not break live loop
                self.logger.debug("Live validation skipped due to error.", exc_info=True)

        positions = self._refresh_positions()

        ctx = StrategyContext(
            equity=equity,
            position=self.execution_client.get_position(),
            index=self.iteration,
        )
        signal = self.strategy.on_bar(row_series, ctx)
        self._process_signal(signal, row_series, price, ts, day_key, positions)

        self.iteration += 1
        # Heartbeat update every bar
        self._update_heartbeat(ts)
        if self.save_state_every > 0 and self.iteration % self.save_state_every == 0:
            self._write_snapshot(ts)
        return ts

    def shutdown(self) -> None:
        self.logger.info("Shutting down live engine run_id=%s.", self.run_id)
        self._write_snapshot(pd.Timestamp.now(tz="UTC"))

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def stop(self) -> None:
        self.is_running = False

    def flatten_all(self) -> None:
        try:
            positions = self.execution_client.get_open_positions()
        except AttributeError:
            positions = []
        any_closed = False
        for pos in positions or []:
            try:
                if hasattr(self.execution_client, "close_position_market"):
                    self.execution_client.close_position_market(pos)
                else:
                    last_price = getattr(self.execution_client, "_last_price", None) or pos.get(
                        "current_price", pos.get("entry_price")
                    )
                    if last_price is None:
                        continue
                    qty = pos.get("qty") or pos.get("positionAmt")
                    if qty is None:
                        continue
                    qty = abs(float(qty))
                    if qty <= 0:
                        continue
                    ts = getattr(self.execution_client, "_last_timestamp", None) or pd.Timestamp.now(tz="UTC")
                    side_raw = str(pos.get("side", "")).upper()
                    side = "SELL" if "LONG" in side_raw else "BUY"
                    client_order_id = self._make_client_order_id(
                        pos.get("symbol", self.config.symbol),
                        side,
                        ts,
                    )
                    self._submit_order_with_retry(
                        symbol=pos.get("symbol", self.config.symbol),
                        side=side,
                        qty=qty,
                        order_type="MARKET",
                        price=last_price,
                        reduce_only=True,
                        client_order_id=client_order_id,
                    )
                any_closed = True
            except Exception:
                self.logger.exception("Failed to flatten position: %s", pos)
        self.requested_action = None
        if any_closed:
            ts_snapshot = self._last_bar_timestamp or pd.Timestamp.now(tz="UTC")
            self._write_snapshot(ts_snapshot)
            self._refresh_positions()

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
    def _select_default_execution_client(self) -> ExecutionClientBase | ExecutionClient | None:
        """Pick a default execution client if mapping is provided but single client is not."""

        if not self.execution_clients:
            return None
        cfg_exchange = getattr(self.config, "exchange", None)
        if isinstance(cfg_exchange, ExchangeId) and cfg_exchange in self.execution_clients:
            return self.execution_clients[cfg_exchange]
        if isinstance(cfg_exchange, str):
            for ex_id, client in self.execution_clients.items():
                if getattr(ex_id, "name", "").lower() == cfg_exchange.lower():
                    return client
        return next(iter(self.execution_clients.values()))

    def _route_execution(
        self, internal_symbol: str
    ) -> tuple[ExecutionClientBase | ExecutionClient, str, RoutingDecision | None]:
        """
        Resolve execution client and exchange symbol for an internal symbol.

        Returns (client, exchange_symbol, routing_decision|None).
        """

        if self.routing_engine and self.execution_clients:
            decision = self.routing_engine.choose_exchange(internal_symbol)
            exec_client = self.execution_clients.get(decision.chosen_exchange)
            if exec_client is None:
                raise ValueError(
                    f"No execution client configured for exchange {decision.chosen_exchange}"
                )
            return exec_client, decision.symbol_mapping.exchange_symbol, decision

        return self.execution_client, internal_symbol, None

    def _apply_pending_action(self) -> None:
        action = self.requested_action
        if not action:
            return
        if action == "pause":
            self.pause()
        elif action == "resume":
            self.resume()
        elif action == "stop":
            self.stop()
        elif action == "flatten":
            self.flatten_all()
        self.requested_action = None

    def _load_requested_action(self) -> None:
        try:
            if self.latest_state_path.is_file():
                state = json.loads(self.latest_state_path.read_text(encoding="utf-8"))
                cmd = state.get("requested_action")
                if cmd:
                    self.requested_action = cmd
        except Exception:
            self.logger.exception("Failed to load requested_action from snapshot.")

    def _process_signal(
        self,
        signal,
        row: pd.Series,
        price: float,
        timestamp: pd.Timestamp,
        day_key: pd.Timestamp,
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        positions = positions if positions is not None else self.open_positions
        if signal == "LONG" and not self.execution_client.has_position():
            self._open_position(row, price, timestamp, day_key, side="LONG", positions=positions)
        elif signal == "SHORT" and not self.execution_client.has_position():
            self._open_position(row, price, timestamp, day_key, side="SHORT", positions=positions)
        elif signal == "CLOSE" and self.execution_client.has_position():
            self._close_position(price, timestamp, day_key)

    def _make_client_order_id(
        self,
        symbol: str,
        side: str,
        timestamp: pd.Timestamp,
    ) -> str:
        ts_ms = int(pd.Timestamp(timestamp).timestamp() * 1000)
        return f"{self.run_id}_{symbol}_{ts_ms}_{side.upper()}_{self.iteration}"

    def _submit_order_with_retry(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        exec_client, exchange_symbol, decision = self._route_execution(symbol)
        route_label = (
            f"{decision.chosen_exchange.name}:{decision.reason}" if decision else "single_exchange"
        )
        attempts = max(1, int(getattr(self.config, "order_retry_limit", 1) or 1))
        wait_seconds = max(0.0, float(getattr(self.config, "order_timeout_seconds", 0)))
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                submit_fn = getattr(exec_client, "submit_order", None)
                place_fn = getattr(exec_client, "place_order", None) if submit_fn is None else None
                if callable(submit_fn):
                    return submit_fn(
                        symbol=exchange_symbol,
                        side=side,
                        qty=qty,
                        order_type=order_type,
                        price=price,
                        reduce_only=reduce_only,
                        client_order_id=client_order_id,
                        **extra,
                    )
                if callable(place_fn):
                    return place_fn(
                        symbol=exchange_symbol,
                        side=side,
                        qty=qty,
                        order_type=order_type,
                        price=price,
                        reduce_only=reduce_only,
                        client_order_id=client_order_id,
                        **extra,
                    )
                raise TypeError("Execution client lacks submit_order/place_order implementation.")
            except ExchangeRiskLimitError as exc:
                self._handle_exchange_limit_error(exc)
                return {}
            except Exception as exc:  # pragma: no cover - network/exchange errors
                last_exc = exc
                self.logger.exception(
                    "Order submit failed attempt %s/%s [%s]: %s", attempt + 1, attempts, route_label, exc
                )
                if attempt < attempts - 1 and wait_seconds > 0:
                    time.sleep(min(wait_seconds, 1.0))
        if last_exc:
            raise last_exc
        return {}

    def _handle_exchange_limit_error(self, exc: Exception) -> None:
        msg = f"Order blocked by exchange risk limits: {exc}"
        self.logger.warning(msg)
        if self.kill_switch:
            try:
                self.kill_switch.register_exception(dt.datetime.now(dt.timezone.utc))
            except Exception:
                self.logger.exception("Failed to register kill-switch exception for limit error.")
        if self.notifier:
            try:
                self.notifier.warn(msg)
            except Exception:
                self.logger.exception("Failed to send notifier warning for limit error.")

    def _open_position(
        self,
        row: pd.Series,
        price: float,
        timestamp: pd.Timestamp,
        day_key: pd.Timestamp,
        *,
        side: str,
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        positions = positions if positions is not None else self.open_positions
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
            open_positions=positions,
            max_open_trades=getattr(self.config, "max_open_trades", None),
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

        current_notional = self._current_notional(self.config.symbol, positions)
        qty = self.risk_engine.calc_position_size(
            equity=current_equity,
            price=price,
            atr=row.get("atr_14"),
            row=row,
            current_notional=current_notional,
            max_position_notional=getattr(self.config, "max_position_notional", None),
        )
        if qty <= 0:
            self.logger.debug("Risk engine produced non-positive size; skipping.")
            return

        order_side = "BUY" if side == "LONG" else "SELL"
        client_order_id = self._make_client_order_id(self.config.symbol, order_side, timestamp)
        trade = self._submit_order_with_retry(
            symbol=self.config.symbol,
            side=order_side,
            qty=qty,
            order_type="MARKET",
            price=price,
            reduce_only=False,
            client_order_id=client_order_id,
        )
        if trade:
            self.logger.info(
                "[ENTRY] side=%s qty=%.4f price=%.2f equity=%.2f",
                side,
                qty,
                price,
                current_equity,
            )
            self._refresh_positions()

    def _close_position(
        self,
        price: float,
        timestamp: pd.Timestamp,
        day_key: pd.Timestamp,
    ) -> None:
        position = self.execution_client.get_position()
        qty = None
        pos_side = ""
        if isinstance(position, Position):
            qty = position.qty
            pos_side = position.side.upper()
        elif isinstance(position, dict):
            qty = position.get("qty") or position.get("positionAmt")
            pos_side = str(position.get("side", "")).upper()
        if qty is None:
            open_positions = self.execution_client.get_open_positions()
            if open_positions:
                qty = open_positions[0].get("qty") or open_positions[0].get("positionAmt")
                pos_side = str(open_positions[0].get("side", "")).upper()
        if qty is None:
            self.logger.debug("No position qty found; skip close.")
            return
        qty = abs(float(qty))
        if qty <= 0:
            return
        order_side = "SELL" if "LONG" in pos_side else "BUY"
        client_order_id = self._make_client_order_id(self.config.symbol, order_side, timestamp)
        trade = self._submit_order_with_retry(
            symbol=self.config.symbol,
            side=order_side,
            qty=qty,
            order_type="MARKET",
            price=price,
            reduce_only=True,
            client_order_id=client_order_id,
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
            self._refresh_positions()

    def _refresh_positions(self) -> List[Dict[str, Any]]:
        try:
            raw_positions = self.execution_client.get_open_positions()
        except Exception:
            self.logger.exception("Failed to fetch open positions.")
            raw_positions = []
        normalized = self._normalize_positions(raw_positions)
        self.open_positions = normalized
        self._last_position_sync_iter = self.iteration
        return normalized

    def _normalize_positions(
        self,
        positions: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not positions:
            return normalized
        for raw in positions:
            if raw is None:
                continue
            symbol = str(raw.get("symbol", self.config.symbol))
            qty_val = raw.get("qty")
            side = raw.get("side")
            entry_price = raw.get("entry_price")
            if qty_val is None and raw.get("positionAmt") is not None:
                try:
                    amt = float(raw.get("positionAmt"))
                except (TypeError, ValueError):
                    amt = 0.0
                if amt == 0:
                    continue
                side = "LONG" if amt > 0 else "SHORT"
                qty = abs(amt)
            else:
                try:
                    qty = float(qty_val)
                except (TypeError, ValueError):
                    qty = 0.0
            if qty == 0:
                continue
            try:
                entry_price = float(
                    entry_price
                    if entry_price not in (None, "")
                    else raw.get("entryPrice", 0.0)
                )
            except (TypeError, ValueError):
                entry_price = 0.0
            pnl_val = raw.get("pnl", raw.get("unrealizedPnl"))
            try:
                pnl_val = float(pnl_val) if pnl_val is not None else 0.0
            except (TypeError, ValueError):
                pnl_val = 0.0
            normalized.append(
                {
                    "symbol": symbol,
                    "side": (side or "LONG").upper(),
                    "qty": qty,
                    "entry_price": entry_price,
                    "pnl": pnl_val,
                }
            )
        return normalized

    def _current_notional(
        self,
        symbol: str,
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        positions = positions if positions is not None else self.open_positions
        total = 0.0
        for pos in positions or []:
            if str(pos.get("symbol")) != symbol:
                continue
            try:
                qty = float(pos.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            try:
                entry_price = float(pos.get("entry_price", 0.0) or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            total += abs(qty) * entry_price
        return total

    def _update_heartbeat(self, ts: pd.Timestamp) -> None:
        try:
            payload = {
                "run_id": self.run_id,
                "status": self.status,
                "last_bar_time": ts.isoformat() if ts is not None else None,
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            tmp_path = self.heartbeat_path.with_suffix(".tmp")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp_path, self.heartbeat_path)
        except Exception:
            self.logger.exception("Failed to update heartbeat file.")

    def _evaluate_kill_switch(
        self,
        equity: float,
        day_key: Optional[pd.Timestamp],
        timestamp: pd.Timestamp,
        reason_source: Optional[str] = None,
        *,
        force: bool = False,
    ) -> bool:
        if not self.kill_switch:
            return False
        interval = self._kill_switch_eval_interval or 1
        if not force and self._kill_switch_last_eval_iter >= 0:
            if (self.iteration - self._kill_switch_last_eval_iter) < interval:
                return False
        self._kill_switch_last_eval_iter = self.iteration
        realized = float(self.daily_realized_pnl.get(day_key, 0.0) if day_key is not None else 0.0)
        ks_state = self.kill_switch.evaluate(
            dt.datetime.now(dt.timezone.utc),
            equity=equity,
            daily_realized_pnl=realized,
        )
        if not ks_state.is_triggered:
            return False
        if self.status != "STOPPED_BY_KILL_SWITCH":
            self.status = "STOPPED_BY_KILL_SWITCH"
            self.logger.error(
                "Kill switch triggered reason=%s source=%s",
                ks_state.reason.value,
                reason_source or "evaluation",
            )
            self._notify_kill_switch(ks_state, equity, realized)
            self.kill_switch_triggered_flag = True
            self.kill_switch_triggered_reason = (
                ks_state.reason.value
                if isinstance(ks_state.reason, KillSwitchReason)
                else str(ks_state.reason)
            )
            self.kill_switch_triggered_ts = timestamp
            try:
                self.flatten_all()
            except Exception:
                self.logger.exception("Failed to flatten positions during kill switch.")
            self.stop()
            self._write_snapshot(timestamp)
        return True

    def _notify_kill_switch(
        self,
        ks_state: KillSwitchState,
        equity: float,
        daily_realized: float,
    ) -> None:
        if not self.notifier:
            return
        msg = (
            f"Kill-switch triggered! reason={ks_state.reason.value} "
            f"equity={equity:.2f} daily_pnl={daily_realized:.2f} run_id={self.run_id}"
        )
        try:
            self.notifier.critical(msg)
        except Exception:
            self.logger.exception("Failed to send kill-switch notification.")

    def _write_snapshot(self, timestamp: pd.Timestamp) -> None:
        exec_state = self.execution_client.to_state_dict()
        portfolio = self.execution_client.get_portfolio()
        last_ts = timestamp or self._last_bar_timestamp
        day_key = last_ts.normalize() if last_ts is not None else None
        last_bar_iso = last_ts.isoformat() if last_ts is not None else None
        last_bar_ts = float(last_ts.timestamp()) if last_ts is not None else None
        open_positions = exec_state.get("open_positions", [])
        daily_unrealized = None
        if open_positions:
            try:
                daily_unrealized = float(
                    sum(float(pos.get("pnl", 0.0) or 0.0) for pos in open_positions)
                )
            except (TypeError, ValueError):
                daily_unrealized = None
        snapshot = {
            "run_id": self.run_id,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "strategy": self.strategy_name,
            "mode": getattr(self.config, "mode", "replay"),
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_bar_time": last_bar_iso,
            "last_bar_time_ts": last_bar_ts,
            "equity": portfolio.get("equity"),
            "realized_pnl": exec_state.get("realized_pnl"),
            "unrealized_pnl": exec_state.get("unrealized_pnl"),
            "daily_realized_pnl": float(
                self.daily_realized_pnl.get(day_key, 0.0)
            )
            if day_key is not None
            else None,
            "daily_unrealized_pnl": daily_unrealized,
            "open_positions": open_positions,
            "risk_stats": {
                "blocked_entries": self.blocked_entries,
                "executed_trades": self.executed_trades,
            },
            "requested_action": self.requested_action,
            "data_source": self.data_source_name,
            "stale_data_seconds": self.stale_data_seconds,
            "ws_reconnect_count": self.ws_reconnect_count,
            "last_orders": exec_state.get("recent_orders", []),
            "timestamp": time.time(),
        }
        snapshot["kill_switch_triggered"] = self.kill_switch_triggered_flag
        snapshot["kill_switch_reason"] = self.kill_switch_triggered_reason
        snapshot["kill_switch_ts"] = (
            self.kill_switch_triggered_ts.isoformat() if self.kill_switch_triggered_ts is not None else None
        )
        if self.kill_switch:
            ks_state = self.kill_switch.state
            snapshot["kill_switch"] = {
                "is_triggered": ks_state.is_triggered,
                "reason": ks_state.reason.value if isinstance(ks_state.reason, KillSwitchReason) else str(ks_state.reason),
                "trigger_time": ks_state.trigger_time.isoformat() if ks_state.trigger_time else None,
                "peak_equity": ks_state.peak_equity,
            }
        else:
            snapshot["kill_switch"] = None
        snapshot["validation_issues"] = self.validation_issues
        for target in {self.state_path, self.latest_state_path}:
            with target.open("w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, ensure_ascii=False, indent=2)

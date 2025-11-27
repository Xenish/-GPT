from __future__ import annotations

import pandas as pd
import pytest

from finantradealgo.core.strategy import BaseStrategy, StrategyContext
from finantradealgo.data_engine.live_data_source import FileReplayDataSource
from finantradealgo.execution.paper_client import PaperExecutionClient
from finantradealgo.live_trading.live_engine import LiveEngine
from finantradealgo.risk.risk_engine import RiskConfig, RiskEngine
from finantradealgo.system.config_loader import LiveConfig


def _dummy_df(rows: int = 4) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="15min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100 + i for i in range(rows)],
            "high": [101 + i for i in range(rows)],
            "low": [99 + i for i in range(rows)],
            "close": [100 + i for i in range(rows)],
            "atr_14": [1.0] * rows,
        }
    )


class PingPongStrategy(BaseStrategy):
    def init(self, df: pd.DataFrame) -> None:
        self._should_close = False

    def on_bar(self, row: pd.Series, ctx: StrategyContext):
        if ctx.position is None and not self._should_close:
            self._should_close = True
            return "LONG"
        if ctx.position is not None and self._should_close:
            self._should_close = False
            return "CLOSE"
        if ctx.position is None:
            self._should_close = True
            return "LONG"
        return None


def test_file_replay_iterates_rows():
    df = _dummy_df(rows=3)
    source = FileReplayDataSource(df, bars_limit=3)
    source.connect()
    rows = [source.next_bar() for _ in range(4)]
    assert rows[0].close_time == pd.to_datetime(df.iloc[0]["timestamp"])
    assert rows[1].close_time == pd.to_datetime(df.iloc[1]["timestamp"])
    assert rows[2].close_time == pd.to_datetime(df.iloc[2]["timestamp"])
    assert rows[3] is None


def test_paper_execution_client_long_close(tmp_path):
    client = PaperExecutionClient(
        initial_cash=10_000,
        fee_pct=0.0,
        slippage_pct=0.0,
        output_dir=str(tmp_path / "paper"),
    )
    ts = pd.Timestamp("2025-01-01 00:00:00")
    client.mark_to_market(100.0, ts)
    client.submit_order("TEST", "BUY", 1.0, "MARKET", price=100.0)
    assert client.has_position()
    close_ts = ts + pd.Timedelta(minutes=15)
    client.mark_to_market(110.0, close_ts)
    trade = client.submit_order("TEST", "SELL", 1.0, "MARKET", price=110.0, reduce_only=True)
    assert trade is not None
    assert trade["pnl"] == pytest.approx(10.0)
    snapshot = client.get_portfolio()
    assert snapshot["equity"] == pytest.approx(10010.0)
    trades = client.get_trade_log()
    assert len(trades) == 1
    state = client.to_state_dict()
    assert pytest.approx(state["equity"]) == 10010.0
    assert len(state["closed_trades"]) == 1


def _build_live_config(tmp_path, bars: int) -> LiveConfig:
    return LiveConfig.from_dict(
        {
            "mode": "replay",
            "symbol": "TEST",
            "timeframe": "15m",
            "log_dir": str(tmp_path / "logs"),
            "replay": {"bars_limit": bars},
            "paper": {
                "initial_cash": 5_000,
                "save_state_every_n_bars": 1,
                "state_path": str(tmp_path / "state.json"),
                "output_dir": str(tmp_path / "paper"),
            },
        },
        default_symbol="TEST",
        default_timeframe="15m",
    )


def _system_cfg_from_live(live_cfg: LiveConfig) -> dict:
    return {
        "symbol": live_cfg.symbol,
        "timeframe": live_cfg.timeframe,
        "live": {},
        "live_cfg": live_cfg,
    }


def test_live_engine_replay_executes_trades(tmp_path):
    df = _dummy_df(rows=4)
    strategy = PingPongStrategy()
    strategy.init(df)

    live_cfg = _build_live_config(tmp_path, bars=len(df))
    data_source = FileReplayDataSource(df, bars_limit=len(df))
    risk_engine = RiskEngine(RiskConfig(capital_risk_pct_per_trade=0.5, stop_loss_pct=0.01))
    paper_client = PaperExecutionClient(
        initial_cash=live_cfg.paper.initial_cash,
        fee_pct=0.0,
        slippage_pct=0.0,
        output_dir=live_cfg.paper.output_dir,
        state_path=live_cfg.paper.state_path,
    )
    system_cfg = _system_cfg_from_live(live_cfg)
    engine = LiveEngine(
        system_cfg=system_cfg,
        data_source=data_source,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=paper_client,
        run_id="test_run",
    )

    engine.run_loop()
    engine.shutdown()

    trades = paper_client.get_trade_log()
    assert len(trades) == 2
    assert trades[0]["exit_price"] is not None
    assert engine.iteration == len(df)


class SingleTradeStrategy(BaseStrategy):
    def init(self, df: pd.DataFrame) -> None:
        self._last_index = len(df) - 1

    def on_bar(self, row: pd.Series, ctx: StrategyContext):
        if ctx.position is None and ctx.index == 0:
            return "LONG"
        if ctx.position is not None and ctx.index == self._last_index:
            return "CLOSE"
        return None


def test_live_engine_single_trade_profit(tmp_path):
    df = _dummy_df(rows=6)
    df["close"] = [100, 101, 102, 103, 104, 110]
    strategy = SingleTradeStrategy()
    strategy.init(df)

    live_cfg = _build_live_config(tmp_path, bars=len(df))
    data_source = FileReplayDataSource(df, bars_limit=len(df))
    risk_engine = RiskEngine(RiskConfig(capital_risk_pct_per_trade=0.5, stop_loss_pct=0.01))
    paper_client = PaperExecutionClient(
        initial_cash=live_cfg.paper.initial_cash,
        fee_pct=0.0,
        slippage_pct=0.0,
        output_dir=live_cfg.paper.output_dir,
        state_path=live_cfg.paper.state_path,
    )
    system_cfg = _system_cfg_from_live(live_cfg)
    engine = LiveEngine(
        system_cfg=system_cfg,
        data_source=data_source,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=paper_client,
        run_id="single_trade",
    )
    engine.run_loop()
    engine.shutdown()

    trades = paper_client.get_trade_log()
    assert len(trades) == 1
    portfolio = paper_client.get_portfolio()
    assert portfolio["equity"] > paper_client.portfolio.initial_cash
    state = paper_client.to_state_dict()
    assert len(state["closed_trades"]) == 1


class AlwaysLongStrategy(BaseStrategy):
    def init(self, df: pd.DataFrame) -> None:
        pass

    def on_bar(self, row: pd.Series, ctx: StrategyContext):
        if ctx.position is None:
            return "LONG"
        return "CLOSE"


@pytest.mark.slow
def test_live_engine_risk_blocks_after_loss(tmp_path):
    df = _dummy_df(rows=6)
    df["close"] = [100, 95, 95, 95, 95, 95]
    strategy = AlwaysLongStrategy()
    strategy.init(df)

    live_cfg = _build_live_config(tmp_path, bars=len(df))
    data_source = FileReplayDataSource(df, bars_limit=len(df))
    risk_engine = RiskEngine(
        RiskConfig(
            capital_risk_pct_per_trade=0.5,
            stop_loss_pct=0.01,
            max_daily_loss_pct=0.0001,
        )
    )
    paper_client = PaperExecutionClient(
        initial_cash=live_cfg.paper.initial_cash,
        fee_pct=0.0,
        slippage_pct=0.0,
        output_dir=live_cfg.paper.output_dir,
        state_path=live_cfg.paper.state_path,
    )
    system_cfg = _system_cfg_from_live(live_cfg)
    engine = LiveEngine(
        system_cfg=system_cfg,
        data_source=data_source,
        strategy=strategy,
        risk_engine=risk_engine,
        execution_client=paper_client,
        run_id="risk_block",
    )
    engine.run_loop()
    engine.shutdown()

    trades = paper_client.get_trade_log()
    assert len(trades) == 1
    assert engine.blocked_entries > 0

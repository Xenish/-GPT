from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from finantradealgo.data_engine.data_backend import build_backend
from finantradealgo.data_engine.ingestion.ohlcv import timeframe_to_seconds
from finantradealgo.features.feature_pipeline import build_feature_pipeline_from_system_config
from finantradealgo.system.config_loader import load_config, DataConfig
from finantradealgo.data_engine.ingestion.state import IngestionStateStore

logger = logging.getLogger(__name__)


@dataclass
class FeatureSinkConfig:
    kind: str = "parquet"  # parquet | duckdb
    output_dir: Path = Path("data/features")
    duckdb_path: Optional[Path] = None
    duckdb_table: str = "features"


class FeatureSink:
    def __init__(self, cfg: FeatureSinkConfig):
        self.cfg = cfg
        self._duckdb = None
        if cfg.kind == "duckdb":
            try:
                import duckdb  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("duckdb package required for duckdb sink") from exc
            db_path = Path(cfg.duckdb_path or "features.duckdb")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._duckdb = duckdb.connect(str(db_path))

    def write(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        mode: str = "overwrite",
    ) -> Path | str:
        if df.empty:
            logger.warning("Feature sink write skipped: empty dataframe for %s %s", symbol, timeframe)
            return ""
        if self.cfg.kind == "parquet":
            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.cfg.output_dir / f"{symbol}_{timeframe}_features.parquet"
            if mode == "append" and path.exists():
                existing = pd.read_parquet(path)
                df = (
                    pd.concat([existing, df])
                    .drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            df.to_parquet(path, index=False)
            return path

        if self.cfg.kind == "duckdb":
            assert self._duckdb is not None
            table = f"{self.cfg.duckdb_table}_{timeframe}"
            self._duckdb.register("df_temp", df)
            self._duckdb.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} AS
                SELECT * FROM df_temp WHERE 1=0;
            """)
            # Overwrite/append semantics: remove existing rows for symbol, then insert
            self._duckdb.execute(f"DELETE FROM {table} WHERE symbol = ?;", [symbol])
            self._duckdb.execute(f"INSERT INTO {table} SELECT * FROM df_temp;")
            self._duckdb.unregister("df_temp")
            return f"{self.cfg.duckdb_path or 'features.duckdb'}::{table}"

        raise ValueError(f"Unknown sink kind: {self.cfg.kind}")


class FeatureBuilderService:
    """
    Feature build service supporting batch and incremental modes.
    """

    def __init__(
        self,
        sys_cfg: Optional[dict] = None,
        *,
        sink_cfg: Optional[FeatureSinkConfig] = None,
    ) -> None:
        self.sys_cfg = sys_cfg or load_config("research")
        self.data_cfg: DataConfig = self.sys_cfg["data_cfg"]
        self.sink = FeatureSink(sink_cfg or FeatureSinkConfig(output_dir=Path(self.data_cfg.features_dir)))

    def _load_ohlcv_window(
        self,
        symbol: str,
        timeframe: str,
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        backend = build_backend(self.data_cfg)
        df = backend.load_ohlcv(
            symbol,
            timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return df

    def build_batch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start_ts: Optional[pd.Timestamp] = None,
        end_ts: Optional[pd.Timestamp] = None,
    ) -> tuple[pd.DataFrame, dict]:
        df_ohlcv = self._load_ohlcv_window(symbol, timeframe, start_ts, end_ts)
        if df_ohlcv.empty:
            raise ValueError(f"No OHLCV data for {symbol} {timeframe} in requested window.")

        df_feat, meta = build_feature_pipeline_from_system_config(
            self.sys_cfg,
            symbol=symbol,
            timeframe=timeframe,
            df_ohlcv_override=df_ohlcv,
        )
        sink_path = self.sink.write(df_feat, symbol=symbol, timeframe=timeframe, mode="overwrite")
        meta["sink_path"] = str(sink_path)
        return df_feat, meta

    def build_incremental(
        self,
        symbol: str,
        timeframe: str,
        *,
        state_store,
        job_name: str = "feature_incremental",
        context_bars: int = 500,
    ) -> dict:
        scope = f"{symbol}:{timeframe}"
        last_watermark = state_store.get_watermark(job_name, scope)
        end_ts = pd.Timestamp.utcnow().tz_localize("UTC")
        start_ts = None
        if last_watermark is not None:
            step_seconds = timeframe_to_seconds(timeframe)
            start_ts = last_watermark - pd.Timedelta(seconds=context_bars * step_seconds)

        df_ohlcv = self._load_ohlcv_window(symbol, timeframe, start_ts, end_ts)
        if df_ohlcv.empty:
            logger.info("No new bars for %s %s; skipping increment", symbol, timeframe)
            return {"status": "noop"}

        df_feat, meta = build_feature_pipeline_from_system_config(
            self.sys_cfg,
            symbol=symbol,
            timeframe=timeframe,
            df_ohlcv_override=df_ohlcv,
        )
        sink_path = self.sink.write(df_feat, symbol=symbol, timeframe=timeframe, mode="overwrite")
        latest_ts = pd.to_datetime(df_feat["timestamp"].max(), utc=True)
        state_store.upsert_watermark(job_name, scope, latest_ts)
        return {
            "status": "success",
            "sink_path": str(sink_path),
            "watermark": latest_ts.isoformat(),
            "rows": len(df_feat),
            "features": len(df_feat.columns),
        }

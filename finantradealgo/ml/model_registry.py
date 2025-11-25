from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .model import ModelMetadata, load_sklearn_model

REGISTRY_INDEX = "registry_index.csv"


@dataclass
class RegistryEntry:
    model_id: str
    symbol: str
    timeframe: str
    model_type: str
    created_at: str
    path: str
    cum_return: float | None = None
    sharpe: float | None = None
    status: str = "success"


def _registry_index_path(base_dir: str) -> str:
    return os.path.join(base_dir, REGISTRY_INDEX)


def register_model(
    meta: ModelMetadata,
    base_dir: str,
    status: str = "success",
    max_models: Optional[int] = None,
) -> None:
    os.makedirs(base_dir, exist_ok=True)
    index_path = _registry_index_path(base_dir)
    run_dir = os.path.dirname(meta.model_path)
    row = {
        "model_id": meta.model_id,
        "symbol": meta.symbol,
        "timeframe": meta.timeframe,
        "model_type": meta.model_type,
        "created_at": meta.created_at,
        "path": run_dir,
        "cum_return": meta.metrics.get("cum_return") if meta.metrics else None,
        "sharpe": meta.metrics.get("sharpe") if meta.metrics else None,
        "status": status,
    }

    if os.path.exists(index_path):
        df = pd.read_csv(index_path)
        if "status" not in df.columns:
            df["status"] = "success"
        df = df[~df["model_id"].eq(meta.model_id)]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.sort_values("created_at", inplace=True)

    if max_models and max_models > 0:
        drop_indices: List[int] = []
        grouped = df.groupby(["symbol", "timeframe", "model_type"], sort=False)
        for _, grp in grouped:
            if len(grp) <= max_models:
                continue
            to_remove = grp.sort_values("created_at").iloc[:-max_models]
            drop_indices.extend(to_remove.index.tolist())
            for path in to_remove["path"]:
                shutil.rmtree(path, ignore_errors=True)
        if drop_indices:
            df = df.drop(index=drop_indices)

    df.to_csv(index_path, index=False)


def list_models(
    base_dir: str,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    model_type: Optional[str] = None,
) -> List[RegistryEntry]:
    index_path = _registry_index_path(base_dir)
    if not os.path.exists(index_path):
        return []

    df = pd.read_csv(index_path)
    if symbol:
        df = df[df["symbol"] == symbol]
    if timeframe:
        df = df[df["timeframe"] == timeframe]
    if model_type:
        df = df[df["model_type"].str.lower() == model_type.lower()]
    if "status" not in df.columns:
        df["status"] = "success"

    entries: List[RegistryEntry] = []
    for _, row in df.iterrows():
        entries.append(
            RegistryEntry(
                model_id=row["model_id"],
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                model_type=row["model_type"],
                created_at=row["created_at"],
                path=row["path"],
                cum_return=row.get("cum_return"),
                sharpe=row.get("sharpe"),
                status=row.get("status", "success"),
            )
        )
    return entries


def get_latest_model(
    base_dir: str,
    symbol: str,
    timeframe: str,
    model_type: Optional[str] = None,
) -> Optional[RegistryEntry]:
    entries = list_models(
        base_dir,
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
    )
    entries = [e for e in entries if e.status == "success"]
    if not entries:
        return None
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries[0]


def load_model_by_id(base_dir: str, model_id: str):
    index_path = _registry_index_path(base_dir)
    if not os.path.exists(index_path):
        raise FileNotFoundError("Model registry index not found.")

    df = pd.read_csv(index_path)
    row = df[df["model_id"] == model_id]
    if row.empty:
        raise ValueError(f"Model id {model_id} not found in registry.")

    model_dir = row.iloc[0]["path"]
    return load_sklearn_model(model_dir)

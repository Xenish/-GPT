from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CandleFeatureConfig:
    """
    Candlestick pattern feature ayarları.

    Tüm threshold'lar "range oranı" üzerinden tanımlanmıştır.
    Yani 0.6 -> mum range'inin %60'ı demek.
    """

    # Doji: gövde küçük
    body_small: float = 0.1

    # Hammer / shooting star: gölge/gövde oranları
    shadow_long: float = 0.6
    upper_shadow_max_hammer: float = 0.2  # hammer'da üst gölge çok küçük

    # Marubozu: gövde büyük, gölgeler çok küçük
    marubozu_body_min: float = 0.8
    marubozu_shadow_max: float = 0.1


def _ensure_ohlc_columns(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing OHLC columns: {missing}")


def _add_candle_geometry(df: pd.DataFrame) -> None:
    """
    Mum geometrisini tanımlar:
      - gövde uzunluğu, gövde oranı
      - üst / alt gölge ve oranları
      - direction (bull/bear)
    """

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    body = c - o
    body_abs = body.abs()
    rng = (h - l).replace(0, np.nan)

    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l

    df["cs_body"] = body
    df["cs_body_abs"] = body_abs
    df["cs_range"] = rng

    df["cs_body_pct"] = body_abs / rng
    df["cs_upper_shadow"] = upper_shadow
    df["cs_lower_shadow"] = lower_shadow
    df["cs_upper_pct"] = upper_shadow / rng
    df["cs_lower_pct"] = lower_shadow / rng

    # Yön flags
    df["cs_bull"] = (body > 0).astype(int)
    df["cs_bear"] = (body < 0).astype(int)


def _add_doji(df: pd.DataFrame, cfg: CandleFeatureConfig) -> None:
    # Gövde çok küçük => doji
    df["cdl_doji"] = (df["cs_body_pct"] < cfg.body_small).astype(int)


def _add_hammer_shooting_star(df: pd.DataFrame, cfg: CandleFeatureConfig) -> None:
    """
    Hammer:
      - alt gölge uzun
      - üst gölge kısa
      - gövde çok küçük değil ama range'in belli bir oranı
    Shooting star:
      - tam tersi (üst gölge uzun, alt gölge kısa)
    """
    long_lower = df["cs_lower_pct"] > cfg.shadow_long
    long_upper = df["cs_upper_pct"] > cfg.shadow_long
    small_upper = df["cs_upper_pct"] < cfg.upper_shadow_max_hammer
    small_lower = df["cs_lower_pct"] < cfg.upper_shadow_max_hammer

    # Gövde çok küçük olmasın, tam doji olmasın diye:
    mid_body = (df["cs_body_pct"] >= cfg.body_small) & (df["cs_body_pct"] <= 0.7)

    df["cdl_hammer"] = (long_lower & small_upper & mid_body).astype(int)
    df["cdl_shooting_star"] = (long_upper & small_lower & mid_body).astype(int)


def _add_engulfing(df: pd.DataFrame) -> None:
    """
    Bullish engulfing:
      - önceki mum kırmızı
      - şimdiki mum yeşil
      - şimdiki gövde, öncekinin gövdesini sarıyor / daha büyük
    Bearish engulfing:
      - tam tersi
    """
    o = df["open"]
    c = df["close"]

    o_prev = o.shift(1)
    c_prev = c.shift(1)

    body_prev = (c_prev - o_prev)
    body_prev_abs = body_prev.abs()
    body_curr = (c - o)
    body_curr_abs = body_curr.abs()

    # Bullish engulfing
    cond_prev_bear = c_prev < o_prev
    cond_curr_bull = c > o
    cond_body_bigger = body_curr_abs > body_prev_abs
    cond_swallow = (c > o_prev) & (o < c_prev)

    bull_engulf = cond_prev_bear & cond_curr_bull & cond_body_bigger & cond_swallow

    # Bearish engulfing
    cond_prev_bull = c_prev > o_prev
    cond_curr_bear = c < o
    cond_body_bigger_bear = body_curr_abs > body_prev_abs
    cond_swallow_bear = (c < o_prev) & (o > c_prev)

    bear_engulf = cond_prev_bull & cond_curr_bear & cond_body_bigger_bear & cond_swallow_bear

    df["cdl_bull_engulf"] = bull_engulf.astype(int)
    df["cdl_bear_engulf"] = bear_engulf.astype(int)


def _add_inside_outside_bar(df: pd.DataFrame) -> None:
    """
    Inside bar:
      - current high < prev high
      - current low > prev low
    Outside bar:
      - current high > prev high
      - current low < prev low
    """
    h = df["high"]
    l = df["low"]

    h_prev = h.shift(1)
    l_prev = l.shift(1)

    inside = (h < h_prev) & (l > l_prev)
    outside = (h > h_prev) & (l < l_prev)

    df["cdl_inside_bar"] = inside.astype(int)
    df["cdl_outside_bar"] = outside.astype(int)


def _add_marubozu(df: pd.DataFrame, cfg: CandleFeatureConfig) -> None:
    """
    Marubozu:
      - gövde çok büyük
      - gölgeler çok küçük
    """
    big_body = df["cs_body_pct"] >= cfg.marubozu_body_min
    small_shadows = (df["cs_upper_pct"] <= cfg.marubozu_shadow_max) & (
        df["cs_lower_pct"] <= cfg.marubozu_shadow_max
    )

    maru = big_body & small_shadows
    df["cdl_marubozu"] = maru.astype(int)


def add_candlestick_features(
    df: pd.DataFrame,
    config: Optional[CandleFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Candlestick pattern feature'ları ekler.

    Giriş:
        df: open, high, low, close içeren DataFrame.
    Çıkış:
        Kopya DataFrame + cs_* ve cdl_* sütunları.
    """
    if config is None:
        config = CandleFeatureConfig()

    _ensure_ohlc_columns(df)

    df = df.copy()

    # 1) Mum geometrisi
    _add_candle_geometry(df)

    # 2) Tek tek pattern'ler
    _add_doji(df, config)
    _add_hammer_shooting_star(df, config)
    _add_engulfing(df)
    _add_inside_outside_bar(df)
    _add_marubozu(df, config)

    return df

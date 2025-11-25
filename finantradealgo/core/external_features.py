# finantradealgo/core/external_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import pandas as pd


@dataclass
class ExternalFeatureConfig:
    """
    Dış veri (funding, OI vs.) için config.

    funding_csv : 15m ya da daha sık funding verisi (timestamp + funding_rate)
    oi_csv      : 15m ya da daha sık OI verisi (timestamp + open_interest)
    """

    # Dosya yolları - yoksa None bırak, fonksiyon o kısmı atlar
    funding_csv: Optional[str] = None
    oi_csv: Optional[str] = None

    # Ortak ayarlar
    timestamp_col: str = "timestamp"
    resample_rule: str = "15T"   # 15 dakikalık bara map edeceğiz

    # Funding için kolon adı
    funding_col: str = "funding_rate"

    # OI için kolon adı
    oi_col: str = "open_interest"

    # Merge sonrası forward fill yapalım mı?
    forward_fill: bool = True

    # Rolling window'lar (bar sayısı)
    funding_zscore_window: int = 96   # ~1 gün (96 * 15m)
    oi_change_window_1: int = 4       # 1 saat
    oi_change_window_4: int = 16      # 4 saat
    oi_change_window_16: int = 64     # ~16 saat


def _load_and_resample_single(
    csv_path: str,
    timestamp_col: str,
    value_cols: List[str],
    rule: str,
) -> pd.DataFrame:
    """
    Genel amaçlı: timestamp + value_cols içeren bir CSV'yi yükle,
    timestamp'i index yap, rule ile yeniden örnekle (resample) ve geri döndür.
    """
    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    # Sadece istediğimiz value kolonlarını al
    df = df[value_cols]

    # Resample: burada 'mean' aldım, funding için genelde makul.
    df = df.resample(rule).mean()

    return df


def add_external_features_15m(
    df: pd.DataFrame,
    cfg: Optional[ExternalFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Mevcut 15m OHLCV + TA dataframe'ine funding / OI vb. dış veri ekler.

    Beklenti:
      - df[cfg.timestamp_col] var
      - df 15m barlar şeklinde

    Funding / OI csv yoksa, ilgili kısmı sessizce atlar (warning basar).
    """

    if cfg is None:
        cfg = ExternalFeatureConfig()

    if cfg.timestamp_col not in df.columns:
        raise ValueError(
            f"DataFrame'de '{cfg.timestamp_col}' kolonu yok. "
            "load_ohlcv_csv çıktısı gibi bir DF bekleniyor."
        )

    df = df.copy()
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col])
    df = df.sort_values(cfg.timestamp_col)
    df = df.set_index(cfg.timestamp_col)

    # --------------------------------------------------
    # 1) FUNDING FEATURES
    # --------------------------------------------------
    if cfg.funding_csv is not None:
        try:
            df_funding = _load_and_resample_single(
                csv_path=cfg.funding_csv,
                timestamp_col=cfg.timestamp_col,
                value_cols=[cfg.funding_col],
                rule=cfg.resample_rule,
            )

            # Kolonları daha açıklayıcı isimlerle yeniden adlandıralım
            df_funding = df_funding.rename(
                columns={
                    cfg.funding_col: "funding_rate_raw",
                }
            )

            # Ana DF ile birleştir
            df = df.join(df_funding, how="left")

            if cfg.forward_fill:
                df["funding_rate_raw"] = df["funding_rate_raw"].ffill()

            # Türetilmiş funding feature'ları
            # funding_rate_raw zaten oran, genelde küçük
            df["funding_rate_abs"] = df["funding_rate_raw"].abs()
            df["funding_rate_sign"] = df["funding_rate_raw"].apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            )

            # Rolling z-score: (x - mean) / std
            win = cfg.funding_zscore_window
            roll_mean = df["funding_rate_raw"].rolling(win).mean()
            roll_std = df["funding_rate_raw"].rolling(win).std()

            df["funding_rate_zscore"] = (df["funding_rate_raw"] - roll_mean) / roll_std

        except FileNotFoundError:
            print(f"[WARN] Funding CSV not found: {cfg.funding_csv}, funding features skipped.")

    # --------------------------------------------------
    # 2) OPEN INTEREST FEATURES
    # --------------------------------------------------
    if cfg.oi_csv is not None:
        try:
            df_oi = _load_and_resample_single(
                csv_path=cfg.oi_csv,
                timestamp_col=cfg.timestamp_col,
                value_cols=[cfg.oi_col],
                rule=cfg.resample_rule,
            )

            df_oi = df_oi.rename(
                columns={
                    cfg.oi_col: "oi_raw",
                }
            )

            df = df.join(df_oi, how="left")

            if cfg.forward_fill:
                df["oi_raw"] = df["oi_raw"].ffill()

            # Basit türev ve yüzde değişim feature'ları
            df["oi_change_1"] = df["oi_raw"].diff(cfg.oi_change_window_1)
            df["oi_change_4"] = df["oi_raw"].diff(cfg.oi_change_window_4)
            df["oi_change_16"] = df["oi_raw"].diff(cfg.oi_change_window_16)

            # Yüzde değişim (bölümde 0'a bölmemek için dikkatli ol)
            for col in ["oi_change_1", "oi_change_4", "oi_change_16"]:
                base = df["oi_raw"].shift(cfg.oi_change_window_1)  # kabaca önceki seviye
                df[col + "_pct"] = df[col] / (base.replace(0, pd.NA))

        except FileNotFoundError:
            print(f"[WARN] OI CSV not found: {cfg.oi_csv}, OI features skipped.")

    # --------------------------------------------------
    # Çıkış: index'i tekrar timestamp kolonu olarak dışarı al
    # --------------------------------------------------
    df = df.reset_index().rename(columns={"index": cfg.timestamp_col})

    return df

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from finantradealgo.core.strategy import BaseStrategy, SignalType, StrategyContext


@dataclass
class RuleStrategyConfig:
    """
    Rule-based long-only strateji için config.

    entry_col / exit_col  : DataFrame'deki kural sinyal kolonları
    warmup_bars           : Başta feature'ların oturması için pozisyon almadan geçilecek bar sayısı

    max_hold_bars         : Maksimum bar sayısı (15m bar -> 4 gün = 4 * 24 * 4)
    use_rule_exit         : rule_long_exit / signal=0 durumlarını dikkate al
    use_atr_tp_sl         : ATR tabanlı TP/SL overlay'ini aktif et
    atr_col               : ATR yüzdesi kolonu (ta_features içinde 'atr_14_pct' var)
    tp_atr_mult           : TP = entry_price * (1 + tp_atr_mult * atr_pct)
    sl_atr_mult           : SL = entry_price * (1 - sl_atr_mult * atr_pct)
    """

    entry_col: str = "rule_long_entry"
    exit_col: str = "rule_long_exit"
    warmup_bars: int = 50

    # Overlay 1: max hold
    max_hold_bars: int = 4 * 24 * 4  # 4 gün (15m barlar)
    min_hold_bars: int = 1

    # Overlay 2: rule exit kullan
    use_rule_exit: bool = True

    # Overlay 3: ATR TP/SL
    use_atr_tp_sl: bool = True
    atr_col: str = "atr_14_pct"
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RuleStrategyConfig":
        data = data or {}
        atr_period = data.get("atr_period")
        atr_col = data.get("atr_col")
        if atr_col is None and atr_period:
            atr_col = f"atr_{atr_period}_pct"

        return cls(
            entry_col=data.get("entry_col", cls.entry_col),
            exit_col=data.get("exit_col", cls.exit_col),
            warmup_bars=data.get("warmup_bars", cls.warmup_bars),
            max_hold_bars=data.get("max_hold_bars", cls.max_hold_bars),
            min_hold_bars=data.get("min_hold_bars", cls.min_hold_bars),
            use_rule_exit=data.get("use_rule_exit", cls.use_rule_exit),
            use_atr_tp_sl=data.get("use_atr_tp_sl", cls.use_atr_tp_sl),
            atr_col=atr_col or cls.atr_col,
            tp_atr_mult=data.get("atr_mult_tp", data.get("tp_atr_mult", cls.tp_atr_mult)),
            sl_atr_mult=data.get("atr_mult_sl", data.get("sl_atr_mult", cls.sl_atr_mult)),
        )


class RuleSignalStrategy(BaseStrategy):
    """
    Long-only kural tabanlı strateji.

    Beklenti:
      - DF'de giriş/çıkış için binary kolonlar var:
          entry_col (default: 'rule_long_entry')
          exit_col  (default: 'rule_long_exit')
      - init(df) çağrıldığında bu kolonlardan bir 'signal' kolonu üretilir:
          signal = 0 -> flat
          signal = 1 -> long

    Backtester:
      - init(df) ile stratejiyi başlatır
      - her bar için on_bar(row, ctx) çağırır
      - 'LONG' dönerse long açar
      - 'CLOSE' dönerse pozisyonu kapatır
    """

    def __init__(self, config: Optional[RuleStrategyConfig] = None):
        self.config = config or RuleStrategyConfig()
        self._df: Optional[pd.DataFrame] = None

        # Pozisyon state'i (Backtester ile senkron yürütüyoruz)
        self._in_position: bool = False
        self._entry_price: Optional[float] = None
        self._bars_in_position: int = 0

    # ------------------------
    # İç state reset fonksiyonu
    # ------------------------
    def _reset_position_state(self) -> None:
        self._in_position = False
        self._entry_price = None
        self._bars_in_position = 0

    # ------------------------
    # Signal kolonunu üret
    # ------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates 'long_entry', 'long_exit', 'short_entry', 'short_exit' columns.
        It uses `entry_col` and `exit_col` if they exist, otherwise assumes no signals.
        It maintains a state machine for being in a long position.
        """
        # Ensure entry/exit columns exist, default to False if not.
        if self.config.entry_col not in df.columns:
            df[self.config.entry_col] = False
        if self.config.exit_col not in df.columns:
            df[self.config.exit_col] = False

        entry = df[self.config.entry_col].fillna(0).astype(int)
        exit_ = df[self.config.exit_col].fillna(0).astype(int)

        position = 0
        signals = []

        for i, (e, x) in enumerate(zip(entry, exit_)):
            # Respect warmup period
            if i < self.config.warmup_bars:
                position = 0
            else:
                if position == 0 and e == 1:
                    position = 1  # Enter long
                elif position == 1 and x == 1:
                    position = 0  # Exit long

            signals.append(position)

        df["signal"] = signals

        # Create the four boolean columns required by the test contract
        is_long = pd.Series(signals, index=df.index) == 1
        was_long = is_long.shift(1).fillna(False).infer_objects(copy=False)

        df["long_entry"] = (is_long & ~was_long)
        df["long_exit"] = (~is_long & was_long)
        df["short_entry"] = False  # Long-only strategy
        df["short_exit"] = False   # Long-only strategy

        return df

    # ------------------------
    # BaseStrategy interface
    # ------------------------
    def init(self, df: pd.DataFrame) -> None:
        """
        Backtest başlamadan önce bir kere çağrılır.
        Burada:
          - signal kolonunu üret
          - internal state’i resetle
        """
        df_sig = self.generate_signals(df)
        self._df = df_sig
        self._reset_position_state()
        return None

    def on_bar(self, row: pd.Series, ctx: StrategyContext) -> SignalType:
        """
        Her yeni bar için Backtester tarafından çağrılır.

        Dönüş:
          - "LONG"  -> long aç
          - "CLOSE" -> pozisyonu kapat
          - None    -> hiçbir şey yapma
        """
        if self._df is None:
            return None

        sig = row.get("signal", np.nan)
        if np.isnan(sig):
            return None

        price = float(row["close"])

        # ------------- FLAT STATE -------------
        if not self._in_position:
            # Sadece signal==1 ise ve flat isek LONG gönder
            if sig >= 1.0:
                self._in_position = True
                self._entry_price = price
                self._bars_in_position = 0
                return "LONG"
            return None

        # ------------- IN-POSITION STATE -------------
        # Buraya gelirsek self._in_position = True

        # 1) ATR tabanlı TP/SL overlay
        if self.config.use_atr_tp_sl and self.config.atr_col in row.index:
            atr_val = row[self.config.atr_col]
            try:
                atr_pct = float(atr_val)
            except (TypeError, ValueError):
                atr_pct = np.nan

            if not np.isnan(atr_pct) and self._entry_price is not None:
                tp_price = self._entry_price * (1.0 + self.config.tp_atr_mult * atr_pct)
                sl_price = self._entry_price * (1.0 - self.config.sl_atr_mult * atr_pct)

                # TP veya SL tetiklendi mi?
                if price >= tp_price or price <= sl_price:
                    self._reset_position_state()
                    return "CLOSE"

        # 2) Max/min hold overlay
        self._bars_in_position += 1
        bars_held = self._bars_in_position
        if self.config.max_hold_bars > 0 and bars_held >= self.config.max_hold_bars:
            self._reset_position_state()
            return "CLOSE"

        # 3) Rule exit (signal tekrar 0'a dönmüşse ya da exit_col tetiklenmişse)
        if self.config.use_rule_exit and bars_held >= self.config.min_hold_bars:
            if sig <= 0:
                self._reset_position_state()
                return "CLOSE"

        # Overlay’lerin hiçbiri tetiklenmediyse → pozisyonu koru
        return None

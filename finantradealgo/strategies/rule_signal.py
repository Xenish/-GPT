
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

from finantradealgo.core.risk import RiskEngine


@dataclass
class RuleStrategyConfig:
    warmup_bars: int = 200     # Feature’ların oturması için
    max_hold_bars: int = 4 * 24 * 4  # 4 gün ~ 4*24*4 bar (15m) – istersen 0 yapıp kapatabilirsin


class RuleSignalStrategy:
    
    Rule-based long-only strateji.
    Giriş/çıkış için df içindeki:
      - rule_long_entry
      - rule_long_exit
    kolonlarını kullanır.
    

    def __init__(self, config: Optional[RuleStrategyConfig] = None):
        self.config = config or RuleStrategyConfig()
        self.position_open_bar: Optional[int] = None
        self.current_pos_size: float = 0.0

    def reset(self) -> None:
        self.position_open_bar = None
        self.current_pos_size = 0.0

    def name(self) -> str:
        return "RuleSignalV1"

    def on_bar(
        self,
        i: int,
        row: pd.Series,
        risk_engine: RiskEngine,
        portfolio: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        Backtester her bar için bunu çağırıyor varsayımıyla yazdım.
        Senin Backtester başka bir interface kullanıyorsa,
        MLSignalStrategy'deki fonksiyon imzasını bire bir kopyalayıp,
        içindeki mantığı buradaki gibi değiştir.
       

        signals: Dict[str, Any] = {}

        # Warmup süresinden önce sinyal üretme
        if i < self.config.warmup_bars:
            return signals

        price = float(row["close"])

        in_position = self.current_pos_size > 0

        # === EXIT LOGIC ===
        exit_signal = bool(row.get("rule_long_exit", 0))
        max_hold_exceeded = False
        if in_position and self.config.max_hold_bars > 0 and self.position_open_bar is not None:
            if i - self.position_open_bar >= self.config.max_hold_bars:
                max_hold_exceeded = True

        if in_position and (exit_signal or max_hold_exceeded):
            # Pozisyonu tamamen kapat
            signals["exit_long"] = {
                "size": self.current_pos_size,
                "price": price,
            }
            self.current_pos_size = 0.0
            self.position_open_bar = None
            # Exit yaptıktan sonra aynı barda tekrar entry’e izin verme
            return signals

        # === ENTRY LOGIC ===
        entry_signal = bool(row.get("rule_long_entry", 0))

        if (not in_position) and entry_signal:
            # RiskEngine’den pozisyon büyüklüğünü hesapla
            equity = float(portfolio.get("equity", 0.0))
            size = risk_engine.calc_position_size(
                equity=equity,
                price=price,
            )

            if size > 0:
                signals["enter_long"] = {
                    "size": size,
                    "price": price,
                }
                self.current_pos_size = size
                self.position_open_bar = i

        return signals

        

        """




## 🧠 Bölüm 3 – Strategy Interface & Backtest Engine (**Zorunlu**)

**Amaç:** Tüm stratejiler aynı çerçeveye otursun, backtest motoru net ve deterministik olsun.

- [ ]  Tüm stratejiler ortak bir base class’tan türemek zorunda (örn. `BaseStrategy`):
    - [ ]  Ortak interface (ör: `generate_signals(df) -> pd.Series` veya `on_bar(ctx)` gibi) net tanımlı.
    - [ ]  Strateji engine kodu strategy’nin iç implementasyon detaylarını bilmek zorunda değil.
- [ ]  Backtest konfigürasyonu için tek bir dataclass:
    - [ ]  `BacktestConfig` (symbol, timeframe, start/end, initial_capital, fee, slippage, risk_params vs.)
- [ ]  Backtest sonucu için tek bir dataclass:
    - [ ]  `BacktestResult`:
        - [ ]  `equity_curve: pd.Series`
        - [ ]  `trades: pd.DataFrame` (entry/exit time, size, pnl vs.)
        - [ ]  `metrics: dict[str, float]` (CAGR, Sharpe, maxDD vs.)
- [ ]  Backtest runner:
    - [ ]  `run_backtest(strategy: BaseStrategy, config: BacktestConfig) -> BacktestResult` gibi tek bir fonksiyon/entrypoint.
    - [ ]  Hem CLI, hem API, hem de testler bu fonksiyon üzerinden geçiyor.
- [ ]  Output kaydetme:
    - [ ]  Standart bir `save_backtest_result(result, path)` fonksiyonu var.
    - [ ]  Sonuçlar `results/backtests/{strategy_name}/{run_id}.json` (veya Parquet) formatında kaydediliyor.
    - [ ]  `run_id` olarak en azından timestamp + random suffix var (collision riski yok).
- [ ]  En az bir **deterministik** birim test:
    - [ ]  Küçük bir sentetik OHLCV datasında basit bir strategy (ör: “her bar long aç, bir bar sonra kapa”) için:
        - [ ]  Trade sayısı, toplam PnL ve maxDD sabit beklenen değerlerle eşleşiyor.

---

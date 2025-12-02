## 📊 Bölüm 7 – Web UI (Next.js) – Backtest Paneli & Raporlar (**Zorunlu v1 seviyesi**)

**Amaç:** UI’dan strateji seçip backtest çalıştırabildiğin ve *kullanılabilir* bir rapor görebildiğin bir panel.

- [ ]  UI’da “Backtest” için ayrı bir sayfa / route var:
    - [ ]  Örn: `/backtests`
- [ ]  Bu sayfada:
    - [ ]  Strategy seçimi için dropdown (backend’den gelen strateji listesi veya frontend’de sabit bir liste).
    - [ ]  Symbol ve timeframe seçimi input’ları.
    - [ ]  Tarih aralığı seçimi için datepicker veya min. iki input.
    - [ ]  Strateji parametreleri için form:
        - [ ]  En azından integer/float slider veya input (ör: ema_fast, ema_slow).
- [ ]  “Run Backtest” butonuna basınca:
    - [ ]  Backend’de `POST /backtests/run` çağrılıyor.
    - [ ]  UI loading state gösteriyor (spinner vs.).
    - [ ]  Success case’de:
        - [ ]  Backtest sonucunu görüntüleme alanı açılıyor.
- [ ]  Rapor bileşenleri:
    - [ ]  Equity curve chart (zaman serisi)
    - [ ]  Drawdown chart (zaman serisi veya alt panel)
    - [ ]  Özet metrikler:
        - [ ]  Sharpe, toplam getir, maxDD, trade sayısı, win rate.
    - [ ]  Trade listesi tablosu:
        - [ ]  Entry time, exit time, direction, size, PnL (en az bunlar).
- [ ]  Chart overlay (minimum):
    - [ ]  Candlestick chart üzerinde:
        - [ ]  Entry’ler için bir marker (ok/ikon).
        - [ ]  Exit’ler için başka bir marker.
    - [ ]  Marker pozisyonları backend’ten gelen trade log ile uyumlu.
- [ ]  UI testleri (veya en azından manuel kabul kriteri):
    - [ ]  Örnek bir config ile:
        - [ ]  Form doldurulup backtest çalıştırılınca, UI’da gerçekten equity curve vs. görüntüleniyor.
        - [ ]  Eğer backend hata dönerse (ör: data yok), UI anlamlı bir hata mesajı gösteriyor (boş sayfa değil).

---
## 🔬 Bölüm 4 – Research & Strategy Search (CLI/Programatik) (**Zorunlu v1 seviyesi**)

**Amaç:** En azından CLI’dan parametre araması yapabildiğin, sonuçları dosyaya kaydeden bir research iskeleti olsun.

- [ ]  Strategy search için tek bir high-level API var:
    - [ ]  Örn: `run_param_search(strategy_name, param_grid, config) -> list[BacktestResultSummary]`
- [ ]  Param grid şu tipte destekleniyor:
    - [ ]  Dictionary + list’ler (örn. `{"ema_fast": [10, 20], "ema_slow": [50, 100]}`)
- [ ]  Param kombinasyonları otomatik üretiliyor (itertools.product vs.).
- [ ]  Her kombinasyon için:
    - [ ]  Backtest çalışıyor.
    - [ ]  Sonuç, en azından şu bilgileri taşıyan bir “summary” objesi ile kaydediliyor:
        - [ ]  strategy_name
        - [ ]  params (dict)
        - [ ]  sharpe, return, maxDD, trades, win_rate
        - [ ]  run_id / result_path
- [ ]  Sonuçlar tek bir file’a da özetleniyor:
    - [ ]  Örn: `results/search/{strategy_name}/{search_id}.parquet` veya `.json` (satır = param kombinasyonu).
- [ ]  Basit CLI komutu:
    - [ ]  `python -m finantradealgo.research.run_search --config config/system.research.yml --strategy ema_example --search config/search/ema_search.yml`
    - [ ]  Çalıştığında:
        - [ ]  Tüm kombinasyonları dener.
        - [ ]  En iyi N kombinasyonu summary olarak ekrana yazdırır.
        - [ ]  Tam sonuçları `results/search/...` altına kaydeder.
- [ ]  En az bir test:
    - [ ]  Küçük bir param grid ile (örn. 2×2 = 4 kombinasyon) toplam kombinasyon sayısı ve summary dosyası boyutu/şekli assert ediliyor.

*(Burada henüz “job queue / async / UI job builder” istemiyorum; onlar Opsiyonel’e gidecek.)*

---

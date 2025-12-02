## 🧱 Bölüm 1 – Config Profilleri & Ortam Ayrımı (**Zorunlu**)

**Amaç:** Research ile live birbirine **karışmasın**, config tekil ve tutarlı olsun.

- [ ]  `config/system.research.yml` dosyan var ve:
    - [ ]  İçinde **sadece** backtest/research için gereken şeyler var (exchange.type: `"backtest"` veya `"mock"`).
    - [ ]  Hiçbir canlı API key / secret *düz metin* olarak tutulmuyor.
- [ ]  `config/system.live.yml` dosyan var ve:
    - [ ]  Canlı ortamda kullanılacak exchange ayarlarını içeriyor.
    - [ ]  API key / secret gibi bilgiler sadece `ENV` referansıyla tutuluyor (ör: `api_key_env: BINANCE_API_KEY`).
- [ ]  `finantradealgo/system/config_loader.py` içinde:
    - [ ]  `load_config(profile: Literal["research", "live"])` benzeri tek bir entrypoint ile hem research hem live config yüklenebiliyor.
    - [ ]  Hatalı profil string’i verilirse temiz bir exception atıyor (ValueError vs.).
- [ ]  Testler:
    - [ ]  `pytest` default olarak **research** / test config kullanıyor; live endpoint’e istek atma ihtimali yok.
    - [ ]  En az bir test, `load_config("research")` ve `load_config("live")` için doğru tipte config döndüğünü assert ediyor.
- [ ]  En az iki örnek strateji config’i var:
    - [ ]  `config/strategies/ema_example.yml`
    - [ ]  `config/strategies/rsi_example.yml` (veya benzeri)
    - [ ]  Bu dosyalarla backtest CLI/API’si gerçekten çalışıyor.

---

## 📂 Bölüm 2 – Data Layer & Sembol / Timeframe Modeli (**Zorunlu**)

**Amaç:** Data erişimi düzenli, genişletilebilir ve güvenilir olsun.

- [ ]  Global `data_root` config’te tek bir yerde tanımlı.
- [ ]  OHLCV için tekil bir path şablonu var, örneğin:
    - [ ]  `ohlcv_path_template: "{data_root}/ohlcv/{symbol}/{timeframe}.parquet"`
- [ ]  Data loader:
    - [ ]  `load_ohlcv(symbol, timeframe, config)` gibi tek bir fonksiyon üzerinden kullanılıyor.
    - [ ]  Dosya yoksa anlamlı bir exception raise ediyor (ör: `DataNotFoundError`).
    - [ ]  Boş dosya / çok az veri varsa uyarı logluyor ve fail ediyor (sessizce saçma backtest yapmıyor).
    - [ ]  Zaman kolonu strictly artan ve duplicate timestamp yok → ihlal varsa log + exception.
- [ ]  Birim testler:
    - [ ]  Sahte / küçük bir CSV/Parquet dosyası ile:
        - [ ]  Normal case: Data yükleniyor, satır sayısı ve kolonlar assert ediliyor.
        - [ ]  Duplicate time içeren bir versiyon için hata fırlatıldığı test ediliyor.

---

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

## ⚡ Bölüm 5 – Live Engine & Risk Guardrails (**Zorunlu v1 seviyesi**)

**Amaç:** En azından **paper trading + basit risk limitleri** olan, kafa rahat çalışabilir minimal live motor.

- [ ]  Live engine için tek bir entrypoint var:
    - [ ]  Örn: `python -m finantradealgo.live.run --config config/system.live.yml --mode paper`
- [ ]  En az 2 mod:
    - [ ]  `"paper"`: Emirler gerçekte borsaya gitmiyor, local ledger üzerinden simüle ediliyor.
    - [ ]  `"live"`: Gerçek exchange API’sine emir atıyor (v1’de bunu kapalı tutsan bile mekanizma hazır).
- [ ]  Risk config:
    - [ ]  `max_daily_loss` (ör: % veya USD)
    - [ ]  `max_position_per_symbol`
    - [ ]  `max_global_notional`
- [ ]  Risk enforcement:
    - [ ]  Her yeni trade öncesi bu limitler kontrol ediliyor.
    - [ ]  Limit aşıldığında **trade açılmıyor**, log’da açık bir warning/error var, sistem crash olmuyor.
- [ ]  Günlük reset mantığı:
    - [ ]  Her günün PnL’i hesaplanıyor.
    - [ ]  `max_daily_loss` aşıldığında o gün için sistem trade açmayı bırakıyor.
- [ ]  State & restart:
    - [ ]  Engine yeniden başlatıldığında:
        - [ ]  Açık pozisyon bilgisi exchange’ten (veya paper ledger’dan) okunuyor.
        - [ ]  Aynı pozisyonu yeniden açmıyor, double exposure oluşturmuyor.
- [ ]  En az bir integration test (mock exchange ile):
    - [ ]  Gün içi belli sayıda loss sonrası yeni trade açılmadığı assert ediliyor.

---

## 🌐 Bölüm 6 – Backend API (FastAPI) (**Zorunlu**)

**Amaç:** UI ve dış dünya backend’le düzgün konuşabilsin.

- [ ]  FastAPI app’in tek bir module/entrypoint altında:
    - [ ]  Örn: `finantradealgo/api/app.py` → `app = FastAPI(...)`
- [ ]  Temel endpoint’ler:
    - [ ]  `GET /health` → `{"status": "ok"}` (test ve Docker health check için).
    - [ ]  `POST /backtests/run`:
        - [ ]  Body: strategy_name, symbol, timeframe, tarih aralığı, param dict.
        - [ ]  Behavior: `run_backtest(...)` çalıştırır ve `BacktestResult`’ı JSON olarak döner *veya* bir `job_id` döner (senin v1 tasarımına göre).
    - [ ]  `GET /backtests/{run_id}`:
        - [ ]  JSON olarak kaydedilmiş backtest sonucunu döner.
- [ ]  Pydantic modeller:
    - [ ]  Request için: `BacktestRequest`
    - [ ]  Response için: `BacktestResponse` (BacktestResult’ı JSON-serializable hale getirir).
- [ ]  API testleri:
    - [ ]  `TestClient` ile:
        - [ ]  `/health` 200 ve `"ok"` döndüğü test ediliyor.
        - [ ]  Örnek bir `POST /backtests/run` çağrısı gerçek bir run_id veya inline result dönüyor.
        - [ ]  Hatalı input’ta (ör: bilinmeyen strategy_name) 4xx ve anlamlı mesaj dönüyor.

---

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

## 🧪 Bölüm 8 – CI, Test & Docker (**Zorunlu**)

**Amaç:** Bozulmuş bir şeyi master’a ittirmen zor olsun, “tek komutla ayağa kalkan” bir sistem olsun.

- [ ]  GitHub Actions:
    - [ ]  Backend job:
        - [ ]  Python 3.11
        - [ ]  `pip install -r requirements.txt`
        - [ ]  `pytest -q -m "not slow"` sorunsuz çalışıyor.
    - [ ]  Frontend job:
        - [ ]  Node 20
        - [ ]  `npm install`
        - [ ]  `npm run lint`
        - [ ]  `npm run build`
- [ ]  CI’de tüm job’lar yeşil → v1 için kırmızı hiçbir test kalmıyor.
- [ ]  Local’de:
    - [ ]  `pytest -q -m "not slow"` tek komutla koşuyor ve full yeşil.
- [ ]  Docker:
    - [ ]  Root’ta bir `docker-compose.yml` var.
    - [ ]  En azından şu servisler:
        - [ ]  `api` (FastAPI)
        - [ ]  `web` (Next.js)
    - [ ]  `docker-compose up --build` ile:
        - [ ]  API container ayağa kalkıyor ve `/health` 200 döndürüyor.
        - [ ]  Web container ayağa kalkıyor ve `/backtests` sayfası browser’dan açılabiliyor.
- [ ]  README’de:
    - [ ]  “Quickstart” bölümünde:
        - [ ]  5–7 adımda repo’yu klonlayıp, env ayarlayıp, docker-compose ile sistemi ayağa kaldırmayı anlatan net komutlar var.

---

## 📜 Bölüm 9 – Logging & Basit Monitoring (**Zorunlu v1 seviyesi**)

**Amaç:** En azından bir şey patladığında ne olduğunu görebilesin; canlıda karanlıkta kalmayasın.

- [ ]  Tek bir logging config’in var:
    - [ ]  Örn: `logging.yml` veya Python `dictConfig`.
- [ ]  Backtest ve live ayrı log dosyalarına yazıyor:
    - [ ]  `logs/backtest.log`
    - [ ]  `logs/live.log`
- [ ]  Live log’unda:
    - [ ]  Her order denemesi (symbol, direction, size, price) INFO level’da kayıtlı.
    - [ ]  Hata / exception’lar WARN/ERROR level’da kayıtlı.
- [ ]  Günlük log rotation mevcut (dev ortamında bile tek dosya 10GB’lara şişmiyor).
- [ ]  En az bir kritik path’te (örn. risk limit breach, API throttling):
    - [ ]  Log’da açık mesaj var (ör: `MAX_DAILY_LOSS_REACHED`, `EXCHANGE_RATE_LIMIT` gibi anahtar kelime ile).
- [ ]  Kunta kinte seviye monitoring istemiyorum, ama:
    - [ ]  `GET /health` endpoint’i:
        - [ ]  Uygulamanın çalıştığı, config yüklendiği ve temel bağımlılıkların ayakta olduğu anlamına geliyor (kuru “ok” değil).

---

## 📚 Bölüm 10 – Kullanım Senaryoları & Dokümantasyon (**Zorunlu**)

**Amaç:** Sen 3 ay projeye ara verdiğinde bile geri dönüp “Bu neydi?” demeden kullanabilesin.

- [ ]  README veya `docs/` altında:
    - [ ]  “Kişisel Quant Lab v1 – Kullanım Akışı” başlıklı bir bölüm var.
- [ ]  En az 3 net kullanım senaryosu dokümante:
    1. **Basit backtest**:
        - [ ]  Örn: EMA cross stratejisini BTCUSDT 1h için 1 yıl boyunca backtest et → CLI komutları + UI adımları tek tek yazılı.
    2. **Parametre araması**:
        - [ ]  Search config → CLI komutu → beklenen output path → sonuçları nasıl okuyacağın (ör. pandas ile).
    3. **Paper trading başlatma**:
        - [ ]  Live config’in doldurulması (mock/paper), engine’in çalıştırılması, log’ların kontrolü, pozisyonların nasıl görüntüleneceği.
- [ ]  Dokümanda:
    - [ ]  “Risk uyarıları” kısmı var:
        - [ ]  Bu sistemin **beta/v1** olduğu, gerçek sermaye riskine girmeden önce uzun süre paper test yapılması gerektiği açıkça yazıyor.
- [ ]  Strateji geliştirme için mini rehber:
    - [ ]  Yeni bir stratejinin `BaseStrategy`’den türetilerek nasıl ekleneceğini adım adım açıklayan kısa bir bölüm:
        - [ ]  Class nerede tanımlanacak.
        - [ ]  Config’e nasıl eklenecek.
        - [ ]  UI’da seçilebilir hale getirmek için ne yapılacak.

---

## ⭐  V1.1+ (Bitirince Bonus, v1 için ŞART DEĞİL)

Bunlar *ekstra kas*. V1 tanımına dahil etmiyorum ama uzun vadede isteyeceksin:

- [ ]  Research job queue (SQLite + background worker, Celery/RQ vs.)
- [ ]  UI’dan full “job builder” (param aralığı slider, random search, vs.)
- [ ]  Gelişmiş raporlar:
    - [ ]  Regime-based performance (bull/bear/sideways)
    - [ ]  Intraday saatlere göre PnL
- [ ]  Telegram/Discord uyarıları (live’da kritik hatalarda DM atma)
- [ ]  Prometheus/Grafana gibi metrik entegrasyonları

---
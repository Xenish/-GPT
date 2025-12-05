# FinanTradeAlgo

[![CI](https://github.com/<username>/TradeProject/actions/workflows/ci.yml/badge.svg)](https://github.com/<username>/TradeProject/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Node 20](https://img.shields.io/badge/node-20-green.svg)](https://nodejs.org/)

## Proje Özeti
- **15m kripto araştırma ortamı**: Şu anda AIAUSDT ve BTCUSDT odaklı ama yapı diğer sembollere de kolayca genişliyor.
- **Feature pipeline** fiyat/TA + microstructure + market structure + funding/OI + flow + sentiment kaynaklarını tek veri setinde birleştiriyor.
- **Rule & ML stratejileri** portföy backtester, senaryo motoru, live/paper trading engine ve FastAPI + Next.js tabanlı UI ile uçtan uca deney alanı sağlıyor.
- **Registry & explainability**: ML modelleri için registry, feature importance, hyperparam grid search ve CLI wrapper destekleniyor.
- **API & Frontend**: FastAPI backend ve Next.js frontend, kullanıcıya chart/portfolio/strategy lab/ML lab/live kontrol paneli sunuyor.

## Mimari Overview
- **Backend katmanları**
  - `finantradealgo/data_engine`: OHLCV, funding, OI, flow, sentiment vb. loader ve veri kaynakları.
  - `finantradealgo/features`: TA, micro/macro structure, flow, sentiment ve rule sinyalleri dahil feature pipeline.
  - `finantradealgo/strategies`: Rule, ML ve diğer strateji sınıfları + StrategyEngine.
  - `finantradealgo/risk`: RiskEngine, pozisyon boyutlama, günlük limit vb.
  - `finantradealgo/backtester`: BacktestEngine, PortfolioBacktestEngine, ScenarioEngine, Walkforward araçları.
  - `finantradealgo/ml`: Labeling, modeller, registry, hyperparam search, importance.
  - `finantradealgo/live_trading`: LiveEngine, replay data source, execution client, snapshot sistemi.
  - `finantradealgo/api`: FastAPI sunucusu; backtest, portfolio, scenario, ML model ve live control endpoint’leri.
- **Frontend**
  - `frontend/web`: Next.js + lightweight-charts tabanlı UI; tabs: Single instrument chart, Portfolio, Strategy Lab, ML Lab, Live Control.

## Kurulum
```bash
git clone https://github.com/<you>/TradeProject.git
cd TradeProject

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## CI Sözleşmesi (V1)
CI şu kontrolleri çalıştırır:
1) Lint: `ruff check finantradealgo services tests scripts` ve `black --check ...`
2) Typecheck: `mypy --config-file mypy.ini finantradealgo/system finantradealgo/risk finantradealgo/live_trading`
3) Test + coverage: `pytest -m "not slow" --cov=finantradealgo --cov=services --cov-report=xml --cov-report=term-missing --cov-fail-under=60`
4) Config/risk guardrails: `python scripts/check_config_sanity.py`, `python scripts/check_strategy_dependency.py`, `python scripts/check_research_imports.py`
5) CLI smoke: `pytest -q tests/test_run_*_cli_*.py tests/test_run_test_risk_overlays_rg1.py`
6) DB integration (eğer `FT_TIMESCALE_DSN` veya `FT_POSTGRES_DSN` secret'ı tanımlıysa): `alembic history -q` + `pytest -m "db"`; normal PR CI'da DB testleri skip edilir.
Hepsinin geçmesi gerekir; coverage XML artefakt olarak yüklenir.
Branch protection önerisi: main/master için yukarıdaki CI check'leri zorunlu status check olarak tanımlayın (lint, typecheck, backend+coverage, guardrails, CLI smoke; varsa DB job).

### Lokal kalite
- `pip install pre-commit && pre-commit install` ile aynı lint/format kurallarını commit öncesi çalıştırabilirsiniz.

## Hızlı Başlangıç
### Data / Feature
```bash
python scripts/run_build_features_15m.py
# veya CLI
finantrade build-features --symbol AIAUSDT --tf 15m
```

### ML Train + Backtest
```bash
python scripts/run_ml_train_15m.py
python scripts/run_ml_backtest_15m.py
```

### API
```bash
python scripts/run_api.py
# veya uvicorn
uvicorn finantradealgo.api.server:create_app --factory --reload
```

### Frontend
```bash
cd frontend/web
cp .env.local.example .env.local
# gerekirse NEXT_PUBLIC_API_BASE_URL'i .env.local içinde güncelle
npm install
npm run dev
```

### Docker Quickstart
```bash
docker-compose up --build
# API → http://localhost:8000/docs
# UI  → http://localhost:3000
```
Docker içinde frontend servisi `NEXT_PUBLIC_API_BASE_URL=http://finantrade_api:8000` ile gelir; farklı ortamlar için `docker-compose.override.yml` veya environment injector kullanabilirsiniz.

### CLI kullanımı
```bash
pip install -e .

# Örnek komutlar:
finantrade build-features --symbol AIAUSDT --tf 15m
finantrade backtest --strategy rule --symbol AIAUSDT --tf 15m
finantrade ml-train --symbol AIAUSDT --tf 15m --preset extended
finantrade live-paper --symbol AIAUSDT --tf 15m
```

Backend exchange erişimi için çevre değişkenleri:
```bash
cp .env.example .env
# BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET değerlerini doldurun
# Ardından `source .env` (Linux/macOS) veya `Set-Content Env:*` (Windows) ile env'e yükleyin
```

### Config profilleri (research vs live)
- Profiller: `research` backtest/research içindir, `live` paper/exchange içindir. Ortak ayarlar `config/system.base.yml`'de, profil farkları `config/system.research.yml` ve `config/system.live.yml` içinde override edilir.
- Tek giriş noktası: YAML doğrudan okunmaz, her zaman loader kullanılır.
  ```python
  from finantradealgo.system.config_loader import load_config, load_config_from_env

  cfg = load_config("research")          # profil ismiyle
  cfg = load_config_from_env()           # FINANTRADE_PROFILE env (yoksa research)
  ```
- Profil seçimi: CLI/script parametresi vermeden ortamdan seçmek için:
  ```bash
  export FINANTRADE_PROFILE=live   # veya research
  ```
  `--profile` verilmezse tüm CLI/script akışları bu env'i dikkate alır.
- Güvenlik: Config her yüklemede validate edilir (required alanlar, aralıklar, live güvenlik). Hatalı kombinasyonlar yükleme aşamasında patlar.

Testnet dry-run:
```bash
# config/system.live.yml içinde exchange.dry_run=true iken
python scripts/run_exchange_dry_test.py
# Binance testnet endpointlerine bağlanır, account info basar, order göndermeden çıkar.
```

### Live WS debug
Binance WS kaynağını hızlıca test etmek için:
```bash
# exchange/testnet ayarlarını yaptıktan sonra
python scripts/run_live_ws_debug.py
# veya belirli semboller:
python scripts/run_live_ws_debug.py --symbol BTCUSDT --symbol ETHUSDT --count 5
```
Script, gelen agregasyon barlarını stdout'a yazar; CTRL+C ile çıkabilirsiniz.

### Live exchange run (testnet/dry-run)
Exchange modunu denemek için:
```bash
python scripts/run_live_exchange_15m.py
```
`config/system.live.yml` içindeki `exchange.testnet=true` ve `exchange.dry_run=true` olduğundan emin olun. Testnet dışına çıkmadan önce minimum notional / düşük kaldıraçla deneyin ve `dry_run` bayrağını kaldırmadan önce gerçek API key'lerinizi kontrol edin.

## Kısaltılmış Dosya Ağacı
```
config/
data/
docs/
finantradealgo/
  data_engine/
  features/
  strategies/
  risk/
  backtester/
  ml/
  live_trading/
  api/
frontend/web/
scripts/
tests/
outputs/
```

## 📚 Dokümantasyon

Kapsamlı Türkçe dokümantasyon için:

**[📖 Dokümantasyon Ana Sayfa](docs/README.md)**

### Hızlı Linkler
- 🚀 **[Hızlı Başlangıç](docs/kullanici/01-hizli-baslangic.md)** - 15 dakikada sistemi çalıştırın
- 📘 **[Kullanıcı Kılavuzu](docs/README.md#-kullanc-klavuzu)** - Sistemi nasıl kullanırsınız
- 🔧 **[Strateji Geliştirici Kılavuzu](docs/README.md#-strateji-gelitirici-klavuzu)** - Özel stratejiler geliştirin
- ⚙️ **[Konfigürasyon Referansı](docs/konfigürasyon/)** - Tüm parametreler
- 📖 **[Örnekler](docs/ornekler/)** - Adım adım uygulamalar
- 🌐 **[API Dokümantasyonu](docs/api/)** - REST API kullanımı

## Notlar
- Bu depo araştırma ve prototipleme amaçlıdır; **production trading riski size aittir.**
- Binance / diğer veri kaynakları için rate limit / API key / mevzuat sorumluluğu size aittir.
- Gönderilen CLI (`finantrade ...`) tüm temel script'leri tek çatı altında toplar.

## Veri Deposu / Backend Se?imi
- Varsay?lan CSV: data.backend: csv, yollar data/ohlcv/{symbol}_{timeframe}.csv.
- Timescale/Postgres: data.backend: timescale, data.backend_params.dsn:  (Alembic 0002/0003 migration).
- DuckDB/Parquet: data.backend: duckdb, data.backend_params.database: data/catalog.duckdb.
- Live/paper i?in live.data_source: replay_db depodan replay; WS i?in inance_ws.

## Ingestion / Feature Build / Monitoring
- Tarihsel/catch-up ingest: python scripts/ingest_marketdata.py historical --symbols BTCUSDT --timeframes 1m --lookback-days 30
- Scheduler (cron + Prometheus metrics :9200): python scripts/schedule_ingestion.py --config config/system.live.yml
- Incremental feature build: python scripts/run_feature_builder.py incremental --symbols BTCUSDT --timeframes 15m --dsn 
- Status API (FastAPI): uvicorn scripts.status_api:app --port 8001 (watermark/runs endpoints)

# FinanTradeAlgo – V1 Done Manifesto

Bu repo kişisel quant lab V1’dir. Backtest, live, ML, risk, raporlama ve temel UI çekirdek sözleşmelerle testlidir. Bu doküman, faz bazında neyi sabitlediğimi ve hangi boşlukları bilinçli olarak V2+’ya ittiğimi anlatır.

## Fazlar & Kapsam
- Faz 1 – Config & Profiller
- Faz 2 – Data & Storage
- Faz 3 – Backtester & Strategy Engine
- Faz 4 – ML & Walkforward
- Faz 5 – Live & Risk
- Faz 6 – Reporting & Research API
- Faz 7 – Core Web UI (Research + Live)
- Faz 8 – Market & Microstructure Features

## Faz Bazlı Garanti Özeti

### Faz 1 – Config & Profiller
- Tek entrypoint: `load_config` + `load_config_from_env`; legacy `--config` yolları CI’da reddedilir (`tests/test_cli_config_usage.py`, `tests/test_no_legacy_config_flag.py`).
- `research` / `live` profilleri base config’ten merge edilir; required/range/cross-field validation testlerle zorunlu, hatalı profil ValueError/RuntimeError ile fail-fast.
- Warehouse/DB config: Timescale/none path’leri testli (`tests/test_warehouse_config_rc1.py`); env yoksa graceful skip, varsa gerçek DSN roundtrip’i DB job’unda koşar.

### Faz 2 – Data & Storage
- Ingestion writer factory sözleşmesi kilitli (`tests/test_ingestion_writer_factory.py`); unsupported backend ValueError fırlatır.
- OHLCV ingestion DB’ye yazıyor (`tests/test_ingest_ohlcv_db.py` @db) ve CLI wiring mock’la assertli.
- TimeSeriesDBClient için gerçek DB roundtrip testi var (`tests/test_timeseries_db_client_db.py` @db); Postgres/Redis helper’ları smoke-testli.
- Migration → ingest → feature → backtest zinciri happy path ile testli (`tests/test_db_lifecycle_happy_path.py` @db).

### Faz 3 – Backtester & Strategy Engine
- Strategy registry + param_space kontratı tek testte kilitli (`tests/test_strategy_param_space_contract.py`); param space eksik/bozuksa CI kırmızı.
- Event/bar invariants (monotonic ts, volume conservation, orderbook spread) testli (`tests/test_event_bars_validation.py`).
- Basic backtest path smoke (`tests/test_backtest_basic_run.py`) ve golden regression (rule-only EMA cross) + üretim scripti (`scripts/generate_backtest_golden.py`, `tests/test_backtest_golden_regression.py`) mevcut.

### Faz 4 – ML & Walkforward
- MLConfig/LabelConfig kontratı ve label üretimi testli (`tests/test_ml_config_contract_rc1.py`, `tests/test_ml_labels_rc1.py`).
- SklearnLongModel seed reproducibility garanti (`tests/test_ml_model_reproducibility.py`); model registry roundtrip testli (`tests/test_ml_model_registry_rc1.py`).
- Walkforward pipeline proba/signal kolonlarıyla testli (`tests/test_ml_walkforward_rc1.py`); live ML replay entegrasyonu smoke-testli (`tests/integration/test_live_engine_replay_ml_lc1.py`).

### Faz 5 – Live & Risk
- LiveEngine loop kontratı mock execution + replay source ile testli (`tests/test_live_engine_replay_loop.py`); data source factory mapping testli (`tests/test_live_data_source_factory.py`).
- Risk guardrails (max notional/size, max open trades, daily loss) kapsanıyor (`tests/test_risk_engine_backtest_rg1.py`, `tests/test_risk.py`); tail/VaR kalkülasyonu temel seviyede yoklanıyor.
- Kill-switch entegrasyonu çalışır (`tests/test_live_engine_kill_integration.py`); kill tetiklenince engine stop ediyor.

### Faz 6 – Reporting & Research API
- Unified `Report` base kontratı ve generator’lar (`backtest`, `strategy_search`, `live`) tek testte doğrulanıyor (`tests/test_report_contract.py`).
- Portfolio/MonteCarlo/Scenario API’leri FastAPI smoke’larıyla güvence altında (`tests/test_extended_reports_rs7.py`).
- Visualization smoke: Report HTML + chart render dosya üretimi testli (`tests/test_reporting_visualization_smoke.py`).

### Faz 7 – Core Web UI (Research + Live)
- Jest/RTL altyapısı kurulu (next/jest, jest-dom setup); API mock pattern hazır (`frontend/web/__mocks__/api.ts`).
- Smoke testler: backtests list/detail, strategy-search list/detail, live dashboard UI render’ları (Jest) mevcut; CI frontend job’u `npm test` çalıştırıyor.
- Lint/build CI’da koşuyor; Jest testleri backend çağrısına düşmeden mock’larla izole.

### Faz 8 – Market & Microstructure Features
- Market structure feature kontratı (ms_* kolon seti) testli (`tests/test_market_structure_features_contract.py`).
- Microstructure feature kontratı (ms_imbalance, ms_sweep_*, ms_chop, burst, vol_regime, exhaustion, parabolic_trend) testli (`tests/test_microstructure_features_contract.py`).
- Feature builder entegrasyonu: market/micro enable/disable flag’leri kolon var/yok ile doğrulanıyor; enriched feature set ile backtest smoke (`tests/test_feature_builder_market_micro_integration.py`, `tests/test_ms_micro_features_strategy_compat.py`).

## Bilinçli Eksikler & V2+ İşleri
- Portfolio & Monte Carlo Web UI: Backend/API hazır; Next.js tarafında `/portfolio` ve `/montecarlo` sayfaları yok, Jest smoke da yok. V2’de yazılacak.
- Monitoring/observability: Temel heartbeat/loglar dışında Prometheus/Grafana/otel trace entegrasyonu yok; prod-grade gözlemcilik V2+ işi.
- Deployment/infra: Docker/compose iskeleti var; Kubernetes/rolling deploy/blue-green yok. Prod orkestrasyon V2+.
- İleri risk/compliance: Tail-risk/VaR guard sadece temel hesaplama seviyesinde; düzenli compliance raporları yok. Gelişmiş risk/compliance V2+.
- UI test coverage: Core sayfalar (backtests, strategy-search, live) smoke’lu; Portfolio/MC UI yok. E2E (Playwright/Cypress) henüz yazılmadı.

## V1’in Bilinçli Olarak Olmadığı Şeyler
- Tam kurumsal, çok kullanıcılı SaaS değil.
- 7/24, multi-region, fully redundant prod sistemi değil; lab düzeyi.
- Otomatik para basma makinesi değil; backtest sonuçları gelecek getirisi garantilemez.
- Regülasyon/compliance kapsamlı bir ürün değil; kişisel lab.

## Son Not & İlerisi
Bu manifesto V1’in gerçekten bittiğini belgelemek için yazıldı. Bundan sonraki işler V2/RS11+/RS12 kapsamına girer. Yeni feature eklerken faz kontratlarını bozarsan önce testleri/golden’ları güncellemek zorundasın; aksi halde CI kırmızıya düşer. V1, çalışır bir temel sağlar; prod-grade orkestrasyon, ileri risk/compliance ve eksik UI’lar bilinçli olarak sonraya bırakıldı.

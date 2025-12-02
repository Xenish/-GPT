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

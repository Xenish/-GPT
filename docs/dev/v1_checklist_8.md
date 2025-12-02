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

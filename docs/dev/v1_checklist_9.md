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
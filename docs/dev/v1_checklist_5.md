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
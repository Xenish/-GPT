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


Madde 1: Research profilinin içeriği (backtest/mocks)
Durum: ⚠️
Dayanak: config/system.research.yml, config/system.base.yml
Açıklama: Research profili var ve exchange.type: backtest+dry_run: true; plaintext key yok. Ancak YAML içinde tam bir live bloğu ve boş api_key/secret_key alanları duruyor, yani dosya “sadece” research/backtest ayarlarıyla sınırlı değil.
Tasks:

 config/system.research.yml içindeki live (ve gereksiz kill_switch/notifications) bloklarını researche özel minimal haline indir; sadece backtest/research alanlarını bırak.
 Research profilde api_key/secret_key alanlarını tamamen kaldır ya da sadece ${ENV} placeholder’ları ile bırak; boş string tutma.
Madde 2: Live profil ve ENV referansları
Durum: ✅
Dayanak: config/system.live.yml
Açıklama: Live/paper profili mevcut; exchange.type: live, testnet/paper ayarları ve tüm hassas bilgiler ${ENV} placeholder’ları + _env alanlarıyla tutuluyor, düz metin anahtar yok.

Madde 3: Tek entrypoint ile profil yükleme (Literal["research","live"])
Durum: ❌
Dayanak: finantradealgo/system/config_loader.py, finantradealgo/system/__init__.py, tests/unit/test_system_config_loader.py
Açıklama: Yükleyici yalnızca load_system_config(path=None) sağlıyor; profil ismi alan bir load_config(profile: Literal["research","live"]) yok. Yanlış profil string’i için ValueError yerine yalnızca dosya yoksa FileNotFoundError geliyor.
Tasks:

 finantradealgo/system/config_loader.py içine load_config(profile: Literal["research","live"]) ekle; "research"/"live" için ilgili YAML’e yönlendir, başka değerlerde ValueError fırlat.
 finantradealgo/system/__init__.py üzerinden load_config’i export et.
 tests/unit/test_system_config_loader.py içine araştırma ve live profilleri için başarı, geçersiz profil için ValueError testleri ekle.
Madde 4: Pytest’in varsayılanı research config ve profil testleri
Durum: ❌
Dayanak: pytest.ini, config/system.yml, tests/unit/test_system_config_loader.py
Açıklama: Pytest default’ta config/system.yml’i kullanıyor; research profiline pinlenmiş bir fixture yok, dolayısıyla live endpoint’e yönelmemeyi garantileyen bir ayar/test bulunmuyor. Ayrıca load_config("research")/("live") çağrılarının doğru tip döndürdüğünü doğrulayan test yok.
Tasks:

 tests/conftest.py içinde FT_CONFIG_PATH=config/system.research.yml’i pytest başlangıcında set eden bir fixture/auto-use hook ekle; gerektiğinde network çağrılarını stubla.
 Yeni load_config API’si için hem "research" hem "live" profillerini döndüren pozitif test ekle; default fixture’ın research kullandığını assert eden bir test ekle.
 Live/real endpoint çağrılarını engellemek için ilgili client/HTTP katmanını pytest’te monkeypatch et (örn. binance client mock).
Madde 5: Örnek strateji config’leri ve backtest CLI uyumu
Durum: ⚠️
Dayanak: config/ema_example.yml, config/rsi_example.yml, scripts/run_backtest.py
Açıklama: EMA ve RSI örnek config’leri var ama config/strategies/ dizini yok ve dosya yolları checklist’te istenen isimlerle eşleşmiyor. Backtest CLI mevcut system config’i okuyor; bu örnek YAML’larla çalışan bir CLI/API akışı veya test yok.
Tasks:

 config/strategies/ema_example.yml ve config/strategies/rsi_example.yml olarak yeniden konumlandır ya da symlink/kopya ekle; README/doc’ta yeni yolları belirt.
 scripts/run_backtest.py veya yeni bir CLI argümanı ile bu örnek strateji YAML’lerini okuyacak akış ekle (örn. --strategy-config).
 Pytest’te hafif bir entegrasyon testi ekle: örnek strateji config’iyle mock veri üzerinden backtest runner’ın çalıştığını doğrula.

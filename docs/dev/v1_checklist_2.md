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

Madde 1: Global data_root tanımı
Durum: ❌
Dayanak: config/system.yml, config/system.base.yml, finantradealgo/system/config_loader.py (lines 66-106) (DataConfig)
Açıklama: Config’te tekil bir data_root/base_dir tanımı yok; DataConfig’te base_dir alanı var ama hiçbir profil dosyasında set edilmiyor ve OHLCV şablonları doğrudan data/… ile hardcoded geliyor, dolayısıyla kök yol tek noktadan yönetilmiyor.
Tasks:

 config/system.base.yml içine data.base_dir ekle ve diğer path’leri (ohlcv_dir, flow_dir, sentiment_dir) bu kökten türet.
 finantradealgo/system/config_loader.py DataConfig varsayılanlarını base_dir ile bağla (örn. ohlcv_dir = f"{base_dir}/ohlcv").
 Dokümanlarda (örn. docs/core_config_profiles.md) data_root kullanımını örnekle.
Madde 2: OHLCV path şablonu
Durum: ⚠️
Dayanak: config/system.yml:data.ohlcv_path_template, config/system.base.yml:data.ohlcv_path_template, finantradealgo/system/config_loader.py (line 69)
Açıklama: Tekil bir şablon mevcut (data/ohlcv/{symbol}_{timeframe}.csv) ancak data_root placeholder’ı yok ve sembol/timeframe klasör hiyerarşisi yerine düz CSV adı kullanılıyor; checklist’teki {data_root}/ohlcv/{symbol}/{timeframe}.parquet benzeri yapı sağlanmıyor.
Tasks:

 data.ohlcv_path_template’i {data_root}/ohlcv/{symbol}/{timeframe}.parquet (veya CSV) formatına güncelle; data_root referansını kullan.
 İlgili loader’ları bu yeni şablona uyacak şekilde güncelle ve eski yol varsayımlarını temizle.
 Gerekirse Parquet desteği için loader’a read_parquet opsiyonu ekle.
Madde 3: Data loader tek entrypoint ve kalite kontrolleri
Durum: ❌
Dayanak: finantradealgo/data_engine/loader.py (load_ohlcv_csv, load_ohlcv_for_symbol_tf), finantradealgo/validation/ohlcv_validator.py
Açıklama: Sembole/timeframe’e özel load_ohlcv_for_symbol_tf var ama dosya yoksa özel bir DataNotFoundError yok, boş/az veri için fail veya uyarı yok, duplicate/artan timestamp kontrolü yapılmıyor (sadece sort). Validation modülü ayrı fakat loader bunu çağırmıyor.
Tasks:

 finantradealgo/data_engine/loader.py içine class DataNotFoundError(FileNotFoundError): ... ekle ve dosya mevcut değilse bunu raise et.
 load_ohlcv_for_symbol_tf/load_ohlcv_csv içinde boş/çok az satır için warning + ValueError ekle (örn. min_rows parametresi).
 Timestamp için sıkı kontrol ekle: monotonic değilse ve/veya duplicate varsa log + exception (ValueError).
 Opsiyonel: validate_ohlcv entegrasyonu için validate=True bayrağı ekle.
Madde 4: Data loader birim testleri (normal + duplicate hata)
Durum: ⚠️
Dayanak: tests/test_data_loader_lookback.py (normal yükleme/lookback), tests/test_data_validation.py:test_duplicate_timestamps (validation bağımsız)
Açıklama: Loader için normal yükleme ve lookback filtrelemesi test ediliyor ancak dosya yok/boş dosya/duplicate timestamp senaryoları loader seviyesinde test edilmiyor; duplicate testi validation modülüne ait ve loader’a bağlanmıyor.
Tasks:

 Yeni test ekle: küçük sahte CSV ile normal yükleme (satır/kolon assert) + DataNotFoundError senaryosu için dosya yokken çağrı.
 Duplicate timestamp içeren CSV için loader’ın exception/log attığını assert eden test yaz (tests/test_data_loader_quality.py).
 Boş veya çok az satırlı CSV’de warning + fail bekleyen test ekle.

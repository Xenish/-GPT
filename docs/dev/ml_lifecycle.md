# ML Model Lifecycle & Registry

## 1. Registry Yapısı
- Her model run’ı `outputs/ml_models/{model_id}/` altında saklanır:
  - `model.joblib`: sklearn/xgboost modeli.
  - `meta.json`: `ModelMetadata` (feature cols, label config, pipeline version, metrics, feature importance).
  - `metrics.csv`: opsiyonel ek metrikler.
  - `feature_importances.csv`: meta’da da saklanan önemler CSV formatında.
- `registry_index.csv` şeması (`finantradealgo/ml/model_registry.py`):
  - `model_id,symbol,timeframe,model_type,created_at,path,cum_return,sharpe,status`.
  - `ModelRegistryEntry` satırları temsil eder; `ModelMetadata` ayrıntıların disk formatıdır.

## 2. Model Status Alanları
- Varsayılan statüler:
  - `success`: eğitim tamamlandı, artefaktlar mevcut.
  - `failed`: train sırasında hata; entries cleanup sırasında loglanır.
  - Gelecek aşama (M1+): `candidate`, `production`, `archived` gibi ekstra statüler planlanır.
- Status nasıl set edilir?
  - `scripts/run_ml_train_15m.py` içinde train başarıyla bittiğinde `register_model(..., status="success")` çağrılır.
  - Hata alınırsa entry eklenmez; log üzerinden takip edilir (ileride failure entries eklenebilir).

## 3. Train & Promote Akışı
1. **Train** (`scripts/run_ml_train_15m.py`):
   - Parametreler: symbol, timeframe, feature_preset, label config.
   - Çıktı: `model_id = {symbol}_{timeframe}_{model_type}_{timestamp}`. Model + meta + metrics yazılır, registry’e entry eklenir.
2. **Değerlendirme** (`scripts/run_ml_backtest_15m.py`, Strategy Lab, ML Lab):
   - Yeni modelin Sharpe, max drawdown, classification metrikleri incelenir.
3. **Promote** (M1 hedefi):
   - Karşılaştırma: `new_model` vs `current_production`.
   - `promote_model(model_id, stage="production")` helper’ı eklenecek; registry index’te status güncellenir.
4. **Inference**:
   - API ve live sistem `get_latest_model(..., model_type)` veya `get_latest_production_model` helper’ını kullanacak.

## 4. Retrain Stratejisi
- Örnek politika:
  - Haftalık veya iki haftada bir, son X gün verisiyle retrain.
  - Script: `scripts/run_ml_retrain_schedule.py` (plan aşamasında).
  - Parametreler: `lookback_days`, `train_start/end`, `test_start/end`.
- Kayıt & kıyas:
  - `rf_grid_results_*.csv` + Scenario/Strategy Lab ile yeni modellerin performansı geçmişe göre analiz edilir.
  - `registry_index.csv` zaman damgası sayesinde hangi retrain’in üretimde olduğunu izlemek kolaydır.

## 5. Live Entegrasyonu
- Live config (`config/system.yml`):
  - `ml.production_model`: `"auto"` veya belirli `model_id`.
- `MLSignalStrategy` davranışı:
  - `auto` → `ModelRegistry` içinden seçilen symbol/timeframe için en güncel `production` model’i çek.
  - Belirli `model_id` verilirse o model ile inference yapılır (A/B test, rollback vb.).
- LiveEngine snapshot’ında `used_model_id` tutulur; UI bu bilgiyi Live tabında gösterebilir.

## 6. Model Silme / Temizleme
- Eski modeller:
  - `status="archived"` (manuel atama) → live/registry tarafından kullanılmaz fakat disk üzerinde kalır.
  - `scripts/run_clean_registry.py` → artefaktları eksik olan entry’leri düşürür, isteğe bağlı olarak klasörleri de siler.
- Dosya bozulması:
  - `validate_registry_entry(base_dir, entry)` helper’ı model joblib/meta dosyalarını kontrol eder.
  - Eksik dosyalar `get_latest_model` sırasında otomatik atılır; logda uyarı görünür.
- Küçültme (max_models):
  - `register_model(..., max_models_per_symbol_tf=N)` parametresi her sembol/timeframe/model_type set’i için en fazla N modeli saklar; eskileri ve klasörlerini siler.

## 7. ML Lab UI
- `/api/ml/models/{symbol}/{timeframe}` registry’den filtrelenmiş liste döndürür (status=success + artefakt valide).
- `/api/ml/models/{model_id}/importance` meta veya CSV’den importance verisini döndürür; UI bunları normalleştirerek bar chart olarak gösterir.
- Böylece modellerin performansı ve importance değerleri UI’da takip edilebilir; promote/cleanup kararlarına yardımcı olur.

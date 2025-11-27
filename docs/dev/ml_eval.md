# ML Evaluation & Validation Guidelines

## 1. Problem Tanımı
- Öğrenilen sinyal: 15 dakikalık horizon için ileriye dönük getiri `%return_{t→t+15m}` pozitif mi?
- Label şeması `LabelConfig` üzerinden tanımlanır:
  - `horizon` (bar sayısı)
  - `pos_threshold` / `neg_threshold` (fee/slippage dahil)
  - `fee_slippage`
  - `method` (simple, percentile vb.)
- `scripts/run_ml_train_15m.py` ve `finantradealgo/ml/labels.py` bu yapılandırmayı kullanır; `label_long` sadece `pos_threshold` üzerinde kalan getirileri 1 olarak işaretler.

## 2. Train/Test Ayrımı
- **Random split uygunsuz**: zaman serilerinde ileriye bakma (future leakage) riskini artırır; geçmiş veriye göre eğitilip gelecekte test edilmelidir.
- Kullanılan şemalar:
  - **Single split**: `run_ml_backtest_15m.py` ilk %70 train, son %30 test olacak şekilde basit bir kesme uygular.
  - **Rolling / expanding window**: `finantradealgo/ml/walkforward.py` içinde bulunan walkforward döngüsü; her blokta belirli bir train periyodu + onu takip eden test periyodu vardır.
- Kaynak: `walkforward.run_walkforward()` fonksiyonu train->fit, test->predict akışını izole şekilde çalıştırır.

## 3. Cross-Validation Stratejisi
- Zaman serilerinde klasik k-fold shuffle kullanmak leakage’a neden olur; fold’lar arasında geleceğe ait barlar train set’e sızabilir.
- Yöntemler:
  - Walkforward motoru ile dilimlenmiş evaluation (her blok bir “fold” gibi kullanılabilir).
  - Hyperparam grid search için: yalnızca train set içinde `StratifiedKFold` kullanılır (`run_rf_grid_search`). Burada veriler zaten train periyoduna ait olduğundan leakage oluşmaz.
- Örnek: `scripts/run_ml_rf_grid_15m.py` -> features + labels hazırlandıktan sonra `run_rf_grid_search(X, y, param_grid)` çağrısı yapar.

## 4. Metrikler
- **Classification**: accuracy, precision, recall, f1, roc_auc. `SklearnLongModel.evaluate` bu metrikleri üretir.
- **Trading**: Sharpe, max drawdown, final equity, trade_count, win_rate. `finantradealgo/backtester/report.py` + `metrics.py` tarafından hesaplanır.
- Nerede raporlanır?
  - `scripts/run_ml_train_15m.py` → train set boyutu ve sınıflandırma metrikleri.
  - `scripts/run_ml_backtest_15m.py` → classification + equity + trade istatistiklerini yazdırır.
  - Strategy/ML Lab UI → `/api/backtests/run` ve `/api/scenarios/*` çıktıları.

## 5. Leakage’ten Kaçınma
- **Zaman bilgisi**: Feature pipeline’da geleceğe bakmak yasaktır (örneğin `shift(-1)` kullanımlarını kontrol edin).
- **Rolling normalization**: Eğer scaler/normalizer kullanılacaksa, train fit’i yalnızca train verisi üzerinde yapılır. Live/predict aşamasında sadece `transform`.
- **Walkforward flow**:
  1. Train periyodu: feature pipeline fit (gerekliyse), model fit.
  2. Test periyodu: sadece transform + predict.
  3. Sonraki blokta periyodlar kaydırılır; geçmişte öğrenilen parametreler yeni blok için “tekrar” fit edilir.
- `run_ml_rf_from_pipeline_15m.py` gibi scriptler bu akışı örnekler; pipeline meta (`feature_cols`, `pipeline_version`) kaydedilerek inference sırasında tutarlılık sağlanır.

## 6. Önerilen Workflow Örneği
1. `python scripts/run_build_features_15m.py` → Güncel data + feature pipeline CSV üret.
2. `python scripts/run_ml_train_15m.py` → Model eğit, registry’e kaydet (meta + importance + metrics).
3. `python scripts/run_ml_backtest_15m.py` → Eğitilen model veya saved model ile backtest çalıştır, Sharpe vs. incele.
4. `python scripts/run_ml_rf_grid_15m.py` → Hyperparam grid search; sonuçları `outputs/ml_models/rf_grid_results_*.csv` olarak değerlendir.
5. UI (Strategy Lab / ML Lab) → `/api/scenarios/run` + `/api/ml/models/*` sonuçlarını görselleştir, hangi model/param kombinasyonunun daha iyi olduğunu takip et.

## 7. Ek Uyarılar
- Tüm ML scriptlerinin log’larını (train/test boyutu, pipeline versiyonu, model_id) saklayın; LiveEngine farklı pipeline versiyonu bulursa hata/log üretir.
- Registry’den model yüklerken (`load_model_by_id`) `feature_cols` ile feature frame’in kolonlarını bire bir karşılaştırıyoruz; mismatch → ValueError (kullanıcı yeniden train etmeli).

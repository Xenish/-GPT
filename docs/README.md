# FinanTradeAlgo Dokümantasyon

FinanTradeAlgo kripto vadeli işlemler için geliştirilmiş profesyonel seviyede bir kantitatif ticaret araştırma ve backtesting platformudur.

## 🎯 Hızlı Başlangıç

İlk kez mi kullanıyorsunuz? Buradan başlayın:

1. **[Hızlı Başlangıç](kullanici/01-hizli-baslangic.md)** - 15 dakikada sistemi çalıştırın
2. **[Kurulum Detayları](kullanici/02-kurulum-detay.md)** - Detaylı kurulum kılavuzu
3. **[Temel Kavramlar](kullanici/03-temel-kavramlar.md)** - Trading ve sistem terimleri

## 📚 Dokümantasyon Kategorileri

### 👤 Kullanıcı Kılavuzu

Sistemi nasıl kullanırsınız - başlangıçtan canlı işleme:

- [01. Hızlı Başlangıç](kullanici/01-hizli-baslangic.md) - İlk adımlar
- [02. Kurulum Detayları](kullanici/02-kurulum-detay.md) - Sistem kurulumu
- [03. Temel Kavramlar](kullanici/03-temel-kavramlar.md) - Terminoloji ve kavramlar
- [04. Veri Hazırlama](kullanici/04-veri-hazirlama.md) - OHLCV ve external data
- [05. Feature Pipeline](kullanici/05-feature-pipeline.md) - Özellik mühendisliği
- [06. Backtest Çalıştırma](kullanici/06-backtest-calistirma.md) - Backtesting rehberi
- [07. Strateji Seçimi](kullanici/07-strateji-secimi.md) - Strateji türleri
- [08. Risk Yönetimi](kullanici/08-risk-yonetimi.md) - Risk parametreleri
- [09. Portfolio Backtest](kullanici/09-portfolio-backtest.md) - Çoklu sembol
- [10. ML Workflow](kullanici/10-ml-workflow.md) - Makine öğrenimi
- [11. Canlı Ticaret](kullanici/11-canli-ticaret.md) - Paper ve live trading
- [12. API Kullanımı](kullanici/12-api-kullanimi.md) - REST API
- [13. UI Kullanımı](kullanici/13-ui-kullanimi.md) - Frontend arayüzü
- [14. Sorun Giderme](kullanici/14-sorun-giderme.md) - Yaygın hatalar ve çözümler

### 🔧 Strateji Geliştirici Kılavuzu

Kendi stratejilerinizi ve özelliklerinizi nasıl geliştirirsiniz:

- [01. Yeni Strateji Ekleme](strateji-gelistirici/01-yeni-strateji-ekleme.md) - BaseStrategy kullanımı
- [02. Feature Oluşturma](strateji-gelistirici/02-feature-olusturma.md) - Özel özellikler
- [03. Market Structure Detay](strateji-gelistirici/03-market-structure-detay.md) - Piyasa yapısı
- [04. Microstructure Detay](strateji-gelistirici/04-microstructure-detay.md) - Mikro yapı sinyalleri
- [05. ML Modeli Geliştirme](strateji-gelistirici/05-ml-modeli-gelistirme.md) - Özel ML modelleri
- [06. Scenario Grid](strateji-gelistirici/06-scenario-grid.md) - Parametre optimizasyonu
- [07. Test Yazma](strateji-gelistirici/07-test-yazma.md) - Test stratejileri
- [08. En İyi Uygulamalar](strateji-gelistirici/08-en-iyi-uygulamalar.md) - Best practices

### ⚙️ Konfigürasyon Referansı

Tüm sistem parametreleri:

- [System.yml Referansı](konfigürasyon/system-yml-referansi.md) - Ana konfigürasyon dosyası
- [Strateji Konfigürasyonu](konfigürasyon/strateji-konfig.md) - Strateji parametreleri
- [Risk Konfigürasyonu](konfigürasyon/risk-konfig.md) - Risk yönetimi ayarları
- [ML Konfigürasyonu](konfigürasyon/ml-konfig.md) - Makine öğrenimi ayarları
- [Live Konfigürasyonu](konfigürasyon/live-konfig.md) - Canlı ticaret ayarları

### 📖 Örnekler

Adım adım uygulamalı örnekler:

- [Örnek 1: Basit Backtest](ornekler/ornek-1-basit-backtest.md) - İlk backtest
- [Örnek 2: ML Workflow](ornekler/ornek-2-ml-workflow.md) - End-to-end ML
- [Örnek 3: Portfolio](ornekler/ornek-3-portfolio.md) - Çoklu varlık stratejisi
- [Örnek 4: Özel Strateji](ornekler/ornek-4-özel-strateji.md) - Sıfırdan strateji geliştirme
- [Örnek 5: Canlı Deploy](ornekler/ornek-5-canlı-deploy.md) - Production deployment

### 📓 Jupyter Notebooks

İnteraktif öğretim materyalleri:

- [01. Giriş ve Veri Yükleme](notebooks/01-giris-ve-veri-yukleme.ipynb)
- [02. Feature Keşfetme](notebooks/02-feature-kesfetme.ipynb)
- [03. Market Structure Örnekleri](notebooks/03-market-structure-ornekleri.ipynb)
- [04. Microstructure Sinyalleri](notebooks/04-microstructure-sinyalleri.ipynb)
- [05. Strateji Geliştirme](notebooks/05-strateji-gelistirme.ipynb)
- [06. ML Modeli Eğitme](notebooks/06-ml-modeli-egitme.ipynb)
- [07. Backtest Analizi](notebooks/07-backtest-analizi.ipynb)

### 🌐 API Dokümantasyonu

REST API kullanımı:

- [Endpoints](api/endpoints.md) - Tüm API endpoint'leri
- [Veri Modelleri](api/veri-modelleri.md) - Request/Response şemaları
- [Örnekler](api/ornekler.md) - API kullanım örnekleri

### 📜 Script Kılavuzu

Mevcut script'lerin kullanımı:

- [Script Kılavuzu](scripts/script-kilavuzu.md) - 48 script'in detaylı açıklaması

### 🏗️ Mimari

Sistem tasarımı ve mimarisi:

- [Genel Bakış](mimari/genel-bakis.md) - Sistem mimarisi
- [Veri Akışı](mimari/veri-akisi.md) - Data flow diyagramları
- [Modül Referansı](mimari/modul-referansi.md) - Her modülün amacı
- [API Referansı](mimari/api-referansi.md) - API mimarisi

### 💻 Geliştirici Dokümantasyonu

Teknik derinlemesine bilgiler:

**Mevcut:**
- [Market Structure](dev/market_structure.md) - Piyasa yapısı modülü
- [Microstructure](dev/microstructure.md) - Mikro yapı modülü
- [ML Lifecycle](dev/ml_lifecycle.md) - ML yaşam döngüsü
- [Strategies](dev/strategies.md) - Strateji türleri
- [Live Trading](dev/live.md) - Canlı ticaret
- [Testing](dev/testing.md) - Test sistemi

**Gelecek:**
- Feature Pipeline Detay - Feature mühendisliği derinlemesine
- Backtester Engine - Backtesting motoru
- Risk Engine - Risk yönetimi motoru
- Execution Clients - Emir yürütme
- Data Engine - Veri kaynakları

### 📖 Yardımcı Kaynaklar

- [Terimler Sözlüğü](sozluk.md) - Trading ve sistem terimleri
- [SSS (Sıkça Sorulan Sorular)](sss.md) - Yaygın sorular ve cevaplar

---

## 🎓 Öğrenme Yolları

### Yeni Kullanıcı
Hiç deneyiminiz yok mu? Bu sırayı takip edin:

1. [Hızlı Başlangıç](kullanici/01-hizli-baslangic.md) ⭐
2. [Kurulum Detayları](kullanici/02-kurulum-detay.md)
3. [Temel Kavramlar](kullanici/03-temel-kavramlar.md)
4. [Backtest Çalıştırma](kullanici/06-backtest-calistirma.md)
5. [Örnek 1: Basit Backtest](ornekler/ornek-1-basit-backtest.md)

### Strateji Geliştirmek İstiyorum
Kendi stratejinizi mi yazmak istiyorsunuz?

1. [Yeni Strateji Ekleme](strateji-gelistirici/01-yeni-strateji-ekleme.md) ⭐
2. [Feature Oluşturma](strateji-gelistirici/02-feature-olusturma.md)
3. [Market Structure Detay](strateji-gelistirici/03-market-structure-detay.md)
4. [Örnek 4: Özel Strateji](ornekler/ornek-4-özel-strateji.md)
5. [Test Yazma](strateji-gelistirici/07-test-yazma.md)

### ML ile Çalışmak İstiyorum
Makine öğrenimi stratejileri geliştirmek için:

1. [ML Workflow](kullanici/10-ml-workflow.md) ⭐
2. [Örnek 2: ML Workflow](ornekler/ornek-2-ml-workflow.md)
3. [ML Modeli Geliştirme](strateji-gelistirici/05-ml-modeli-gelistirme.md)
4. [ML Konfigürasyonu](konfigürasyon/ml-konfig.md)
5. [Notebook: ML Modeli Eğitme](notebooks/06-ml-modeli-egitme.ipynb)

### Canlı İşlem Yapmak İstiyorum
Production'a geçmek için:

1. [Canlı Ticaret](kullanici/11-canli-ticaret.md) ⭐
2. [Risk Yönetimi](kullanici/08-risk-yonetimi.md)
3. [Örnek 5: Canlı Deploy](ornekler/ornek-5-canlı-deploy.md)
4. [Live Konfigürasyonu](konfigürasyon/live-konfig.md)
5. [Sorun Giderme](kullanici/14-sorun-giderme.md)

### API Kullanacağım
Sistemi API üzerinden kullanmak için:

1. [API Kullanımı](kullanici/12-api-kullanimi.md) ⭐
2. [API Endpoints](api/endpoints.md)
3. [API Örnekleri](api/ornekler.md)
4. [API Referansı](mimari/api-referansi.md)

---

## 🔗 Dışarıdan Linkler

- **Ana Proje:** [README.md](../README.md)
- **Konfigürasyon:** [config/system.yml](../config/system.yml)
- **GitHub Issues:** Sorun bildirmek için
- **Discussions:** Topluluk tartışmaları için

---

## 📝 Katkıda Bulunma

Bu dokümantasyonu geliştirmek ister misiniz? Katkılarınızı bekliyoruz!

- Hata bulduysanız issue açın
- Eksik gördüğünüz konu varsa önerin
- Pull request gönderin

---

**Son Güncelleme:** Kasım 2025
**Versiyon:** 1.0
**Dil:** Türkçe

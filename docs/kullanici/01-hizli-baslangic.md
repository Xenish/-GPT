# Hızlı Başlangıç

> **Hedef Kitle:** Yeni kullanıcılar
> **Süre:** ~15 dakika
> **Önkoşul:** Python 3.10+ kurulu olmalı

Bu kılavuz, FinanTradeAlgo'yu 15 dakikada çalışır hale getirmenizi sağlar. İlk backtest'inizi çalıştıracak ve sonuçları görüntüleyeceksiniz.

---

## İçindekiler
- [Adım 1: Projeyi İndirin](#adım-1-projeyi-indirin)
- [Adım 2: Sanal Ortam Oluşturun](#adım-2-sanal-ortam-oluşturun)
- [Adım 3: Bağımlılıkları Yükleyin](#adım-3-bağımlılıkları-yükleyin)
- [Adım 4: İlk Backtest'inizi Çalıştırın](#adım-4-ilk-backtestinizi-çalıştırın)
- [Adım 5: Sonuçları İnceleyin](#adım-5-sonuçları-inceleyin)
- [Sonraki Adımlar](#sonraki-adımlar)
- [Sorun mu Yaşıyorsunuz?](#sorun-mu-yaşıyorsunuz)

---

## Adım 1: Projeyi İndirin

Projeyi klonlayın veya ZIP olarak indirin:

```bash
git clone https://github.com/<you>/TradeProject.git
cd TradeProject
```

**Not:** Eğer git yoksa, projeyi ZIP olarak indirip çıkartabilirsiniz.

---

## Adım 2: Sanal Ortam Oluşturun

Python sanal ortamı oluşturun (önerilir):

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows (CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Başarılı olduğunuzda:**
Terminal'inizde `(.venv)` ön eki görünecektir.

---

## Adım 3: Bağımlılıkları Yükleyin

Gerekli Python paketlerini yükleyin:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Beklenen süre:** 2-3 dakika

**Sorun yaşarsanız:**
- Python sürümünüzü kontrol edin: `python --version` (3.10+ olmalı)
- [Sorun Giderme](14-sorun-giderme.md#bağımlılık-yükleme-hataları) sayfasına bakın

---

## Adım 4: İlk Backtest'inizi Çalıştırın

### Seçenek A: Hazır Script ile (Önerilen)

En basit yol, hazır bir script kullanmaktır:

```bash
python scripts/run_backtest.py
```

Bu script:
1. `data/ohlcv/` klasöründen OHLCV verisi yükler
2. Temel feature'ları oluşturur (returns, volume, vb.)
3. Basit bir EMA cross stratejisi çalıştırır
4. Backtest sonuçlarını ekrana yazdırır

**Beklenen çıktı:**
```
=== EMA Cross Backtest Report ===
Equity:
  Initial cash : 10000.0
  Final equity : 11234.56
  Total return : 12.35%
  Max drawdown : -5.67%
  Sharpe ratio : 1.89
...
```

### Seçenek B: CLI ile

Eğer CLI kullanmak isterseniz, önce kurulumu yapın:

```bash
pip install -e .
```

Sonra backtest çalıştırın:

```bash
finantrade backtest --strategy rule --symbol AIAUSDT --tf 15m
```

**Parametreler:**
- `--strategy`: Strateji türü (`rule`, `ml`, `trend_continuation`, vb.)
- `--symbol`: Sembol adı (örn: AIAUSDT, BTCUSDT)
- `--tf`: Timeframe (örn: 15m, 1h)

---

## Adım 5: Sonuçları İnceleyin

Backtest tamamlandığında, terminalde şu bilgileri göreceksiniz:

### Equity Metrikleri
- **Initial cash:** Başlangıç sermayesi (varsayılan: 10,000 USDT)
- **Final equity:** Son sermaye (kâr/zarar sonrası)
- **Total return:** Toplam getiri (%)
- **Max drawdown:** Maksimum düşüş (%)
- **Sharpe ratio:** Risk-ayarlı getiri

### Trade İstatistikleri
- **Total trades:** Toplam işlem sayısı
- **Win rate:** Kazanan işlem oranı (%)
- **Avg profit:** Ortalama kâr
- **Avg loss:** Ortalama zarar
- **Profit factor:** Kazanç / zarar oranı

### Örnek Çıktı
```
=== EMA Cross Backtest Report ===
Equity:
  Initial cash : 10000.0
  Final equity : 11234.56
  Total return : 12.35%
  Max drawdown : -5.67%
  Sharpe ratio : 1.89

Trades:
  Total trades : 45
  Win rate     : 55.56%
  Avg profit   : +234.50 USDT
  Avg loss     : -156.30 USDT
  Profit factor: 1.85
```

**Bu ne anlama geliyor?**
- %12.35 getiri elde ettiniz
- Maksimum %5.67 düşüş yaşadınız
- 45 işlemden %55.56'sı kazançlı
- Sharpe ratio 1.89 (1'den büyükse iyi sayılır)

---

## Sonraki Adımlar

Tebrikler! İlk backtest'inizi çalıştırdınız. 🎉

Şimdi ne yapalım?

### Yeni Kullanıcılar İçin
1. **[Temel Kavramlar](03-temel-kavramlar.md)** - Trading terimlerini öğrenin
2. **[Kurulum Detayları](02-kurulum-detay.md)** - Daha fazla konfigürasyon seçeneği
3. **[Backtest Çalıştırma](06-backtest-calistirma.md)** - Detaylı backtesting rehberi

### Daha İleri Seviye
1. **[Veri Hazırlama](04-veri-hazirlama.md)** - Kendi verilerinizi kullanın
2. **[Feature Pipeline](05-feature-pipeline.md)** - Özellik mühendisliği
3. **[Strateji Seçimi](07-strateji-secimi.md)** - Farklı stratejiler deneyin
4. **[ML Workflow](10-ml-workflow.md)** - Makine öğrenimi ile strateji geliştirin

### Pratik Örnekler
1. **[Örnek 1: Basit Backtest](../ornekler/ornek-1-basit-backtest.md)** - Adım adım detaylı örnek
2. **[Örnek 2: ML Workflow](../ornekler/ornek-2-ml-workflow.md)** - End-to-end ML
3. **[Örnek 4: Özel Strateji](../ornekler/ornek-4-özel-strateji.md)** - Kendi stratejinizi yazın

### Frontend ile Görselleştirme
Backtest sonuçlarını web arayüzünde görmek ister misiniz?

1. **API'yi başlatın:**
   ```bash
   python scripts/run_api.py
   ```

2. **Frontend'i başlatın:**
   ```bash
   cd frontend/web
   npm install
   npm run dev
   ```

3. **Tarayıcınızda açın:** http://localhost:3000

Detaylar için: [UI Kullanımı](13-ui-kullanimi.md)

---

## Sorun mu Yaşıyorsunuz?

### Yaygın Hatalar

**1. "ModuleNotFoundError: No module named 'finantradealgo'"**

**Çözüm:**
```bash
# Sanal ortamı aktif ettiğinizden emin olun
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows

# Bağımlılıkları tekrar yükleyin
pip install -r requirements.txt
```

**2. "FileNotFoundError: data/ohlcv/BTCUSDT_15m.csv"**

**Çözüm:**
Veri dosyası yok. İki seçenek:

a) Örnek veriyi indirin (eğer varsa):
```bash
# Örnek veri indirme scripti
python scripts/download_sample_data.py
```

b) Kendi verinizi hazırlayın:
```bash
python scripts/fetch_binance_data.py --symbol BTCUSDT --tf 15m
```

Detaylar: [Veri Hazırlama](04-veri-hazirlama.md)

**3. "Python version too old"**

**Çözüm:**
Python 3.10+ gerekli. Güncelleyin:
- [Python.org](https://www.python.org/downloads/) adresinden indirin
- Linux: `sudo apt-get install python3.10` (Ubuntu/Debian)
- macOS: `brew install python@3.10`

### Daha Fazla Yardım

- **[Sorun Giderme](14-sorun-giderme.md)** - Detaylı troubleshooting
- **[Terimler Sözlüğü](../sozluk.md)** - Terimlerin açıklamaları
- **[SSS](../sss.md)** - Sıkça sorulan sorular
- **GitHub Issues** - Sorun bildirin

---

## Özet

Bu kılavuzda:
- ✅ Projeyi indirdiniz
- ✅ Sanal ortam oluşturdunuz
- ✅ Bağımlılıkları yüklediniz
- ✅ İlk backtest'inizi çalıştırdınız
- ✅ Sonuçları yorumladınız

**Sonraki adım:** [Temel Kavramlar](03-temel-kavramlar.md) - Trading ve sistem terimlerini öğrenin

---

**İlgili Dokümantasyon:**
- [Kurulum Detayları](02-kurulum-detay.md)
- [Backtest Çalıştırma](06-backtest-calistirma.md)
- [Örnek 1: Basit Backtest](../ornekler/ornek-1-basit-backtest.md)
- [Sorun Giderme](14-sorun-giderme.md)

**Kaynak Dosyalar:**
- [scripts/run_backtest.py](../../scripts/run_backtest.py)
- [config/system.research.yml](../../config/system.research.yml)

---

**Geri:** [Dokümantasyon Ana Sayfa](../README.md)
**İleri:** [Kurulum Detayları](02-kurulum-detay.md)

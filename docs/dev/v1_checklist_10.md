## 📚 Bölüm 10 – Kullanım Senaryoları & Dokümantasyon (**Zorunlu**)

**Amaç:** Sen 3 ay projeye ara verdiğinde bile geri dönüp “Bu neydi?” demeden kullanabilesin.

- [ ]  README veya `docs/` altında:
    - [ ]  “Kişisel Quant Lab v1 – Kullanım Akışı” başlıklı bir bölüm var.
- [ ]  En az 3 net kullanım senaryosu dokümante:
    1. **Basit backtest**:
        - [ ]  Örn: EMA cross stratejisini BTCUSDT 1h için 1 yıl boyunca backtest et → CLI komutları + UI adımları tek tek yazılı.
    2. **Parametre araması**:
        - [ ]  Search config → CLI komutu → beklenen output path → sonuçları nasıl okuyacağın (ör. pandas ile).
    3. **Paper trading başlatma**:
        - [ ]  Live config’in doldurulması (mock/paper), engine’in çalıştırılması, log’ların kontrolü, pozisyonların nasıl görüntüleneceği.
- [ ]  Dokümanda:
    - [ ]  “Risk uyarıları” kısmı var:
        - [ ]  Bu sistemin **beta/v1** olduğu, gerçek sermaye riskine girmeden önce uzun süre paper test yapılması gerektiği açıkça yazıyor.
- [ ]  Strateji geliştirme için mini rehber:
    - [ ]  Yeni bir stratejinin `BaseStrategy`’den türetilerek nasıl ekleneceğini adım adım açıklayan kısa bir bölüm:
        - [ ]  Class nerede tanımlanacak.
        - [ ]  Config’e nasıl eklenecek.
        - [ ]  UI’da seçilebilir hale getirmek için ne yapılacak.

---
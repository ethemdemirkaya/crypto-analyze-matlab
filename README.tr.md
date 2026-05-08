# LSTM ile Kripto Para Fiyat Tahmini (Binance API)

![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-blue?logo=mathworks)
![Lisans](https://img.shields.io/badge/Lisans-MIT-green)
![Derin Öğrenme](https://img.shields.io/badge/Derin%20Öğrenme-LSTM-orange)

> **Binance API'den gerçek zamanlı geçmiş veri çekerek kripto para fiyatlarını tahmin eden MATLAB tabanlı LSTM derin öğrenme uygulaması.**

---

## Özellikler

- Binance REST API üzerinden canlı veri çekimi (API anahtarı gerektirmez)
- 1000'den fazla mum için otomatik sayfalama (pagination)
- CSV önbellek — tekrar çalıştırmalarda gereksiz API çağrısını önler
- Veri sızıntısı koruması: min-max normalizasyonu yalnızca eğitim setinden hesaplanır
- LSTM için kayan pencere (sliding window) sekans oluşturma
- Dropout düzenlileştirmeli iki katmanlı LSTM mimarisi
- Parçalı öğrenme hızı programıyla Adam optimizer
- Early stopping destekli validation split (%10)
- Çok adımlı özyinelemeli tahmin (örn. 30 günlük ufuk)
- Beş değerlendirme metriği: RMSE, MAE, MAPE, R², Yön Doğruluğu
- 300 DPI PNG olarak kaydedilen dört adet yayın kaliteli grafik
- Naive/Drift/Mean baseline karşılaştırması (`runBaseline.m`)
- Çok özellikli (OHLCV) vs tek özellikli (Close) karşılaştırması (`runMultiFeature.m`)
- BTC/ETH/BNB için toplu eğitim ve karşılaştırma tablosu (`runAllCoins.m`)
- Walk-forward çapraz doğrulama (`src/walkForwardValidation.m`)
- Programatik MATLAB GUI uygulaması (`app/CryptoPredictorApp.m`)

---

## Gereksinimler

| Gereksinim | Sürüm |
|------------|-------|
| MATLAB | R2023a veya üzeri |
| Deep Learning Toolbox | zorunlu |
| Statistics & Machine Learning Toolbox | zorunlu |

---

## Kurulum

```bash
git clone https://github.com/ethemdemirkaya/crypto-analyze-matlab.git
cd crypto-analyze-matlab
```

MATLAB'da `src/` klasörünü path'e ekleyin:

```matlab
addpath(genpath('src'))
```

---

## Kullanım

### Hızlı Başlangıç — Tek Coin

```matlab
% main.m içindeki config'i düzenleyip çalıştırın:
main
```

`main.m` içindeki temel parametreler:

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `config.symbol` | `'BTCUSDT'` | İşlem çifti |
| `config.interval` | `'1d'` | Mum aralığı |
| `config.numCandles` | `1500` | Çekilecek geçmiş mum sayısı |
| `config.sequenceLength` | `60` | Girdi pencere uzunluğu (gün) |
| `config.trainRatio` | `0.8` | Eğitim/test bölünme oranı |
| `config.numHiddenUnits` | `100` | LSTM gizli birim sayısı |
| `config.forecastDays` | `30` | Gelecek tahmin ufku |

### Toplu Çalıştırma (BTC + ETH + BNB)

```matlab
runAllCoins
```

### Baseline Karşılaştırması

```matlab
runBaseline
```

LSTM'i Naive, Drift ve Mean baseline yöntemleriyle karşılaştırır.

### Çok Özellikli Deney (OHLCV vs Close)

```matlab
runMultiFeature
```

### Walk-Forward Doğrulama

```matlab
addpath(genpath('src'))
data = fetchBinanceData('BTCUSDT', '1d', 1500);
results = walkForwardValidation(data.Close, 60, 5, 100);
```

### Grafik Arayüz (GUI)

```matlab
addpath(genpath('src'))
CryptoPredictorApp()
```

---

## Proje Yapısı

```
crypto-analyze-matlab/
├── main.m                              # Giriş noktası — tek coin
├── runAllCoins.m                       # Toplu çalıştırıcı — BTC, ETH, BNB
├── runBaseline.m                       # LSTM vs Baseline karşılaştırması
├── runMultiFeature.m                   # Çok özellikli deney
├── src/
│   ├── fetchBinanceData.m              # Binance API veri çekme + CSV önbellek
│   ├── preprocessData.m               # Normalizasyon + sliding window + bölünme
│   ├── preprocessMultiFeature.m       # OHLCV çok özellikli ön işleme
│   ├── buildLSTMModel.m               # LSTM mimarisi
│   ├── trainModel.m                   # Eğitim + validation split + model kaydetme
│   ├── predictFuture.m                # Özyinelemeli çok adımlı tahmin
│   ├── evaluateModel.m                # RMSE, MAE, MAPE, R², Yön Doğruluğu
│   ├── plotResults.m                  # 4 grafik → 300 DPI PNG
│   ├── baselineForecast.m             # Naive/Drift/Mean baseline
│   ├── walkForwardValidation.m        # Walk-forward çapraz doğrulama
│   └── denormalize.m                  # Min-max ters dönüşüm
├── app/
│   └── CryptoPredictorApp.m           # Programatik MATLAB GUI
├── data/                              # CSV önbellek (git'te yok)
├── models/                            # Kaydedilmiş .mat modeller (git'te yok)
├── figures/                           # Çıktı grafikleri (git'te yok)
├── results/                           # Metrik metin dosyaları (git'te yok)
├── docs/
│   ├── report.md                      # Akademik rapor (Türkçe)
│   └── architecture.md               # Mimari kararlar
└── LICENSE
```

---

## Model Mimarisi

```
Girdi (pencere uzunluğu × özellik sayısı)
        │
  ┌─────▼──────┐
  │  LSTM(100) │  ← uzun vadeli bağımlılıkları öğrenir
  └─────┬──────┘
        │ Dropout(0.2)
  ┌─────▼──────┐
  │  LSTM(50)  │  ← yüksek seviyeli örüntü soyutlama
  └─────┬──────┘
        │ Dropout(0.2)
  ┌─────▼──────┐
  │   FC(25)   │
  └─────┬──────┘
  ┌─────▼──────┐
  │   FC(1)    │  ← skalar fiyat tahmini
  └─────┬──────┘
  Regresyon Katmanı (MSE kaybı)
```

**Eğitim:** Adam optimizer, 100 epoch, batch boyutu 32, parçalı öğrenme hızı programı (her 50 epoch'ta ×0.2 düşüş), gradyan kırpma eşiği 1, `Shuffle='never'` (zaman serisi için kritik), validation patience 10 (early stopping).

---

## Üretilen Grafikler

1. **Gerçek vs Tahmin** — test setinde gerçek ve tahmin fiyatlarının üst üste çizgisi
2. **Artık Hata** — zaman içinde tahmin hatası
3. **Dağılım Korelasyonu** — kimlik doğrusu çizgili gerçek - tahmin dağılımı
4. **Gelecek Tahmini** — son bilinen tarihten itibaren 30 günlük ileri tahmin

---

## Sonuçlar (Örnek)

| Coin | RMSE | MAE | MAPE (%) | R² | Yön Doğruluğu (%) |
|------|------|-----|----------|----|-------------------|
| BTCUSDT | — | — | — | — | — |
| ETHUSDT | — | — | — | — | — |
| BNBUSDT | — | — | — | — | — |

*Gerçek değerleri almak için `main.m` veya `runAllCoins.m` çalıştırın.*

---

## Sınırlamalar ve Uyarılar

> **ÖNEMLİ UYARI:** Bu proje **yalnızca akademik ve eğitim amaçlıdır**. Elde edilen tahminler **finansal yatırım kararı vermek için kullanılmamalıdır**. Kripto para piyasaları son derece yüksek risk içermekte olup geçmiş performans gelecekteki sonuçları garanti etmez.

- Tahminler yalnızca geçmiş fiyat örüntülerine dayanır; düzenleyici haberler, borsa arızaları vb. temel olaylar modellenmemiştir.
- Özyinelemeli çok adımlı tahminlerde hata birikir; 7 günü aşan tahminlerin güvenilirliği azalır.
- Model yalnızca eğitim verisi dağılımı içinde iyi sonuç verir; aşırı değer (black swan) olaylarında başarısız olabilir.

---

## Gelecek Geliştirmeler

- Çok özellikli girdi (OHLCV + RSI + MACD + Bollinger)
- Attention mekanizması ve Transformer karşılaştırması
- Validation loss tabanlı early stopping (zaten temel destek var)
- Duygu analizi entegrasyonu (sosyal medya / haber)
- Hyperparameter optimizasyonu (Bayesian / grid search)
- Gerçek zamanlı tahmin paneli

---

## Lisans

MIT — bakınız [LICENSE](LICENSE).

---

## Yazar

**Ethem Demirkaya**  
Yazılım Mühendisliği, 3. Sınıf  
📧 ethemdemirkaya189@gmail.com

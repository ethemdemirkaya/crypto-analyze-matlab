# MATLAB LSTM Kripto Fiyat Tahmin Projesi — Claude için Detaylı Prompt

> Bu dosyayı Claude'a (veya Claude Code'a) verip projeyi baştan sona ürettirebilirsin. Türkçe yazıldı çünkü hem sen hem hocan için anlaşılır olsun.

---

## ROL VE BAĞLAM

Sen kıdemli bir MATLAB ve makine öğrenmesi geliştiricisisin. Bilişim Matematiği dersi için **orta düzey, akademik kalitede ve GitHub'da gösterilebilecek** bir proje hazırlayacaksın. Proje sahibi 3. sınıf yazılım mühendisliği öğrencisi, MATLAB'a temel düzeyde hakim, Python/JS deneyimi var.

**Proje adı:** Cryptocurrency Price Prediction with LSTM (Binance API)

**Amaç:** Binance API'den çekilen geçmiş kripto para fiyat verilerini kullanarak, LSTM (Long Short-Term Memory) tabanlı bir derin öğrenme modeli ile gelecekteki fiyatları tahmin eden bir MATLAB uygulaması geliştirmek.

---

## TEKNİK GEREKSİNİMLER

### Geliştirme Ortamı
- **MATLAB R2023a veya üzeri**
- **Gerekli Toolbox'lar:**
  - Deep Learning Toolbox (LSTM için zorunlu)
  - Statistics and Machine Learning Toolbox (normalizasyon ve metrikler için)
  - (Opsiyonel) MATLAB App Designer (GUI için)

### Veri Kaynağı
- **Binance Public REST API** kullanılacak — API key gerektirmez (geçmiş kline verisi için).
- **Endpoint:** `https://api.binance.com/api/v3/klines`
- **Parametreler:**
  - `symbol`: BTCUSDT (varsayılan), ETHUSDT, BNBUSDT vb. seçilebilir
  - `interval`: 1d (günlük), 4h, 1h vb.
  - `limit`: 1000 (max). Daha fazla veri için `startTime` parametresiyle pagination yapılacak.
- **Veri formatı:** JSON array, her eleman 12 alanlı bir array (open time, open, high, low, close, volume, ...)

### Çıktı Formatı
- Tahmin grafikleri (.png olarak `figures/` klasörüne kaydedilecek)
- Eğitilmiş model (.mat olarak `models/` klasörüne kaydedilecek)
- Performans metrikleri (RMSE, MAE, MAPE) konsola ve `results/metrics.txt` dosyasına yazılacak

---

## PROJE KLASÖR YAPISI

Şu yapıyı oluştur:

```
crypto-lstm-prediction/
├── README.md                      # Proje tanıtımı, kurulum, kullanım
├── LICENSE                        # MIT License
├── .gitignore                     # *.mat, figures/, results/ vb.
├── main.m                         # Ana çalıştırma scripti (entry point)
├── src/
│   ├── fetchBinanceData.m         # Binance API'den veri çeken fonksiyon
│   ├── preprocessData.m           # Normalizasyon, train/test split, sequence oluşturma
│   ├── buildLSTMModel.m           # LSTM mimarisini kuran fonksiyon
│   ├── trainModel.m               # Modeli eğiten fonksiyon
│   ├── predictFuture.m            # Tahmin yapan fonksiyon
│   ├── evaluateModel.m            # RMSE, MAE, MAPE hesaplayan fonksiyon
│   └── plotResults.m              # Grafik çizen fonksiyon
├── app/
│   └── CryptoPredictorApp.mlapp   # App Designer GUI 
├── data/
│   └── .gitkeep                   # Çekilen veriler buraya cache'lenir
├── models/
│   └── .gitkeep                   # Eğitilmiş modeller
├── figures/
│   └── .gitkeep                   # Üretilen grafikler
├── results/
│   └── .gitkeep                   # Metrikler ve raporlar
└── docs/
    ├── report.md                  # Akademik mini rapor (Türkçe)
    └── architecture.md            # Mimari ve karar açıklamaları
```

---

## DETAYLI MODÜL SPESİFİKASYONLARI

### 1. `fetchBinanceData.m`

**İmza:**
```matlab
function data = fetchBinanceData(symbol, interval, numCandles)
```

**Görev:**
- `webread` veya `webwrite` ile Binance kline endpoint'ine istek at
- `numCandles` 1000'den büyükse pagination yap (her seferinde 1000 mum çek, `endTime` ile geri git)
- Dönen JSON'u tablo (`table`) yapısına çevir
- Sütunlar: `OpenTime`, `Open`, `High`, `Low`, `Close`, `Volume`, `CloseTime`
- `OpenTime`'ı `datetime` formatına çevir (Binance milisaniye timestamp döner)
- Sayısal sütunları `double`'a cast et (API string olarak dönüyor)
- Veriyi `data/<symbol>_<interval>.csv` olarak cache'le; bir sonraki çağrıda cache varsa direkt oradan oku (opsiyonel: `forceRefresh` parametresi)

**Hata yönetimi:**
- Network hatasında anlamlı hata mesajı ver
- Geçersiz symbol için Binance'in döndürdüğü hata mesajını yakala

### 2. `preprocessData.m`

**İmza:**
```matlab
function [XTrain, YTrain, XTest, YTest, normParams] = preprocessData(prices, sequenceLength, trainRatio)
```

**Görev:**
- `prices` vektörünü min-max normalize et (0-1 arası), `normParams` struct'ında min/max değerlerini sakla (sonra denormalize için)
- Sliding window ile sequence oluştur: `sequenceLength` (örn. 60) gün veriden 1 sonraki günü tahmin et
- Veriyi `trainRatio` (örn. 0.8) oranında train/test'e böl — **kronolojik sırayı bozma!** (Time series için shuffle yapılmaz)
- Çıktı boyutları:
  - `XTrain`: cell array, her hücre `[1 x sequenceLength]` boyutunda
  - `YTrain`: `[1 x N]` vektör

### 3. `buildLSTMModel.m`

**İmza:**
```matlab
function layers = buildLSTMModel(numFeatures, numHiddenUnits)
```

**Önerilen mimari:**
```matlab
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits/2, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(25)
    fullyConnectedLayer(1)
    regressionLayer
];
```

`numHiddenUnits` varsayılan 100, parametrik olsun.

### 4. `trainModel.m`

**İmza:**
```matlab
function net = trainModel(layers, XTrain, YTrain, options)
```

**Önerilen training options:**
```matlab
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 50, ...
    'LearnRateDropFactor', 0.2, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ...        % Time series için ÖNEMLİ
    'Plots', 'training-progress', ...
    'Verbose', 1);
```

Eğitim sonunda modeli `models/<symbol>_lstm_<timestamp>.mat` olarak kaydet.

### 5. `predictFuture.m`

**İmza:**
```matlab
function predictions = predictFuture(net, lastSequence, numSteps, normParams)
```

**Görev:**
- Verilen son `sequenceLength` günden başlayarak iteratif tahmin yap (multi-step ahead)
- Her tahmini bir sonraki adımın input'una ekle (recursive forecasting)
- Tahminleri denormalize et ve gerçek fiyat ölçeğine çevir

### 6. `evaluateModel.m`

**İmza:**
```matlab
function metrics = evaluateModel(YTrue, YPred)
```

**Hesaplanacak metrikler:**
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **Directional Accuracy** (yön tahmini doğruluğu — yukarı/aşağı yönü ne kadar isabetli)

`metrics` struct olarak dönsün, `results/metrics.txt`'e yazılsın.

### 7. `plotResults.m`

**İmza:**
```matlab
function plotResults(YTrue, YPred, dates, symbol, savePath)
```

**Üretilecek grafikler:**
1. **Actual vs Predicted** — gerçek ve tahmin fiyatları aynı eksende çizgi grafik
2. **Residuals** — hata grafiği (tahmin − gerçek)
3. **Loss Curve** — eğitim sırasında loss'un düşüşü (training history'den)
4. **Forecast** — gelecek N gün tahmini (mevcut veriyi de gösterip ayrı renkle)

Grafikler `figures/` klasörüne `.png` olarak yüksek çözünürlükte (300 DPI) kaydedilsin. Türkçe başlık ve etiketler kullan.

### 8. `main.m`

Ana akış:
```matlab
%% MATLAB LSTM Kripto Fiyat Tahmin — Ana Script
clear; clc; close all;

% 1. Konfigürasyon
config.symbol = 'BTCUSDT';
config.interval = '1d';
config.numCandles = 1500;
config.sequenceLength = 60;
config.trainRatio = 0.8;
config.numHiddenUnits = 100;
config.forecastDays = 30;

% 2. Veri çekme
fprintf('[1/6] Binance''ten veri çekiliyor...\n');
data = fetchBinanceData(config.symbol, config.interval, config.numCandles);

% 3. Önişleme
fprintf('[2/6] Veri önişleniyor...\n');
[XTrain, YTrain, XTest, YTest, normParams] = preprocessData(...
    data.Close, config.sequenceLength, config.trainRatio);

% 4. Model oluşturma ve eğitim
fprintf('[3/6] LSTM modeli kuruluyor ve eğitiliyor...\n');
layers = buildLSTMModel(1, config.numHiddenUnits);
net = trainModel(layers, XTrain, YTrain);

% 5. Test seti üzerinde tahmin
fprintf('[4/6] Test seti tahmin ediliyor...\n');
YPred = predict(net, XTest);
YPredDenorm = denormalize(YPred, normParams);
YTestDenorm = denormalize(YTest, normParams);

% 6. Değerlendirme
fprintf('[5/6] Performans metrikleri hesaplanıyor...\n');
metrics = evaluateModel(YTestDenorm, YPredDenorm);
disp(metrics);

% 7. Görselleştirme
fprintf('[6/6] Grafikler üretiliyor...\n');
plotResults(YTestDenorm, YPredDenorm, data.OpenTime, config.symbol, 'figures/');

% 8. Gelecek tahmini
futurePrices = predictFuture(net, XTest{end}, config.forecastDays, normParams);
fprintf('Önümüzdeki %d gün için tahmin tamamlandı.\n', config.forecastDays);
```

---

## OPSİYONEL: APP DESIGNER GUI

Eğer GUI yapacaksan `CryptoPredictorApp.mlapp` içinde:
- **Coin seçim dropdown'u** (BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT)
- **Interval dropdown'u** (1d, 4h, 1h)
- **Tarih aralığı seçici** (datetime picker)
- **Sequence length slider** (30-120)
- **Hidden units slider** (50-200)
- **Epoch sayısı input'u**
- **"Veriyi Çek" butonu**
- **"Modeli Eğit" butonu**
- **"Tahmin Yap" butonu**
- **Embed edilmiş axes** — grafikleri göstermek için
- **Metrik tablosu** — RMSE, MAE, MAPE değerlerini göstermek için
- **Status bar** — işlem durumu için

---

## README.md İÇERİĞİ

Aşağıdaki bölümleri içermeli:

1. **Proje Başlığı + Tek Cümlelik Açıklama** + rozet (badges: MATLAB version, License)
2. **Demo GIF veya Ekran Görüntüsü** — tahmin grafiği örneği
3. **Özellikler** (madde madde)
4. **Gereksinimler** (MATLAB sürümü, toolbox'lar)
5. **Kurulum** (clone, path ekleme)
6. **Kullanım** — `main.m` çalıştırma örneği, parametre açıklamaları
7. **Proje Yapısı** — klasör ağacı
8. **Model Mimarisi** — LSTM katmanlarının açıklaması, diyagram
9. **Sonuçlar** — örnek metrikler tablosu, grafikler
10. **Sınırlamalar / Uyarılar** — "Bu proje eğitim amaçlıdır, gerçek yatırım kararı için kullanılmaz" disclaimer'ı
11. **Gelecek Geliştirmeler** (multi-feature input, attention mekanizması, transformer karşılaştırması)
12. **Lisans** (MIT)
13. **Yazar / İletişim**

İngilizce ve Türkçe iki versiyon olabilir (`README.md` İngilizce, `README.tr.md` Türkçe).

---

## AKADEMİK RAPOR (`docs/report.md`)

Bu hocaya sunum için kritik. Şu bölümleri içersin:

1. **Özet** (Abstract) — 150 kelime
2. **Giriş** — problem tanımı, motivasyon, kripto piyasalarının zorluğu
3. **Literatür Taraması** — LSTM, finansal zaman serisi tahmini üzerine 5-10 referans
4. **Yöntem**
   - Veri toplama (Binance API)
   - Önişleme (normalizasyon, sliding window)
   - LSTM mimarisi (matematiksel formüllerle: forget gate, input gate, output gate denklemleri)
   - Eğitim parametreleri
5. **Deneysel Kurulum** — kullanılan veri, train/test oranı, hyperparameter'lar
6. **Sonuçlar** — metrikler tablosu, grafikler, yorumlar
7. **Tartışma** — modelin başarılı/başarısız olduğu durumlar, overfitting riski, veri sızıntısı kontrolü
8. **Sonuç ve Gelecek Çalışmalar**
9. **Referanslar** — IEEE veya APA formatında

---

## KOD KALİTESİ KURALLARI

- **Tüm fonksiyonların docstring'i (header comment) olsun:**
  ```matlab
  function [out1, out2] = myFunc(in1, in2)
  % MYFUNC - Kısa açıklama
  %
  % Girdi:
  %   in1 - açıklama
  %   in2 - açıklama
  %
  % Çıktı:
  %   out1 - açıklama
  %
  % Örnek:
  %   [a, b] = myFunc(x, y);
  ```
- **Magic number kullanma** — sabitleri başta tanımla
- **Try-catch ile hata yakala** — özellikle network ve dosya I/O işlemlerinde
- **Türkçe yorum, İngilizce değişken adı** kullan (akademik convention)
- **`fprintf` ile bilgilendirici loglar** at (her aşamada)
- **Reproducibility için `rng(42)` ile seed sabitle**

---

## .gitignore İÇERİĞİ

```
# MATLAB
*.asv
*.m~
*.mat
*.mex*
*.slxc

# Proje çıktıları
data/*.csv
models/*.mat
figures/*.png
results/*.txt

# Sistem
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/

# .gitkeep dosyalarını koru
!**/.gitkeep
```

---

## TESLİMAT KONTROL LİSTESİ

Proje tamamlandığında şunlar bitmiş olmalı:

- [ ] Tüm `src/` fonksiyonları çalışıyor ve dokümantasyonlu
- [ ] `main.m` baştan sona hatasız çalışıyor
- [ ] En az 3 farklı coin (BTC, ETH, BNB) için test edildi
- [ ] RMSE, MAE, MAPE değerleri hesaplandı ve makul (MAPE < %10 hedef)
- [ ] 4 farklı grafik üretildi ve `figures/` içinde
- [ ] README.md tamamlandı, ekran görüntüleri eklendi
- [ ] Akademik rapor (`docs/report.md`) yazıldı
- [ ] `.gitignore` doğru kuruldu
- [ ] LICENSE dosyası var (MIT)
- [ ] (Opsiyonel) GUI çalışıyor
- [ ] GitHub'a push'landı, repo public

---

## EK TAVSİYELER

1. **Veri sızıntısı (data leakage) kontrolü:** Normalizasyon parametrelerini SADECE training set üzerinden hesapla, test set'e aynı parametreleri uygula. Aksi halde model "hile yapmış" olur.

2. **Overfitting kontrolü:** Validation set ekle (train'in %20'si), early stopping kullan.

3. **Baseline karşılaştırması:** LSTM'in ne kadar iyi olduğunu göstermek için basit bir baseline (örn. son fiyatı tahmin et = naive forecast) ile karşılaştır. Bu rapor için çok güçlü bir argüman.

4. **Multi-feature deneyi (bonus):** Sadece Close fiyatı yerine Open, High, Low, Volume da input olarak ver. Performans karşılaştırması yap.

5. **Walk-forward validation:** Daha sağlam değerlendirme için tek bir train/test split yerine kayan pencere ile çoklu test yap.

6. **Disclaimer kritik:** README ve rapora "Bu proje akademik amaçlıdır, finansal tavsiye değildir" notu ekle. Hem etik hem de yasal koruma sağlar.

---

## BAŞLAMAK İÇİN

Claude'a bu prompt'u verdikten sonra şu sırayla isteyebilirsin:

1. "Önce klasör yapısını ve boş dosyaları oluştur"
2. "Şimdi `fetchBinanceData.m`'i yaz ve test et"
3. "`preprocessData.m`'i yaz"
4. "Modeli kur ve eğit"
5. "Grafikleri üret"
6. "README'yi yaz"
7. "Akademik raporu yaz"

Her adımda Claude'dan örnek çıktı / test sonucu göstermesini iste.

İyi şanslar kanka, GitHub'da yıldızlanacak bir proje çıkacak buradan! 🚀
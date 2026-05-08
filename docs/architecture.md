# Mimari ve Tasarım Kararları

## Genel Akış

```
Binance API
    │
    ▼
fetchBinanceData.m  ──► data/<symbol>_<interval>.csv (cache)
    │
    ▼
preprocessData.m
  - Min-Max normalizasyon (SADECE train seti üzerinden)
  - Sliding window sequence oluşturma
  - Kronolojik train/test bölünmesi (%80/%20)
    │
    ▼
buildLSTMModel.m
  Input(1) → LSTM(100) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → FC(25) → FC(1)
    │
    ▼
trainModel.m (Adam optimizer, 100 epoch)
    │
    ┌─────────────────────┐
    ▼                     ▼
predict(net, XTest)   predictFuture.m
    │                 (recursive forecasting)
    ▼
evaluateModel.m     plotResults.m
(RMSE/MAE/MAPE/R²)  (4 grafik, 300 DPI)
```

## LSTM Katman Detayları

### Neden 2 LSTM Katmanı?
Tek katman basit zaman bağımlılıklarını öğrenir. İkinci LSTM, birincinin öğrendiği yüksek seviyeli örüntüleri işler. Literatürde finansal zaman serisi için 2-3 katman optimum olarak raporlanmıştır.

### Neden Dropout(0.2)?
LSTM'lerin overfitting'e yatkınlığını azaltmak için. %20 dropout oranı finansal veri için yaygın kullanılan değerdir.

### Neden Adam Optimizer?
- Adaptive learning rate: farklı parametreler için farklı hız
- Momentum: gradyan salınımlarını yumuşatır
- Finansal veri gibi gürültülü ortamlarda SGD'ye göre daha hızlı yakınsama

### Neden Shuffle='never'?
Zaman serisi verisinde veri noktaları bağımlıdır. Shuffle yapılırsa gelecek veri geçmiş veri eğitiminde kullanılmış olur (data leakage). Bu modeli geçersiz kılar.

## Normalizasyon Stratejisi

**Kritik:** Normalizasyon parametreleri (min, max) yalnızca eğitim setinden hesaplanır. Test setine aynı parametreler uygulanır. Aksi halde test verisi eğitime sızar ve gerçek dışı başarım metrikleri elde edilir.

## Veri Sızıntısı Kontrolü

1. ✅ Normalizasyon: Sadece train set'ten min/max
2. ✅ Sequence oluşturma: test örnekleri train örneklerini içermiyor
3. ✅ Shuffle=never: kronolojik sıra korunuyor
4. ✅ Gelecek tahmini: sadece geçmiş veri kullanılıyor

## Recursive Forecasting

`predictFuture.m` sliding window yaklaşımı kullanır:
```
t=1: [d60, d61, ..., d119]    → ŷ120
t=2: [d61, d62, ..., ŷ120]   → ŷ121
t=3: [d62, d63, ..., ŷ121]   → ŷ122
...
```
Hata birikimi: Çok adım ötesi tahminlerde her adımda tahmin hatası girdiye eklenir. Bu nedenle 30 günü aşan tahminler güvenilirliğini yitirir.

## Cache Mekanizması

İlk çalıştırmada API'den çekilen veri `data/<symbol>_<interval>.csv`'e yazılır. Sonraki çalıştırmalarda dosya varsa API'ye gidilmez. `forceRefresh=true` ile zorla yenileme yapılır.

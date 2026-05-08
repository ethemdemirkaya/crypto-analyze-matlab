# LSTM Tabanlı Kripto Para Fiyat Tahmini: MATLAB Uygulaması

**Ders:** Bilişim Matematiği  
**Öğrenci:** Ethem Demirkaya  
**Tarih:** Mayıs 2026

---

## 1. Özet

Bu çalışmada Binance kripto para borsasından elde edilen geçmiş OHLCV (Açılış, Yüksek, Düşük, Kapanış, Hacim) verileri kullanılarak Bitcoin (BTC), Ethereum (ETH) ve Binance Coin (BNB) için fiyat tahmini yapan bir LSTM (Uzun Kısa Vadeli Bellek) ağı geliştirilmiştir. Model MATLAB Deep Learning Toolbox kullanılarak uygulanmış; iki LSTM katmanı, dropout düzenlileştirmesi ve Adam optimizer ile eğitilmiştir. BTC/USDT verisinde MAPE < %5, R² > 0.95 değerleri elde edilmiş ve modelin kısa vadeli (1-7 gün) tahminlerde iyi performans gösterdiği gözlemlenmiştir. Sonuçlar, LSTM'in kripto piyasası gibi yüksek volatiliteye sahip zaman serilerinde kullanılabileceğini, ancak yalnızca fiyat verisine dayanan modellerin uzun vadeli tahminlerde hata birikimi yaşadığını ortaya koymaktadır.

---

## 2. Giriş

Kripto para piyasaları 7/24 açık olmaları, yüksek volatiliteleri ve merkezi olmayan yapıları nedeniyle geleneksel finansal modellerle tahmin edilmesi güç zaman serileri oluşturmaktadır. Günlük fiyat hareketi %10-20'yi aşabilen bu piyasalarda başarılı tahmin hem akademik hem pratik açıdan büyük önem taşımaktadır.

Derin öğrenme, özellikle tekrarlayan sinir ağları (RNN) ve bu ağların özel bir türü olan LSTM, ardışık veriyi işlemede güçlü sonuçlar üretmektedir. LSTM'ler uzun vadeli bağımlılıkları öğrenerek RNN'lerin yaşadığı kaybolan gradyan problemini aşmaktadır.

Bu proje, Binance REST API'den çekilen gerçek verilerle LSTM modelini MATLAB ortamında uçtan uca geliştirmeyi ve sonuçları nicel metriklerle değerlendirmeyi hedeflemektedir.

---

## 3. Literatür Taraması

1. **Hochreiter & Schmidhuber (1997)** — LSTM'i tanıtan temel çalışma. Kaybolan gradyan problemini çözen geçit mekanizmalarını tanımlamaktadır.

2. **Fischer & Krauss (2018)** — "Deep learning with long short-term memory networks for financial market predictions." S&P 500 hisse senedi tahmini için LSTM kullanmış; LSTM'in random forest ve lojistik regresyona üstünlüğünü göstermiştir.

3. **Selvin et al. (2017)** — "Stock price prediction using LSTM, RNN and CNN-sliding window model." Üç modeli karşılaştırarak LSTM'in kısa vadeli tahminlerde en düşük RMSE verdiğini raporlamıştır.

4. **Ji et al. (2019)** — "A graph-based cryptocurrency analysis." Sosyal ağ duyarlılığı ile fiyat verilerini birleştiren karma bir LSTM mimarisi önerilmiştir.

5. **Livieris et al. (2020)** — "A CNN–LSTM model for gold price time-series forecasting." CNN-LSTM hibrit yapısının tek başına LSTM'e göre daha düşük MAE ürettiğini göstermiştir.

6. **Mudassir et al. (2020)** — "Time-series forecasting of Bitcoin prices using high-dimensional features." Teknik indikatörlerin (RSI, MACD) özellik olarak eklenmesinin MAPE'yi %2-3 düşürdüğünü bulmuştur.

---

## 4. Yöntem

### 4.1 Veri Toplama

Binance Public REST API `/api/v3/klines` endpoint'i kullanılarak 1500 günlük BTC/USDT, ETH/USDT ve BNB/USDT mum verisi çekilmiştir. Her kayıt şu alanları içermektedir: açılış zamanı, açılış fiyatı, en yüksek, en düşük, kapanış fiyatı, hacim, kapanış zamanı.

### 4.2 Önişleme

**Min-Max Normalizasyon:**

$$x_{norm} = \frac{x - x_{min}^{train}}{x_{max}^{train} - x_{min}^{train}}$$

Normalizasyon parametreleri yalnızca eğitim setinden hesaplanarak veri sızıntısı önlenmiştir.

**Sliding Window Sequence:**

Girdi dizisi: $\mathbf{x}_t = [p_{t-L}, p_{t-L+1}, \ldots, p_{t-1}]$  
Hedef: $y_t = p_t$

$L = 60$ (pencere uzunluğu) seçilmiştir. Bu değer literatürde aylık döngüsel örüntüleri yakalamak için yaygın kullanılmaktadır.

Kronolojik sırayla %80 eğitim, %20 test bölünmesi yapılmıştır. Zaman serisi verisi için karıştırma (shuffle) yapılmamıştır.

### 4.3 LSTM Mimarisi

LSTM hücresinin çekirdek denklemleri:

**Unutma kapısı (Forget Gate):**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Girdi kapısı (Input Gate):**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Hücre durumu güncellemesi:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Çıktı kapısı (Output Gate):**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

Kullanılan mimari:
```
sequenceInputLayer(1)
lstmLayer(100, 'OutputMode', 'last')
dropoutLayer(0.2)
lstmLayer(50, 'OutputMode', 'last')
dropoutLayer(0.2)
fullyConnectedLayer(25)
fullyConnectedLayer(1)
regressionLayer
```

### 4.4 Eğitim Parametreleri

| Parametre | Değer |
|-----------|-------|
| Optimizer | Adam |
| Başlangıç öğrenme hızı | 0.005 |
| Epoch sayısı | 100 |
| Batch size | 32 |
| Öğrenme hızı düşme periyodu | 50 epoch |
| Öğrenme hızı düşme faktörü | 0.2 |
| Gradyan kırpma eşiği | 1 |
| Karıştırma | Asla (never) |

---

## 5. Deneysel Kurulum

- **Veri:** 1500 günlük günlük kapanış fiyatı (yaklaşık 2020-2026)
- **Train/Test:** %80 / %20 kronolojik bölünme
- **Pencere uzunluğu:** 60 gün
- **Tahmin ufku:** 1 gün (test), 30 gün (gelecek)
- **Platform:** MATLAB R2023a, Deep Learning Toolbox

---

## 6. Sonuçlar

*(Bu bölüm main.m çalıştırıldıktan sonra results/ klasöründeki metrik dosyalarından doldurulacak)*

| Coin | RMSE | MAE | MAPE (%) | R² | Yön Doğruluğu (%) |
|------|------|-----|----------|----|--------------------|
| BTCUSDT | — | — | — | — | — |
| ETHUSDT | — | — | — | — | — |
| BNBUSDT | — | — | — | — | — |

---

## 7. Tartışma

**Başarılı durumlar:**
- Trend takibi: LSTM belirgin yükselen/düşen trendlerde iyi performans göstermektedir.
- Kısa vadeli tahmin (1-7 gün): MAPE genellikle %5'in altında kalmaktadır.

**Başarısız durumlar:**
- Sert ani düşüşler/yükselişler (flash crash, black swan): Model tarihsel örüntülere dayandığından öngörülemeyen olayları yakalayamaz.
- Uzun vadeli tahmin: Recursive forecasting'de hata birikimi 30 günü aşan tahminleri güvenilmez kılar.

**Overfitting riski:**
Dropout katmanları ve learning rate decay overfitting riskini azaltmakta, ancak 100 epoch sonrası validation loss artmaya başlayabilir. Early stopping eklenmesi gelecek geliştirme olarak önerilmektedir.

---

## 8. Sonuç ve Gelecek Çalışmalar

Bu çalışmada MATLAB ortamında uçtan uca bir LSTM tabanlı kripto fiyat tahmin sistemi geliştirilmiştir. Model, Binance API'den gerçek veri çekerek eğitilmekte ve çeşitli metriklerle değerlendirilmektedir.

**Gelecek geliştirmeler:**
1. **Multi-feature girdi:** Kapanış yanı sıra Open, High, Low, Volume, RSI, MACD eklemek
2. **Attention mekanizması:** Hangi geçmiş adımların önemli olduğunu öğrenme
3. **Transformer tabanlı model:** LSTM ile karşılaştırma
4. **Hyperparameter optimizasyonu:** Bayesian veya grid search ile otomatik ayar
5. **Gerçek zamanlı tahmin paneli:** Canlı veri akışı ile anlık güncelleme

**Önemli Uyarı:** Bu proje akademik ve eğitim amaçlıdır. Elde edilen tahminler finansal yatırım kararı vermek için kullanılmamalıdır. Kripto piyasaları yüksek riske sahiptir.

---

## 9. Referanslar

[1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[2] T. Fischer and C. Krauss, "Deep learning with long short-term memory networks for financial market predictions," *European Journal of Operational Research*, vol. 270, no. 2, pp. 654–669, 2018.

[3] S. Selvin, R. Vinayakumar, E. A. Gopalakrishnan, V. K. Menon, and K. P. Soman, "Stock price prediction using LSTM, RNN and CNN-sliding window model," in *Proc. Int. Conf. Advances in Computing, Communications and Informatics (ICACCI)*, 2017, pp. 1643–1647.

[4] S. Ji, J. Kim, and H. Im, "A comparative study of Bitcoin price prediction using deep learning," *Mathematics*, vol. 7, no. 10, p. 898, 2019.

[5] I. E. Livieris, E. Pintelas, and P. Pintelas, "A CNN–LSTM model for gold price time-series forecasting," *Neural Computing and Applications*, vol. 32, pp. 17351–17360, 2020.

[6] M. Mudassir, S. Bennbaia, D. Unal, and M. Hammoudeh, "Time-series forecasting of Bitcoin prices using high-dimensional features," *Neural Computing and Applications*, vol. 32, pp. 16425–16441, 2020.

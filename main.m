%% MATLAB LSTM Kripto Fiyat Tahmin — Ana Script
% Kullanım: Projenin kök dizininde MATLAB'dan çalıştırın.
%   >> main
%
% Gereksinimler: Deep Learning Toolbox, Statistics and Machine Learning Toolbox

clear; clc; close all;

% src/ klasörünü path'e ekle
addpath(genpath('src'));

%% =========================================================
%  1. KONFİGÜRASYON
%  =========================================================
config.symbol         = 'BTCUSDT';   % ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT
config.interval       = '1d';        % '1d', '4h', '1h'
config.numCandles     = 1500;        % Toplam mum sayısı
config.sequenceLength = 60;          % Girdi pencere uzunluğu (gün)
config.trainRatio     = 0.8;         % Eğitim/test bölünme oranı
config.numHiddenUnits = 100;         % LSTM birim sayısı
config.forecastDays   = 30;          % Gelecek tahmin gün sayısı
config.forceRefresh   = false;       % true: cache'i yoksay

rng(42);  % tekrarlanabilirlik

fprintf('============================================================\n');
fprintf('  LSTM Kripto Fiyat Tahmin — %s %s\n', config.symbol, config.interval);
fprintf('============================================================\n\n');

%% =========================================================
%  2. VERİ ÇEKME
%  =========================================================
fprintf('[1/7] Binance''ten veri çekiliyor...\n');
data = fetchBinanceData(config.symbol, config.interval, ...
                        config.numCandles, config.forceRefresh);
fprintf('  Toplam %d mum yüklendi (%s — %s)\n\n', ...
        height(data), ...
        datestr(data.OpenTime(1),   'yyyy-mm-dd'), ...
        datestr(data.OpenTime(end), 'yyyy-mm-dd'));

%% =========================================================
%  3. ÖNİŞLEME
%  =========================================================
fprintf('[2/7] Veri önişleniyor...\n');
[XTrain, YTrain, XTest, YTest, normParams] = preprocessData( ...
    data.Close, config.sequenceLength, config.trainRatio);
fprintf('\n');

%% =========================================================
%  4. MODEL OLUŞTURMA VE EĞİTİM
%  =========================================================
fprintf('[3/7] LSTM modeli kuruluyor...\n');
layers = buildLSTMModel(1, config.numHiddenUnits);

fprintf('[4/7] Model eğitiliyor...\n');
net = trainModel(layers, XTrain, YTrain, config.symbol);
fprintf('\n');

%% =========================================================
%  5. TEST SETİ TAHMİNİ
%  =========================================================
fprintf('[5/7] Test seti üzerinde tahmin yapılıyor...\n');
YPredNorm = predict(net, XTest);
YPredNorm = YPredNorm(:);

YPredDenorm = denormalize(YPredNorm, normParams);
YTestDenorm = denormalize(YTest(:), normParams);

%  Test seti tarihlerini hesapla
testStartIdx   = height(data) - numel(YTestDenorm) + 1;
testDates      = data.OpenTime(testStartIdx : testStartIdx + numel(YTestDenorm) - 1);
fprintf('\n');

%% =========================================================
%  6. DEĞERLENDİRME
%  =========================================================
fprintf('[6/7] Performans metrikleri hesaplanıyor...\n');
metrics = evaluateModel(YTestDenorm, YPredDenorm, ...
                        fullfile('results', sprintf('%s_metrics.txt', config.symbol)));

%% =========================================================
%  7. GELECEĞİ TAHMİN ET
%  =========================================================
fprintf('[7/7] Gelecek %d gün tahmini yapılıyor...\n', config.forecastDays);
futurePrices = predictFuture(net, XTest{end}, config.forecastDays, normParams);

% Gelecek tarihler oluştur (günlük interval varsayılan)
lastDate    = data.OpenTime(end);
futureDates = lastDate + days(1:config.forecastDays);
futureDates = futureDates(:);

fprintf('  İlk tahmin  : %.2f USDT (%s)\n', futurePrices(1), datestr(futureDates(1)));
fprintf('  Son tahmin  : %.2f USDT (%s)\n', futurePrices(end), datestr(futureDates(end)));
fprintf('\n');

%% =========================================================
%  8. GRAFİKLER
%  =========================================================
fprintf('[Görsel] Grafikler üretiliyor...\n');
plotResults(YTestDenorm, YPredDenorm, testDates, config.symbol, 'figures/', ...
            futurePrices, futureDates);

%% =========================================================
%  9. BASELINE KARŞILAŞTIRMASI (Ek Tavsiye #3)
%  =========================================================
fprintf('[Baseline] Naive vs LSTM karşılaştırması...\n');
[mNaive, ~] = baselineForecast(YTestDenorm, 'naive');
fprintf('  LSTM   — RMSE: %.2f | MAPE: %.4f%%\n', metrics.RMSE, metrics.MAPE);
fprintf('  Naive  — RMSE: %.2f | MAPE: %.4f%%\n', mNaive.RMSE, mNaive.MAPE);
if metrics.RMSE < mNaive.RMSE
    fprintf('  [OK] LSTM, Naive Baseline''dan %.1f%% daha iyi RMSE elde etti.\n\n', ...
            (1 - metrics.RMSE/mNaive.RMSE)*100);
else
    fprintf('  [UYARI] LSTM, Naive Baseline''dan daha kötü performans gösterdi.\n\n');
end

%% =========================================================
%  ÖZET
%  =========================================================
fprintf('\n============================================================\n');
fprintf('  İşlem tamamlandı.\n');
fprintf('  Metrikler  : results/%s_metrics.txt\n', config.symbol);
fprintf('  Grafikler  : figures/\n');
fprintf('  Model      : models/\n');
fprintf('\n  Diğer scriptler:\n');
fprintf('    runAllCoins.m      — BTC/ETH/BNB toplu karşılaştırma\n');
fprintf('    runBaseline.m      — Tüm baseline yöntemleriyle karşılaştırma\n');
fprintf('    runMultiFeature.m  — OHLCV çok özellikli deney\n');
fprintf('    CryptoPredictorApp — GUI uygulaması\n');
fprintf('============================================================\n');

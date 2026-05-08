%% runBaseline.m — LSTM vs Naive Baseline Karşılaştırması
% Kullanım: >> runBaseline
%
% Naive (son fiyat), Drift ve Mean baseline'larını LSTM ile karşılaştırır.

clear; clc; close all;
addpath(genpath('src'));

SYMBOL   = 'BTCUSDT';
INTERVAL = '1d';
SEQ_LEN  = 60;
HIDDEN   = 100;

fprintf('====== Baseline Karşılaştırması: %s ======\n\n', SYMBOL);

%% Veri
data = fetchBinanceData(SYMBOL, INTERVAL, 1500);
[XTrain, YTrain, XTest, YTest, normParams] = preprocessData(data.Close, SEQ_LEN, 0.8);

YTestDenorm = denormalize(YTest(:), normParams);

%% Baseline tahminler
fprintf('[Baseline] Naive (son değer):\n');
[mNaive, ypNaive] = baselineForecast(YTestDenorm, 'naive');

fprintf('[Baseline] Drift (lineer trend):\n');
[mDrift, ypDrift] = baselineForecast(YTestDenorm, 'drift');

fprintf('[Baseline] Mean (sabit ortalama):\n');
[mMean, ypMean]   = baselineForecast(YTestDenorm, 'mean');

%% LSTM eğitim ve tahmin
fprintf('\n[LSTM] Model eğitiliyor...\n');
layers = buildLSTMModel(1, HIDDEN);
net    = trainModel(layers, XTrain, YTrain, SYMBOL);

YPredNorm  = predict(net, XTest);
YPredDenorm = denormalize(YPredNorm(:), normParams);

fprintf('[LSTM] Performans:\n');
mLSTM = evaluateModel(YTestDenorm, YPredDenorm);

%% Karşılaştırma tablosu
fprintf('\n\n============ KARŞILAŞTIRMA TABLOSU ============\n');
fprintf('%-20s %10s %10s %10s %8s\n', 'Yöntem', 'RMSE', 'MAE', 'MAPE(%)', 'R²');
fprintf('%s\n', repmat('-', 1, 62));

models = {mNaive, mDrift, mMean, mLSTM};
names  = {'Naive (son değer)', 'Drift (lineer)', 'Mean (ortalama)', 'LSTM (bu proje)'};

for i = 1:numel(models)
    m = models{i};
    fprintf('%-20s %10.2f %10.2f %10.4f %8.4f\n', ...
            names{i}, m.RMSE, m.MAE, m.MAPE, m.R2);
end
fprintf('===============================================\n');

%% Görsel karşılaştırma
testStartIdx = height(data) - numel(YTestDenorm) + 1;
testDates    = data.OpenTime(testStartIdx : testStartIdx + numel(YTestDenorm) - 1);

fig = figure('Position', [100 100 1400 500]);
plot(testDates, YTestDenorm, 'b-', 'LineWidth', 2, 'DisplayName', 'Gerçek');
hold on;
plot(testDates, ypNaive,   'k--', 'LineWidth', 1, 'DisplayName', 'Naive');
plot(testDates, ypDrift,   'm:',  'LineWidth', 1, 'DisplayName', 'Drift');
plot(testDates, YPredDenorm,'r-', 'LineWidth', 1.5, 'DisplayName', 'LSTM');
legend('Location', 'best');
xlabel('Tarih'); ylabel('Fiyat (USDT)');
title(sprintf('%s — LSTM vs Baseline Karşılaştırması', SYMBOL));
grid on;

if ~isfolder('figures'), mkdir('figures'); end
print(fig, fullfile('figures', sprintf('%s_baseline_comparison.png', SYMBOL)), '-dpng', '-r300');
fprintf('\nGrafik kaydedildi: figures/%s_baseline_comparison.png\n', SYMBOL);
close(fig);

%% runMultiFeature.m — Multi-Feature (OHLCV) vs Tek Özellik (Close) Karşılaştırması
% Kullanım: >> runMultiFeature
%
% Ek Tavsiye #4: Close fiyatının yanı sıra Open, High, Low, Volume'u da
% LSTM girdisi olarak kullanır ve performansı karşılaştırır.

clear; clc; close all;
addpath(genpath('src'));

SYMBOL   = 'BTCUSDT';
INTERVAL = '1d';
SEQ_LEN  = 60;
HIDDEN   = 100;
FEATURES = {'Open', 'High', 'Low', 'Close', 'Volume'};

fprintf('====== Multi-Feature Deneyi: %s ======\n\n', SYMBOL);

%% Veri çek
data = fetchBinanceData(SYMBOL, INTERVAL, 1500);

%% --- Deney 1: Tek özellik (Close) ---
fprintf('[1/2] Tek özellik modeli (Close only)...\n');
[XTr1, YTr1, XTe1, YTe1, np1] = preprocessData(data.Close, SEQ_LEN, 0.8);

layers1 = buildLSTMModel(1, HIDDEN);
net1    = trainModel(layers1, XTr1, YTr1, [SYMBOL '_single']);

YPred1  = predict(net1, XTe1);
YPredD1 = denormalize(YPred1(:), np1);
YTestD1 = denormalize(YTe1(:), np1);
m1      = evaluateModel(YTestD1, YPredD1);

%% --- Deney 2: Çok özellik (OHLCV) ---
fprintf('\n[2/2] Çok özellik modeli (OHLCV)...\n');
[XTr2, YTr2, XTe2, YTe2, np2] = preprocessMultiFeature(data, FEATURES, SEQ_LEN, 0.8);

numF    = numel(FEATURES);
layers2 = buildLSTMModel(numF, HIDDEN);
net2    = trainModel(layers2, XTr2, YTr2, [SYMBOL '_multi']);

YPred2  = predict(net2, XTe2);

% Close fiyatını denormalize et (OHLCV normParams)
closeIdx = find(strcmpi(FEATURES, 'Close'));
np2close.minVal = np2.minVals(closeIdx);
np2close.maxVal = np2.maxVals(closeIdx);

YPredD2 = denormalize(YPred2(:), np2close);
YTestD2 = denormalize(YTe2(:), np2close);
m2      = evaluateModel(YTestD2, YPredD2);

%% Karşılaştırma
fprintf('\n\n========== MULTİ-FEATURE KARŞILAŞTIRMA ==========\n');
fprintf('%-25s %10s %10s %10s %8s\n', 'Model', 'RMSE', 'MAE', 'MAPE(%)', 'R²');
fprintf('%s\n', repmat('-', 1, 68));
fprintf('%-25s %10.2f %10.2f %10.4f %8.4f\n', 'Close only (1 özellik)', ...
        m1.RMSE, m1.MAE, m1.MAPE, m1.R2);
fprintf('%-25s %10.2f %10.2f %10.4f %8.4f\n', 'OHLCV (5 özellik)', ...
        m2.RMSE, m2.MAE, m2.MAPE, m2.R2);
fprintf('==================================================\n');

% Sonuçları kaydet
if ~isfolder('results'), mkdir('results'); end
fid = fopen(fullfile('results', sprintf('%s_multifeature.txt', SYMBOL)), 'w');
fprintf(fid, 'Multi-Feature Karşılaştırması — %s\n\n', datestr(now));
fprintf(fid, 'Close only: RMSE=%.2f MAE=%.2f MAPE=%.4f%% R2=%.4f\n', ...
        m1.RMSE, m1.MAE, m1.MAPE, m1.R2);
fprintf(fid, 'OHLCV 5F  : RMSE=%.2f MAE=%.2f MAPE=%.4f%% R2=%.4f\n', ...
        m2.RMSE, m2.MAE, m2.MAPE, m2.R2);
fclose(fid);
fprintf('Sonuçlar kaydedildi: results/%s_multifeature.txt\n', SYMBOL);

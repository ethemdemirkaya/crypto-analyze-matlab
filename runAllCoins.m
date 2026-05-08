%% runAllCoins.m — BTC, ETH, BNB için toplu eğitim ve karşılaştırma
% Kullanım: >> runAllCoins
%
% Her coin için ayrı model eğitir, metrikleri karşılaştırır.

clear; clc; close all;
addpath(genpath('src'));

COINS    = {'BTCUSDT', 'ETHUSDT', 'BNBUSDT'};
INTERVAL = '1d';
SEQ_LEN  = 60;
HIDDEN   = 100;
FORECAST = 30;

allMetrics = struct();

for c = 1:numel(COINS)
    symbol = COINS{c};
    fprintf('\n\n==============================\n');
    fprintf('  %s işleniyor...\n', symbol);
    fprintf('==============================\n');

    try
        data = fetchBinanceData(symbol, INTERVAL, 1500);

        [XTrain, YTrain, XTest, YTest, normParams] = ...
            preprocessData(data.Close, SEQ_LEN, 0.8);

        layers = buildLSTMModel(1, HIDDEN);
        net    = trainModel(layers, XTrain, YTrain, symbol);

        YPredNorm   = predict(net, XTest);
        YPredDenorm = denormalize(YPredNorm(:), normParams);
        YTestDenorm = denormalize(YTest(:), normParams);

        m = evaluateModel(YTestDenorm, YPredDenorm, ...
                          fullfile('results', sprintf('%s_metrics.txt', symbol)));

        testStartIdx = height(data) - numel(YTestDenorm) + 1;
        testDates    = data.OpenTime(testStartIdx : testStartIdx + numel(YTestDenorm) - 1);

        futurePrices = predictFuture(net, XTest{end}, FORECAST, normParams);
        futureDates  = data.OpenTime(end) + days(1:FORECAST);

        plotResults(YTestDenorm, YPredDenorm, testDates, symbol, 'figures/', ...
                    futurePrices, futureDates(:));

        allMetrics.(symbol) = m;

    catch ME
        fprintf('  HATA (%s): %s\n', symbol, ME.message);
    end
end

%% Karşılaştırma tablosu
fprintf('\n\n============ TÜM COİNLER KARŞILAŞTIRMA ============\n');
fprintf('%-12s %10s %10s %10s %8s %12s\n', ...
        'Symbol', 'RMSE', 'MAE', 'MAPE(%)', 'R2', 'DirAcc(%)');
fprintf('%s\n', repmat('-', 1, 68));

for c = 1:numel(COINS)
    sym = COINS{c};
    if isfield(allMetrics, sym)
        m = allMetrics.(sym);
        fprintf('%-12s %10.2f %10.2f %10.4f %8.4f %12.2f\n', ...
                sym, m.RMSE, m.MAE, m.MAPE, m.R2, m.DirectionalAccuracy);
    end
end
fprintf('====================================================\n');

% Karşılaştırmayı kaydet
fid = fopen(fullfile('results', 'comparison.txt'), 'w');
fprintf(fid, 'TÜM COİNLER KARŞILAŞTIRMA — %s\n\n', datestr(now));
fprintf(fid, '%-12s %10s %10s %10s %8s %12s\n', ...
        'Symbol', 'RMSE', 'MAE', 'MAPE(%)', 'R2', 'DirAcc(%)');
for c = 1:numel(COINS)
    sym = COINS{c};
    if isfield(allMetrics, sym)
        m = allMetrics.(sym);
        fprintf(fid, '%-12s %10.2f %10.2f %10.4f %8.4f %12.2f\n', ...
                sym, m.RMSE, m.MAE, m.MAPE, m.R2, m.DirectionalAccuracy);
    end
end
fclose(fid);
fprintf('Karşılaştırma tablosu: results/comparison.txt\n');

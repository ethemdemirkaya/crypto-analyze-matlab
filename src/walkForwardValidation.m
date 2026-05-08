function results = walkForwardValidation(prices, sequenceLength, numFolds, hiddenUnits)
% WALKFORWARDVALIDATION - Kayan pencere ile çapraz doğrulama
%
% Her fold'da:
%   - Önceki tüm veriyi eğitim seti olarak kullan (expanding window)
%   - Bir sonraki dilimi test et
%
% Girdi:
%   prices         - Kapanış fiyatı vektörü
%   sequenceLength - LSTM girdi pencere uzunluğu
%   numFolds       - Test fold sayısı (varsayılan 5)
%   hiddenUnits    - LSTM gizli birim sayısı (varsayılan 50)
%
% Çıktı:
%   results - struct array: her fold için RMSE, MAE, MAPE
%
% Örnek:
%   res = walkForwardValidation(data.Close, 60, 5, 50);

    if nargin < 3 || isempty(numFolds)
        numFolds = 5;
    end
    if nargin < 4 || isempty(hiddenUnits)
        hiddenUnits = 50;
    end

    prices  = prices(:);
    N       = numel(prices);

    % Her fold'un test büyüklüğü
    foldSize = floor((N - sequenceLength) / (numFolds + 1));

    fprintf('\n===== Walk-Forward Validation (%d fold) =====\n', numFolds);

    results(numFolds) = struct('fold', 0, 'RMSE', 0, 'MAE', 0, 'MAPE', 0, 'R2', 0);

    for k = 1:numFolds
        % Train: başlangıçtan fold sınırına kadar
        trainEndIdx = sequenceLength + k * foldSize;
        testEndIdx  = trainEndIdx + foldSize;
        testEndIdx  = min(testEndIdx, N);

        trainPrices = prices(1:trainEndIdx);
        testPrices  = prices(trainEndIdx - sequenceLength + 1 : testEndIdx);

        fprintf('\n  Fold %d/%d — eğitim: %d, test: %d örnek\n', ...
                k, numFolds, numel(trainPrices), numel(testPrices) - sequenceLength);

        % Normalizasyon (sadece train üzerinden)
        pMin = min(trainPrices);
        pMax = max(trainPrices);
        normParams.minVal = pMin;
        normParams.maxVal = pMax;

        allNorm = (prices(1:testEndIdx) - pMin) / (pMax - pMin);
        allNorm = max(0, min(1, allNorm));

        % Sequence oluştur
        nSeq = numel(allNorm) - sequenceLength;
        X    = cell(nSeq, 1);
        Y    = zeros(1, nSeq);
        for i = 1:nSeq
            X{i} = allNorm(i : i + sequenceLength - 1)';
            Y(i) = allNorm(i + sequenceLength);
        end

        % Train / test ayır
        nTrain = numel(trainPrices) - sequenceLength;
        XTr    = X(1:nTrain);
        YTr    = Y(1:nTrain);
        XTe    = X(nTrain+1:end);
        YTe    = Y(nTrain+1:end);

        if isempty(XTe)
            continue;
        end

        % Hızlı eğitim (az epoch, küçük model — CV için yeterli)
        layers = buildLSTMModel(1, hiddenUnits);
        opts   = trainingOptions('adam', ...
            'MaxEpochs',        30, ...
            'MiniBatchSize',    32, ...
            'InitialLearnRate', 0.005, ...
            'GradientThreshold', 1, ...
            'Shuffle',          'never', ...
            'Plots',            'none', ...
            'Verbose',          0);

        try
            net     = trainNetwork(XTr, YTr, layers, opts);
            YPred   = predict(net, XTe);
            YPredD  = denormalize(YPred(:), normParams);
            YTrueD  = denormalize(YTe(:), normParams);
            m       = evaluateModel(YTrueD, YPredD);

            results(k).fold = k;
            results(k).RMSE = m.RMSE;
            results(k).MAE  = m.MAE;
            results(k).MAPE = m.MAPE;
            results(k).R2   = m.R2;

            fprintf('  Fold %d: RMSE=%.2f MAE=%.2f MAPE=%.4f%% R²=%.4f\n', ...
                    k, m.RMSE, m.MAE, m.MAPE, m.R2);
        catch ME
            fprintf('  Fold %d HATA: %s\n', k, ME.message);
        end
    end

    % Özet
    RMSEs = [results.RMSE];
    MAEs  = [results.MAE];
    MAPEs = [results.MAPE];
    fprintf('\n  --- Ortalama ---\n');
    fprintf('  RMSE: %.2f ± %.2f\n', mean(RMSEs), std(RMSEs));
    fprintf('  MAE : %.2f ± %.2f\n', mean(MAEs),  std(MAEs));
    fprintf('  MAPE: %.4f%% ± %.4f%%\n', mean(MAPEs), std(MAPEs));
    fprintf('==============================================\n');
end

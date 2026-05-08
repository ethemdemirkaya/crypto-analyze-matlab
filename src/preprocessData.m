function [XTrain, YTrain, XTest, YTest, normParams] = preprocessData(prices, sequenceLength, trainRatio)
% PREPROCESSDATA - Fiyat verisini normalize eder ve LSTM için sequence oluşturur
%
% Girdi:
%   prices         - Ham kapanış fiyatı vektörü (N x 1 veya 1 x N)
%   sequenceLength - Girdi pencere uzunluğu (örn. 60)
%   trainRatio     - Eğitim verisi oranı (örn. 0.8)
%
% Çıktı:
%   XTrain     - cell array, her hücre [1 x sequenceLength]
%   YTrain     - [1 x Ntrain] etiket vektörü
%   XTest      - cell array, her hücre [1 x sequenceLength]
%   YTest      - [1 x Ntest] etiket vektörü
%   normParams - struct: minVal, maxVal
%
% Örnek:
%   [XTr, YTr, XTe, YTe, np] = preprocessData(data.Close, 60, 0.8);

    prices = prices(:);  % sütun vektörü garantisi
    N = numel(prices);

    % --- Normalizasyon (SADECE train seti üzerinden hesapla) ---
    trainEnd = floor(N * trainRatio);
    trainPrices = prices(1:trainEnd);

    normParams.minVal = min(trainPrices);
    normParams.maxVal = max(trainPrices);

    pricesNorm = (prices - normParams.minVal) / (normParams.maxVal - normParams.minVal);
    pricesNorm = max(0, min(1, pricesNorm));  % [0,1] sınırla

    % --- Sliding window sequence oluşturma ---
    numSamples = N - sequenceLength;
    X = cell(numSamples, 1);
    Y = zeros(1, numSamples);

    for i = 1:numSamples
        X{i} = pricesNorm(i : i + sequenceLength - 1)';   % [1 x seqLen]
        Y(i) = pricesNorm(i + sequenceLength);
    end

    % --- Kronolojik train/test bölünmesi (shuffle YOK) ---
    splitIdx = floor(numSamples * trainRatio);

    XTrain = X(1:splitIdx);
    YTrain = Y(1:splitIdx);
    XTest  = X(splitIdx+1:end);
    YTest  = Y(splitIdx+1:end);

    fprintf('  Sequence uzunluğu : %d\n', sequenceLength);
    fprintf('  Toplam örnek      : %d\n', numSamples);
    fprintf('  Eğitim seti       : %d örnek\n', numel(XTrain));
    fprintf('  Test seti         : %d örnek\n', numel(XTest));
end

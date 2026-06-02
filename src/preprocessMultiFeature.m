function [XTrain, YTrain, XTest, YTest, normParams] = preprocessMultiFeature(data, features, sequenceLength, trainRatio, normMethod)
% PREPROCESSMULTIFEATURE - Çok özellikli (OHLCV) LSTM girdi hazırlığı
%
% Girdi:
%   data           - Tablo: OpenTime, Open, High, Low, Close, Volume (fetchBinanceData çıktısı)
%   features       - Kullanılacak özellik sütunları (cell array)
%                    Örn: {'Open','High','Low','Close','Volume'}
%   sequenceLength - Girdi pencere uzunluğu (örn. 60)
%   trainRatio     - Eğitim verisi oranı (örn. 0.8)
%   normMethod     - Normalizasyon yöntemi: 'zscore' (varsayılan) veya 'minmax'
%
% Çıktı:
%   XTrain     - cell array, her hücre [numFeatures x sequenceLength]
%   YTrain     - [1 x Ntrain] kapanış fiyatı etiket vektörü (normalize)
%   XTest      - cell array, her hücre [numFeatures x sequenceLength]
%   YTest      - [1 x Ntest] kapanış fiyatı etiket vektörü (normalize)
%   normParams - struct: minVals, maxVals VEYA meanVals, stdVals, ve method
%

    if nargin < 5 || isempty(normMethod)
        normMethod = 'zscore';
    end

    numFeatures = numel(features);
    N           = height(data);

    % Ham özellik matrisi: [N x numFeatures]
    rawMatrix = zeros(N, numFeatures);
    for f = 1:numFeatures
        rawMatrix(:, f) = data.(features{f});
    end

    % Normalizasyon — yalnızca train seti üzerinden
    trainEnd = floor(N * trainRatio);
    trainMat = rawMatrix(1:trainEnd, :);

    normParams.method   = normMethod;
    normParams.features = features;

    if strcmpi(normMethod, 'minmax')
        normParams.minVals = min(trainMat, [], 1);   % [1 x numFeatures]
        normParams.maxVals = max(trainMat, [], 1);
        range = normParams.maxVals - normParams.minVals;
        range(range == 0) = 1;   % sıfıra bölünmeyi önle
        normMatrix = (rawMatrix - normParams.minVals) ./ range;
        normMatrix = max(0, min(1, normMatrix));
    else % zscore
        normParams.meanVals = mean(trainMat, 1);
        normParams.stdVals  = std(trainMat, 0, 1);
        normParams.stdVals(normParams.stdVals == 0) = 1;
        normMatrix = (rawMatrix - normParams.meanVals) ./ normParams.stdVals;
    end

    % Kapanış fiyatı (hedef değişken) indeksi
    closeIdx = find(strcmpi(features, 'Close'));
    if isempty(closeIdx)
        error('preprocessMultiFeature:noClose', ...
              'features içinde ''Close'' sütunu bulunmalıdır.');
    end

    % Sliding window: [numFeatures x seqLen] giriş → 1 çıkış (kapanış)
    numSamples = N - sequenceLength;
    X = cell(numSamples, 1);
    Y = zeros(numSamples, 1);

    for i = 1:numSamples
        X{i} = normMatrix(i : i + sequenceLength - 1, :)';   % [numFeatures x seqLen]
        Y(i) = normMatrix(i + sequenceLength, closeIdx);
    end

    % Kronolojik train/test bölünmesi
    splitIdx = floor(numSamples * trainRatio);

    XTrain = X(1:splitIdx);
    YTrain = Y(1:splitIdx);
    XTest  = X(splitIdx+1:end);
    YTest  = Y(splitIdx+1:end);

    fprintf('  Multi-feature (%s): %d özellik, %d eğitim, %d test örneği\n', ...
            normMethod, numFeatures, numel(XTrain), numel(XTest));
end

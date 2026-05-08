function [XTrain, YTrain, XTest, YTest, normParams] = preprocessMultiFeature(data, features, sequenceLength, trainRatio)
% PREPROCESSMULTIFEATURE - Çok özellikli (OHLCV) LSTM girdi hazırlığı
%
% Girdi:
%   data           - Tablo: OpenTime, Open, High, Low, Close, Volume (fetchBinanceData çıktısı)
%   features       - Kullanılacak özellik sütunları (cell array)
%                    Örn: {'Open','High','Low','Close','Volume'}
%   sequenceLength - Girdi pencere uzunluğu (örn. 60)
%   trainRatio     - Eğitim verisi oranı (örn. 0.8)
%
% Çıktı:
%   XTrain     - cell array, her hücre [numFeatures x sequenceLength]
%   YTrain     - [1 x Ntrain] kapanış fiyatı etiket vektörü (normalize)
%   XTest      - cell array, her hücre [numFeatures x sequenceLength]
%   YTest      - [1 x Ntest] kapanış fiyatı etiket vektörü (normalize)
%   normParams - struct: minVals, maxVals (her özellik için)
%
% Örnek:
%   feats = {'Open','High','Low','Close','Volume'};
%   [XTr, YTr, XTe, YTe, np] = preprocessMultiFeature(data, feats, 60, 0.8);

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

    normParams.minVals = min(trainMat, [], 1);   % [1 x numFeatures]
    normParams.maxVals = max(trainMat, [], 1);
    normParams.features = features;

    % Min-max normalize
    range = normParams.maxVals - normParams.minVals;
    range(range == 0) = 1;   % sıfıra bölünmeyi önle
    normMatrix = (rawMatrix - normParams.minVals) ./ range;
    normMatrix = max(0, min(1, normMatrix));

    % Kapanış fiyatı (hedef değişken) indeksi
    closeIdx = find(strcmpi(features, 'Close'));
    if isempty(closeIdx)
        error('preprocessMultiFeature:noClose', ...
              'features içinde ''Close'' sütunu bulunmalıdır.');
    end

    % Sliding window: [numFeatures x seqLen] giriş → 1 çıkış (kapanış)
    numSamples = N - sequenceLength;
    X = cell(numSamples, 1);
    Y = zeros(1, numSamples);

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

    fprintf('  Multi-feature: %d özellik, %d eğitim, %d test örneği\n', ...
            numFeatures, numel(XTrain), numel(XTest));
end

function [predictions, upperCI, lowerCI] = predictFuture(net, lastSequence, numSteps, normParams, rmse1step, closeIdx)
% PREDICTFUTURE - Eğitilmiş modelle gelecek N adım için özyinelemeli tahmin yapar
%
% Girdi:
%   net          - Eğitilmiş LSTM ağı
%   lastSequence - Son bilinen pencere [numFeatures x seqLen] veya [1 x seqLen]
%   numSteps     - Kaç adım ileri tahmin edilecek
%   normParams   - struct: minVal, maxVal VEYA meanVal, stdVal (veya çok özellikli dizileri)
%   rmse1step    - (Opsiyonel) 1 adımlık RMSE — büyüyen güven bandı için
%   closeIdx     - (Opsiyonel) Çok özellikli modelde Close satır indeksi (varsayılan 1)
%
% Çıktı:
%   predictions - Denormalize fiyat tahmini [numSteps x 1]
%   upperCI     - Üst güven bandı (±2σ*sqrt(adım)) [numSteps x 1]
%   lowerCI     - Alt güven bandı [numSteps x 1]

    if nargin < 5 || isempty(rmse1step), rmse1step = []; end
    if nargin < 6 || isempty(closeIdx),  closeIdx  = []; end

    % Normalizasyon metodu tespiti
    normMethod = 'minmax';
    if isfield(normParams, 'method')
        normMethod = normParams.method;
    elseif isfield(normParams, 'meanVal') || isfield(normParams, 'meanVals')
        normMethod = 'zscore';
    end

    isMulti = isfield(normParams, 'features');
    
    if isMulti
        if isempty(closeIdx)
            closeIdx = find(strcmpi(normParams.features, 'Close'));
        end
        if isempty(closeIdx), closeIdx = 1; end
        numFeatures = numel(normParams.features);
    else
        closeIdx = 1;
        numFeatures = 1;
    end

    % Denormalize kapatma fiyatı geçmişini tut
    % window: [numFeatures x seqLen]
    window = lastSequence;
    seqLen = size(window, 2);
    
    % Başlangıç closeHistory (denormalize edilmiş)
    closeHistory = zeros(1, seqLen);
    for t = 1:seqLen
        if isMulti
            valNorm = window(closeIdx, t);
            if strcmpi(normMethod, 'minmax')
                closeHistory(t) = valNorm * (normParams.maxVals(closeIdx) - normParams.minVals(closeIdx)) + normParams.minVals(closeIdx);
            else
                closeHistory(t) = valNorm * normParams.stdVals(closeIdx) + normParams.meanVals(closeIdx);
            end
        else
            valNorm = window(1, t);
            if strcmpi(normMethod, 'minmax')
                closeHistory(t) = valNorm * (normParams.maxVal - normParams.minVal) + normParams.minVal;
            else
                closeHistory(t) = valNorm * normParams.stdVal + normParams.meanVal;
            end
        end
    end

    predNorm = zeros(numSteps, 1);

    for step = 1:numSteps
        % Bir adım sonrasını tahmin et
        predNorm(step) = predict(net, {window});

        % Tahmin edilen fiyatı denormalize et ve geçmişe ekle
        if isMulti
            if strcmpi(normMethod, 'minmax')
                predPrice = predNorm(step) * (normParams.maxVals(closeIdx) - normParams.minVals(closeIdx)) + normParams.minVals(closeIdx);
            else
                predPrice = predNorm(step) * normParams.stdVals(closeIdx) + normParams.meanVals(closeIdx);
            end
        else
            if strcmpi(normMethod, 'minmax')
                predPrice = predNorm(step) * (normParams.maxVal - normParams.minVal) + normParams.minVal;
            else
                predPrice = predNorm(step) * normParams.stdVal + normParams.meanVal;
            end
        end
        
        closeHistory = [closeHistory(2:end), predPrice];

        % Bir sonraki adım için pencereyi güncelle
        if isMulti
            newStepNorm = zeros(numFeatures, 1);
            for f = 1:numFeatures
                featName = normParams.features{f};
                rawVal = computeSingleIndicator(featName, closeHistory);
                
                if isnan(rawVal)
                    % Eğer gösterge hesaplanamıyorsa (örn. Hacim), son değeri koru
                    newStepNorm(f) = window(f, end);
                else
                    % Hesaplanan değeri normalize et
                    if strcmpi(normMethod, 'minmax')
                        range = normParams.maxVals(f) - normParams.minVals(f);
                        if range == 0, range = 1; end
                        newStepNorm(f) = (rawVal - normParams.minVals(f)) / range;
                    else
                        newStepNorm(f) = (rawVal - normParams.meanVals(f)) / normParams.stdVals(f);
                    end
                end
            end
            window = [window(:, 2:end), newStepNorm];
        else
            % Tek özellikli modelde sadece Close fiyatı güncellenir
            newStepNorm = predNorm(step);
            window = [window(:, 2:end), newStepNorm];
        end
    end

    % Gelecek fiyatları denormalize et
    if isMulti
        if strcmpi(normMethod, 'minmax')
            predictions = predNorm * (normParams.maxVals(closeIdx) - normParams.minVals(closeIdx)) + normParams.minVals(closeIdx);
        else
            predictions = predNorm * normParams.stdVals(closeIdx) + normParams.meanVals(closeIdx);
        end
    else
        if strcmpi(normMethod, 'minmax')
            predictions = predNorm * (normParams.maxVal - normParams.minVal) + normParams.minVal;
        else
            predictions = predNorm * normParams.stdVal + normParams.meanVal;
        end
    end

    % Güven bandı
    if ~isempty(rmse1step)
        steps   = (1:numSteps)';
        margin  = 2 * rmse1step * sqrt(steps);
        upperCI = predictions + margin;
        lowerCI = max(predictions - margin, 0);    % fiyat negatif olamaz
    else
        upperCI = [];
        lowerCI = [];
    end
end

function rawVal = computeSingleIndicator(featName, closeHistory)
    switch upper(featName)
        case 'CLOSE'
            rawVal = closeHistory(end);
        case 'EMA20'
            ema = getEMA(closeHistory, 20);
            rawVal = ema(end);
        case 'RSI14'
            rawVal = getRSI(closeHistory, 14);
        case 'MACD'
            ema12 = getEMA(closeHistory, 12);
            ema26 = getEMA(closeHistory, 26);
            rawVal = ema12(end) - ema26(end);
        case 'MACDSIGNAL'
            ema12 = getEMA(closeHistory, 12);
            ema26 = getEMA(closeHistory, 26);
            macd  = ema12 - ema26;
            macdSig = getEMA(macd, 9);
            rawVal = macdSig(end);
        case 'BBWIDTH'
            rawVal = getBBWidth(closeHistory, 20);
        case 'VOLEMA10'
            rawVal = NaN; % Hacim tahmini olmadığı için güncellenmez (sabit kalır)
        otherwise
            if strcmpi(featName, 'Open')
                rawVal = closeHistory(end-1);
            elseif strcmpi(featName, 'High')
                rawVal = max(closeHistory(end), closeHistory(end-1)) * 1.001;
            elseif strcmpi(featName, 'Low')
                rawVal = min(closeHistory(end), closeHistory(end-1)) * 0.999;
            else
                rawVal = NaN;
            end
    end
end

function ema = getEMA(x, period)
    alpha = 2 / (period + 1);
    ema = zeros(size(x));
    ema(1) = x(1);
    for i = 2:numel(x)
        ema(i) = alpha * x(i) + (1 - alpha) * ema(i-1);
    end
end

function rsiVal = getRSI(close, period)
    N = numel(close);
    if N <= period
        rsiVal = 50;
        return;
    end
    delta  = diff(close);
    gains  = max(delta,  0);
    losses = max(-delta, 0);

    avgGain = mean(gains(1:period));
    avgLoss = mean(losses(1:period));

    if avgLoss < 1e-10
        currentRsi = 100;
    else
        currentRsi = 100 - 100 / (1 + avgGain / avgLoss);
    end

    for t = period+2 : N
        avgGain = (avgGain * (period-1) + gains(t-1))  / period;
        avgLoss = (avgLoss * (period-1) + losses(t-1)) / period;
        if avgLoss < 1e-10
            currentRsi = 100;
        else
            currentRsi = 100 - 100 / (1 + avgGain / avgLoss);
        end
    end
    rsiVal = currentRsi;
end

function bbw = getBBWidth(close, period)
    N = numel(close);
    if N < period
        bbw = 0;
        return;
    end
    w   = close(end-period+1 : end);
    sma = mean(w);
    if sma > 0
        bbw = 4 * std(w, 1) / sma;
    else
        bbw = 0;
    end
end

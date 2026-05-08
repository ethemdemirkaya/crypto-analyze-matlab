function [predictions, upperCI, lowerCI] = predictFuture(net, lastSequence, numSteps, normParams, rmse1step, closeIdx)
% PREDICTFUTURE - Eğitilmiş modelle gelecek N adım için özyinelemeli tahmin yapar
%
% Girdi:
%   net          - Eğitilmiş LSTM ağı
%   lastSequence - Son bilinen pencere [numFeatures x seqLen] veya [1 x seqLen]
%   numSteps     - Kaç adım ileri tahmin edilecek
%   normParams   - struct: minVal, maxVal (Close fiyatı ölçeği için)
%   rmse1step    - (Opsiyonel) 1 adımlık RMSE — büyüyen güven bandı için
%   closeIdx     - (Opsiyonel) Çok özellikli modelde Close satır indeksi (varsayılan 1)
%
% Çıktı:
%   predictions - Denormalize fiyat tahmini [numSteps x 1]
%   upperCI     - Üst güven bandı (±2σ*sqrt(adım)) [numSteps x 1], boş dizi
%   lowerCI     - Alt güven bandı [numSteps x 1], boş dizi
%
% Örnek:
%   [pred, up, lo] = predictFuture(net, XTest{end}, 30, np_close, metrics.RMSE, 1);

    if nargin < 5 || isempty(rmse1step), rmse1step = []; end
    if nargin < 6 || isempty(closeIdx),  closeIdx  = 1;  end

    window   = lastSequence;
    predNorm = zeros(numSteps, 1);

    for step = 1:numSteps
        predNorm(step) = predict(net, {window});

        % Pencereyi kaydır: en eski timestep'i at, yeni tahmini ekle
        newStep           = window(:, end);        % son timestep'in tüm özellikleri
        newStep(closeIdx) = predNorm(step);         % yalnızca Close'u güncelle
        window            = [window(:, 2:end), newStep];
    end

    predictions = predNorm * (normParams.maxVal - normParams.minVal) + normParams.minVal;

    % Güven bandı: hata sqrt(adım) ile büyür (AR modeli varsayımı)
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

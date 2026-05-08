function predictions = predictFuture(net, lastSequence, numSteps, normParams)
% PREDICTFUTURE - Eğitilmiş modelle gelecek N adım için tahmin yapar
%
% Girdi:
%   net          - Eğitilmiş LSTM ağı
%   lastSequence - Son bilinen normalize edilmiş dizi [1 x seqLen]
%   numSteps     - Kaç adım ilerisi tahmin edilecek
%   normParams   - struct: minVal, maxVal (denormalizasyon için)
%
% Çıktı:
%   predictions - Denormalize edilmiş tahmin vektörü [numSteps x 1]
%
% Örnek:
%   preds = predictFuture(net, XTest{end}, 30, normParams);

    seqLen   = numel(lastSequence);
    window   = lastSequence(:)';   % [1 x seqLen] satır vektörü

    predNorm = zeros(numSteps, 1);

    for step = 1:numSteps
        % predict için cell array formatı
        inputSeq    = {window};
        predNorm(step) = predict(net, inputSeq);

        % Pencereyi kaydır: ilk elemanı at, yeni tahmini sona ekle
        window = [window(2:end), predNorm(step)];
    end

    % Denormalizasyon
    predictions = predNorm * (normParams.maxVal - normParams.minVal) + normParams.minVal;
end

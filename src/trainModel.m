function net = trainModel(layers, XTrain, YTrain, symbol, customOptions)
% TRAINMODEL - LSTM modelini eğitir ve diske kaydeder
%
% Girdi:
%   layers        - buildLSTMModel'den gelen katman dizisi
%   XTrain        - cell array eğitim dizileri
%   YTrain        - eğitim etiketleri [1 x N]
%   symbol        - Coin sembolü (kayıt dosya adı için)
%   customOptions - (Opsiyonel) trainingOptions nesnesi
%
% Çıktı:
%   net - Eğitilmiş SeriesNetwork nesnesi
%
% Örnek:
%   net = trainModel(layers, XTrain, YTrain, 'BTCUSDT');

    if nargin < 4 || isempty(symbol)
        symbol = 'COIN';
    end

    rng(42);  % tekrarlanabilirlik için

    if nargin < 5 || isempty(customOptions)
        options = trainingOptions('adam', ...
            'MaxEpochs',          100, ...
            'MiniBatchSize',      32, ...
            'InitialLearnRate',   0.005, ...
            'LearnRateSchedule',  'piecewise', ...
            'LearnRateDropPeriod', 50, ...
            'LearnRateDropFactor', 0.2, ...
            'GradientThreshold',  1, ...
            'Shuffle',            'never', ...
            'ValidationData',     [], ...
            'Plots',              'training-progress', ...
            'Verbose',            1);
    else
        options = customOptions;
    end

    fprintf('  Eğitim başlıyor... (MaxEpochs=%d, BatchSize=%d)\n', ...
            options.MaxEpochs, options.MiniBatchSize);

    net = trainNetwork(XTrain, YTrain, layers, options);

    % Modeli kaydet
    if ~isfolder('models')
        mkdir('models');
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    savePath  = fullfile('models', sprintf('%s_lstm_%s.mat', symbol, timestamp));
    save(savePath, 'net');
    fprintf('  Model kaydedildi: %s\n', savePath);
end

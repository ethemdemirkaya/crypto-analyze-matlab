function net = trainModel(layers, XTrain, YTrain, symbol, customOptions, valRatio)
% TRAINMODEL - LSTM modelini eğitir, validation split uygular ve diske kaydeder
%
% Girdi:
%   layers        - buildLSTMModel'den gelen katman dizisi
%   XTrain        - cell array eğitim dizileri
%   YTrain        - eğitim etiketleri [1 x N]
%   symbol        - Coin sembolü (kayıt dosya adı için, varsayılan 'COIN')
%   customOptions - (Opsiyonel) trainingOptions nesnesi; boş bırakılırsa varsayılan kullanılır
%   valRatio      - (Opsiyonel) validation oranı (0-1 arası, varsayılan 0.1)
%
% Çıktı:
%   net - Eğitilmiş SeriesNetwork nesnesi
%
% Örnek:
%   net = trainModel(layers, XTrain, YTrain, 'BTCUSDT');
%   net = trainModel(layers, XTrain, YTrain, 'BTCUSDT', [], 0.1);

    if nargin < 4 || isempty(symbol)
        symbol = 'COIN';
    end
    if nargin < 6 || isempty(valRatio)
        valRatio = 0.1;
    end

    rng(42);

    % --- Validation split (kronolojik sıra korunur) ---
    nTotal = numel(XTrain);
    nVal   = floor(valRatio * nTotal);
    nTr    = nTotal - nVal;

    XVal   = XTrain(nTr+1:end);
    YVal   = YTrain(nTr+1:end);
    YVal   = YVal(:);
    XTr    = XTrain(1:nTr);
    YTr    = YTrain(1:nTr);
    YTr    = YTr(:);

    fprintf('  Eğitim: %d, Validation: %d örnek\n', numel(XTr), numel(XVal));

    if nargin < 5 || isempty(customOptions)
        options = trainingOptions('adam', ...
            'MaxEpochs',            100, ...
            'MiniBatchSize',        32, ...
            'InitialLearnRate',     0.005, ...
            'LearnRateSchedule',    'piecewise', ...
            'LearnRateDropPeriod',  50, ...
            'LearnRateDropFactor',  0.2, ...
            'GradientThreshold',    1, ...
            'Shuffle',              'never', ...
            'ValidationData',       {XVal, YVal}, ...
            'ValidationFrequency',  50, ...
            'ValidationPatience',   10, ...
            'Plots',                'training-progress', ...
            'Verbose',              1);
    else
        options = customOptions;
    end

    fprintf('  Eğitim başlıyor... (MaxEpochs=%d, BatchSize=%d)\n', ...
            options.MaxEpochs, options.MiniBatchSize);

    net = trainNetwork(XTr, YTr, layers, options);

    % Model kaydet
    if ~isfolder('models')
        mkdir('models');
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    savePath  = fullfile('models', sprintf('%s_lstm_%s.mat', symbol, timestamp));
    save(savePath, 'net');
    fprintf('  Model kaydedildi: %s\n', savePath);
end

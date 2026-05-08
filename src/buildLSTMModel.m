function layers = buildLSTMModel(numFeatures, numHiddenUnits, useBiLSTM)
% BUILDLSTMMODEL - LSTM sinir ağı mimarisini oluşturur
%
% Girdi:
%   numFeatures    - Girdi özellik sayısı (tek fiyat için 1)
%   numHiddenUnits - İlk LSTM/BiLSTM katmanındaki birim sayısı (varsayılan 128)
%   useBiLSTM      - (Opsiyonel) true: BiLSTM ilk katman, varsayılan false
%
% Çıktı:
%   layers - MATLAB katman dizisi (trainNetwork'e verilecek)
%
% Örnek:
%   layers = buildLSTMModel(7, 128, true);   % çok özellikli BiLSTM
%   layers = buildLSTMModel(1, 100);         % tek özellikli LSTM

    if nargin < 2, numHiddenUnits = 128;  end
    if nargin < 3, useBiLSTM     = false; end

    if useBiLSTM
        firstLayer = bilstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm1');
        typeStr    = sprintf('BiLSTM(%d)', numHiddenUnits);
    else
        firstLayer = lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm1');
        typeStr    = sprintf('LSTM(%d)', numHiddenUnits);
    end

    layers = [
        sequenceInputLayer(numFeatures, 'Name', 'input')

        firstLayer
        dropoutLayer(0.2, 'Name', 'drop1')

        lstmLayer(floor(numHiddenUnits / 2), 'OutputMode', 'last', 'Name', 'lstm2')
        dropoutLayer(0.2, 'Name', 'drop2')

        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(1,  'Name', 'fc2')

        regressionLayer('Name', 'output')
    ];

    fprintf('  Mimari: input(%d) → %s → Drop(0.2) → LSTM(%d) → Drop(0.2) → FC(64) → ReLU → FC(1)\n', ...
            numFeatures, typeStr, floor(numHiddenUnits/2));
end

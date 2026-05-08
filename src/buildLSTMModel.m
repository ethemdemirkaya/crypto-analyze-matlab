function layers = buildLSTMModel(numFeatures, numHiddenUnits)
% BUILDLSTMMODEL - LSTM sinir ağı mimarisini oluşturur
%
% Girdi:
%   numFeatures    - Girdi özellik sayısı (tek fiyat için 1)
%   numHiddenUnits - İlk LSTM katmanındaki birim sayısı (varsayılan 100)
%
% Çıktı:
%   layers - MATLAB katman dizisi (trainNetwork'e verilecek)
%
% Örnek:
%   layers = buildLSTMModel(1, 100);

    if nargin < 2
        numHiddenUnits = 100;
    end

    layers = [
        sequenceInputLayer(numFeatures, 'Name', 'input')

        lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm1')
        dropoutLayer(0.2, 'Name', 'drop1')

        lstmLayer(floor(numHiddenUnits / 2), 'OutputMode', 'last', 'Name', 'lstm2')
        dropoutLayer(0.2, 'Name', 'drop2')

        fullyConnectedLayer(25, 'Name', 'fc1')
        fullyConnectedLayer(1,  'Name', 'fc2')

        regressionLayer('Name', 'output')
    ];

    fprintf('  LSTM mimarisi: input(%d) -> LSTM(%d) -> Dropout(0.2) -> LSTM(%d) -> Dropout(0.2) -> FC(25) -> FC(1)\n', ...
            numFeatures, numHiddenUnits, floor(numHiddenUnits/2));
end

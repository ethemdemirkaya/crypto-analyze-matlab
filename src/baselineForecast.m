function [metrics, YPred] = baselineForecast(YTrue, method)
% BASELINEFORECAST - Naive baseline tahmin yöntemleri (LSTM karşılaştırması için)
%
% Girdi:
%   YTrue  - Gerçek fiyat değerleri vektörü (denormalize edilmiş)
%   method - 'naive'   : önceki günün fiyatını tahmin et (varsayılan)
%            'mean'    : train setinin ortalamasını sabit tahmin et
%            'drift'   : lineer trend uzatma
%
% Çıktı:
%   metrics - struct: RMSE, MAE, MAPE, R2, DirectionalAccuracy
%   YPred   - Tahmin vektörü (YTrue ile aynı boyut)
%
% Örnek:
%   [m, yp] = baselineForecast(YTestDenorm, 'naive');

    if nargin < 2
        method = 'naive';
    end

    YTrue = YTrue(:);
    n     = numel(YTrue);

    switch lower(method)
        case 'naive'
            % Her t için tahmin = t-1 değeri
            YPred = [YTrue(1); YTrue(1:end-1)];

        case 'mean'
            % Sabit ortalama tahmini
            YPred = repmat(mean(YTrue), n, 1);

        case 'drift'
            % İlk ve son değer arasındaki lineer eğimi uzat
            slope = (YTrue(end) - YTrue(1)) / (n - 1);
            YPred = YTrue(1) + slope * (0:n-1)';

        otherwise
            error('baselineForecast:unknownMethod', ...
                  'Bilinmeyen yöntem: %s. naive | mean | drift kullanın.', method);
    end

    metrics = evaluateModel(YTrue, YPred);
    fprintf('  Baseline (%s): RMSE=%.2f | MAE=%.2f | MAPE=%.4f%% | R²=%.4f\n', ...
            method, metrics.RMSE, metrics.MAE, metrics.MAPE, metrics.R2);
end

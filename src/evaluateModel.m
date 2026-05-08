function metrics = evaluateModel(YTrue, YPred, savePath)
% EVALUATEMODEL - Regresyon performans metriklerini hesaplar
%
% Girdi:
%   YTrue    - Gerçek değerler vektörü
%   YPred    - Tahmin değerleri vektörü
%   savePath - (Opsiyonel) metriklerin yazılacağı dosya yolu
%
% Çıktı:
%   metrics - struct: RMSE, MAE, MAPE, R2, DirectionalAccuracy
%
% Örnek:
%   m = evaluateModel(YTestDenorm, YPredDenorm, 'results/metrics.txt');

    YTrue = YTrue(:);
    YPred = YPred(:);

    n = numel(YTrue);

    % RMSE
    metrics.RMSE = sqrt(mean((YTrue - YPred).^2));

    % MAE
    metrics.MAE = mean(abs(YTrue - YPred));

    % MAPE (sıfıra bölünmeyi önle)
    nonzero = YTrue ~= 0;
    metrics.MAPE = mean(abs((YTrue(nonzero) - YPred(nonzero)) ./ YTrue(nonzero))) * 100;

    % R² (determinasyon katsayısı)
    ssTot = sum((YTrue - mean(YTrue)).^2);
    ssRes = sum((YTrue - YPred).^2);
    metrics.R2 = 1 - ssRes / ssTot;

    % Directional Accuracy: yön (yukarı/aşağı) ne kadar doğru tahmin edildi
    if n > 1
        trueDir = sign(diff(YTrue));
        predDir = sign(diff(YPred));
        metrics.DirectionalAccuracy = mean(trueDir == predDir) * 100;
    else
        metrics.DirectionalAccuracy = NaN;
    end

    % Konsola yazdır
    fprintf('\n========== MODEL PERFORMANS METRİKLERİ ==========\n');
    fprintf('  RMSE                : %.4f\n', metrics.RMSE);
    fprintf('  MAE                 : %.4f\n', metrics.MAE);
    fprintf('  MAPE                : %.4f%%\n', metrics.MAPE);
    fprintf('  R²                  : %.4f\n', metrics.R2);
    fprintf('  Yön Doğruluğu       : %.2f%%\n', metrics.DirectionalAccuracy);
    fprintf('==================================================\n\n');

    % Dosyaya kaydet
    if nargin >= 3 && ~isempty(savePath)
        dirPart = fileparts(savePath);
        if ~isempty(dirPart) && ~isfolder(dirPart)
            mkdir(dirPart);
        end
        fid = fopen(savePath, 'w');
        fprintf(fid, 'MODEL PERFORMANS METRİKLERİ\n');
        fprintf(fid, 'Tarih: %s\n\n', datestr(now));
        fprintf(fid, 'RMSE                : %.4f\n', metrics.RMSE);
        fprintf(fid, 'MAE                 : %.4f\n', metrics.MAE);
        fprintf(fid, 'MAPE                : %.4f%%\n', metrics.MAPE);
        fprintf(fid, 'R2                  : %.4f\n', metrics.R2);
        fprintf(fid, 'Yön Doğruluğu       : %.2f%%\n', metrics.DirectionalAccuracy);
        fclose(fid);
        fprintf('  Metrikler kaydedildi: %s\n', savePath);
    end
end

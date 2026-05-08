function values = denormalize(normValues, normParams)
% DENORMALIZE - Min-max normalize edilmiş değerleri orijinal ölçeğe çevirir
%
% Girdi:
%   normValues - Normalize edilmiş değerler
%   normParams - struct: minVal, maxVal
%
% Çıktı:
%   values - Orijinal ölçekteki değerler
%
% Örnek:
%   prices = denormalize(YPred, normParams);

    values = normValues * (normParams.maxVal - normParams.minVal) + normParams.minVal;
end

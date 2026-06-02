function values = denormalize(normValues, normParams)
% DENORMALIZE - Normalize edilmiş değerleri orijinal ölçeğe çevirir (Min-Max veya Z-Score)
%
% Girdi:
%   normValues - Normalize edilmiş değerler
%   normParams - struct: minVal, maxVal VEYA meanVal, stdVal
%
% Çıktı:
%   values - Orijinal ölçekteki değerler

    if isfield(normParams, 'method') && strcmpi(normParams.method, 'zscore')
        values = normValues * normParams.stdVal + normParams.meanVal;
    elseif isfield(normParams, 'meanVal') && isfield(normParams, 'stdVal')
        % Z-Score method explicitly detected by fields
        values = normValues * normParams.stdVal + normParams.meanVal;
    else
        % Default/Min-Max
        values = normValues * (normParams.maxVal - normParams.minVal) + normParams.minVal;
    end
end

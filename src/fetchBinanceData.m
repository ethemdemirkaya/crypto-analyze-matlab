function data = fetchBinanceData(symbol, interval, numCandles, forceRefresh)
% FETCHBINANCEDATA - Binance API'den geçmiş OHLCV kline verisi çeker
%
% Girdi:
%   symbol       - Kripto çifti (örn. 'BTCUSDT')
%   interval     - Mum aralığı ('1d', '4h', '1h' vb.)
%   numCandles   - Çekilecek mum sayısı (1000 üzeri için pagination)
%   forceRefresh - (Opsiyonel) true ise cache'i yoksay, varsayılan false
%
% Çıktı:
%   data - Tablo: OpenTime, Open, High, Low, Close, Volume, CloseTime
%
% Örnek:
%   data = fetchBinanceData('BTCUSDT', '1d', 1500);

    if nargin < 4
        forceRefresh = false;
    end

    BASE_URL    = 'https://api.binance.com/api/v3/klines';
    MAX_LIMIT   = 1000;
    CACHE_DIR   = 'data';

    % Cache dosyası
    cacheFile = fullfile(CACHE_DIR, sprintf('%s_%s.csv', symbol, interval));

    % Cache kontrolü
    if ~forceRefresh && isfile(cacheFile)
        fprintf('  [Cache] %s okunuyor...\n', cacheFile);
        data = readtable(cacheFile, 'TextType', 'string');
        data.OpenTime  = datetime(data.OpenTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
        data.CloseTime = datetime(data.CloseTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
        return;
    end

    % Pagination ile veri çekme
    allRows  = {};
    fetched  = 0;
    endTime  = [];   % ilk istekte boş (en güncel veriden başla)

    while fetched < numCandles
        batchSize = min(MAX_LIMIT, numCandles - fetched);

        params = weboptions('Timeout', 30, 'ContentType', 'json');
        urlStr = sprintf('%s?symbol=%s&interval=%s&limit=%d', ...
                         BASE_URL, symbol, interval, batchSize);
        if ~isempty(endTime)
            urlStr = sprintf('%s&endTime=%d', urlStr, endTime);
        end

        try
            raw = webread(urlStr, params);
        catch ME
            error('fetchBinanceData:networkError', ...
                  'API isteği başarısız: %s\nURL: %s', ME.message, urlStr);
        end

        if isempty(raw)
            break;
        end

        % JSON yanıtı hata mesajı mı kontrol et
        if isstruct(raw) && isfield(raw, 'code')
            error('fetchBinanceData:apiError', ...
                  'Binance API hatası %d: %s', raw.code, raw.msg);
        end

        % raw: cell array of cell arrays
        batch = raw;
        allRows = [batch; allRows];  %#ok<AGROW>  % kronolojik sıra için önüne ekle
        fetched = fetched + numel(batch);

        % Bir sonraki batch için endTime: mevcut batch'in ilk mumun açılış zamanı - 1 ms
        firstOpenTime = str2double(batch{1}{1});
        endTime = firstOpenTime - 1;

        if numel(batch) < batchSize
            break;  % API'nin döndüreceği daha veri yok
        end
    end

    if isempty(allRows)
        error('fetchBinanceData:noData', 'Veri çekilemedi: %s %s', symbol, interval);
    end

    % Tablo oluşturma
    n = numel(allRows);
    OpenTime  = NaT(n, 1);
    Open      = zeros(n, 1);
    High      = zeros(n, 1);
    Low       = zeros(n, 1);
    Close     = zeros(n, 1);
    Volume    = zeros(n, 1);
    CloseTime = NaT(n, 1);

    for i = 1:n
        row = allRows{i};
        OpenTime(i)  = datetime(str2double(row{1}) / 1000, ...
                                'ConvertFrom', 'posixtime', 'TimeZone', 'UTC');
        Open(i)      = str2double(row{2});
        High(i)      = str2double(row{3});
        Low(i)       = str2double(row{4});
        Close(i)     = str2double(row{5});
        Volume(i)    = str2double(row{6});
        CloseTime(i) = datetime(str2double(row{7}) / 1000, ...
                                'ConvertFrom', 'posixtime', 'TimeZone', 'UTC');
    end

    data = table(OpenTime, Open, High, Low, Close, Volume, CloseTime);

    % Chronolojik sıralama
    data = sortrows(data, 'OpenTime');

    % Cache kaydet
    if ~isfolder(CACHE_DIR)
        mkdir(CACHE_DIR);
    end
    writetable(data, cacheFile);
    fprintf('  [Cache] %d mum %s''e yazıldı.\n', height(data), cacheFile);
end

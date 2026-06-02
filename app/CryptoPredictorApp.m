function CryptoPredictorApp()
% CRYPTOPREDICTORAPP - LSTM Kripto Fiyat Tahmin Uygulaması (Gelişmiş Modern GUI)
%
% Kullanım:
%   CryptoPredictorApp()
%
% Gereksinim: MATLAB R2019b+ (uifigure desteği, Deep Learning Toolbox)

    % src/ klasörünü path'e ekle
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

    %% ---- Uygulama Durumu ----
    app.data       = [];
    app.net        = [];
    app.normParams = [];
    app.XTest      = {};
    app.YTest      = [];

    %% ---- Ana Pencere Tasarımı (Premium Tema) ----
    fig = uifigure('Name', 'LSTM & BiLSTM Kripto Fiyat Tahmin Paneli', ...
                   'Position', [100 100 1200 730], ...
                   'Color', [0.95 0.96 0.98], ...
                   'Resize', 'on');

    %% ---- Sol Panel: Sekmeli Kontroller ----
    tabGroup = uitabgroup(fig, 'Position', [15 45 285 670]);
    
    % Sekme 1: Veri & Model
    tabData = uitab(tabGroup, 'Title', 'Veri & Model Yapısı');
    
    % Sekme 2: Eğitim & Tahmin
    tabTrain = uitab(tabGroup, 'Title', 'Eğitim & Tahmin');

    %% ---- SEKME 1: VERİ & MODEL YAPISI İÇERİĞİ ----
    rowY = 600;
    rowH = 28;
    gap  = 12;

    function y = nextRowData(h)
        if nargin < 1, h = rowH; end
        rowY = rowY - h - gap;
        y = rowY;
    end

    % Başlık
    uilabel(tabData, 'Text', 'Veri Kaynağı & Zaman Serisi', ...
            'Position', [15 605 240 22], ...
            'FontWeight', 'bold', 'FontSize', 12, 'FontName', 'Segoe UI');

    % Coin seçim
    uilabel(tabData, 'Text', 'Kripto Çifti:', 'Position', [15 nextRowData() 120 22], 'FontName', 'Segoe UI');
    ddCoin = uidropdown(tabData, ...
        'Items', {'BTCUSDT','ETHUSDT','BNBUSDT','ADAUSDT','SOLUSDT'}, ...
        'Value', 'BTCUSDT', ...
        'Position', [15 nextRowData() 245 28], 'FontName', 'Segoe UI');

    % Interval
    uilabel(tabData, 'Text', 'Zaman Aralığı (Interval):', 'Position', [15 nextRowData() 180 22], 'FontName', 'Segoe UI');
    ddInterval = uidropdown(tabData, ...
        'Items', {'1d','4h','1h'}, ...
        'Value', '1d', ...
        'Position', [15 nextRowData() 245 28], 'FontName', 'Segoe UI');

    % Mum sayısı
    uilabel(tabData, 'Text', 'Çekilecek Mum Sayısı:', 'Position', [15 nextRowData() 180 22], 'FontName', 'Segoe UI');
    efCandles = uieditfield(tabData, 'numeric', ...
        'Value', 1500, 'Limits', [100 5000], ...
        'Position', [15 nextRowData() 245 28], 'FontName', 'Segoe UI');

    % Özellik Modu
    uilabel(tabData, 'Text', 'Özellik Seçimi (Features):', 'Position', [15 nextRowData() 180 22], 'FontName', 'Segoe UI');
    ddFeatMode = uidropdown(tabData, ...
        'Items', {'Tek Özellik (Kapanış Fiyatı)', 'Çok Özellikli (Teknik Göstergeler)'}, ...
        'ItemsData', {'single', 'multi'}, ...
        'Value', 'single', ...
        'Position', [15 nextRowData() 245 28], 'FontName', 'Segoe UI');

    % Normalizasyon Tipi
    uilabel(tabData, 'Text', 'Normalizasyon Tipi:', 'Position', [15 nextRowData() 180 22], 'FontName', 'Segoe UI');
    ddNormType = uidropdown(tabData, ...
        'Items', {'Z-Score (Tavsiye Edilen)', 'Min-Max (Ölçekli)'}, ...
        'ItemsData', {'zscore', 'minmax'}, ...
        'Value', 'zscore', ...
        'Position', [15 nextRowData() 245 28], 'FontName', 'Segoe UI');

    % Model Tipi
    uilabel(tabData, 'Text', 'Model Ağ Mimarisi:', 'Position', [15 nextRowData() 180 22], 'FontName', 'Segoe UI');
    ddModelType = uidropdown(tabData, ...
        'Items', {'LSTM Model (Hızlı)', 'BiLSTM Model (Yönlü Hafıza)'}, ...
        'ItemsData', {'lstm', 'bilstm'}, ...
        'Value', 'bilstm', ...
        'Position', [15 nextRowData() 245 28], 'FontName', 'Segoe UI');

    % Veriyi Çek Butonu
    btnFetch = uibutton(tabData, 'push', 'Text', 'Veriyi Çek & Analiz Et', ...
        'Position', [15 25 245 42], ...
        'BackgroundColor', [0.13 0.43 0.81], 'FontColor', 'white', ...
        'FontWeight', 'bold', 'FontSize', 12, 'FontName', 'Segoe UI', ...
        'ButtonPushedFcn', @(~,~) onFetch());


    %% ---- SEKME 2: EĞİTİM & TAHMİN İÇERİĞİ ----
    rowY2 = 610;
    function y = nextRowTrain(h)
        if nargin < 1, h = rowH; end
        rowY2 = rowY2 - h - gap;
        y = rowY2;
    end

    % Başlık
    uilabel(tabTrain, 'Text', 'Hiperparametreler', ...
            'Position', [15 615 240 22], ...
            'FontWeight', 'bold', 'FontSize', 12, 'FontName', 'Segoe UI');

    % Sequence length slider
    uilabel(tabTrain, 'Text', sprintf('Pencere Uzunluğu (Seq): %d', 60), ...
            'Position', [15 nextRowTrain() 245 22], 'Tag', 'lblSeq', 'FontName', 'Segoe UI');
    slSeq = uislider(tabTrain, ...
        'Limits', [10 120], 'Value', 60, ...
        'MajorTicks', [10 30 60 90 120], ...
        'Position', [20 nextRowTrain(20) 235 3]);
    slSeq.ValueChangedFcn = @(s,~) updateLabel(fig, 'lblSeq', ...
                                   sprintf('Pencere Uzunluğu (Seq): %d', round(s.Value)));

    % Hidden units slider
    uilabel(tabTrain, 'Text', sprintf('Gizli Birim (LSTM): %d', 128), ...
            'Position', [15 nextRowTrain() 245 22], 'Tag', 'lblHidden', 'FontName', 'Segoe UI');
    slHidden = uislider(tabTrain, ...
        'Limits', [32 256], 'Value', 128, ...
        'MajorTicks', [32 64 128 192 256], ...
        'Position', [20 nextRowTrain(20) 235 3]);
    slHidden.ValueChangedFcn = @(s,~) updateLabel(fig, 'lblHidden', ...
                                      sprintf('Gizli Birim (LSTM): %d', round(s.Value)));

    % Epoch sayısı
    uilabel(tabTrain, 'Text', 'Eğitim Epoch Sayısı:', 'Position', [15 nextRowTrain() 180 22], 'FontName', 'Segoe UI');
    efEpochs = uieditfield(tabTrain, 'numeric', ...
        'Value', 100, 'Limits', [5 1000], ...
        'Position', [15 nextRowTrain() 245 28], 'FontName', 'Segoe UI');

    % Öğrenme Oranı (Learning Rate)
    uilabel(tabTrain, 'Text', 'Öğrenme Oranı (Learn Rate):', 'Position', [15 nextRowTrain() 180 22], 'FontName', 'Segoe UI');
    efLR = uieditfield(tabTrain, 'numeric', ...
        'Value', 0.002, 'Limits', [0.0001 0.1], ...
        'Position', [15 nextRowTrain() 245 28], 'FontName', 'Segoe UI');

    % Batch Size
    uilabel(tabTrain, 'Text', 'Yığın Boyutu (Batch Size):', 'Position', [15 nextRowTrain() 180 22], 'FontName', 'Segoe UI');
    ddBatch = uidropdown(tabTrain, ...
        'Items', {'16', '32', '64', '128'}, ...
        'Value', '32', ...
        'Position', [15 nextRowTrain() 245 28], 'FontName', 'Segoe UI');

    % Validation Split
    uilabel(tabTrain, 'Text', 'Doğrulama Oranı (Validation):', 'Position', [15 nextRowTrain() 180 22], 'FontName', 'Segoe UI');
    efValSplit = uieditfield(tabTrain, 'numeric', ...
        'Value', 0.1, 'Limits', [0 0.5], ...
        'Position', [15 nextRowTrain() 245 28], 'FontName', 'Segoe UI');

    % Forecast günleri
    uilabel(tabTrain, 'Text', 'Gelecek Tahmini (Adım/Gün):', 'Position', [15 nextRowTrain() 180 22], 'FontName', 'Segoe UI');
    efForecast = uieditfield(tabTrain, 'numeric', ...
        'Value', 30, 'Limits', [1 180], ...
        'Position', [15 nextRowTrain() 245 28], 'FontName', 'Segoe UI');

    % Eğitim ve Tahmin Düğmeleri
    btnTrain = uibutton(tabTrain, 'push', 'Text', 'Modeli Eğit (Detaylı)', ...
        'Position', [15 65 245 35], ...
        'BackgroundColor', [0.18 0.65 0.35], 'FontColor', 'white', ...
        'FontWeight', 'bold', 'FontName', 'Segoe UI', ...
        'ButtonPushedFcn', @(~,~) onTrain());

    btnPredict = uibutton(tabTrain, 'push', 'Text', 'Geleceği Tahmin Et', ...
        'Position', [15 20 245 35], ...
        'BackgroundColor', [0.85 0.40 0.10], 'FontColor', 'white', ...
        'FontWeight', 'bold', 'FontName', 'Segoe UI', ...
        'ButtonPushedFcn', @(~,~) onPredict());


    %% ---- Merkez Panel: Grafik Alanı ----
    ax = uiaxes(fig, 'Position', [315 220 865 480], ...
                    'FontName', 'Segoe UI', 'FontSize', 10, ...
                    'BackgroundColor', [0.98 0.98 0.99]);
    title(ax, 'Kripto Para Analiz Paneli', 'FontSize', 12, 'FontWeight', 'bold');
    ax.XLabel.String = 'Tarih';
    ax.YLabel.String = 'Fiyat (USDT)';
    grid(ax, 'on');

    %% ---- Alt Panel: Performans Metrik Tablosu ----
    metricPanel = uipanel(fig, 'Title', ' Model Performans Karşılaştırma Sonuçları', ...
                          'Position', [315 45 865 165], ...
                          'FontSize', 10, 'FontWeight', 'bold', ...
                          'FontName', 'Segoe UI', 'BackgroundColor', 'white');

    tbl = uitable(metricPanel, ...
        'ColumnName', {'Metrik Adı','Değer (Eğitilen Model)','Durum / Anlamı'}, ...
        'Data', { ...
            'RMSE','—','Kök Ortalama Kare Hata (Düşük olması istenir)'; ...
            'MAE','—','Ortalama Mutlak Hata (Fiyat cinsinden sapma)'; ...
            'MAPE (%)','—','Ortalama Mutlak Yüzde Hata (Yüzdesel sapma)'; ...
            'R²','—','Determinasyon Katsayısı (1.0''e yakın olması istenir)'; ...
            'Yön Doğruluğu (%)','—','Fiyat yönü (yukarı/aşağı) tahmin başarısı' ...
        }, ...
        'ColumnWidth', {220 180 430}, ...
        'Position', [10 10 845 130], ...
        'RowName', {}, 'FontName', 'Segoe UI');

    %% ---- Status Bar (Durum Çubuğu) ----
    statusPanel = uipanel(fig, 'Position', [0 0 1200 35], ...
                          'BorderType', 'none', 'BackgroundColor', [0.15 0.15 0.17]);
    
    lblStatus = uilabel(statusPanel, ...
        'Text', ' HAZIR: Lütfen önce sol panelden "Veriyi Çek & Analiz Et" düğmesine basarak başlayın.', ...
        'Position', [15 7 1170 20], ...
        'FontColor', [0.9 0.9 0.92], 'FontWeight', 'bold', 'FontName', 'Segoe UI');

    %% ---- Callback Fonksiyonları ----

    function onFetch()
        setStatus('Kripto para verisi Binance API üzerinden çekiliyor...');
        try
            symbol   = ddCoin.Value;
            interval = ddInterval.Value;
            nCandles = round(efCandles.Value);
            
            % Veri çekme (forceRefresh=true yapılarak her seferinde güncel veri çekilmesi sağlanıyor)
            app.data = fetchBinanceData(symbol, interval, nCandles, true);
            n        = height(app.data);
            
            setStatus(sprintf('Veri başarıyla çekildi: %d mum yüklendi (%s ile %s arası)', n, ...
                              datestr(app.data.OpenTime(1), 'yyyy-mm-dd HH:MM'), ...
                              datestr(app.data.OpenTime(end), 'yyyy-mm-dd HH:MM')));
                          
            % Arayüz grafik alanını temizle ve güncel fiyatları çiz
            cla(ax);
            plot(ax, app.data.OpenTime, app.data.Close, 'Color', [0.13 0.43 0.81], 'LineWidth', 1.5);
            title(ax, sprintf('%s %s — Tarihsel Kapanış Fiyatı (USDT)', symbol, interval), ...
                  'FontName', 'Segoe UI', 'FontSize', 12, 'FontWeight', 'bold');
            grid(ax, 'on');
            legend(ax, 'Fiyat (USDT)', 'Location', 'best');
        catch ME
            setStatus(['VERİ HATASI: ' ME.message]);
            uialert(fig, ME.message, 'Veri Çekme Hatası');
        end
    end

    function onTrain()
        if isempty(app.data)
            uialert(fig, 'Eğitime başlamadan önce lütfen veriyi çekin.', 'Uyarı'); 
            return;
        end
        
        setStatus('Veri önişleme yapılıyor ve model eğitimi başlatılıyor...');
        drawnow;
        
        try
            seqLen       = round(slSeq.Value);
            hidden       = round(slHidden.Value);
            maxEp        = round(efEpochs.Value);
            symbol       = ddCoin.Value;
            featMode     = ddFeatMode.Value;
            normMethod   = ddNormType.Value;
            modelType    = ddModelType.Value;
            lr           = efLR.Value;
            batchSize    = str2double(ddBatch.Value);
            valSplit     = efValSplit.Value;
            useBiLSTM    = strcmpi(modelType, 'bilstm');

            % 1. Özellik Moduna Göre Ön İşleme
            if strcmpi(featMode, 'multi')
                setStatus('Teknik göstergeler hesaplanıyor...');
                dataFeat = computeIndicators(app.data);
                features = {'Close', 'EMA20', 'RSI14', 'MACD', 'MACDSignal', 'BBWidth', 'VolEMA10'};
                
                [XTrain, YTrain, XTest, YTest, normP] = ...
                    preprocessMultiFeature(dataFeat, features, seqLen, 0.8, normMethod);
                numFeatures = numel(features);
            else
                [XTrain, YTrain, XTest, YTest, normP] = ...
                    preprocessData(app.data.Close, seqLen, 0.8, normMethod);
                numFeatures = 1;
            end

            app.normParams = normP;
            app.XTest      = XTest;
            app.YTest      = YTest;

            % 2. Katman Yapısını Kur (Stacked LSTM Hata Düzeltmeli)
            layers = buildLSTMModel(numFeatures, hidden, useBiLSTM);
            
            % 3. Validation Set Hazırlığı
            if valSplit > 0
                nTotal = numel(XTrain);
                nVal   = floor(valSplit * nTotal);
                nTr    = nTotal - nVal;
                
                XVal   = XTrain(nTr+1:end);
                YVal   = YTrain(nTr+1:end);
                YVal   = YVal(:);
                XTr    = XTrain(1:nTr);
                YTr    = YTrain(1:nTr);
                YTr    = YTr(:);
                
                validationData = {XVal, YVal};
                validationFreq = max(1, floor(nTr/batchSize));
            else
                XTr = XTrain;
                YTr = YTrain(:);
                validationData = {};
                validationFreq = [];
            end

            % 4. Eğitim Seçenekleri (Detaylı Parametreler & Shuffle Aktif)
            if valSplit > 0
                opts = trainingOptions('adam', ...
                    'MaxEpochs',          maxEp, ...
                    'MiniBatchSize',      batchSize, ...
                    'InitialLearnRate',   lr, ...
                    'LearnRateSchedule',  'piecewise', ...
                    'LearnRateDropPeriod', round(maxEp/2), ...
                    'LearnRateDropFactor', 0.2, ...
                    'GradientThreshold',  1, ...
                    'Shuffle',            'every-epoch', ...
                    'ValidationData',     validationData, ...
                    'ValidationFrequency', validationFreq, ...
                    'Plots',              'training-progress', ...
                    'Verbose',            0);
            else
                opts = trainingOptions('adam', ...
                    'MaxEpochs',          maxEp, ...
                    'MiniBatchSize',      batchSize, ...
                    'InitialLearnRate',   lr, ...
                    'LearnRateSchedule',  'piecewise', ...
                    'LearnRateDropPeriod', round(maxEp/2), ...
                    'LearnRateDropFactor', 0.2, ...
                    'GradientThreshold',  1, ...
                    'Shuffle',            'every-epoch', ...
                    'Plots',              'training-progress', ...
                    'Verbose',            0);
            end

            setStatus(sprintf('Eğitim başladı: %s katmanı (%d unit), Epoch=%d, LR=%.4f...', ...
                              upper(modelType), hidden, maxEp, lr));
            drawnow;
            
            % Ağı eğit (MATLAB eğitim ekranı açılır)
            app.net = trainNetwork(XTr, YTr, layers, opts);
            
            setStatus('Eğitim başarıyla tamamlandı. Gelecek tahminleri görmek için "Geleceği Tahmin Et" düğmesine basın.');
        catch ME
            setStatus(['EĞİTİM HATASI: ' ME.message]);
            uialert(fig, ME.message, 'Model Eğitim Hatası');
        end
    end

    function onPredict()
        if isempty(app.net)
            uialert(fig, 'Tahmin yapabilmek için önce modeli eğitmeniz gerekmektedir.', 'Uyarı'); 
            return;
        end
        
        setStatus('Test seti üzerinde doğrulama ve gelecek tahminleri yapılıyor...');
        drawnow;
        
        try
            symbol      = ddCoin.Value;
            forecastN   = round(efForecast.Value);
            normP       = app.normParams;
            featMode    = ddFeatMode.Value;

            % Test seti tahminlerini al
            YPredNorm   = predict(app.net, app.XTest);
            
            % Close fiyata ait normalizasyon parametrelerini ayır
            if strcmpi(featMode, 'multi')
                closeIdx = find(strcmpi(normP.features, 'Close'));
                np_close.method = normP.method;
                if strcmpi(normP.method, 'minmax')
                    np_close.minVal = normP.minVals(closeIdx);
                    np_close.maxVal = normP.maxVals(closeIdx);
                else
                    np_close.meanVal = normP.meanVals(closeIdx);
                    np_close.stdVal  = normP.stdVals(closeIdx);
                end
            else
                np_close = normP;
                closeIdx = 1;
            end
            
            % Denormalizasyon
            YPredDenorm = denormalize(YPredNorm(:), np_close);
            YTestDenorm = denormalize(app.YTest(:), np_close);

            % Performans değerlendir
            m = evaluateModel(YTestDenorm, YPredDenorm);

            % Tabloyu güncelle
            tbl.Data = { ...
                'RMSE', sprintf('%.4f', m.RMSE), 'Hata boyutu (USDT)'; ...
                'MAE',  sprintf('%.4f', m.MAE),  'Ortalama sapma boyutu (USDT)'; ...
                'MAPE (%)', sprintf('%.4f%%', m.MAPE), 'Yüzdesel ortalama sapma'; ...
                'R²',   sprintf('%.4f', m.R2),   'Model açıklayıcılık gücü'; ...
                'Yön Doğruluğu (%)', sprintf('%.2f%%', m.DirectionalAccuracy), 'Fiyat yönü tahmin yüzdesi' ...
            };

            % Test grafiği çizimi
            n = numel(YTestDenorm);
            testStartIdx = height(app.data) - n + 1;
            testDates = app.data.OpenTime(testStartIdx : testStartIdx + n - 1);

            cla(ax);
            plot(ax, testDates, YTestDenorm, 'Color', [0.15 0.15 0.15], 'LineWidth', 1.5, 'DisplayName', 'Gerçek Fiyat');
            hold(ax, 'on');
            plot(ax, testDates, YPredDenorm, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Model Test Tahmini');

            % 2. Güvenilir Gelecek Tahmini ( predictFuture.m içindeki yeni dinamik gösterge güncellemeli yapı )
            [futurePrices, upperCI, lowerCI] = predictFuture( ...
                app.net, app.XTest{end}, forecastN, normP, m.RMSE, closeIdx);
            
            futureDates = app.data.OpenTime(end) + days(1:forecastN);
            futureDates = futureDates(:);
            
            % Gelecek Tahmin Çizgisi
            plot(ax, futureDates, futurePrices, 'Color', [0.18 0.65 0.35], 'LineStyle', '-', 'Marker', 'o', ...
                 'LineWidth', 2, 'MarkerSize', 5, 'DisplayName', 'Gelecek Tahmini');
             
            % Güven Sınırları Çizimi (Belirsizlik Bandı)
            if ~isempty(upperCI) && ~isempty(lowerCI)
                fill(ax, [futureDates; flipud(futureDates)], [upperCI; flipud(lowerCI)], ...
                     [0.18 0.65 0.35], 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'DisplayName', '95% Güven Bandı');
            end
            
            % Ayraç çizgisi
            xline(ax, app.data.OpenTime(end), 'k--', 'LineWidth', 1.2, 'DisplayName', 'Tahmin Başlangıcı');

            hold(ax, 'off');
            legend(ax, 'Location', 'best', 'FontName', 'Segoe UI');
            title(ax, sprintf('%s — Gerçek Fiyat vs Model Tahmini (%d Günlük Gelecek)', ...
                               symbol, forecastN), 'FontName', 'Segoe UI', 'FontSize', 12, 'FontWeight', 'bold');
            grid(ax, 'on');
            
            setStatus(sprintf('Tahminler başarıyla üretildi. Test Seti MAPE = %.4f%%, R² = %.4f', m.MAPE, m.R2));
        catch ME
            setStatus(['TAHMİN HATASI: ' ME.message]);
            uialert(fig, ME.message, 'Tahmin Hatası');
        end
    end

    function setStatus(msg)
        lblStatus.Text = [' [' datestr(now,'HH:MM:SS') ']  ' msg];
        drawnow;
    end
end

function updateLabel(fig, tag, newText)
    lbl = findobj(fig, 'Tag', tag);
    if ~isempty(lbl)
        lbl.Text = newText;
    end
end

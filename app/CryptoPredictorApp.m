function CryptoPredictorApp()
% CRYPTOPREDICTORAPP - LSTM Kripto Fiyat Tahmin Uygulaması (Programmatic GUI)
%
% Kullanım:
%   CryptoPredictorApp()
%
% Gereksinim: MATLAB R2019b+ (uifigure desteği)

    % src/ klasörünü path'e ekle
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

    %% ---- Uygulama durumu ----
    app.data       = [];
    app.net        = [];
    app.normParams = [];
    app.XTest      = {};
    app.YTest      = [];

    %% ---- Ana pencere ----
    fig = uifigure('Name', 'LSTM Kripto Fiyat Tahmin', ...
                   'Position', [100 100 1150 720], ...
                   'Resize', 'on');

    %% ---- Sol panel: Kontroller ----
    leftPanel = uipanel(fig, 'Title', 'Parametreler', ...
                        'Position', [10 10 260 700], ...
                        'FontSize', 11, 'FontWeight', 'bold');

    row = 680;
    rowH = 28;
    gap  = 6;

    function y = nextRow()
        row = row - rowH - gap;
        y   = row;
    end

    % Coin seçim
    uilabel(leftPanel, 'Text', 'Kripto Çifti:', 'Position', [10 nextRow() 120 22]);
    ddCoin = uidropdown(leftPanel, ...
        'Items', {'BTCUSDT','ETHUSDT','BNBUSDT','ADAUSDT','SOLUSDT'}, ...
        'Value', 'BTCUSDT', ...
        'Position', [10 nextRow() 230 28]);

    % Interval
    uilabel(leftPanel, 'Text', 'Zaman Aralığı:', 'Position', [10 nextRow() 120 22]);
    ddInterval = uidropdown(leftPanel, ...
        'Items', {'1d','4h','1h'}, ...
        'Value', '1d', ...
        'Position', [10 nextRow() 230 28]);

    % Mum sayısı
    uilabel(leftPanel, 'Text', 'Mum Sayısı:', 'Position', [10 nextRow() 120 22]);
    efCandles = uieditfield(leftPanel, 'numeric', ...
        'Value', 1500, 'Limits', [100 5000], ...
        'Position', [10 nextRow() 230 28]);

    % Sequence length slider
    uilabel(leftPanel, 'Text', sprintf('Pencere Uzunluğu: %d', 60), ...
            'Position', [10 nextRow() 230 22], 'Tag', 'lblSeq');
    slSeq = uislider(leftPanel, ...
        'Limits', [30 120], 'Value', 60, ...
        'MajorTicks', [30 60 90 120], ...
        'Position', [10 nextRow() 220 20]);
    slSeq.ValueChangedFcn = @(s,~) updateLabel(fig, 'lblSeq', ...
                                   sprintf('Pencere Uzunluğu: %d', round(s.Value)));

    % Hidden units slider
    uilabel(leftPanel, 'Text', sprintf('Gizli Birim: %d', 100), ...
            'Position', [10 nextRow() 230 22], 'Tag', 'lblHidden');
    slHidden = uislider(leftPanel, ...
        'Limits', [50 200], 'Value', 100, ...
        'MajorTicks', [50 100 150 200], ...
        'Position', [10 nextRow() 220 20]);
    slHidden.ValueChangedFcn = @(s,~) updateLabel(fig, 'lblHidden', ...
                                      sprintf('Gizli Birim: %d', round(s.Value)));

    % Epoch sayısı
    uilabel(leftPanel, 'Text', 'Epoch Sayısı:', 'Position', [10 nextRow() 120 22]);
    efEpochs = uieditfield(leftPanel, 'numeric', ...
        'Value', 100, 'Limits', [10 500], ...
        'Position', [10 nextRow() 230 28]);

    % Forecast günleri
    uilabel(leftPanel, 'Text', 'Gelecek Tahmin (gün):', 'Position', [10 nextRow() 200 22]);
    efForecast = uieditfield(leftPanel, 'numeric', ...
        'Value', 30, 'Limits', [1 180], ...
        'Position', [10 nextRow() 230 28]);

    nextRow();  % boşluk

    % Butonlar
    btnFetch = uibutton(leftPanel, 'push', 'Text', 'Veriyi Çek', ...
        'Position', [10 nextRow() 230 28], ...
        'BackgroundColor', [0.2 0.6 1], 'FontColor', 'white', 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(~,~) onFetch());

    btnTrain = uibutton(leftPanel, 'push', 'Text', 'Modeli Eğit', ...
        'Position', [10 nextRow() 230 28], ...
        'BackgroundColor', [0.2 0.7 0.3], 'FontColor', 'white', 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(~,~) onTrain());

    btnPredict = uibutton(leftPanel, 'push', 'Text', 'Tahmin Yap', ...
        'Position', [10 nextRow() 230 28], ...
        'BackgroundColor', [0.85 0.4 0.1], 'FontColor', 'white', 'FontWeight', 'bold', ...
        'ButtonPushedFcn', @(~,~) onPredict());

    %% ---- Merkez panel: Grafik ----
    ax = uiaxes(fig, 'Position', [285 220 850 470]);
    title(ax, 'Grafik burada görünecek');
    ax.XLabel.String = 'Tarih';
    ax.YLabel.String = 'Fiyat (USDT)';
    grid(ax, 'on');

    %% ---- Alt panel: Metrik tablosu ----
    metricPanel = uipanel(fig, 'Title', 'Performans Metrikleri', ...
                          'Position', [285 10 850 200], ...
                          'FontSize', 10);

    tbl = uitable(metricPanel, ...
        'ColumnName', {'Metrik','Değer'}, ...
        'Data', {'RMSE','—'; 'MAE','—'; 'MAPE (%)','—'; 'R²','—'; 'Yön Doğruluğu (%)','—'}, ...
        'ColumnWidth', {200 150}, ...
        'Position', [10 10 500 165], ...
        'RowName', {});

    %% ---- Status bar ----
    lblStatus = uilabel(fig, ...
        'Text', 'Hazır.', ...
        'Position', [10 2 1130 20], ...
        'FontColor', [0.3 0.3 0.3]);

    %% ---- Callback fonksiyonları ----

    function onFetch()
        setStatus('Veri çekiliyor...');
        try
            symbol   = ddCoin.Value;
            interval = ddInterval.Value;
            nCandles = round(efCandles.Value);
            app.data = fetchBinanceData(symbol, interval, nCandles, false);
            n        = height(app.data);
            setStatus(sprintf('Veri yüklendi: %d mum (%s — %s)', n, ...
                              datestr(app.data.OpenTime(1), 'yyyy-mm-dd'), ...
                              datestr(app.data.OpenTime(end), 'yyyy-mm-dd')));
            % Fiyat grafiğini göster
            cla(ax);
            plot(ax, app.data.OpenTime, app.data.Close, 'b-', 'LineWidth', 1);
            title(ax, sprintf('%s %s — Kapanış Fiyatı', symbol, interval));
        catch ME
            setStatus(['HATA: ' ME.message]);
            uialert(fig, ME.message, 'Veri Hatası');
        end
    end

    function onTrain()
        if isempty(app.data)
            uialert(fig, 'Önce veri çekin.', 'Uyarı'); return;
        end
        setStatus('Ön işleme ve eğitim başlıyor...');
        try
            seqLen  = round(slSeq.Value);
            hidden  = round(slHidden.Value);
            maxEp   = round(efEpochs.Value);
            symbol  = ddCoin.Value;

            [XTrain, YTrain, XTest, YTest, normP] = ...
                preprocessData(app.data.Close, seqLen, 0.8);

            app.normParams = normP;
            app.XTest      = XTest;
            app.YTest      = YTest;

            layers = buildLSTMModel(1, hidden);
            opts   = trainingOptions('adam', ...
                'MaxEpochs',          maxEp, ...
                'MiniBatchSize',      32, ...
                'InitialLearnRate',   0.005, ...
                'LearnRateSchedule',  'piecewise', ...
                'LearnRateDropPeriod', round(maxEp/2), ...
                'LearnRateDropFactor', 0.2, ...
                'GradientThreshold',  1, ...
                'Shuffle',            'never', ...
                'Plots',              'training-progress', ...
                'Verbose',            0);

            app.net = trainNetwork(XTrain, YTrain, layers, opts);
            setStatus('Eğitim tamamlandı. "Tahmin Yap" butonuna basın.');
        catch ME
            setStatus(['HATA: ' ME.message]);
            uialert(fig, ME.message, 'Eğitim Hatası');
        end
    end

    function onPredict()
        if isempty(app.net)
            uialert(fig, 'Önce modeli eğitin.', 'Uyarı'); return;
        end
        setStatus('Tahmin yapılıyor...');
        try
            symbol      = ddCoin.Value;
            forecastN   = round(efForecast.Value);
            normP       = app.normParams;

            YPredNorm   = predict(app.net, app.XTest);
            YPredDenorm = denormalize(YPredNorm(:), normP);
            YTestDenorm = denormalize(app.YTest(:), normP);

            m = evaluateModel(YTestDenorm, YPredDenorm);

            % Tablo güncelle
            tbl.Data = {'RMSE', sprintf('%.4f', m.RMSE);
                        'MAE',  sprintf('%.4f', m.MAE);
                        'MAPE (%)', sprintf('%.4f', m.MAPE);
                        'R²',   sprintf('%.4f', m.R2);
                        'Yön Doğruluğu (%)', sprintf('%.2f', m.DirectionalAccuracy)};

            % Test grafiği
            n = numel(YTestDenorm);
            testStartIdx = height(app.data) - n + 1;
            testDates = app.data.OpenTime(testStartIdx : testStartIdx + n - 1);

            cla(ax);
            plot(ax, testDates, YTestDenorm, 'b-', 'LineWidth', 1.5, ...
                 'DisplayName', 'Gerçek');
            hold(ax, 'on');
            plot(ax, testDates, YPredDenorm, 'r--', 'LineWidth', 1.5, ...
                 'DisplayName', 'LSTM Tahmini');

            % Gelecek tahmin
            futurePrices = predictFuture(app.net, app.XTest{end}, forecastN, normP);
            futureDates  = app.data.OpenTime(end) + days(1:forecastN);
            plot(ax, futureDates(:), futurePrices, 'g-o', ...
                 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Gelecek');
            xline(ax, app.data.OpenTime(end), 'k--');

            hold(ax, 'off');
            legend(ax, 'Location', 'best');
            title(ax, sprintf('%s — Gerçek vs LSTM Tahmini + %d Günlük Forecast', ...
                              symbol, forecastN));
            setStatus(sprintf('Tahmin tamamlandı. MAPE=%.4f%%, R²=%.4f', m.MAPE, m.R2));
        catch ME
            setStatus(['HATA: ' ME.message]);
            uialert(fig, ME.message, 'Tahmin Hatası');
        end
    end

    function setStatus(msg)
        lblStatus.Text = ['[' datestr(now,'HH:MM:SS') '] ' msg];
        drawnow;
    end
end

function updateLabel(fig, tag, newText)
    lbl = findobj(fig, 'Tag', tag);
    if ~isempty(lbl)
        lbl.Text = newText;
    end
end

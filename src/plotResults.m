function plotResults(YTrue, YPred, dates, symbol, savePath, futurePred, futureDates)
% PLOTRESULTS - Model sonuçlarını görselleştirir ve kaydeder
%
% Girdi:
%   YTrue       - Gerçek fiyat değerleri
%   YPred       - Tahmin fiyat değerleri
%   dates       - Tarih vektörü (datetime)
%   symbol      - Coin sembolü (başlık için)
%   savePath    - Grafiklerin kaydedileceği klasör
%   futurePred  - (Opsiyonel) Gelecek tahminleri
%   futureDates - (Opsiyonel) Gelecek tahmin tarihleri
%
% Örnek:
%   plotResults(YTestDenorm, YPredDenorm, dates, 'BTCUSDT', 'figures/');

    if nargin < 6
        futurePred  = [];
        futureDates = [];
    end

    if ~isfolder(savePath)
        mkdir(savePath);
    end

    YTrue = YTrue(:);
    YPred = YPred(:);

    % --- Grafik 1: Gerçek vs Tahmin ---
    fig1 = figure('Visible', 'off', 'Position', [100 100 1200 500]);
    plot(dates, YTrue, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Fiyat');
    hold on;
    plot(dates, YPred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Tahmin Fiyat');
    xlabel('Tarih');
    ylabel('Fiyat (USDT)');
    title(sprintf('%s — Gerçek vs LSTM Tahmini', symbol));
    legend('Location', 'best');
    grid on;
    saveFig(fig1, fullfile(savePath, sprintf('%s_actual_vs_predicted.png', symbol)));
    close(fig1);

    % --- Grafik 2: Artık (Residuals) ---
    fig2 = figure('Visible', 'off', 'Position', [100 100 1200 400]);
    residuals = YPred - YTrue;
    plot(dates, residuals, 'k-', 'LineWidth', 1);
    yline(0, 'r--', 'LineWidth', 1.5);
    xlabel('Tarih');
    ylabel('Hata (Tahmin − Gerçek)');
    title(sprintf('%s — Artık Hata Grafiği', symbol));
    grid on;
    saveFig(fig2, fullfile(savePath, sprintf('%s_residuals.png', symbol)));
    close(fig2);

    % --- Grafik 3: Scatter (Gerçek vs Tahmin korelasyonu) ---
    fig3 = figure('Visible', 'off', 'Position', [100 100 600 600]);
    scatter(YTrue, YPred, 15, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    mn = min([YTrue; YPred]); mx = max([YTrue; YPred]);
    plot([mn mx], [mn mx], 'r--', 'LineWidth', 2);
    xlabel('Gerçek Fiyat (USDT)');
    ylabel('Tahmin Fiyat (USDT)');
    title(sprintf('%s — Korelasyon Grafiği', symbol));
    grid on;
    saveFig(fig3, fullfile(savePath, sprintf('%s_scatter.png', symbol)));
    close(fig3);

    % --- Grafik 4: Gelecek Tahmini ---
    if ~isempty(futurePred) && ~isempty(futureDates)
        fig4 = figure('Visible', 'off', 'Position', [100 100 1400 500]);
        % Son 90 günlük gerçek veri
        showN = min(90, numel(YTrue));
        plot(dates(end-showN+1:end), YTrue(end-showN+1:end), ...
             'b-', 'LineWidth', 1.5, 'DisplayName', 'Gerçek (Son 90 Gün)');
        hold on;
        plot(futureDates, futurePred, 'r-o', 'LineWidth', 2, ...
             'MarkerSize', 4, 'DisplayName', 'Gelecek Tahmini');
        xline(dates(end), 'k--', 'Tahmin Başlangıcı', 'LineWidth', 1.5);
        xlabel('Tarih');
        ylabel('Fiyat (USDT)');
        title(sprintf('%s — %d Günlük Gelecek Tahmini', symbol, numel(futurePred)));
        legend('Location', 'best');
        grid on;
        saveFig(fig4, fullfile(savePath, sprintf('%s_forecast.png', symbol)));
        close(fig4);
    end

    fprintf('  Grafikler kaydedildi: %s\n', savePath);
end

% ---- Yardımcı: 300 DPI PNG kaydet ----
function saveFig(fig, filepath)
    print(fig, filepath, '-dpng', '-r300');
end

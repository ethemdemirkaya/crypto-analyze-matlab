function indData = computeIndicators(data)
% COMPUTEINDICATORS - OHLCV verisinden teknik göstergeler hesaplar
%
% Girdi:
%   data    - Tablo: fetchBinanceData çıktısı (OpenTime, Open, High, Low, Close, Volume)
%
% Çıktı:
%   indData - Orijinal tablo + EMA20, RSI14, MACD, MACDSignal, BBWidth, VolEMA10
%
% Örnek:
%   dataFeat = computeIndicators(data);

    close  = data.Close(:);
    volume = data.Volume(:);

    ema20    = calcEMA(close,  20);
    rsi14    = calcRSI(close,  14);
    ema12    = calcEMA(close,  12);
    ema26    = calcEMA(close,  26);
    macd     = ema12 - ema26;
    macdSig  = calcEMA(macd,    9);
    bbWidth  = calcBBWidth(close, 20);
    volEma10 = calcEMA(volume, 10);

    indData            = data;
    indData.EMA20      = ema20;
    indData.RSI14      = rsi14;
    indData.MACD       = macd;
    indData.MACDSignal = macdSig;
    indData.BBWidth    = bbWidth;
    indData.VolEMA10   = volEma10;

    fprintf('  Teknik göstergeler hesaplandı: EMA20, RSI14, MACD, MACDSignal, BBWidth, VolEMA10\n');
end

function y = calcEMA(x, period)
    alpha = 2 / (period + 1);
    y     = zeros(size(x));
    y(1)  = x(1);
    for i = 2:numel(x)
        y(i) = alpha * x(i) + (1 - alpha) * y(i-1);
    end
end

function rsi = calcRSI(close, period)
    N      = numel(close);
    delta  = diff(close);
    gains  = max(delta,  0);
    losses = max(-delta, 0);
    rsi    = zeros(N, 1);

    avgGain = mean(gains(1:period));
    avgLoss = mean(losses(1:period));

    rsi(1:period) = 50;

    if avgLoss < 1e-10
        rsi(period+1) = 100;
    else
        rsi(period+1) = 100 - 100 / (1 + avgGain / avgLoss);
    end

    for t = period+2 : N
        avgGain = (avgGain * (period-1) + gains(t-1))  / period;
        avgLoss = (avgLoss * (period-1) + losses(t-1)) / period;
        if avgLoss < 1e-10
            rsi(t) = 100;
        else
            rsi(t) = 100 - 100 / (1 + avgGain / avgLoss);
        end
    end
end

function bbw = calcBBWidth(close, period)
    N   = numel(close);
    bbw = zeros(N, 1);
    for t = period : N
        w   = close(t-period+1 : t);
        sma = mean(w);
        if sma > 0
            bbw(t) = 4 * std(w, 1) / sma;
        end
    end
    bbw(1:period-1) = bbw(period);
end

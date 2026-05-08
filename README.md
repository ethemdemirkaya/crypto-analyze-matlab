# Cryptocurrency Price Prediction with LSTM (Binance API)

![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-blue?logo=mathworks)
![License](https://img.shields.io/badge/License-MIT-green)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM-orange)

> **A MATLAB-based LSTM deep learning pipeline that fetches real-time historical data from Binance API and predicts cryptocurrency prices.**

---

## Features

- **Live data ingestion** via Binance Public REST API (no API key required)
- **Automatic pagination** for fetching more than 1000 candles
- **Local CSV cache** — avoids redundant API calls on re-runs
- **Min-max normalization** computed exclusively on train set (no data leakage)
- **Sliding-window sequence** creation for LSTM input
- **Two-layer LSTM** architecture with dropout regularization
- **Adam optimizer** with piecewise learning rate schedule
- **Multi-step recursive forecasting** (e.g. 30-day ahead)
- **Five evaluation metrics:** RMSE, MAE, MAPE, R², Directional Accuracy
- **Four publication-quality plots** saved as 300 DPI PNGs
- **Multi-coin batch runner** (`runAllCoins.m`) for BTC, ETH, BNB comparison

---

## Requirements

| Requirement | Version |
|-------------|---------|
| MATLAB | R2023a or later |
| Deep Learning Toolbox | required |
| Statistics & Machine Learning Toolbox | required |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/crypto-analyze-matlab.git
cd crypto-analyze-matlab
```

Open MATLAB and add the source folder to your path:

```matlab
addpath(genpath('src'))
```

---

## Usage

### Quick Start — Single Coin

```matlab
% Edit config inside main.m then run:
main
```

Key configuration parameters in `main.m`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config.symbol` | `'BTCUSDT'` | Trading pair |
| `config.interval` | `'1d'` | Candle interval |
| `config.numCandles` | `1500` | Historical candles to fetch |
| `config.sequenceLength` | `60` | Input window (days) |
| `config.trainRatio` | `0.8` | Train/test split ratio |
| `config.numHiddenUnits` | `100` | LSTM hidden units |
| `config.forecastDays` | `30` | Future forecast horizon |

### Multi-Coin Batch Run

```matlab
runAllCoins
```

Trains models for BTC, ETH, BNB and produces a comparison table in `results/comparison.txt`.

---

## Project Structure

```
crypto-analyze-matlab/
├── main.m                         # Entry point — single coin
├── runAllCoins.m                  # Batch runner — BTC, ETH, BNB
├── src/
│   ├── fetchBinanceData.m         # Binance API data fetching + CSV cache
│   ├── preprocessData.m           # Normalization + sliding window + split
│   ├── buildLSTMModel.m           # LSTM architecture
│   ├── trainModel.m               # Training + model save
│   ├── predictFuture.m            # Recursive multi-step forecasting
│   ├── evaluateModel.m            # RMSE, MAE, MAPE, R², Dir. Accuracy
│   ├── plotResults.m              # 4 plots → 300 DPI PNG
│   └── denormalize.m              # Min-max inverse transform
├── data/                          # CSV cache (git-ignored)
├── models/                        # Saved .mat models (git-ignored)
├── figures/                       # Output plots (git-ignored)
├── results/                       # Metric text files (git-ignored)
├── docs/
│   ├── report.md                  # Academic report (Turkish)
│   └── architecture.md            # Architecture decisions
└── LICENSE
```

---

## Model Architecture

```
Input (sequence length × 1 features)
        │
  ┌─────▼──────┐
  │  LSTM(100) │  ← learns long-term dependencies
  └─────┬──────┘
        │ Dropout(0.2)
  ┌─────▼──────┐
  │  LSTM(50)  │  ← higher-level pattern abstraction
  └─────┬──────┘
        │ Dropout(0.2)
  ┌─────▼──────┐
  │   FC(25)   │
  └─────┬──────┘
  ┌─────▼──────┐
  │   FC(1)    │  ← scalar price prediction
  └─────┬──────┘
  Regression Layer (MSE loss)
```

**Training:** Adam optimizer, 100 epochs, batch size 32, piecewise LR schedule (drop ×0.2 every 50 epochs), gradient clipping at 1, `Shuffle='never'` (critical for time series).

---

## Results (Example)

| Coin | RMSE | MAE | MAPE | R² | Dir. Acc. |
|------|------|-----|------|----|-----------|
| BTCUSDT | — | — | — | — | — |
| ETHUSDT | — | — | — | — | — |
| BNBUSDT | — | — | — | — | — |

*Run `main.m` or `runAllCoins.m` to populate with real values.*

---

## Generated Plots

1. **Actual vs Predicted** — overlay of real and predicted prices on the test set
2. **Residuals** — prediction error over time
3. **Scatter Correlation** — actual vs predicted scatter with identity line
4. **Future Forecast** — 30-day ahead prediction beyond the last known date

---

## Limitations & Disclaimer

- **Academic purpose only.** This project is developed for a university course in Computational Mathematics. It is **not** financial advice.
- Predictions are based solely on historical price patterns; fundamental events (regulations, exchange failures, etc.) are not modeled.
- Multi-step recursive forecasting accumulates error; reliability decreases beyond 7 days.
- Past performance does not guarantee future results.

---

## Future Work

- Multi-feature input (OHLCV + RSI + MACD)
- Attention mechanism and Transformer comparison
- Walk-forward cross-validation
- Early stopping based on validation loss
- Sentiment analysis integration (social media / news)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Ethem Demirkaya**  
Software Engineering, 3rd Year  
📧 ethemdemirkaya189@gmail.com

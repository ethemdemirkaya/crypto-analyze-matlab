# Cryptocurrency Price Prediction with LSTM (Binance API)

![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-blue?logo=mathworks)
![License](https://img.shields.io/badge/License-MIT-green)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM-orange)
![Türkçe](https://img.shields.io/badge/README-Türkçe-red?logo=googletranslate)

> **A MATLAB-based LSTM deep learning pipeline that fetches real-time historical data from Binance API and predicts cryptocurrency prices.**
>
> Türkçe dokümantasyon için: [README.tr.md](README.tr.md)

---

## Features

- **Live data ingestion** via Binance Public REST API (no API key required)
- **Automatic pagination** for fetching more than 1000 candles
- **Local CSV cache** — avoids redundant API calls on re-runs
- **Min-max normalization** computed exclusively on train set (no data leakage)
- **Sliding-window sequence** creation for LSTM input
- **Two-layer LSTM** architecture with dropout regularization
- **Adam optimizer** with piecewise learning rate schedule + early stopping (validation patience 10)
- **Multi-step recursive forecasting** (e.g. 30-day ahead)
- **Five evaluation metrics:** RMSE, MAE, MAPE, R², Directional Accuracy
- **Four publication-quality plots** saved as 300 DPI PNGs
- **Naive / Drift / Mean baseline comparison** (`runBaseline.m`)
- **Multi-feature OHLCV experiment** vs close-only (`runMultiFeature.m`)
- **Walk-forward cross-validation** (`src/walkForwardValidation.m`)
- **Multi-coin batch runner** (`runAllCoins.m`) for BTC, ETH, BNB comparison
- **Programmatic GUI** with uifigure (`app/CryptoPredictorApp.m`)

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
git clone https://github.com/ethemdemirkaya/crypto-analyze-matlab.git
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
runAllCoins   % trains BTC + ETH + BNB, prints comparison table
```

### Baseline Comparison

```matlab
runBaseline   % LSTM vs Naive, Drift, Mean — bar chart + table
```

### Multi-Feature Experiment (OHLCV vs Close-Only)

```matlab
runMultiFeature   % trains single-feature and 5-feature models, compares metrics
```

### Walk-Forward Cross-Validation

```matlab
results = walkForwardValidation(data.Close, 60, 5, 100);
```

### GUI Application

```matlab
addpath(genpath('src'))
CryptoPredictorApp()
```

Provides dropdown menus for coin/interval selection, sliders for sequence length and hidden units, one-click fetch/train/predict, embedded axes, and a live metrics table.

---

## Project Structure

```
crypto-analyze-matlab/
├── main.m                              # Entry point — single coin pipeline
├── runAllCoins.m                       # Batch trainer — BTC, ETH, BNB
├── runBaseline.m                       # LSTM vs naive/drift/mean baseline
├── runMultiFeature.m                   # OHLCV vs Close-only experiment
├── src/
│   ├── fetchBinanceData.m              # Binance API + CSV cache + pagination
│   ├── preprocessData.m               # Normalize + sliding window + split
│   ├── preprocessMultiFeature.m       # Multi-feature (OHLCV) preprocessing
│   ├── buildLSTMModel.m               # Parametric LSTM architecture
│   ├── trainModel.m                   # Train + validation split + model save
│   ├── predictFuture.m                # Recursive multi-step forecasting
│   ├── evaluateModel.m                # RMSE, MAE, MAPE, R², Directional Acc.
│   ├── plotResults.m                  # 4 plot types → 300 DPI PNG
│   ├── baselineForecast.m             # Naive / Drift / Mean baselines
│   ├── walkForwardValidation.m        # Walk-forward expanding-window CV
│   └── denormalize.m                  # Min-max inverse transform
├── app/
│   └── CryptoPredictorApp.m           # Programmatic uifigure GUI
├── data/                              # CSV cache (git-ignored)
├── models/                            # Saved .mat networks (git-ignored)
├── figures/                           # Output PNG plots (git-ignored)
├── results/                           # Metric text files (git-ignored)
├── docs/
│   ├── report.md                      # Academic report (Turkish, IEEE format)
│   └── architecture.md               # Design decisions & data-flow diagram
├── README.md                          # This file (English)
├── README.tr.md                       # Turkish documentation
└── LICENSE                            # MIT
```

---

## Model Architecture

```
Input (sequence length × numFeatures)
        │
  ┌─────▼──────┐
  │  LSTM(100) │  ← learns long-term temporal dependencies
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

**Training:** Adam optimizer · 100 epochs · batch 32 · piecewise LR (drop ×0.2 at epoch 50) · gradient clipping 1 · `Shuffle='never'` (critical for time series) · validation split 10% · early stopping patience 10.

---

## Generated Plots

| Plot | Description |
|------|-------------|
| `*_actual_vs_predicted.png` | Overlay of real and predicted prices on test set |
| `*_residuals.png` | Prediction error (Predicted − Actual) over time |
| `*_scatter.png` | Actual vs predicted scatter with identity line |
| `*_forecast.png` | N-day ahead recursive forecast beyond last date |

---

## Results (Example)

| Coin | RMSE | MAE | MAPE (%) | R² | Dir. Acc. (%) |
|------|------|-----|----------|----|---------------|
| BTCUSDT | — | — | — | — | — |
| ETHUSDT | — | — | — | — | — |
| BNBUSDT | — | — | — | — | — |

*Run `main.m` or `runAllCoins.m` and check `results/` for real values.*

---

## Limitations & Disclaimer

> **IMPORTANT:** This project is developed **for academic and educational purposes only** as part of a university Computational Mathematics course. The predictions generated by this model are **not financial advice** and must **not** be used to make real investment decisions.

- Predictions rely solely on historical price patterns — regulatory events, exchange failures, and black-swan events are not modeled.
- Recursive multi-step forecasting accumulates error; reliability degrades beyond 7 days.
- The model performs within the distribution of training data; extreme outlier events will cause large errors.
- Past performance does not guarantee future results.

---

## Future Work

- Multi-feature OHLCV + RSI + MACD + Bollinger Bands inputs
- Attention mechanism and Transformer architecture comparison
- Hyperparameter optimization (Bayesian / grid search)
- Sentiment analysis integration (social media / news feeds)
- Real-time streaming prediction dashboard

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Ethem Demirkaya**  
Software Engineering, 3rd Year  
📧 ethemdemirkaya189@gmail.com  
🔗 [github.com/ethemdemirkaya](https://github.com/ethemdemirkaya)

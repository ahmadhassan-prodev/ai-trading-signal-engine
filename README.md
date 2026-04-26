# AI Trading Signal Engine

**An experimental AI-powered trading signal generator for SOL/USDT using technical indicators, classical ML, and deep learning models.**

This repository implements an end-to-end pipeline for generating bullish trading signals on Solana (SOL/USDT) using 1-hour candlestick data. It combines traditional technical analysis with modern machine learning and deep learning models to predict significant upward price movements.

## Overview

The project fetches historical market data from Binance, engineers rich technical features, creates forward-looking labels for "significant upward moves," and trains multiple models to generate trading signals:

- **Classical ML**: Random Forest classifier
- **Deep Learning**: BiLSTM + Attention, GRU + Attention, and Transformer-based models

It serves as a practical demonstration of applying time-series feature engineering and sequence modeling to cryptocurrency trading signal generation.

## Repository Purpose

This is an **experimental / learning project** focused on building an AI trading signal engine. The goal is to explore how different machine learning architectures perform on predicting future price appreciation (≥3% within the next 12 hours) using engineered technical indicators.

## Key Features

- **Data Pipeline**: Automated fetching of SOL/USDT 1h data via CCXT
- **Feature Engineering**: 20+ technical indicators including EMAs, RSI, MACD, Bollinger Bands, ATR, volume z-score, market structure (HH/LL), and detailed candle features
- **Labeling Strategy**: Forward-looking binary label (`target_up`) based on reaching +3% from current close within the next 12 candles
- **Multiple Models**:
  - Random Forest (with hyperparameter tuning)
  - Bidirectional LSTM + Custom Attention
  - Bidirectional GRU + Custom Attention
  - Transformer Encoder with Positional Encoding
- **Evaluation**: Accuracy, classification report, and confusion matrix for all models
- **Class Imbalance Handling**: Weighted loss and adjusted decision thresholds

## Technologies Used

- **Language**: Python 3
- **Data Processing**: pandas, NumPy
- **Technical Analysis**: `ta` library
- **Machine Learning**: scikit-learn (RandomForestClassifier)
- **Deep Learning**: TensorFlow / Keras (LSTM, GRU, Transformer, custom Attention layer)
- **Data Fetching**: CCXT (Binance API)
- **Visualization**: Matplotlib (feature importance plots)

## Project Structure

```bash
ai-trading-signal-engine/
├── src/
│   ├── data/
│   │   └── data_fetch.py          # Fetches raw SOL/USDT 1h data
│   ├── features/
│   │   └── feature_engineering.py # Creates technical indicators & candle features
│   └── models/
│       ├── create_labels.py       # Generates target_up labels
│       ├── scaling.py             # Feature scaling for classical ML
│       ├── train_rf.py            # Basic Random Forest training
│       ├── train_rf_tuned.py      # Tuned Random Forest with probability threshold
│       ├── bilstm_model.py        # BiLSTM + Attention model
│       ├── gru_model.py           # BiGRU + Attention model
│       └── transformer_model.py   # Transformer-based sequence model
├── results/                       # Model evaluation outputs & plots
├── labeled_data.csv               # Final labeled dataset (after processing)
├── features.csv                   # Engineered features
├── scaled_features.csv            # Scaled version for classical ML
├── sol_1h_raw.csv                 # Raw OHLCV data
└── README.md
```

**How the Code Works**

The pipeline follows these steps:

- **Data Acquisition** (data_fetch.py)
  - Connects to Binance via CCXT
  - Downloads ~1100 days of 1-hour SOL/USDT candlestick data
- **Feature Engineering** (feature_engineering.py)
  - Computes EMA ratios and slopes
  - Adds momentum (RSI, MACD), volatility (BB width, ATR), and volume features
  - Extracts market structure (higher highs / lower lows)
  - Generates detailed candle statistics (body, wicks, ratios)
- **Label Creation** (create_labels.py)
  - Creates binary target: 1 if price reaches +3% within next 12 hours (from current close to future high)
- **Model Training**
  - **Random Forest**: Trains on scaled tabular features
  - **Sequence Models** (BiLSTM, BiGRU, Transformer): Use 24-hour lookback windows with custom attention mechanisms
- **Evaluation**
  - All models output accuracy, classification reports, and confusion matrices
  - Deep learning models use class weighting to handle imbalance

**Installation & Setup**

Bash

_\# Clone the repository_

git clone <https://github.com/ahmadhassan-prodev/ai-trading-signal-engine.git>

cd ai-trading-signal-engine

_\# Create and activate virtual environment (recommended)_

python -m venv venv

source venv/bin/activate _# On Windows: venv\\Scripts\\activate_

_\# Install dependencies_

pip install pandas numpy scikit-learn tensorflow ta ccxt matplotlib

**Usage**

Run the pipeline step by step:

**1\. Fetch Raw Data**

Bash

python src/data/data_fetch.py

**2\. Engineer Features**

Bash

python src/features/feature_engineering.py

**3\. Create Labels**

Bash

python src/models/create_labels.py

**4\. Scale Features (for Random Forest)**

Bash

python src/models/scaling.py

**5\. Train & Evaluate Models**

**Random Forest:**

Bash

python src/models/train_rf.py

_\# or tuned version:_

python src/models/train_rf_tuned.py

**Deep Learning Models:**

Bash

python src/models/bilstm_model.py

python src/models/gru_model.py

python src/models/transformer_model.py

Each script will print training progress, final test accuracy, classification report, and confusion matrix.

**Learning Outcomes**

By studying this repository, you can learn:

- End-to-end time series feature engineering for trading
- How to create forward-looking labels for supervised learning on price data
- Implementation of custom Attention layers in Keras
- Sequence modeling with LSTM, GRU, and Transformer architectures
- Handling class imbalance in financial prediction tasks
- Practical differences between classical ML and deep learning approaches for trading signals

**Notes**

- This is an **experimental educational project**. The models are not financial advice and have not been validated in live trading.
- Performance on the test set reflects historical backtesting only - past performance does not guarantee future results.
- Market conditions change; models may require retraining with recent data.
- Training deep learning models requires a machine with sufficient RAM/GPU for best performance.

**Future Improvements**

- Live signal generation & paper trading integration
- Ensemble model combining RF + deep learning predictions
- Hyperparameter optimization (Optuna / Keras Tuner)
- Walk-forward validation for more realistic time-series evaluation
- Additional data sources (on-chain metrics, sentiment)
- Backtesting framework with risk management

**Built as a learning project to explore AI applications in cryptocurrency trading signal generation.**

Feel free to explore, experiment, and extend the codebase

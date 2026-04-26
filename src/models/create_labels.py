import pandas as pd
import numpy as np

def create_label():
    # Load processed data
    df = pd.read_csv("features.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # =========================
    # PARAMETERS
    # =========================
    future_candles = 12 # e.g., next 6 hours
    threshold_pct = 3   # 3% rise

    # =========================
    # CREATE DIRECTION LABEL
    # =========================
    # Calculate the future high for next N candles
    df['future_max'] = df['high'].shift(-1).rolling(window=future_candles, min_periods=1).max()

    # Calculate % change from current close to future max
    df['future_return_pct'] = (df['future_max'] - df['close']) / df['close'] * 100

    # Label: 1 if price rises ≥ threshold_pct, else 0
    df['target_up'] = (df['future_return_pct'] >= threshold_pct).astype(int)

    # Drop helper columns
    df = df.drop(columns=['future_max', 'future_return_pct'])

    # Drop last N rows as they cannot have future data
    df = df[:-future_candles]

    # Save final dataset
    df.to_csv("labeled_data.csv")
    print("✅ Target column 'target_up' created and dataset saved to labeled_data.csv")
    print(df[['close', 'target_up']].tail(10))

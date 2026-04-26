import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features():
    # Load labeled data
    df = pd.read_csv("labeled_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    features = [
        'ema9', 'ema21', 'ema50', 'ema100',
        'ema_ratio_9_21', 'ema_ratio_21_50', 'ema_ratio_50_100',
        'ema_slope_9', 'ema_slope_21', 'ema_slope_50',
        'rsi', 'macd_diff', 'bb_width', 'atr',
        'vol_zscore', 'hh_ll',
        'candle_bullish', 'candle_body', 'upper_wick', 'lower_wick', 'wick_ratio'
    ]

    # Scale features
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    # Save scaled dataset for classical ML
    df_scaled.to_csv("scaled_features.csv")
    print("✅ Scaled dataset saved to scaled_features.csv")

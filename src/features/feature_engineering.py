import pandas as pd
import numpy as np
import ta

def create_features():
    # Load your data
    df = pd.read_csv("sol_1h_raw.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # =========================
    # TREND INDICATORS
    # =========================
    df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema100'] = ta.trend.EMAIndicator(df['close'], window=100).ema_indicator()

    # EMA ratios & slopes
    df['ema_ratio_9_21'] = df['ema9'] / df['ema21']
    df['ema_ratio_21_50'] = df['ema21'] / df['ema50']
    df['ema_ratio_50_100'] = df['ema50'] / df['ema100']
    df['ema_slope_9'] = df['ema9'].diff()
    df['ema_slope_21'] = df['ema21'].diff()
    df['ema_slope_50'] = df['ema50'].diff()

    # =========================
    # MOMENTUM INDICATORS
    # =========================
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd_diff'] = macd.macd_diff()

    # =========================
    # VOLATILITY INDICATORS
    # =========================
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']

    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    ).average_true_range()

    # =========================
    # VOLUME INDICATORS
    # =========================
    df['vol_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

    # =========================
    # MARKET STRUCTURE (Higher High / Lower Low)
    # =========================
    df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    df['hh_ll'] = np.where(df['higher_high'], 1, np.where(df['lower_low'], -1, 0))

    # =========================
    # CANDLE FEATURES
    # =========================
    df['candle_bullish'] = (df['close'] > df['open']).astype(int)      # 1 = bullish, 0 = bearish
    df['candle_body'] = abs(df['close'] - df['open'])                 # body size
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['wick_ratio'] = df['upper_wick'] / (df['lower_wick'] + 1e-6)  # avoid division by zero

    # =========================
    # CLEAN DATA
    # =========================
    df = df.dropna()

    # Save processed file
    df.to_csv("features.csv")
    print("✅ All indicators and candle features added to features.csv")
    print(df.tail())

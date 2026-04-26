import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    # =========================
    # LOAD SCALED DATA
    # =========================
    df = pd.read_csv("scaled_features.csv")
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
    target = 'target_up'

    X = df[features]
    y = df[target]

    # =========================
    # TRAIN-TEST SPLIT
    # =========================
    # For time series, avoid shuffling; use first 80% for train, last 20% for test
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # =========================
    # RANDOM FOREST CLASSIFIER
    # =========================
    rf_model = RandomForestClassifier(
        n_estimators=500,      # number of trees
        max_depth=10,          # max depth to prevent overfitting
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )

    # Train
    rf_model.fit(X_train, y_train)

    # =========================
    # PREDICTION & EVALUATION
    # =========================
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%\n")

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    # import matplotlib.pyplot as plt

    # importances = rf_model.feature_importances_
    # indices = importances.argsort()[::-1]

    # plt.figure(figsize=(12,6))
    # plt.title("Feature Importance - Random Forest")
    # plt.bar(range(len(features)), importances[indices], align='center')
    # plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    # plt.tight_layout()
    # plt.show()

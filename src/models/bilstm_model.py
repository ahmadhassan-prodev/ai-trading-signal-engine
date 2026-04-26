import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Layer
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# LOAD LABELED DATA
# =========================
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
target = 'target_up'

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
y = df[target].values

# =========================
# TRAIN-TEST SPLIT
# =========================
split_ratio = 0.8
split_index = int(len(df) * split_ratio)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# =========================
# CREATE SEQUENCES FOR LSTM/BiLSTM
# =========================
sequence_length = 24

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

print("Sequence shapes:")
print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("X_test_seq:", X_test_seq.shape, "y_test_seq:", y_test_seq.shape)

# =========================
# ATTENTION LAYER
# =========================
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# =========================
# BUILD MODEL
# =========================
num_features = X_train_seq.shape[2]

inputs = Input(shape=(sequence_length, num_features))
x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Dropout(0.3)(x)
x = Attention()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=30,
    batch_size=64,
    class_weight={0:1, 1:5},  # handle class imbalance
    verbose=2
)

# =========================
# EVALUATION
# =========================
loss, accuracy = model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")

# =========================
# PREDICTION EXAMPLE
# =========================
y_pred_prob = model.predict(X_test_seq)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test_seq, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test_seq, y_pred))

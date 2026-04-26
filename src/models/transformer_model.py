import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
from keras.models import Model

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
# CREATE SEQUENCES
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
# POSitional Encoding
# =========================
class PositionalEncoding(Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding = self.add_weight(
            name="position_emb",
            shape=(seq_len, d_model),
            initializer="uniform",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_embedding

# =========================
# TRANSFORMER BLOCK
# =========================
def transformer_block(x, num_heads, ff_dim, dropout_rate=0.2):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = Dense(ff_dim, activation='relu')(out1)
    ff_output = Dense(x.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff_output)
    return out2

# =========================
# BUILD MODEL
# =========================
num_features = X_train_seq.shape[2]
inputs = Input(shape=(sequence_length, num_features))
x = PositionalEncoding(sequence_length, num_features)(inputs)
x = transformer_block(x, num_heads=4, ff_dim=64)
x = transformer_block(x, num_heads=4, ff_dim=64)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
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
    epochs=20,          # laptop-friendly
    batch_size=64,
    class_weight={0:1, 1:5},  # handle class imbalance
    verbose=2
)

# =========================
# EVALUATION
# =========================
loss, accuracy = model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")

y_pred_prob = model.predict(X_test_seq)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test_seq, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test_seq, y_pred))

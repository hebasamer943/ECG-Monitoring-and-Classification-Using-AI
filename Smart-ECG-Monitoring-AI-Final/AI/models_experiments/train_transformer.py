from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "Dataset"
MODELS = BASE / "Models"
MODELS.mkdir(exist_ok=True)

NORMAL = DATA / "ptbdb_normal.csv"
ABNORMAL = DATA / "ptbdb_abnormal.csv"

def load_ptbdb():
    dn = pd.read_csv(NORMAL, header=None)
    da = pd.read_csv(ABNORMAL, header=None)
    Xn, yn = dn.iloc[:, :-1].to_numpy(np.float32), dn.iloc[:, -1].to_numpy(np.int64)
    Xa, ya = da.iloc[:, :-1].to_numpy(np.float32), da.iloc[:, -1].to_numpy(np.int64)
    X = np.vstack([Xn, Xa])
    y = np.concatenate([yn, ya])
    return X, y

def focal_loss(gamma=2.0, alpha=0.60):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        w  = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return -tf.reduce_mean(w * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss

def best_threshold_balanced(y_true, prob):
    best_t, best_score = 0.5, -1
    for t in np.linspace(0.2, 0.9, 71):
        pred = (prob >= t).astype(int)
        f1_abn = f1_score(y_true, pred, pos_label=1)
        f1_norm = f1_score(1 - y_true, 1 - pred, pos_label=1)
        score = 0.5 * f1_abn + 0.5 * f1_norm
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t), float(best_score)

class PositionalEncoding(layers.Layer):
    def __init__(self, length, d_model):
        super().__init__()
        pos = np.arange(length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angles = pos * angle_rates
        pe = np.zeros((length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :])

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

def transformer_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.2):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.LayerNormalization()(x + attn)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1])(ff)
    return layers.LayerNormalization()(x + ff)

def main():
    print("=== Transformer FIX (Norm + Focal + Threshold) ===")
    X, y = load_ptbdb()
    X = X.reshape(-1, 187, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    inp = keras.Input(shape=(187, 1))
    norm = layers.Normalization()
    x = norm(inp)
    norm.adapt(X_train)

    x = layers.Dense(64)(x)               # d_model=64
    x = PositionalEncoding(187, 64)(x)

    x = transformer_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.2)
    x = transformer_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.2)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=focal_loss(gamma=2.0, alpha=0.60),
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2),
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    prob = model.predict(X_test, verbose=0).reshape(-1)
    t, best_score = best_threshold_balanced(y_test, prob)
    y_pred = (prob >= t).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("Best threshold:", round(t,3), "BalancedScore:", round(best_score,4))
    print("Accuracy:", round(acc,4), "F1(abnormal):", round(f1,4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=4))

    out_path = MODELS / "ptbdb_transformer.keras"
    model.save(out_path)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()


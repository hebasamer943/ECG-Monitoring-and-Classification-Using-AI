from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import f1_score
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

def focal_loss(gamma=2.0, alpha=0.75):
    # alpha أعلى شوي لصالح abnormal (label=1)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        w = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return -tf.reduce_mean(w * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss

def res_block(x, filters, k=7, downsample=False, dropout=0.2):
    stride = 2 if downsample else 1
    shortcut = x

    x = layers.Conv1D(filters, k, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters, k, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet1d():
    inp = keras.Input(shape=(187, 1))
    norm = layers.Normalization()
    x = norm(inp)

    x = layers.Conv1D(64, 7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(2)(x)

    x = res_block(x, 64, k=7, downsample=False)
    x = res_block(x, 128, k=5, downsample=True)
    x = res_block(x, 128, k=5, downsample=False)
    x = res_block(x, 256, k=3, downsample=True)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out), norm



def best_threshold_balanced(y_true, prob):
    best_t, best_score = 0.5, -1
    for t in np.linspace(0.2, 0.9, 71):
        pred = (prob >= t).astype(int)
        f1_abn = f1_score(y_true, pred, pos_label=1)
        f1_norm = f1_score(1 - y_true, 1 - pred, pos_label=1)
        score = 0.5 * f1_abn + 0.5 * f1_norm  # توازن
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t), float(best_score)


def main():
    print("=== ResNet1D FIX (FocalLoss + Threshold tuning) ===")
    X, y = load_ptbdb()
    X = X.reshape(-1, 187, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, norm = build_resnet1d()
    norm.adapt(X_train)

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
    print("Best threshold:", round(t, 3), "BalancedScore:", round(best_score, 4))

    print("Accuracy:", round(acc, 4), "F1(abnormal):", round(f1, 4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=4))

    out = MODELS / "ptbdb_resnet1d_fix.keras"
    model.save(out)
    print("\nSaved:", out)

if __name__ == "__main__":
    main()

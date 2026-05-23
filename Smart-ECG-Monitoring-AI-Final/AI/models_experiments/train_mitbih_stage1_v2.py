from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class EvalResult:
    threshold: float
    accuracy: float
    precision_abnormal: float
    recall_abnormal: float
    f1_abnormal: float
    specificity_normal: float
    roc_auc: float
    tp: int
    tn: int
    fp: int
    fn: int
    confusion_matrix: list[list[int]]
    classification_report: dict


def load_mitbih(dataset_dir: Path):
    train_path = dataset_dir / "mitbih_train.csv"
    test_path = dataset_dir / "mitbih_test.csv"

    if not train_path.exists():
        raise FileNotFoundError("Missing Dataset/mitbih_train.csv")

    if not test_path.exists():
        raise FileNotFoundError("Missing Dataset/mitbih_test.csv")

    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)

    x_train_all = df_train.iloc[:, :187].to_numpy(dtype=np.float32)
    y_train_raw = df_train.iloc[:, 187].to_numpy(dtype=np.int64)

    x_test = df_test.iloc[:, :187].to_numpy(dtype=np.float32)
    y_test_raw = df_test.iloc[:, 187].to_numpy(dtype=np.int64)

    y_train_all = (y_train_raw != 0).astype(np.int64)
    y_test = (y_test_raw != 0).astype(np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train_all,
    )

    return x_train, y_train, x_val, y_val, x_test, y_test


def z_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def oversample_abnormal(x: np.ndarray, y: np.ndarray, target_ratio: float = 1.0):
    """
    target_ratio=1.0 means abnormal count will be close to normal count.
    """
    rng = np.random.default_rng(SEED)

    idx_normal = np.where(y == 0)[0]
    idx_abnormal = np.where(y == 1)[0]

    n_normal = len(idx_normal)
    n_abnormal = len(idx_abnormal)

    target_abnormal = int(n_normal * target_ratio)

    if n_abnormal >= target_abnormal:
        idx_final = np.concatenate([idx_normal, idx_abnormal])
    else:
        extra = rng.choice(
            idx_abnormal,
            size=target_abnormal - n_abnormal,
            replace=True,
        )
        idx_final = np.concatenate([idx_normal, idx_abnormal, extra])

    rng.shuffle(idx_final)

    return x[idx_final], y[idx_final]


def binary_focal_loss(alpha_abnormal: float = 3.0, gamma: float = 2.0):
    """
    Higher alpha_abnormal pushes the model to care more about abnormal beats.
    """
    alpha = tf.constant(alpha_abnormal, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.clip_by_value(tf.reshape(y_pred, [-1]), 1e-7, 1.0 - 1e-7)

        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0)

        ce = -tf.math.log(pt)
        fl = alpha_t * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(fl)

    return loss


def se_block(x, ratio: int = 8):
    channels = int(x.shape[-1])
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(channels // ratio, 8), activation="relu")(s)
    s = layers.Dense(channels, activation="sigmoid")(s)
    s = layers.Reshape((1, channels))(s)
    return layers.Multiply()([x, s])


def res_block(x, filters: int, kernel_size: int, dropout: float):
    shortcut = x

    if int(x.shape[-1]) != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(filters, kernel_size, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.SpatialDropout1D(dropout)(y)

    y = layers.Conv1D(filters, kernel_size, padding="same")(y)
    y = layers.BatchNormalization()(y)
    y = se_block(y)

    y = layers.Add()([shortcut, y])
    y = layers.Activation("relu")(y)

    return y


def build_model():
    inp = keras.Input(shape=(187, 1))

    x = layers.Conv1D(32, 7, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = res_block(x, 32, 7, 0.10)
    x = res_block(x, 64, 5, 0.15)
    x = layers.MaxPool1D(2)(x)

    x = res_block(x, 96, 5, 0.20)
    x = res_block(x, 128, 3, 0.25)
    x = layers.MaxPool1D(2)(x)

    x = res_block(x, 192, 3, 0.25)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=binary_focal_loss(alpha_abnormal=3.0, gamma=2.0),
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def augment_batch(x, y):
    batch = tf.shape(x)[0]

    # light gaussian noise
    x = x + tf.random.normal(tf.shape(x), stddev=0.015, dtype=x.dtype)

    # amplitude scale
    scale = tf.random.uniform((batch, 1, 1), 0.95, 1.05, dtype=x.dtype)
    x = x * scale

    # small shift
    shift = tf.random.uniform((batch,), minval=-3, maxval=4, dtype=tf.int32)

    x = tf.map_fn(
        lambda item: tf.roll(item[0], shift=item[1], axis=0),
        (x, shift),
        fn_output_signature=tf.float32,
    )

    return x, y


def make_ds(x, y, batch_size: int, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if training:
        ds = ds.shuffle(min(len(y), 20000), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    if training:
        ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def evaluate_threshold(y_true, y_prob, threshold: float) -> EvalResult:
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    return EvalResult(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_abnormal=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_abnormal=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_abnormal=float(f1_score(y_true, y_pred, zero_division=0)),
        specificity_normal=float(specificity),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
    )


def find_best_threshold(y_true, y_prob):
    results = []

    for thr in np.arange(0.10, 0.91, 0.02):
        r = evaluate_threshold(y_true, y_prob, float(thr))
        results.append(r)

    # We prefer a useful screening model:
    # recall at least 0.65, then highest F1.
    candidates = [r for r in results if r.recall_abnormal >= 0.65]

    if candidates:
        best = max(candidates, key=lambda r: r.f1_abnormal)
    else:
        best = max(results, key=lambda r: r.f1_abnormal)

    return best, results


def main():
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"
    reports_dir = project_root / "Reports"

    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    print("Loading MIT-BIH...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_mitbih(dataset_dir)

    print("Before oversampling:")
    print("Train normal:", int((y_train == 0).sum()), "abnormal:", int((y_train == 1).sum()))
    print("Val   normal:", int((y_val == 0).sum()), "abnormal:", int((y_val == 1).sum()))
    print("Test  normal:", int((y_test == 0).sum()), "abnormal:", int((y_test == 1).sum()))

    x_train = z_norm(x_train)
    x_val = z_norm(x_val)
    x_test = z_norm(x_test)

    x_train, y_train = oversample_abnormal(x_train, y_train, target_ratio=1.0)

    print("After oversampling:")
    print("Train normal:", int((y_train == 0).sum()), "abnormal:", int((y_train == 1).sum()))

    x_train = x_train.reshape(-1, 187, 1)
    x_val = x_val.reshape(-1, 187, 1)
    x_test = x_test.reshape(-1, 187, 1)

    model = build_model()
    model.summary()

    batch_size = 128

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    print("Training...")
    history = model.fit(
        make_ds(x_train, y_train, batch_size=batch_size, training=True),
        validation_data=make_ds(x_val, y_val, batch_size=batch_size, training=False),
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )

    print("Predicting validation...")
    val_prob = model.predict(x_val, verbose=0, batch_size=256).reshape(-1)
    best_val, val_threshold_results = find_best_threshold(y_val, val_prob)

    print("\nBest threshold from validation:")
    print(best_val)

    print("Predicting test...")
    test_prob = model.predict(x_test, verbose=0, batch_size=256).reshape(-1)
    test_result = evaluate_threshold(y_test, test_prob, threshold=best_val.threshold)

    print("\n================ TEST RESULT ================")
    print(json.dumps(asdict(test_result), indent=2))

    model_path = models_dir / "mitbih_stage1_binary_v2.keras"
    meta_path = models_dir / "mitbih_stage1_binary_v2_meta.json"

    model.save(model_path)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": "mitbih_stage1_binary_v2",
        "task": "MIT-BIH Stage 1 binary classification: Normal vs Abnormal",
        "normal_label": 0,
        "abnormal_label": 1,
        "preprocessing": "z-score per beat",
        "best_threshold_from_val": float(best_val.threshold),
        "best_val_result": asdict(best_val),
        "test_result_at_best_val_threshold": asdict(test_result),
        "notes": [
            "This v2 model is trained with abnormal oversampling and focal loss.",
            "Selection prioritizes abnormal recall >= 0.65, then abnormal F1.",
            "Do not replace the production model unless this result is better than the old Stage 1 model.",
        ],
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"mitbih_stage1_v2_report_{stamp}.json"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
        "best_val_threshold": asdict(best_val),
        "validation_threshold_results": [asdict(r) for r in val_threshold_results],
        "test_result": asdict(test_result),
        "saved_model": str(model_path),
        "saved_meta": str(meta_path),
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(model_path)
    print(meta_path)
    print(report_path)


if __name__ == "__main__":
    main()
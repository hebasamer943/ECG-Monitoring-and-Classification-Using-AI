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


# MIT-BIH original labels:
# 0 = N
# 1 = S
# 2 = V
# 3 = F
# 4 = Q
#
# Stage 2 uses only:
# S -> 0
# V -> 1
# F -> 2

MIT_TO_LOCAL = {1: 0, 2: 1, 3: 2}
LOCAL_TO_LABEL = {0: "S", 1: "V", 2: "F"}

DISPLAY_NAMES = {
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion beat",
    "Q": "Unknown / low confidence",
}


@dataclass
class MultiClassResult:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict


@dataclass
class RejectionResult:
    confidence_threshold: float
    coverage: float
    unknown_rate: float
    covered_accuracy: float
    covered_macro_f1: float
    covered_weighted_f1: float
    covered_confusion_matrix: list[list[int]]
    covered_classification_report: dict


def load_mitbih_stage2(dataset_dir: Path):
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

    # Keep only S/V/F = labels 1,2,3
    train_mask = np.isin(y_train_raw, [1, 2, 3])
    test_mask = np.isin(y_test_raw, [1, 2, 3])

    x_train_all = x_train_all[train_mask]
    y_train_raw = y_train_raw[train_mask]

    x_test = x_test[test_mask]
    y_test_raw = y_test_raw[test_mask]

    y_train_all = np.asarray([MIT_TO_LOCAL[int(v)] for v in y_train_raw], dtype=np.int64)
    y_test = np.asarray([MIT_TO_LOCAL[int(v)] for v in y_test_raw], dtype=np.int64)

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


def oversample_classes(x: np.ndarray, y: np.ndarray):
    """
    Balance S, V, F by oversampling minority classes.
    Stage 2 old model was weak mostly because S and F are underrepresented.
    """
    rng = np.random.default_rng(SEED)

    idx_s = np.where(y == 0)[0]
    idx_v = np.where(y == 1)[0]
    idx_f = np.where(y == 2)[0]

    max_count = max(len(idx_s), len(idx_v), len(idx_f))

    def upsample(idx: np.ndarray, target: int):
        if len(idx) == 0:
            return idx
        if len(idx) >= target:
            return idx
        extra = rng.choice(idx, size=target - len(idx), replace=True)
        return np.concatenate([idx, extra])

    idx_s2 = upsample(idx_s, max_count)
    idx_v2 = upsample(idx_v, max_count)
    idx_f2 = upsample(idx_f, max_count)

    idx_final = np.concatenate([idx_s2, idx_v2, idx_f2])
    rng.shuffle(idx_final)

    return x[idx_final], y[idx_final]


def sparse_focal_loss(class_weights=(1.8, 1.0, 3.5), gamma=2.0):
    """
    Class weights:
    S gets higher weight than V.
    F gets strongest weight because Fusion beats are rare and were very weak before.
    """
    cw = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        y_one = tf.one_hot(y_true, depth=3)
        ce = -tf.reduce_sum(y_one * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_one * y_pred, axis=-1)
        alpha = tf.gather(cw, y_true)

        fl = alpha * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(fl)

    return loss


def se_block(x, ratio=8):
    c = int(x.shape[-1])
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(c // ratio, 8), activation="relu")(s)
    s = layers.Dense(c, activation="sigmoid")(s)
    s = layers.Reshape((1, c))(s)
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


def inception_block(x, filters: int, dropout: float):
    """
    Multi-kernel block helps because S/V/F morphology can appear at different widths.
    """
    b1 = layers.Conv1D(filters, 3, padding="same")(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation("relu")(b1)

    b2 = layers.Conv1D(filters, 5, padding="same")(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation("relu")(b2)

    b3 = layers.Conv1D(filters, 9, padding="same")(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation("relu")(b3)

    y = layers.Concatenate()([b1, b2, b3])
    y = layers.Conv1D(filters * 2, 1, padding="same")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.SpatialDropout1D(dropout)(y)
    y = se_block(y)

    shortcut = layers.Conv1D(filters * 2, 1, padding="same")(x)
    shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Add()([shortcut, y])
    y = layers.Activation("relu")(y)

    return y


def build_model():
    inp = keras.Input(shape=(187, 1))

    x = layers.Conv1D(32, 7, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = inception_block(x, 32, 0.10)
    x = layers.MaxPool1D(2)(x)

    x = res_block(x, 96, 5, 0.15)
    x = inception_block(x, 48, 0.20)
    x = layers.MaxPool1D(2)(x)

    x = res_block(x, 160, 3, 0.25)
    x = res_block(x, 192, 3, 0.25)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)

    out = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inp, out)

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=sparse_focal_loss(class_weights=(1.8, 1.0, 3.5), gamma=2.0),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )

    return model


def augment_batch(x, y):
    batch = tf.shape(x)[0]

    # Light noise
    x = x + tf.random.normal(tf.shape(x), stddev=0.015, dtype=x.dtype)

    # Amplitude scaling
    scale = tf.random.uniform((batch, 1, 1), 0.95, 1.05, dtype=x.dtype)
    x = x * scale

    # Small temporal shift
    shift = tf.random.uniform((batch,), minval=-4, maxval=5, dtype=tf.int32)
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


class ValMacroF1Callback(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, save_path: Path):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.save_path = save_path
        self.best_macro_f1 = -1.0

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.x_val, verbose=0, batch_size=256)
        pred = np.argmax(probs, axis=1)

        macro_f1 = f1_score(self.y_val, pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(self.y_val, pred, average="weighted", zero_division=0)

        print(f"\n[val custom] epoch={epoch + 1} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")

        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.model.save_weights(self.save_path)
            print(f"[val custom] saved best weights: macro_f1={macro_f1:.4f}")


def evaluate_known(y_true, probs) -> MultiClassResult:
    pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])

    report = classification_report(
        y_true,
        pred,
        labels=[0, 1, 2],
        target_names=["S", "V", "F"],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    return MultiClassResult(
        accuracy=float(accuracy_score(y_true, pred)),
        macro_precision=float(precision_score(y_true, pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, pred, average="macro", zero_division=0)),
        macro_f1=float(f1_score(y_true, pred, average="macro", zero_division=0)),
        weighted_precision=float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        weighted_recall=float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        weighted_f1=float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
    )


def evaluate_with_rejection(y_true, probs, threshold: float) -> RejectionResult:
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)

    covered = conf >= threshold
    coverage = float(np.mean(covered))
    unknown_rate = 1.0 - coverage

    if not np.any(covered):
        return RejectionResult(
            confidence_threshold=float(threshold),
            coverage=0.0,
            unknown_rate=1.0,
            covered_accuracy=0.0,
            covered_macro_f1=0.0,
            covered_weighted_f1=0.0,
            covered_confusion_matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            covered_classification_report={},
        )

    y_cov = y_true[covered]
    pred_cov = pred[covered]

    cm = confusion_matrix(y_cov, pred_cov, labels=[0, 1, 2])

    report = classification_report(
        y_cov,
        pred_cov,
        labels=[0, 1, 2],
        target_names=["S", "V", "F"],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    return RejectionResult(
        confidence_threshold=float(threshold),
        coverage=float(coverage),
        unknown_rate=float(unknown_rate),
        covered_accuracy=float(accuracy_score(y_cov, pred_cov)),
        covered_macro_f1=float(f1_score(y_cov, pred_cov, average="macro", zero_division=0)),
        covered_weighted_f1=float(f1_score(y_cov, pred_cov, average="weighted", zero_division=0)),
        covered_confusion_matrix=cm.astype(int).tolist(),
        covered_classification_report=report,
    )


def find_best_conf_threshold(y_true, probs):
    results = []

    for thr in np.arange(0.40, 0.91, 0.02):
        r = evaluate_with_rejection(y_true, probs, float(thr))
        results.append(r)

    # Require at least 60% coverage, then maximize covered macro F1.
    candidates = [r for r in results if r.coverage >= 0.60]

    if candidates:
        best = max(candidates, key=lambda r: r.covered_macro_f1)
    else:
        best = max(results, key=lambda r: r.covered_macro_f1)

    return best, results


def main():
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"
    reports_dir = project_root / "Reports"

    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    print("Loading MIT-BIH Stage 2 data...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_mitbih_stage2(dataset_dir)

    print("Before oversampling:")
    print("Train S:", int((y_train == 0).sum()), "V:", int((y_train == 1).sum()), "F:", int((y_train == 2).sum()))
    print("Val   S:", int((y_val == 0).sum()), "V:", int((y_val == 1).sum()), "F:", int((y_val == 2).sum()))
    print("Test  S:", int((y_test == 0).sum()), "V:", int((y_test == 1).sum()), "F:", int((y_test == 2).sum()))

    x_train = z_norm(x_train)
    x_val = z_norm(x_val)
    x_test = z_norm(x_test)

    x_train, y_train = oversample_classes(x_train, y_train)

    print("After oversampling:")
    print("Train S:", int((y_train == 0).sum()), "V:", int((y_train == 1).sum()), "F:", int((y_train == 2).sum()))

    x_train = x_train.reshape(-1, 187, 1)
    x_val = x_val.reshape(-1, 187, 1)
    x_test = x_test.reshape(-1, 187, 1)

    model = build_model()
    model.summary()

    best_weights_path = models_dir / "_tmp_mitbih_stage2_v2_best.weights.h5"

    callbacks = [
        ValMacroF1Callback(x_val, y_val, best_weights_path),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            restore_best_weights=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    batch_size = 128

    print("Training Stage 2 v2...")
    history = model.fit(
        make_ds(x_train, y_train, batch_size=batch_size, training=True),
        validation_data=make_ds(x_val, y_val, batch_size=batch_size, training=False),
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )

    if best_weights_path.exists():
        print("Loading best macro-F1 weights...")
        model.load_weights(best_weights_path)

    print("Evaluating validation...")
    val_probs = model.predict(x_val, verbose=0, batch_size=256)
    val_known = evaluate_known(y_val, val_probs)
    best_reject, rejection_results = find_best_conf_threshold(y_val, val_probs)

    print("\nValidation known-class result:")
    print(json.dumps(asdict(val_known), indent=2))

    print("\nBest confidence threshold from validation:")
    print(json.dumps(asdict(best_reject), indent=2))

    print("Evaluating test...")
    test_probs = model.predict(x_test, verbose=0, batch_size=256)
    test_known = evaluate_known(y_test, test_probs)
    test_reject = evaluate_with_rejection(
        y_test,
        test_probs,
        threshold=best_reject.confidence_threshold,
    )

    print("\n================ TEST KNOWN RESULT ================")
    print(json.dumps(asdict(test_known), indent=2))

    print("\n================ TEST WITH REJECTION RESULT ================")
    print(json.dumps(asdict(test_reject), indent=2))

    model_path = models_dir / "mitbih_stage2_subtypes_svf_v2.keras"
    meta_path = models_dir / "mitbih_stage2_subtypes_svf_v2_meta.json"

    model.save(model_path)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": "mitbih_stage2_subtypes_svf_v2",
        "task": "MIT-BIH Stage 2 subtype classification: S/V/F",
        "train_class_names": ["S", "V", "F"],
        "display_names": DISPLAY_NAMES,
        "preprocessing": "z-score per beat",
        "recommended_confidence_threshold": float(best_reject.confidence_threshold),
        "validation_known_result": asdict(val_known),
        "validation_best_rejection_result": asdict(best_reject),
        "test_known_result": asdict(test_known),
        "test_rejection_result": asdict(test_reject),
        "notes": [
            "Stage 2 is trained only on MIT-BIH abnormal subtypes S/V/F.",
            "The model uses oversampling, focal loss, SE blocks, and multi-kernel convolution.",
            "Confidence rejection should be used: low-confidence predictions should be reported as Q / Unknown.",
            "Do not replace production Stage 2 unless v2 improves macro F1 and rare-class performance.",
        ],
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"mitbih_stage2_v2_report_{stamp}.json"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
        "validation_known_result": asdict(val_known),
        "validation_best_rejection_result": asdict(best_reject),
        "validation_rejection_results": [asdict(r) for r in rejection_results],
        "test_known_result": asdict(test_known),
        "test_rejection_result": asdict(test_reject),
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
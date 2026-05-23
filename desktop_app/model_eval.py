from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model

        pos = np.arange(length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angles = pos * angle_rates

        pe = np.zeros((length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :])

    def call(self, x):
        return x + self.pe[:, : tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "d_model": self.d_model})
        return cfg


@dataclass
class BinaryMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float | None
    tp: int
    tn: int
    fp: int
    fn: int
    n_samples: int
    positive_rate_pred: float
    positive_rate_true: float


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def _binarize_labels(y: np.ndarray, positive_nonzero: bool) -> np.ndarray:
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    if positive_nonzero:
        return (y != 0).astype(np.int32)
    return y.astype(np.int32)


def _normalize_beats(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D beat array, got shape {x.shape}")
    med = np.median(x, axis=1, keepdims=True)
    x0 = x - med
    scale = np.max(np.abs(x0), axis=1, keepdims=True) + 1e-9
    return x0 / scale


def _load_csv_rows(path: Path, *, limit: int | None = None) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            if not row:
                continue
            rows.append([float(v) for v in row])
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return np.asarray(rows, dtype=np.float32)


def load_ptbdb(dataset_dir: Path, *, limit_per_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    normal = _load_csv_rows(dataset_dir / "ptbdb_normal.csv", limit=limit_per_class)
    abnormal = _load_csv_rows(dataset_dir / "ptbdb_abnormal.csv", limit=limit_per_class)

    x_n = normal[:, :187]
    x_a = abnormal[:, :187]

    y_n = np.zeros((x_n.shape[0],), dtype=np.int32)
    y_a = np.ones((x_a.shape[0],), dtype=np.int32)

    x = np.concatenate([x_n, x_a], axis=0)
    y = np.concatenate([y_n, y_a], axis=0)
    return x, y


def load_labeled_csv(
    path: Path,
    *,
    limit: int | None = None,
    positive_nonzero: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    arr = _load_csv_rows(path, limit=limit)
    if arr.shape[1] < 188:
        raise ValueError(f"{path} must have at least 188 columns (187 features + label)")
    x = arr[:, :187]
    y = _binarize_labels(arr[:, 187], positive_nonzero=positive_nonzero)
    return x, y


def _predict_binary_model(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    x3 = np.asarray(x, dtype=np.float32).reshape((-1, 187, 1))
    probs = model.predict(x3, batch_size=batch_size, verbose=0).reshape(-1)
    return np.asarray(probs, dtype=np.float32)


def predict_ptbdb_ensemble(
    x: np.ndarray,
    models_dir: Path,
    *,
    batch_size: int = 256,
    unc_low: float = 0.30,
    unc_high: float = 0.70,
    w_resnet: float = 0.50,
    w_tcn: float = 0.30,
    w_trans: float = 0.20,
) -> np.ndarray:
    resnet = tf.keras.models.load_model(models_dir / "ptbdb_resnet1d_fix.keras", compile=False)
    p_res = _predict_binary_model(resnet, x, batch_size=batch_size)

    mask = (p_res >= unc_low) & (p_res <= unc_high)
    if not np.any(mask):
        return p_res

    tcn = tf.keras.models.load_model(models_dir / "ptbdb_tcn.keras", compile=False)
    trans = tf.keras.models.load_model(
        models_dir / "ptbdb_transformer_fix.keras",
        compile=False,
        custom_objects={"PositionalEncoding": PositionalEncoding},
    )

    p_final = p_res.copy()
    x_unc = x[mask]
    p_tcn = _predict_binary_model(tcn, x_unc, batch_size=batch_size)
    p_trans = _predict_binary_model(trans, x_unc, batch_size=batch_size)
    p_final[mask] = (w_resnet * p_res[mask]) + (w_tcn * p_tcn) + (w_trans * p_trans)
    return p_final


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> BinaryMetrics:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    y_pred = (y_prob >= float(threshold)).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = int(y_true.size)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, n)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    return BinaryMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1),
        roc_auc=_roc_auc_binary(y_true, y_prob),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        n_samples=n,
        positive_rate_pred=float(np.mean(y_pred)),
        positive_rate_true=float(np.mean(y_true)),
    )


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> BinaryMetrics:
    best: BinaryMetrics | None = None
    for thr in np.linspace(0.05, 0.95, 91):
        m = compute_metrics(y_true, y_prob, float(thr))
        if best is None:
            best = m
            continue
        if (m.f1 > best.f1) or (m.f1 == best.f1 and m.accuracy > best.accuracy):
            best = m
    assert best is not None
    return best


def load_optional_threshold(meta_path: Path, fallback: float) -> float:
    if not meta_path.exists():
        return float(fallback)
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return float(fallback)

    for key in ("best_threshold_from_val", "threshold"):
        val = data.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return float(fallback)


def evaluate_preset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    *,
    threshold: float,
) -> dict:
    x_norm = _normalize_beats(x)
    probs = predictor(x_norm)
    fixed = compute_metrics(y, probs, threshold=threshold)
    best = find_best_threshold(y, probs)
    return {
        "preset": name,
        "threshold_used": float(threshold),
        "metrics_at_threshold": asdict(fixed),
        "best_f1_threshold": asdict(best),
        "n_samples": int(len(y)),
    }


def build_report(project_root: Path, *, limit_per_class: int | None = None, batch_size: int = 256) -> dict:
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"

    ptbdb_x, ptbdb_y = load_ptbdb(dataset_dir, limit_per_class=limit_per_class)
    mitbih_x, mitbih_y = load_labeled_csv(
        dataset_dir / "mitbih_test.csv",
        limit=None if limit_per_class is None else limit_per_class * 2,
        positive_nonzero=True,
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "datasets": {
            "ptbdb": int(len(ptbdb_y)),
            "mitbih_test_binary": int(len(mitbih_y)),
        },
        "notes": [
            "This report evaluates only the models/pipelines currently used by the application UI.",
            "PTBDB result matches the live beat pipeline logic: ResNet, then conditional TCN/Transformer ensemble only in the uncertain range.",
            "MITBIH result is a proxy for the image classifier model on beat vectors, not a full-image extraction benchmark.",
        ],
        "results": [],
    }

    report["results"].append(
        evaluate_preset(
            "ptbdb_live_pipeline on PTBDB",
            ptbdb_x,
            ptbdb_y,
            predictor=lambda z: predict_ptbdb_ensemble(z, models_dir, batch_size=batch_size),
            threshold=0.60,
        )
    )

    inception = tf.keras.models.load_model(models_dir / "mitbih_binary_inception.keras", compile=False)
    inc_thr = load_optional_threshold(models_dir / "mitbih_binary_label_map.json", fallback=0.50)
    report["results"].append(
        evaluate_preset(
            "mitbih_image_inception on MITBIH test",
            mitbih_x,
            mitbih_y,
            predictor=lambda z: _predict_binary_model(inception, z, batch_size=batch_size),
            threshold=inc_thr,
        )
    )

    return report


def print_report(report: dict):
    print(f"Generated: {report['generated_at']}")
    print(f"PTBDB samples: {report['datasets']['ptbdb']}")
    print(f"MITBIH samples: {report['datasets']['mitbih_test_binary']}")
    print("")
    for item in report["results"]:
        cur = item["metrics_at_threshold"]
        best = item["best_f1_threshold"]
        print(f"[{item['preset']}]")
        print(
            "  threshold={:.2f} acc={:.4f} prec={:.4f} rec={:.4f} spec={:.4f} f1={:.4f} auc={}".format(
                cur["threshold"],
                cur["accuracy"],
                cur["precision"],
                cur["recall"],
                cur["specificity"],
                cur["f1"],
                "n/a" if cur["roc_auc"] is None else f"{cur['roc_auc']:.4f}",
            )
        )
        print(
            "  confusion: TP={} TN={} FP={} FN={}".format(
                cur["tp"], cur["tn"], cur["fp"], cur["fn"]
            )
        )
        print(
            "  best_f1_threshold={:.2f} -> f1={:.4f} acc={:.4f}".format(
                best["threshold"], best["f1"], best["accuracy"]
            )
        )
        print("")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG models on local labeled datasets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing Dataset/ and Models/",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap for PTBDB per class; MITBIH uses roughly 2x this limit.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to Reports/model_eval_<timestamp>.json",
    )
    args = parser.parse_args()

    report = build_report(
        args.project_root,
        limit_per_class=args.limit_per_class,
        batch_size=args.batch_size,
    )
    print_report(report)

    out_path = args.out
    if out_path is None:
        reports_dir = args.project_root / "Reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = reports_dir / f"model_eval_{stamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()

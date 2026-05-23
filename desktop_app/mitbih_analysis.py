from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass
class MITBIHBeatPrediction:
    final_label: str
    final_display: str
    stage1_prob_abnormal: float
    final_confidence: float
    stage2_label: str | None = None
    stage2_display: str | None = None
    stage2_confidence: float | None = None


@dataclass
class MITBIHSessionSummary:
    total_beats: int
    counts: dict[str, int]
    percentages: dict[str, float]
    dominant_abnormal_label: str = ""
    dominant_abnormal_display: str = ""
    dominant_abnormal_count: int = 0
    dominant_abnormal_pct: float = 0.0


class MITBIHTwoStageAnalyzer:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)

        self.stage1_model_path = self.models_dir / "mitbih_stage1_binary_v2.keras"
        self.stage1_meta_path = self.models_dir / "mitbih_stage1_binary_v2_meta.json"
        self.stage2_model_path = self.models_dir / "mitbih_stage2_subtypes_svf_v2.keras"
        self.stage2_meta_path = self.models_dir / "mitbih_stage2_subtypes_svf_v2_meta.json"

        self.stage1_model = None
        self.stage2_model = None
        self._loaded = False

        # Stage 1 v2 threshold:
        # p >= 0.76 means abnormal and goes to Stage 2.
        # p <= 0.30 is treated as normal-like.
        # between them is Unknown / low confidence.
        self.stage1_low = 0.30
        self.stage1_high = 0.76
        # Stage 2 v2 confidence rejection:
        # if subtype confidence < 0.90, report Unknown.
        self.stage2_conf_thr = 0.90
        # Do not use an extra strict Fusion-only guard now.
        # Stage 2 v2 already uses confidence rejection.
        self.f_guard_conf_thr = 0.90

        self.display_names = {
            "N": "Normal-like",
            "S": "Supraventricular ectopic",
            "V": "Ventricular ectopic",
            "F": "Fusion beat",
            "Q": "Unknown / low confidence",
        }
        self.stage2_labels = ["S", "V", "F"]

    def available(self) -> bool:
        needed = [
            self.stage1_model_path,
            self.stage1_meta_path,
            self.stage2_model_path,
            self.stage2_meta_path,
        ]
        return all(p.exists() for p in needed)

    def load(self):
        if self._loaded:
            return
        if not self.available():
            missing = [
                str(p.name)
                for p in [
                    self.stage1_model_path,
                    self.stage1_meta_path,
                    self.stage2_model_path,
                    self.stage2_meta_path,
                ]
                if not p.exists()
            ]
            raise FileNotFoundError(f"Missing MIT-BIH analysis files: {', '.join(missing)}")

        with open(self.stage1_meta_path, "r", encoding="utf-8") as f:
            stage1_meta = json.load(f)
        with open(self.stage2_meta_path, "r", encoding="utf-8") as f:
            stage2_meta = json.load(f)

        # Stage 1 threshold loading.
       # New v2 meta stores best_threshold_from_val = 0.76.
        band = stage1_meta.get("uncertainty_band", {})
        self.stage1_low = float(band.get("normal_if_prob_abnormal_le", self.stage1_low))
        self.stage1_high = float(band.get("abnormal_if_prob_abnormal_ge", self.stage1_high))

        best_thr = stage1_meta.get("best_threshold_from_val")
        if isinstance(best_thr, (int, float)):
          self.stage1_high = float(best_thr)
          self.stage1_low = 0.30

       # Stage 2 v2 confidence threshold.
        self.stage2_conf_thr = float(
         stage2_meta.get("recommended_confidence_threshold", self.stage2_conf_thr)
       )

      # Keep the Fusion guard equal to the general confidence threshold.
       # Otherwise F predictions may be unfairly hidden.
        self.f_guard_conf_thr = self.stage2_conf_thr

        stage2_display = stage2_meta.get("display_names", {})
        for key, value in stage2_display.items():
            self.display_names[str(key)] = str(value)

        labels = stage2_meta.get("train_class_names", self.stage2_labels)
        if isinstance(labels, list) and len(labels) == 3:
            self.stage2_labels = [str(x) for x in labels]

        self.stage1_model = tf.keras.models.load_model(self.stage1_model_path, compile=False)
        self.stage2_model = tf.keras.models.load_model(self.stage2_model_path, compile=False)
        self._loaded = True

    @staticmethod
    def _prep_one(beat: np.ndarray) -> np.ndarray:
        x = np.asarray(beat, dtype=np.float32).reshape(-1)
        if x.size > 187:
            x = x[:187]
        if x.size < 187:
            x = np.pad(x, (0, 187 - x.size))
        mu = float(np.mean(x))
        sd = float(np.std(x)) + 1e-6
        x = (x - mu) / sd
        return x.reshape(187, 1)

    def _prep_batch(self, beats: list[np.ndarray]) -> np.ndarray:
        if not beats:
            return np.zeros((0, 187, 1), dtype=np.float32)
        rows = [self._prep_one(beat) for beat in beats]
        return np.stack(rows).astype(np.float32)

    def predict_batch(self, beats: list[np.ndarray]) -> list[MITBIHBeatPrediction]:
        self.load()
        if not beats:
            return []

        x = self._prep_batch(beats)
        stage1_probs = self.stage1_model.predict(x, verbose=0, batch_size=256).reshape(-1)

        out: list[MITBIHBeatPrediction | None] = [None] * len(beats)
        stage2_indices: list[int] = []

        for i, prob_abn in enumerate(stage1_probs):
            p = float(prob_abn)
            if p <= self.stage1_low:
                out[i] = MITBIHBeatPrediction(
                    final_label="N",
                    final_display=self.display_names["N"],
                    stage1_prob_abnormal=p,
                    final_confidence=max(0.0, min(1.0, 1.0 - p)),
                )
            elif p < self.stage1_high:
                out[i] = MITBIHBeatPrediction(
                    final_label="Q",
                    final_display=self.display_names["Q"],
                    stage1_prob_abnormal=p,
                    final_confidence=max(0.0, min(1.0, 1.0 - abs(p - 0.5) * 2.0)),
                )
            else:
                stage2_indices.append(i)

        if stage2_indices:
            x_stage2 = x[stage2_indices]
            stage2_probs = self.stage2_model.predict(x_stage2, verbose=0, batch_size=256)

            for local_idx, probs in enumerate(stage2_probs):
                i = stage2_indices[local_idx]
                p_stage1 = float(stage1_probs[i])
                probs = np.asarray(probs, dtype=np.float32).reshape(-1)
                arg = int(np.argmax(probs))
                conf = float(np.max(probs))
                stage2_label = self.stage2_labels[arg]
                stage2_display = self.display_names.get(stage2_label, stage2_label)

                final_label = stage2_label
                final_display = stage2_display
                if conf < self.stage2_conf_thr:
                    final_label = "Q"
                    final_display = self.display_names["Q"]
                elif stage2_label == "F" and conf < self.f_guard_conf_thr:
                    final_label = "Q"
                    final_display = self.display_names["Q"]

                out[i] = MITBIHBeatPrediction(
                    final_label=final_label,
                    final_display=final_display,
                    stage1_prob_abnormal=p_stage1,
                    final_confidence=conf,
                    stage2_label=stage2_label,
                    stage2_display=stage2_display,
                    stage2_confidence=conf,
                )

        finalized: list[MITBIHBeatPrediction] = []
        for i, item in enumerate(out):
            if item is None:
                p = float(stage1_probs[i]) if i < len(stage1_probs) else 0.5
                item = MITBIHBeatPrediction(
                    final_label="Q",
                    final_display=self.display_names["Q"],
                    stage1_prob_abnormal=p,
                    final_confidence=0.0,
                )
            finalized.append(item)
        return finalized

    def predict_abnormal_batch(self, beats: list[np.ndarray]) -> list[MITBIHBeatPrediction]:
        self.load()
        if not beats:
            return []

        x = self._prep_batch(beats)
        stage2_probs = self.stage2_model.predict(x, verbose=0, batch_size=256)

        out: list[MITBIHBeatPrediction] = []
        for probs in stage2_probs:
            probs = np.asarray(probs, dtype=np.float32).reshape(-1)
            arg = int(np.argmax(probs))
            conf = float(np.max(probs))
            stage2_label = self.stage2_labels[arg]
            stage2_display = self.display_names.get(stage2_label, stage2_label)

            final_label = stage2_label
            final_display = stage2_display
            if conf < self.stage2_conf_thr:
                final_label = "Q"
                final_display = self.display_names["Q"]
            elif stage2_label == "F" and conf < self.f_guard_conf_thr:
                final_label = "Q"
                final_display = self.display_names["Q"]

            out.append(
                MITBIHBeatPrediction(
                    final_label=final_label,
                    final_display=final_display,
                    stage1_prob_abnormal=1.0,
                    final_confidence=conf,
                    stage2_label=stage2_label,
                    stage2_display=stage2_display,
                    stage2_confidence=conf,
                )
            )
        return out

    def summarize_predictions(self, predictions: list[MITBIHBeatPrediction]) -> MITBIHSessionSummary:
        labels = ["N", "S", "V", "F", "Q"]
        counts = {label: 0 for label in labels}
        for pred in predictions:
            counts[pred.final_label] = counts.get(pred.final_label, 0) + 1

        total = len(predictions)
        percentages = {
            label: (100.0 * float(counts[label]) / float(total)) if total > 0 else 0.0
            for label in labels
        }

        dominant_abnormal_label = ""
        dominant_abnormal_count = 0
        for label in ["V", "S", "F"]:
            if counts.get(label, 0) > dominant_abnormal_count:
                dominant_abnormal_label = label
                dominant_abnormal_count = counts[label]

        dominant_abnormal_display = self.display_names.get(dominant_abnormal_label, dominant_abnormal_label) if dominant_abnormal_label else ""
        dominant_abnormal_pct = (100.0 * float(dominant_abnormal_count) / float(total)) if total > 0 else 0.0

        return MITBIHSessionSummary(
            total_beats=total,
            counts=counts,
            percentages=percentages,
            dominant_abnormal_label=dominant_abnormal_label,
            dominant_abnormal_display=dominant_abnormal_display,
            dominant_abnormal_count=dominant_abnormal_count,
            dominant_abnormal_pct=dominant_abnormal_pct,
        )

    def analyze_beats(self, beats: list[np.ndarray]) -> tuple[list[MITBIHBeatPrediction], MITBIHSessionSummary]:
        predictions = self.predict_batch(beats)
        summary = self.summarize_predictions(predictions)
        return predictions, summary

    def analyze_abnormal_beats(self, beats: list[np.ndarray]) -> tuple[list[MITBIHBeatPrediction], MITBIHSessionSummary]:
        predictions = self.predict_abnormal_batch(beats)
        summary = self.summarize_predictions(predictions)
        return predictions, summary

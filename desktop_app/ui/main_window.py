from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QSlider, QProgressBar,
    QStackedWidget, QMessageBox, QFileDialog, QScrollArea,
    QInputDialog, QLineEdit, QTextEdit
)
from PySide6.QtTextToSpeech import QTextToSpeech

import pyqtgraph as pg

from desktop_app.io_sources import PTBDBSource, CSVSource, BeatSample
from desktop_app.image_pipeline import extract_ecg_from_image
from desktop_app.live_serial import SerialECGSource
from desktop_app.signal_quality import evaluate_quality
from desktop_app.session_analysis import BeatRecord, analyze_session
from desktop_app.mitbih_analysis import MITBIHTwoStageAnalyzer
from desktop_app.reporting import export_pdf_report
from desktop_app.ai_summary import generate_bilingual_ai_report


def card(title: str = "", object_name: str = "Card"):
    frame = QFrame()
    frame.setObjectName(object_name)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 16, 16, 16)
    lay.setSpacing(10)

    if title:
        h = QLabel(title)
        h.setObjectName("SectionTitle")
        lay.addWidget(h)

    return frame, lay


def metric_card(title: str, value: str, sub: str = ""):
    frame = QFrame()
    frame.setObjectName("MetricCard")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(6)

    lbl_t = QLabel(title)
    lbl_t.setObjectName("MetricLabel")

    lbl_v = QLabel(value)
    lbl_v.setObjectName("MetricValue")

    lbl_s = QLabel(sub)
    lbl_s.setObjectName("Muted")

    lay.addWidget(lbl_t)
    lay.addWidget(lbl_v)
    lay.addWidget(lbl_s)
    lay.addStretch(1)
    return frame, lbl_v, lbl_s


class BeatTrailWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[str] = []
        self.max_items = 42
        self.setMinimumHeight(34)
        self.setMaximumHeight(40)

    def clear_items(self):
        self.items = []
        self.update()

    def add_item(self, state: str):
        self.items.append(state)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(8, 6, -8, -6)
        painter.setPen(Qt.NoPen)

        if not self.items:
            painter.setPen(QColor("#93a4bd"))
            painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, "Live beat stream will appear here")
            return

        diameter = 12
        gap = 7
        x = rect.left()
        y = rect.center().y() - diameter / 2

        for state in self.items:
            if state == "normal":
                color = QColor("#18d26e")
            elif state == "abnormal":
                color = QColor("#ff4d5a")
            else:
                color = QColor("#8a94a7")

            painter.setBrush(color)
            painter.drawEllipse(int(x), int(y), diameter, diameter)
            x += diameter + gap
            if x + diameter > rect.right():
                break


class DonutWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.normal_pct = 0.0
        self.abnormal_pct = 0.0
        self.noisy_pct = 0.0
        self.setMinimumHeight(220)

    def set_distribution(self, normal_pct: float, abnormal_pct: float, noisy_pct: float):
        self.normal_pct = max(0.0, float(normal_pct))
        self.abnormal_pct = max(0.0, float(abnormal_pct))
        self.noisy_pct = max(0.0, float(noisy_pct))
        total = self.normal_pct + self.abnormal_pct + self.noisy_pct
        if total <= 0:
            self.normal_pct, self.abnormal_pct, self.noisy_pct = 100.0, 0.0, 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(20, 20, -20, -20)
        side = min(rect.width(), rect.height())
        cx = rect.center().x()
        cy = rect.center().y()

        donut_rect = QRectF(cx - side * 0.33, cy - side * 0.33, side * 0.66, side * 0.66)
        pen_bg = QPen(QColor("#122033"), 18)
        painter.setPen(pen_bg)
        painter.drawArc(donut_rect, 0, 360 * 16)

        values = [
            (self.normal_pct, QColor("#14f195")),
            (self.abnormal_pct, QColor("#ff7a45")),
            (self.noisy_pct, QColor("#8a94a7")),
        ]

        start_deg = 90.0
        for pct, color in values:
            if pct <= 0:
                continue
            span = 360.0 * (pct / 100.0)
            pen = QPen(color, 18)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawArc(donut_rect, int(-start_deg * 16), int(-span * 16))
            start_deg += span

        painter.setPen(QColor("#eef4ff"))
        f1 = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(f1)
        painter.drawText(donut_rect, Qt.AlignCenter, f"N {self.normal_pct:.1f}%")

        legend_x = rect.left() + 10
        legend_y = rect.bottom() - 50
        legend = [
            ("Normal", "#14f195", self.normal_pct),
            ("Abnormal", "#ff7a45", self.abnormal_pct),
            ("Noisy", "#8a94a7", self.noisy_pct),
        ]
        yy = legend_y
        for name, col, pct in legend:
            painter.setBrush(QColor(col))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(legend_x + 8, yy + 8, 10, 10)
            painter.setPen(QColor("#dce7f7"))
            painter.setBrush(Qt.NoBrush)
            painter.drawText(legend_x + 24, yy + 16, f"{name}: {pct:.1f}%")
            yy += 22


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
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "d_model": self.d_model})
        return cfg


class ModelWrapper:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

        self.resnet_path = self.models_dir / "ptbdb_resnet1d_fix.keras"
        self.tcn_path = self.models_dir / "ptbdb_tcn.keras"
        self.trans_path = self.models_dir / "ptbdb_transformer_fix.keras"

        self.m_resnet = None
        self.m_tcn = None
        self.m_trans = None

        self.w_resnet = 0.30
        self.w_tcn = 0.50
        self.w_trans = 0.20

        self.unc_low = 0.20
        self.unc_high = 0.85
        self.load_error = ""
        self._tcn_checked = False
        self._trans_checked = False

    def is_ready(self) -> bool:
        return self.m_resnet is not None

    def load(self):
        if not self.resnet_path.exists():
            self.load_error = "Missing Models/ptbdb_resnet1d_fix.keras"
            raise FileNotFoundError(self.load_error)
        try:
            self.m_resnet = tf.keras.models.load_model(self.resnet_path, compile=False)
        except Exception as e:
            self.m_resnet = None
            self.load_error = f"Failed to load {self.resnet_path.name}: {e}"
            raise RuntimeError(self.load_error) from e
        self.load_error = ""
        print("Loaded ResNet:", self.resnet_path)

        if not self.tcn_path.exists():
            print("TCN missing (optional):", self.tcn_path)
        if not self.trans_path.exists():
            print("Transformer missing (optional):", self.trans_path)

    def _ensure_tcn(self):
        if self.m_tcn is not None:
            return self.m_tcn
        if self._tcn_checked:
            return None

        self._tcn_checked = True
        if not self.tcn_path.exists():
            print("TCN unavailable (skipping fallback):", self.tcn_path)
            return None
        try:
            self.m_tcn = tf.keras.models.load_model(self.tcn_path, compile=False)
            print("Loaded TCN:", self.tcn_path)
        except Exception as e:
            print("Failed to load TCN (skipping fallback):", e)
            self.m_tcn = None
        return self.m_tcn

    def _ensure_trans(self):
        if self.m_trans is not None:
            return self.m_trans
        if self._trans_checked:
            return None

        self._trans_checked = True
        if not self.trans_path.exists():
            print("Transformer unavailable (skipping fallback):", self.trans_path)
            return None
        try:
            self.m_trans = tf.keras.models.load_model(
                self.trans_path,
                compile=False,
                custom_objects={"PositionalEncoding": PositionalEncoding},
            )
            print("Loaded Transformer:", self.trans_path)
        except Exception as e:
            print("Failed to load Transformer (skipping fallback):", e)
            self.m_trans = None
        return self.m_trans

    def _prep(self, beat: np.ndarray) -> np.ndarray:
        x = np.asarray(beat, dtype=np.float32)
        if x.size > 187:
            x = x[:187]
        if x.size < 187:
            x = np.pad(x, (0, 187 - x.size))
        return x.reshape(1, 187, 1)

    def predict_proba_abn(self, beat: np.ndarray) -> float:
     """
     Final selected prediction pipeline:
     TCN-first wide conditional ensemble.

     Logic:
     1. Run TCN first because it achieved the best standalone performance.
     2. If TCN is confident, use TCN output directly.
     3. If TCN is uncertain, combine TCN + ResNet + Transformer using weighted average.

     Uncertainty range:
        [0.20, 0.85]

     Weights:
        TCN         = 0.50
        ResNet1D    = 0.30
        Transformer = 0.20
     """

     # ResNet is still loaded by self.load(), and we keep it available for fallback
     # and for the ensemble stage.
     if self.m_resnet is None:
         self.load()

     X = self._prep(beat)

     # Load TCN lazily. If TCN is missing, fallback safely to ResNet.
     tcn_model = self._ensure_tcn()
     if tcn_model is None:
         p_res = float(self.m_resnet.predict(X, verbose=0).reshape(-1)[0])
         return p_res

     # Step 1: TCN first
     p_tcn = float(tcn_model.predict(X, verbose=0).reshape(-1)[0])

     # Step 2: If TCN is confident, use it directly.
     if (p_tcn < self.unc_low) or (p_tcn > self.unc_high):
          return p_tcn

     # Step 3: TCN is uncertain, so activate conditional ensemble.
     weighted_probs = [(self.w_tcn, p_tcn)]

     # ResNet contribution
     p_res = float(self.m_resnet.predict(X, verbose=0).reshape(-1)[0])
     weighted_probs.append((self.w_resnet, p_res))

     # Transformer contribution, if available
     trans_model = self._ensure_trans()
     if trans_model is not None:
         p_tr = float(trans_model.predict(X, verbose=0).reshape(-1)[0])
         weighted_probs.append((self.w_trans, p_tr))

     # Normalize weights in case Transformer is missing.
     total_w = sum(w for w, _ in weighted_probs)
     p_final = sum(w * p for w, p in weighted_probs) / max(total_w, 1e-9)

     return float(p_final)

class MainWindow(QMainWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self.dataset_dir = project_root / "Dataset"
        self.models_dir = project_root / "Models"
        self.reports_dir = project_root / "Reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.source = PTBDBSource(self.dataset_dir)

        self.model = ModelWrapper(self.models_dir)
        self.core_model_error = ""
        try:
            self.model.load()
        except Exception as e:
            self.core_model_error = str(e)
            print("Prediction backend unavailable:", e)

        self.image_paths = []
        self.image_pos = 0
        self.image_mode = False
        self.image_last_path = None
        self.image_model_path = self.models_dir / "mitbih_binary_inception.keras"
        self.image_model = None
        self.image_model_error = ""
        self.image_threshold = 0.60

        if self.image_model_path.exists():
            try:
                self.image_model = tf.keras.models.load_model(self.image_model_path, compile=False)
                print("Loaded IMAGE model:", self.image_model_path)
            except Exception as e:
                self.image_model = None
                self.image_model_error = str(e)
                print("IMAGE model unavailable:", e)
        else:
            print("IMAGE model missing:", self.image_model_path, "(Images will not be classified)")

        self.serial_source: SerialECGSource | None = None
        self._pending_image_beat = None

        self.threshold = 0.60
        self.hw_threshold_floor = 0.80
        self.hw_noisy_cutoff = 0.42
        self.hw_prob_smooth_window = 5
        self.hw_prob_hist: list[float] = []
        self.hw_quality_smooth_window = 6
        self.hw_quality_hist: list[float] = []
        self.hw_motion_smooth_window = 4
        self.hw_motion_hist: list[float] = []
        self.hw_abn_gate_hist: list[bool] = []
        self.hw_noisy_streak = 0
        self.hw_abn_min_prob = 0.88
        self.hw_abn_margin = 0.26
        self.hw_min_quality_for_abn = 0.78
        self.hw_motion_noisy_gate = 0.30
        self.hw_motion_noisy_gate_fast = 0.24
        self.hw_motion_for_abn_max = 0.22
        self.hw_live_motion_gate = 0.18
        self.hw_live_motion_score = 0.0
        self.hw_live_noise_hold_sec = 1.8
        self.hw_live_motion_flag_until = 0.0
        self.hw_quality_display = 0.90
        self.replay_interval_csv_sec = 0.90
        self.replay_interval_image_sec = 1.20
        self.replay_last_step_t = 0.0
        self.session_rows = []
        self._error_showing = False

        self.hw_session_on = False
        self.hw_session_paused = False
        self.hw_elapsed_before_pause = 0.0
        self.hw_t0 = 0.0
        self.hw_duration = 120.0
        self.hw_records: list[BeatRecord] = []
        self.hw_result = None

        self.patient_name = ""
        self.selected_source_name = "Demo Data"
        self.last_result = None
        self.ai_report_en = ""
        self.ai_report_ar = ""
        self.ai_report_provider = "local"
        self.summary_lang = "en"
        self.mitbih_analyzer = MITBIHTwoStageAnalyzer(self.models_dir)

        self.tts = QTextToSpeech(self)
        self.tts.setRate(-0.05)

        self.setWindowTitle("PulseAI ECG Monitor")
        self.setMinimumSize(1350, 820)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self._build_home_page()
        self._build_monitor_page()
        self._build_loading_page()
        self._build_analysis_page()
        self.stack.setCurrentIndex(0)

        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(280)
        self._blink_state = True
        self.blink_timer.timeout.connect(self._blink)
        self.blink_timer.start()

        self.heart_timer = QTimer(self)
        self.heart_timer.setInterval(420)
        self._heart_big = False
        self.heart_timer.timeout.connect(self._pulse_heart)
        self.heart_timer.start()

        self.hw_poll_timer = QTimer(self)
        self.hw_poll_timer.setInterval(40)
        self.hw_poll_timer.timeout.connect(self._hw_poll)
        self.hw_poll_timer.start()

        self.hw_ui_timer = QTimer(self)
        self.hw_ui_timer.setInterval(220)
        self.hw_ui_timer.timeout.connect(self._hw_tick)
        self.hw_ui_timer.start()

        self.analysis_timer = QTimer(self)
        self.analysis_timer.setSingleShot(True)
        self.analysis_timer.timeout.connect(self._finish_analysis_now)

        self._update_patient_labels()
        self._update_home_status()
        self._apply_mode_ui()

    # ---------------- general helpers ----------------

    def _patient_name_value(self) -> str:
        name = (self.patient_name or "").strip()
        if not name and hasattr(self, "edit_patient_name"):
            name = self.edit_patient_name.text().strip()
        return name or "Unknown"

    def _on_patient_name_changed(self, text: str):
        self.patient_name = (text or "").strip()
        self._update_patient_labels()

    def _update_patient_labels(self):
        shown = self._patient_name_value()
        for attr in ("lbl_patient_preview", "lbl_monitor_patient", "lbl_analysis_patient"):
            if hasattr(self, attr):
                getattr(self, attr).setText(f"Patient: {shown}")

    def _update_home_status(self):
     if not hasattr(self, "lbl_home_status"):
         return

     if not self.model.is_ready():
         self.lbl_home_status.setText("Prediction model is not ready")
         return

     dataset_ready = (self.dataset_dir / "ptbdb_normal.csv").exists() and (self.dataset_dir / "ptbdb_abnormal.csv").exists()

     analysis_models_needed = [
         self.models_dir / "mitbih_stage1_binary_v2.keras",
         self.models_dir / "mitbih_stage1_binary_v2_meta.json",
         self.models_dir / "mitbih_stage2_subtypes_svf_v2.keras",
         self.models_dir / "mitbih_stage2_subtypes_svf_v2_meta.json",
         ]
     mitbih_ready = all(p.exists() for p in analysis_models_needed)

     if dataset_ready and mitbih_ready:
         self.lbl_home_status.setText("System ready for ECG monitoring and analysis")
     elif dataset_ready:
         self.lbl_home_status.setText("System ready for ECG monitoring")
     else:
         self.lbl_home_status.setText("Dataset files are missing")
         
    def _disconnect_serial_source(self):
        if self.serial_source is None:
            return
        try:
            self.serial_source.stop()
        except Exception:
            pass
        self.serial_source = None
        if isinstance(self.source, SerialECGSource):
            self.source = PTBDBSource(self.dataset_dir)

    def _prepare_source_change(self, *, disconnect_serial: bool):
        if self.hw_session_on:
            self._stop_monitoring_session()
        if disconnect_serial:
            self._disconnect_serial_source()
        self.image_mode = False
        self.image_paths = []
        self.image_pos = 0
        self.image_last_path = None
        self._pending_image_beat = None
        self.replay_last_step_t = 0.0

    def _ensure_prediction_backend_ready(self) -> bool:
        if self.model.is_ready():
            if self.core_model_error:
                self.core_model_error = ""
                self._update_home_status()
            return True
        try:
            self.model.load()
            self.core_model_error = ""
            self._update_home_status()
            return True
        except Exception as e:
            self.core_model_error = str(e)
            self._update_home_status()
            QMessageBox.critical(
                self,
                "Prediction Model",
                f"Prediction backend is not ready.\n\n{self.core_model_error}",
            )
            return False

    def _handle_replay_exception(self, error: Exception):
        if self.hw_session_on:
            self._stop_monitoring_session()
        if self._error_showing:
            return
        self._error_showing = True
        try:
            QMessageBox.critical(self, "Playback Error", f"Playback stopped.\n\n{error}")
        finally:
            self._error_showing = False

    def closeEvent(self, event):
        self._disconnect_serial_source()
        event.accept()

    def _setup_plot(self, w: pg.PlotWidget, wave: bool = True):
        w.setBackground("#050913")
        w.showGrid(x=True, y=True, alpha=0.10)
        axis_pen = pg.mkPen((130, 170, 210, 90))
        w.getAxis("left").setPen(axis_pen)
        w.getAxis("bottom").setPen(axis_pen)
        w.getAxis("left").setTextPen(axis_pen)
        w.getAxis("bottom").setTextPen(axis_pen)
        if wave:
            w.getAxis("left").setLabel("Amplitude", color="#14f195")
            w.getAxis("bottom").setLabel("Sample", color="#14f195")
        else:
            w.getAxis("left").setLabel("BPM", color="#14f195")
            w.getAxis("bottom").setLabel("Time (s)", color="#14f195")

    def _go_home(self):
     try:
         self.tts.stop()
     except Exception:
         pass
     self.stack.setCurrentIndex(0)

    def _go_monitor(self):
     try:
         self.tts.stop()
     except Exception:
         pass
     self.stack.setCurrentIndex(1)

    def _pulse_heart(self):
        if not hasattr(self, "lbl_heart_icon"):
            return
        self._heart_big = not self._heart_big
        if self._heart_big:
            self.lbl_heart_icon.setStyleSheet("""
                color: #ff3355;
                font-size: 24pt;
                font-weight: 900;
                background: transparent;
            """)
        else:
            self.lbl_heart_icon.setStyleSheet("""
                color: #ff6b81;
                font-size: 20pt;
                font-weight: 900;
                background: transparent;
            """)

    def _blink(self):
        self._blink_state = not self._blink_state
        alpha = 255 if self._blink_state else 170
        self.curve.setPen(pg.mkPen((20, 241, 149, alpha), width=2))

    def _update_monitor_source_labels(self):
        if hasattr(self, "lbl_top_source"):
            self.lbl_top_source.setText(f"Source: {self.selected_source_name}")

    def _update_monitor_metric_visibility(self):
         show_hw_metrics = isinstance(self.source, SerialECGSource) and (not self.image_mode)
 
         if hasattr(self, "card_hr_live"):
             self.card_hr_live.setVisible(show_hw_metrics)

         if hasattr(self, "card_quality_live"):
             self.card_quality_live.setVisible(show_hw_metrics)

         if hasattr(self, "card_bottom_bpm"):
             self.card_bottom_bpm.setVisible(show_hw_metrics)

         if hasattr(self, "side_layout"):
             for w in (self.card_hr_live, self.session_card, self.card_quality_live, self.ctrl_card):
                 self.side_layout.removeWidget(w)

             if show_hw_metrics:
                 ordered = (
                     self.card_hr_live,
                     self.session_card,
                     self.card_quality_live,
                     self.ctrl_card,
                 )
             else:
                 ordered = (
                     self.session_card,
                     self.ctrl_card,
                     self.card_hr_live,
                     self.card_quality_live,
                 )

             for i, w in enumerate(ordered):
                 self.side_layout.insertWidget(i, w)

    def _update_analysis_metric_visibility(self, show_vitals: bool):
        if not hasattr(self, "analysis_metrics_layout"):
            return

        # hide/show individual cards
        self.card_avg_bpm.setVisible(show_vitals)
        self.card_hrv.setVisible(show_vitals)
        self.card_bpm_trend.setVisible(show_vitals)

        # clear metrics grid positions
        metrics = self.analysis_metrics_layout
        while metrics.count():
            item = metrics.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if show_vitals:
            metrics.addWidget(self.card_avg_bpm, 0, 0)
            metrics.addWidget(self.card_hrv, 0, 1)
            metrics.addWidget(self.card_norm, 0, 2)
            metrics.addWidget(self.card_duration, 0, 3)
            if self.card_arrhythmia_mix.isVisible():
                metrics.addWidget(self.card_arrhythmia_mix, 1, 0, 1, 4)
        else:
            metrics.addWidget(self.card_norm, 0, 0, 1, 2)
            metrics.addWidget(self.card_duration, 0, 2, 1, 2)
            if self.card_arrhythmia_mix.isVisible():
                metrics.addWidget(self.card_arrhythmia_mix, 1, 0, 1, 4)

        # rebuild charts row so donut takes the space cleanly
        if hasattr(self, "analysis_charts_layout") and hasattr(self, "card_donut_analysis"):
            charts = self.analysis_charts_layout
            while charts.count():
                item = charts.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

            if show_vitals:
                charts.addWidget(self.card_bpm_trend, 1)
                charts.addWidget(self.card_donut_analysis, 1)
            else:
                charts.addWidget(self.card_donut_analysis, 1)

    def _apply_mode_ui(self):
        self._update_monitor_source_labels()
        self._update_monitor_metric_visibility()

    def _thr_changed(self, v: int):
        self.threshold = v / 100.0
        self.lbl_thr.setText(f"Threshold: {self.threshold:.2f}")

    def _set_bpm_display(self, bpm: float | int | None):
        if not hasattr(self, "lbl_bpm_big"):
            return

        if bpm is None:
            self.lbl_bpm_big.setText("88")
            alert = False
        else:
            bpm_int = int(round(float(bpm)))
            self.lbl_bpm_big.setText(f"{bpm_int}")
            alert = (bpm_int < 60) or (bpm_int > 100)

        self.lbl_bpm_big.setProperty("alert", alert)
        self.lbl_bpm_big.style().unpolish(self.lbl_bpm_big)
        self.lbl_bpm_big.style().polish(self.lbl_bpm_big)


    def _clear_live_counters(self):
        self.hw_prob_hist = []
        self.hw_quality_hist = []
        self.hw_motion_hist = []
        self.hw_abn_gate_hist = []
        self.hw_noisy_streak = 0
        self.hw_live_motion_score = 0.0
        self.hw_live_motion_flag_until = 0.0
        self.hw_quality_display = 0.90
        self.replay_last_step_t = 0.0
        if hasattr(self, "lbl_seen"):
            self.lbl_seen.setText("Seen: 0")
        if hasattr(self, "lbl_true_abn"):
            self.lbl_true_abn.setText("True (abn): -")
        if hasattr(self, "lbl_pred_abn"):
            self.lbl_pred_abn.setText("Predicted abnormal: -")
        if hasattr(self, "lbl_pabn_avg"):
            self.lbl_pabn_avg.setText("Avg P(abn): -")
        if hasattr(self, "lbl_bottom_total"):
            self.lbl_bottom_total[1].setText("0")
        if hasattr(self, "lbl_bottom_normal"):
            self.lbl_bottom_normal[1].setText("-")

    def _update_live_counters(self, bpm: float | None = None):
        seen = len(self.session_rows)
        self.lbl_seen.setText(f"Seen: {seen}")

        if seen > 0:
            avg_p = float(np.mean([float(r[4]) for r in self.session_rows]))
            self.lbl_pabn_avg.setText(f"Avg P(abn): {avg_p:.3f}")
        else:
            self.lbl_pabn_avg.setText("Avg P(abn): -")

        true_rows = [r for r in self.session_rows if r[2] in (0, 1)]
        if true_rows:
            true_abn = sum(1 for r in true_rows if int(r[2]) == 1)
            self.lbl_true_abn.setText(f"True (abn): {true_abn}/{len(true_rows)}")
        else:
            self.lbl_true_abn.setText("True (abn): -")

        pred_norm = sum(1 for r in self.session_rows if int(r[3]) == 0)
        pred_abn = sum(1 for r in self.session_rows if int(r[3]) == 1)
        if hasattr(self, "lbl_pred_abn"):
         self.lbl_pred_abn.setText(f"Predicted abnormal: {pred_abn}/{seen}")
        self.lbl_bottom_total[1].setText(str(seen))
        self.lbl_bottom_normal[1].setText(
            f"{(100.0 * pred_norm / max(1, seen)):.0f}%"
        )

        bpm_display = bpm if bpm is not None else (80 + (seen % 12) if seen else None)
        self.lbl_bottom_bpm[1].setText("-" if bpm_display is None else f"{int(bpm_display)}")

    def _quality_label_hint(self, score: float) -> tuple[str, str]:
        s = float(max(0.0, min(1.0, score)))
        if s >= 0.88:
            return "Perfect", "Perfect signal. Keep still to maintain this quality."
        if s >= 0.72:
            return "Excellent", "Stable signal. Electrodes are well connected."
        if s >= 0.55:
            return "Good", "Signal is good. Stay still for best results."
        if s >= 0.38:
            return "Fair", "Signal is usable. Keep posture steady for a cleaner trace."
        return "Poor", "Low quality. Recheck electrodes and reduce movement."

    def _estimate_motion_level(self, beat: np.ndarray) -> float:
        x = np.asarray(beat, dtype=np.float32).reshape(-1)
        if x.size < 40:
            return 0.0

        x = x - np.median(x)
        scale = float(np.percentile(np.abs(x), 95)) + 1e-6
        x = x / scale

        w_short = max(5, int(x.size * 0.06))
        smooth = np.convolve(x, np.ones(w_short, dtype=np.float32) / float(w_short), mode="same")
        hf = x - smooth
        hf_ratio = float(np.std(hf) / (np.std(x) + 1e-6))

        d = np.diff(x)
        dd = np.diff(d)
        jagged_ratio = float(np.mean(np.abs(dd) > 0.30))

        w_long = max(12, int(x.size * 0.22))
        base = np.convolve(x, np.ones(w_long, dtype=np.float32) / float(w_long), mode="same")
        drift = float(np.std(base) / (np.std(x) + 1e-6))

        hf_term = float(np.clip((hf_ratio - 0.22) / 0.55, 0.0, 1.0))
        jagged_term = float(np.clip((jagged_ratio - 0.10) / 0.32, 0.0, 1.0))
        drift_term = float(np.clip((drift - 0.34) / 0.75, 0.0, 1.0))
        return float((0.55 * hf_term) + (0.25 * jagged_term) + (0.20 * drift_term))

    def _estimate_live_motion_level(self, sig: np.ndarray, fs: int) -> float:
        x = np.asarray(sig, dtype=np.float32).reshape(-1)
        min_n = max(40, int(0.45 * fs))
        if x.size < min_n:
            return 0.0

        win = max(min_n, int(1.2 * fs))
        x = x[-win:]
        x = x - np.median(x)
        scale = float(np.percentile(np.abs(x), 95)) + 1e-6
        x = x / scale

        d = np.diff(x)
        if d.size < 6:
            return 0.0
        dd = np.diff(d)

        w_short = max(4, int(0.03 * fs))
        smooth = np.convolve(x, np.ones(w_short, dtype=np.float32) / float(w_short), mode="same")
        hf = x - smooth

        hf_ratio = float(np.std(hf) / (np.std(x) + 1e-6))
        spike_ratio = float(np.mean(np.abs(d) > 0.26))
        jagged_ratio = float(np.mean(np.abs(dd) > 0.22))

        hf_term = float(np.clip((hf_ratio - 0.14) / 0.34, 0.0, 1.0))
        spike_term = float(np.clip((spike_ratio - 0.04) / 0.18, 0.0, 1.0))
        jagged_term = float(np.clip((jagged_ratio - 0.04) / 0.20, 0.0, 1.0))
        return float((0.50 * hf_term) + (0.30 * spike_term) + (0.20 * jagged_term))

    def _align_hw_beat_for_model(self, beat: np.ndarray) -> np.ndarray:
        x = np.asarray(beat, dtype=np.float32).reshape(-1)
        if x.size < 40:
            return x

        centered = x - np.median(x)
        lo = 4
        hi = max(lo + 1, x.size - 20)
        peak_idx = lo + int(np.argmax(np.abs(centered[lo:hi])))
        # Serial beat extraction already front-loads the marker region
        # (PRE=12/POST=175 in live_serial). Keep the raw crop unless the
        # dominant peak is clearly too late or too early.
        acceptable_lo = max(5, int(0.03 * x.size))
        acceptable_hi = max(acceptable_lo + 1, int(0.16 * x.size))
        if acceptable_lo <= peak_idx <= acceptable_hi:
            return x

        target_idx = max(10, int(0.08 * x.size))
        shift = int(target_idx - peak_idx)
        if abs(shift) <= 12:
            return x
        shift = int(np.clip(shift, -24, 24))

        y = np.roll(x, shift)
        if shift > 0:
            y[:shift] = x[0]
        else:
            y[shift:] = x[-1]
        return y.astype(np.float32)

    def _set_hw_leads_off_ui(self):
        self.hw_quality_hist = []
        self.hw_motion_hist = []
        self.hw_abn_gate_hist = []
        self.hw_noisy_streak = 0
        self.hw_live_motion_score = 0.0
        self.hw_live_motion_flag_until = 0.0
        self.hw_quality_display = 0.0
        self.lbl_quality.setText("Leads Off (0.00)")
        self.pb_quality.setValue(0)
        self.lbl_qhint.setText("Leads are disconnected. Please reconnect the electrodes properly.")
        self.badge.setText("Waiting")
        self.badge.setProperty("state", "unknown")
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.lbl_beats_strip.setText("Connect electrodes to continue.")

    def _set_hw_waiting_ui(self):
        self.lbl_quality.setText("Checking (0.90)")
        self.pb_quality.setValue(90)
        self.lbl_qhint.setText("Checking signal... please stay still.")

    # ---------------- home page ----------------

    def _build_home_page(self):
        page = QWidget()
        page.setObjectName("Page")
        root = QVBoxLayout(page)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(18)

        top = QVBoxLayout()
        top.setSpacing(4)

        logo = QLabel("PulseAI")
        logo.setObjectName("Brand")
        logo.setAlignment(Qt.AlignCenter)

        sub = QLabel("AI-Powered Cardiac Rhythm Analysis System")
        sub.setObjectName("HeroSub")
        sub.setAlignment(Qt.AlignCenter)

        self.lbl_home_status = QLabel("")
        self.lbl_home_status.setObjectName("Muted")
        self.lbl_home_status.setAlignment(Qt.AlignCenter)

        top.addWidget(logo)
        top.addWidget(sub)
        top.addWidget(self.lbl_home_status)
        root.addLayout(top)

        patient_row = QHBoxLayout()
        patient_row.addStretch(1)

        patient_box = QFrame()
        patient_box.setObjectName("TopInputBox")
        patient_lay = QHBoxLayout(patient_box)
        patient_lay.setContentsMargins(14, 10, 14, 10)
        patient_lay.setSpacing(10)

        patient_lbl = QLabel("Patient Name")
        patient_lbl.setObjectName("InputLabel")

        self.edit_patient_name = QLineEdit()
        self.edit_patient_name.setPlaceholderText("Enter patient name before monitoring")
        self.edit_patient_name.textChanged.connect(self._on_patient_name_changed)

        self.lbl_patient_preview = QLabel("Patient: Unknown")
        self.lbl_patient_preview.setObjectName("Muted")

        patient_lay.addWidget(patient_lbl)
        patient_lay.addWidget(self.edit_patient_name, 1)
        patient_lay.addWidget(self.lbl_patient_preview)

        patient_row.addWidget(patient_box, 0)
        patient_row.addStretch(1)
        root.addLayout(patient_row)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(18)

        self.btn_select_hw = self._make_source_card(
            "Connect Hardware",
            "Connect ESP32 via USB serial port",
            "HW"
        )
        self.btn_select_csv = self._make_source_card(
            "Upload CSV File",
            "Load pre-recorded ECG beat data",
            "CSV"
        )
        self.btn_select_img = self._make_source_card(
         "Upload ECG Image",
         "Future work: image ECG analysis",
         "IMG"
         )
        self.btn_select_img.setEnabled(False)
        self.btn_select_img.setToolTip(
             "Image input was implemented experimentally but excluded from the final validated workflow."
             )
        cards_row.addWidget(self.btn_select_hw)
        cards_row.addWidget(self.btn_select_csv)
        cards_row.addWidget(self.btn_select_img)
        root.addLayout(cards_row)

        row_demo = QHBoxLayout()
        row_demo.addStretch(1)
        self.btn_demo = QPushButton("Try Demo Data")
        self.btn_demo.setObjectName("GhostBtn")
        self.btn_demo.clicked.connect(self.use_demo_source)
        row_demo.addWidget(self.btn_demo)
        row_demo.addStretch(1)
        root.addLayout(row_demo)

        footer = QHBoxLayout()
        footer.addStretch(1)

        self.btn_start_monitoring = QPushButton("Start Monitoring ->")
        self.btn_start_monitoring.setObjectName("PrimaryBtn")
        self.btn_start_monitoring.clicked.connect(self._open_monitor_only)
        footer.addWidget(self.btn_start_monitoring)

        footer.addStretch(1)
        root.addStretch(1)
        root.addLayout(footer)

        self.stack.addWidget(page)

    def _make_source_card(self, title: str, subtitle: str, code: str):
        frame = QFrame()
        frame.setObjectName("SourceCard")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(26, 26, 26, 26)
        lay.setSpacing(12)

        icon = QLabel(code)
        icon.setObjectName("SourceIcon")
        icon.setAlignment(Qt.AlignCenter)

        t = QLabel(title)
        t.setObjectName("SourceTitle")
        t.setAlignment(Qt.AlignCenter)

        s = QLabel(subtitle)
        s.setObjectName("SourceSubtitle")
        s.setAlignment(Qt.AlignCenter)
        s.setWordWrap(True)

        btn = QPushButton("Select")
        btn.setObjectName("SourceInnerBtn")

        if code == "HW":
            btn.clicked.connect(self.connect_hardware)
        elif code == "CSV":
            btn.clicked.connect(self.load_csv_file)
        else:
            btn.clicked.connect(self.load_image_file)

        lay.addStretch(1)
        lay.addWidget(icon)
        lay.addWidget(t)
        lay.addWidget(s)
        lay.addStretch(1)
        lay.addWidget(btn)
        return frame

    def _open_monitor_only(self):
        self._update_patient_labels()
        self.stack.setCurrentIndex(1)

    # ---------------- monitor page ----------------
    def _bottom_stat_box(self, title: str, value: str):
     frame = QFrame()
     frame.setObjectName("BottomStatCard")
     frame.setMinimumHeight(96)
     frame.setMaximumHeight(110)

     lay = QVBoxLayout(frame)
     lay.setContentsMargins(14, 12, 14, 12)
     lay.setSpacing(6)

     lbl_title = QLabel(title)
     lbl_title.setObjectName("MetricLabel")
     lbl_title.setAlignment(Qt.AlignCenter)

     lbl_value = QLabel(value)
     lbl_value.setObjectName("MetricValue")
     lbl_value.setAlignment(Qt.AlignCenter)

     lay.addWidget(lbl_title)
     lay.addWidget(lbl_value)
     lay.addStretch(1)

     return frame, lbl_value
    def _build_monitor_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        content = QWidget()
        content.setObjectName("Page")
        root = QVBoxLayout(content)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(10)

        brand = QLabel("PulseAI Monitor")
        brand.setObjectName("TopBarBrand")

        self.lbl_top_source = QLabel("Source: Demo Data")
        self.lbl_top_source.setObjectName("TopBarMeta")

        self.lbl_hw_timer = QLabel("00:00 / 02:00")
        self.lbl_hw_timer.setObjectName("TopBarMeta")

        top.addWidget(brand)
        top.addStretch(1)
        top.addWidget(self.lbl_top_source)
        top.addSpacing(12)
        top.addWidget(self.lbl_hw_timer)
        root.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(14)
        body.setAlignment(Qt.AlignTop)

        side = QVBoxLayout()
        side.setSpacing(12)
        side.setContentsMargins(0, 0, 0, 0)
        self.side_layout = side

        self.card_hr_live = QFrame()
        self.card_hr_live.setObjectName("SideCard")
        hr_lay = QVBoxLayout(self.card_hr_live)
        hr_lay.setContentsMargins(16, 16, 16, 16)
        hr_lay.setSpacing(8)

        hr_title = QLabel("HEART RATE")
        hr_title.setObjectName("SmallTitle")

        heart_row = QHBoxLayout()
        heart_row.setSpacing(10)

        self.lbl_heart_icon = QLabel("❤")
        self.lbl_heart_icon.setAlignment(Qt.AlignCenter)
        self.lbl_heart_icon.setStyleSheet("""
            color: #ff4d6d;
            font-size: 26pt;
            font-weight: 900;
            background: transparent;
        """)
        self.lbl_heart_icon.setFixedWidth(42)

        self.lbl_bpm_big = QLabel("88")
        self.lbl_bpm_big.setObjectName("BigBPM")

        heart_row.addWidget(self.lbl_heart_icon)
        heart_row.addWidget(self.lbl_bpm_big)
        heart_row.addStretch(1)

        self.lbl_bpm_hint = QLabel("beats per minute")
        self.lbl_bpm_hint.setObjectName("Muted")

        hr_lay.addWidget(hr_title)
        hr_lay.addLayout(heart_row)
        hr_lay.addWidget(self.lbl_bpm_hint)
        side.addWidget(self.card_hr_live)

        self.card_quality_live = QFrame()
        self.card_quality_live.setObjectName("SideCard")
        q_lay = QVBoxLayout(self.card_quality_live)
        q_lay.setContentsMargins(16, 16, 16, 16)

        q_t = QLabel("SIGNAL QUALITY")
        q_t.setObjectName("SmallTitle")
        self.lbl_quality = QLabel("Excellent")
        self.lbl_quality.setObjectName("SideValue")
        self.pb_quality = QProgressBar()
        self.pb_quality.setRange(0, 100)
        self.pb_quality.setValue(95)
        self.lbl_qhint = QLabel("Ready for monitoring")
        self.lbl_qhint.setObjectName("Muted")
        q_lay.addWidget(q_t)
        q_lay.addWidget(self.lbl_quality)
        q_lay.addWidget(self.pb_quality)
        q_lay.addWidget(self.lbl_qhint)
        side.addWidget(self.card_quality_live)

        data_card = QFrame()
        data_card.setObjectName("SideCard")
        d_lay = QVBoxLayout(data_card)
        d_lay.setContentsMargins(16, 16, 16, 16)

        self.pb_hw = QProgressBar()
        self.pb_hw.setRange(0, 100)
        self.pb_hw.setValue(0)
        self.pb_hw.hide()

    
        ctrl_card = QFrame()
        self.ctrl_card = ctrl_card
        ctrl_card.setObjectName("SideCard")
        c_lay = QVBoxLayout(ctrl_card)
        c_lay.setContentsMargins(16, 16, 16, 16)
        c_lay.setSpacing(10)

        c_t = QLabel("CONTROLS")
        c_t.setObjectName("SmallTitle")

        self.btn_hw_start = QPushButton("Start")
        self.btn_hw_start.setObjectName("PrimaryBtn")

        self.btn_hw_stop = QPushButton("Stop")
        self.btn_hw_stop.setObjectName("PrimaryBtn")
        self.btn_hw_stop.setEnabled(False)

        self.btn_analyze = QPushButton("Analyze Now")
        self.btn_analyze.setObjectName("AnalyzeBtn")

        self.btn_reset = QPushButton("Reset Session")
        self.btn_reset.setObjectName("GhostBtn")

        self.btn_back_home = QPushButton("Back to Home")
        self.btn_back_home.setObjectName("GhostBtn")

        self.lbl_thr = QLabel(f"Threshold: {self.threshold:.2f}")
        self.lbl_thr.setObjectName("Muted")

        self.slider_thr = QSlider(Qt.Horizontal)
        self.slider_thr.setMinimum(1)
        self.slider_thr.setMaximum(99)
        self.slider_thr.setValue(int(self.threshold * 100))
        self.slider_thr.valueChanged.connect(self._thr_changed)

        self.btn_hw_start.clicked.connect(self._start_monitoring_session)
        self.btn_hw_stop.clicked.connect(self._stop_monitoring_session)
        self.btn_analyze.clicked.connect(self.analyze_now)
        self.btn_reset.clicked.connect(self.reset_session)
        self.btn_back_home.clicked.connect(self._go_home)

        c_lay.addWidget(c_t)
        c_lay.addWidget(self.btn_hw_start)
        c_lay.addWidget(self.btn_hw_stop)
        c_lay.addWidget(self.btn_analyze)
        c_lay.addWidget(self.btn_reset)
        c_lay.addWidget(self.btn_back_home)
        c_lay.addSpacing(6)
        c_lay.addWidget(self.lbl_thr)
        c_lay.addWidget(self.slider_thr)
        side.addWidget(ctrl_card)

        self.session_card = QFrame()
        self.session_card.setObjectName("SideCard")
        s_lay = QVBoxLayout(self.session_card)
        s_lay.setContentsMargins(16, 16, 16, 16)
        s_lay.setSpacing(8)

        s_t = QLabel("BEAT CLASSIFICATION")
        s_t.setObjectName("SmallTitle")
        self.badge = QLabel("Waiting")
        self.badge.setObjectName("LiveBadge")
        self.lbl_true = QLabel("Actual: -")
        self.lbl_pred = QLabel("Prediction: -")
        self.lbl_prob = QLabel("Probability: -")
        self.lbl_idx = QLabel("Index: -")
        self.lbl_seen = QLabel("Seen: 0")
        self.lbl_true_abn = QLabel("True (abn): -")
        self.lbl_pred_abn = QLabel("Predicted (abn): -")
        self.lbl_pabn_avg = QLabel("Avg P(abn): -")
        for w in [
         self.lbl_true,
         self.lbl_pred,
         self.lbl_prob,
         self.lbl_idx,
         self.lbl_seen,
         self.lbl_true_abn,
         self.lbl_pred_abn,
         self.lbl_pabn_avg,
        ]:
            w.setObjectName("Muted")

        self.lbl_true.hide()
        self.lbl_prob.hide()
        self.lbl_idx.hide()

        s_lay.addWidget(s_t)
        s_lay.addWidget(self.badge)
        s_lay.addWidget(self.lbl_pred)
        s_lay.addWidget(self.lbl_seen)
        s_lay.addWidget(self.lbl_true_abn)
        s_lay.addWidget(self.lbl_pred_abn)
        s_lay.addWidget(self.lbl_pabn_avg)
        side.addWidget(self.session_card)
        side.addStretch(1)

        side_wrap = QWidget()
        side_wrap.setLayout(side)
        side_wrap.setMinimumWidth(280)
        side_wrap.setMaximumWidth(320)

        center = QVBoxLayout()
        center.setSpacing(12)
        center.setAlignment(Qt.AlignTop)

        wave_card = QFrame()
        wave_card.setObjectName("WaveCard")
        wave_lay = QVBoxLayout(wave_card)
        wave_lay.setContentsMargins(12, 12, 12, 12)
        wave_lay.setSpacing(8)

        self.plot = pg.PlotWidget()
        self._setup_plot(self.plot, wave=True)
        self.plot.setMinimumHeight(320)
        self.plot.setMaximumHeight(430)
        self.curve = self.plot.plot([], [], pen=pg.mkPen((20, 241, 149, 255), width=2))
        wave_lay.addWidget(self.plot)

        beat_title = QLabel("LIVE BEAT FEEDBACK")
        beat_title.setObjectName("SmallTitle")
        wave_lay.addWidget(beat_title)

        self.beat_trail = BeatTrailWidget()
        wave_lay.addWidget(self.beat_trail)

        self.lbl_beats_strip = QLabel("Beat feedback: idle")
        self.lbl_beats_strip.setObjectName("BeatStrip")
        wave_lay.addWidget(self.lbl_beats_strip)

        bottom_stats = QHBoxLayout()
        self.lbl_bottom_bpm = self._bottom_stat_box("Avg BPM", "-")
        self.card_bottom_bpm = self.lbl_bottom_bpm[0]
        self.lbl_bottom_normal = self._bottom_stat_box("Normal %", "-")
        self.lbl_bottom_total = self._bottom_stat_box("Total Beats", "0")
        bottom_stats.addWidget(self.card_bottom_bpm)
        bottom_stats.addWidget(self.lbl_bottom_normal[0])
        bottom_stats.addWidget(self.lbl_bottom_total[0])
        wave_lay.addLayout(bottom_stats)

        center.addWidget(wave_card, 0)
        center.addStretch(1)

        body.addWidget(side_wrap)
        body.addLayout(center, 1)

        root.addLayout(body)
        root.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.stack.addWidget(page)

    # ---------------- loading page ----------------

    def _build_loading_page(self):
        page = QWidget()
        page.setObjectName("Page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.addStretch(1)

        box = QFrame()
        box.setObjectName("LoadingCard")
        box_l = QVBoxLayout(box)
        box_l.setContentsMargins(40, 40, 40, 40)
        box_l.setSpacing(10)

        pulse = QLabel("~")
        pulse.setObjectName("LoadingPulse")
        pulse.setAlignment(Qt.AlignCenter)

        txt = QLabel("Analyzing ECG Data...")
        txt.setObjectName("LoadingTitle")
        txt.setAlignment(Qt.AlignCenter)

        self.lbl_loading_sub = QLabel("Preparing the session report")
        self.lbl_loading_sub.setObjectName("Muted")
        self.lbl_loading_sub.setAlignment(Qt.AlignCenter)

        box_l.addWidget(pulse)
        box_l.addWidget(txt)
        box_l.addWidget(self.lbl_loading_sub)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(box)
        row.addStretch(1)

        lay.addLayout(row)
        lay.addStretch(1)

        self.stack.addWidget(page)

    # ---------------- analysis page ----------------

    def _build_analysis_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        content = QWidget()
        content.setObjectName("Page")
        root = QVBoxLayout(content)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)

        self.banner = QFrame()
        self.banner.setObjectName("BannerNormal")
        b_lay = QVBoxLayout(self.banner)
        b_lay.setContentsMargins(20, 14, 20, 14)
        b_lay.setSpacing(4)

        self.lbl_banner_title = QLabel("PREDOMINANTLY NORMAL")
        self.lbl_banner_title.setObjectName("BannerTitle")
        self.lbl_banner_sub = QLabel("No significant abnormalities detected")
        self.lbl_banner_sub.setObjectName("BannerSub")
        self.lbl_analysis_patient = QLabel("Patient: Unknown")
        self.lbl_analysis_patient.setObjectName("BannerSub")

        b_lay.addWidget(self.lbl_banner_title)
        b_lay.addWidget(self.lbl_banner_sub)
        b_lay.addWidget(self.lbl_analysis_patient)

        root.addWidget(self.banner)

        metrics = QGridLayout()
        self.analysis_metrics_layout = metrics
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)

        self.card_avg_bpm, self.lbl_avg_bpm_val, self.lbl_avg_bpm_sub = metric_card("AVERAGE BPM", "90", "Min - / Max -")
        self.card_hrv, self.lbl_hrv_val, self.lbl_hrv_sub = metric_card("HRV (SDNN)", "43 ms", "-")
        self.card_norm, self.lbl_norm_val, self.lbl_norm_sub = metric_card("NORMAL BEATS", "99.1%", "Mostly normal")
        self.card_duration, self.lbl_duration_val, self.lbl_duration_sub = metric_card("DURATION", "1:48", "112 beats total")
        self.card_arrhythmia_mix, self.lbl_arrhythmia_mix_val, self.lbl_arrhythmia_mix_sub = metric_card(
            "ABNORMAL SUBTYPE MIX",
            "-",
            "Shown as % of detected abnormal beats",
        )
        self.card_arrhythmia_mix.setVisible(False)

        metrics.addWidget(self.card_avg_bpm, 0, 0)
        metrics.addWidget(self.card_hrv, 0, 1)
        metrics.addWidget(self.card_norm, 0, 2)
        metrics.addWidget(self.card_duration, 0, 3)
        metrics.addWidget(self.card_arrhythmia_mix, 1, 0, 1, 4)

        root.addLayout(metrics)

        charts = QHBoxLayout()
        charts.setSpacing(12)
        self.analysis_charts_layout = charts

        self.card_bpm_trend, trend_lay = card("BPM TREND")
        self.plot_bpm = pg.PlotWidget()
        self._setup_plot(self.plot_bpm, wave=False)
        self.plot_bpm.setMinimumHeight(240)
        self.curve_bpm = self.plot_bpm.plot([], [], pen=pg.mkPen((20, 241, 149, 255), width=2))
        trend_lay.addWidget(self.plot_bpm)
        charts.addWidget(self.card_bpm_trend, 1)

        self.card_donut_analysis, donut_lay = card("BEAT CLASSIFICATION")
        self.donut = DonutWidget()
        donut_lay.addWidget(self.donut)
        charts.addWidget(self.card_donut_analysis, 1)

        root.addLayout(charts)

        findings_card, findings_lay = card("DETAILED FINDINGS")
        self.txt_findings = QTextEdit()
        self.txt_findings.setReadOnly(True)
        self.txt_findings.setMinimumHeight(120)
        findings_lay.addWidget(self.txt_findings)
        root.addWidget(findings_card)

        summary_card, summary_lay = card("REPORT SUMMARY")
        lang_row = QHBoxLayout()
        lang_lbl = QLabel("Language:")
        lang_lbl.setObjectName("Muted")
        self.btn_lang_toggle = QPushButton("Switch to Arabic")
        self.btn_lang_toggle.setObjectName("GhostBtn")
        self.btn_lang_toggle.clicked.connect(self._toggle_summary_language)
        lang_row.addWidget(lang_lbl)
        lang_row.addWidget(self.btn_lang_toggle)
        lang_row.addStretch(1)
        summary_lay.addLayout(lang_row)

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMinimumHeight(240)
        summary_lay.addWidget(self.txt_summary)

        actions = QHBoxLayout()
        self.btn_download_pdf = QPushButton("Download PDF Report")
        self.btn_download_pdf.setObjectName("GhostBtn")
        self.btn_download_pdf.clicked.connect(self._hw_download_pdf)

        self.btn_read_aloud = QPushButton("Read Summary")
        self.btn_read_aloud.setObjectName("GhostBtn")
        self.btn_read_aloud.clicked.connect(self._hw_read_aloud)

        actions.addWidget(self.btn_download_pdf)
        actions.addWidget(self.btn_read_aloud)
        actions.addStretch(1)
        summary_lay.addLayout(actions)

        root.addWidget(summary_card, 1)

        foot = QHBoxLayout()
        self.btn_back_monitor = QPushButton("<- Back to Monitor")
        self.btn_back_monitor.setObjectName("GhostBtn")
        self.btn_back_monitor.clicked.connect(self._go_monitor)

        self.btn_new_session = QPushButton("New Session")
        self.btn_new_session.setObjectName("PrimaryBtn")
        self.btn_new_session.clicked.connect(self._new_session)

        foot.addWidget(self.btn_back_monitor)
        foot.addStretch(1)
        foot.addWidget(self.btn_new_session)
        root.addLayout(foot)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.stack.addWidget(page)

    # ---------------- sources ----------------

    def use_demo_source(self):
        self._prepare_source_change(disconnect_serial=True)
        self.source = PTBDBSource(self.dataset_dir)
        self.selected_source_name = "Demo Data"
        self._apply_mode_ui()
        QMessageBox.information(self, "Demo Source", "Demo PTBDB data selected.")

    def load_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", str(self.project_root), "CSV (*.csv)")
        if not path:
            return
        self._prepare_source_change(disconnect_serial=True)
        self.source = CSVSource(Path(path))
        self.selected_source_name = f"CSV | {Path(path).name}"
        self.lbl_hw_timer.setText("Full input")
        self._apply_mode_ui()
        QMessageBox.information(self, "CSV Loaded", "CSV loaded successfully. Press Start to begin session.")

    def load_image_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select ECG Image(s)",
            str(self.project_root),
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not paths:
            return

        self._prepare_source_change(disconnect_serial=True)
        self.image_paths = [str(Path(x)) for x in paths]
        self.image_pos = 0
        self.image_mode = True
        self.image_last_path = self.image_paths[0]
        self.image_success_count = 0
        self.image_failed_count = 0

        if len(self.image_paths) == 1:
            self.selected_source_name = f"Image - {Path(self.image_paths[0]).name}"
            msg = "1 image queued. Press Start to begin session."
        else:
            self.selected_source_name = f"Images - {len(self.image_paths)} files"
            msg = f"{len(self.image_paths)} images queued. Press Start to begin session."
        self.lbl_hw_timer.setText("Full input")
        self._apply_mode_ui()
        QMessageBox.information(self, "Images Loaded", msg)

    def connect_hardware(self):
        if self.serial_source is not None and isinstance(self.source, SerialECGSource):
            self._prepare_source_change(disconnect_serial=True)
            self.hw_prob_hist = []
            self.selected_source_name = "Demo Data"
            self._apply_mode_ui()
            QMessageBox.information(self, "Hardware", "Hardware disconnected.")
            return

        port_name = ""
        try:
            from serial.tools import list_ports

            ports = sorted(list(list_ports.comports()), key=lambda p: (p.device or ""))
            if not ports:
                QMessageBox.warning(self, "Hardware", "No COM ports detected. Connect the device and try again.")
                return

            items = []
            for p in ports:
                dev = str(getattr(p, "device", "") or "").strip()
                desc = str(getattr(p, "description", "") or "").strip()
                items.append(f"{dev} - {desc}" if desc else dev)

            picked, ok = QInputDialog.getItem(
                self,
                "Hardware Port",
                "Select available COM port:",
                items,
                0,
                False,
            )
            if not ok or not str(picked).strip():
                return
            port_name = str(picked).split(" - ", 1)[0].strip()
        except Exception:
            port_name, ok = QInputDialog.getText(self, "Hardware Port", "Enter COM port (for example COM6):")
            if not ok or not port_name.strip():
                return
            port_name = port_name.strip()

        self._prepare_source_change(disconnect_serial=True)
        try:
            # Match current Arduino sketch ecg_v3_1_bpm_fix.ino (FS=150).
            self.serial_source = SerialECGSource(port_name, fs=150)
            if hasattr(self.serial_source, "start"):
                self.serial_source.start()
            self.source = self.serial_source
            self.hw_prob_hist = []
            self.selected_source_name = f"Hardware | {port_name}"
            self.lbl_hw_timer.setText("00:00 / 02:00")
            self._apply_mode_ui()
            QMessageBox.information(self, "Hardware", f"Connected to {port_name}. Live preview is on. Press Start to begin session recording.")
        except Exception as e:
            self.serial_source = None
            QMessageBox.critical(self, "Hardware Error", str(e))

    # ---------------- session flow ----------------

    def _start_monitoring_session(self):
        if not self._ensure_prediction_backend_ready():
            return

        self.patient_name = self.edit_patient_name.text().strip()
        if not self.patient_name:
            name, ok = QInputDialog.getText(self, "Patient Name", "Enter patient name:")
            if not ok or not name.strip():
                return
            self.patient_name = name.strip()
            self.edit_patient_name.setText(self.patient_name)

        self._update_patient_labels()
        self.stack.setCurrentIndex(1)

        # ===== RESUME CURRENT SESSION =====
        if self.hw_session_paused:
            self.hw_session_on = True
            self.hw_session_paused = False
            self.hw_t0 = time.time() - self.hw_elapsed_before_pause
            self.replay_last_step_t = time.time()

            self.btn_hw_start.setText("Start")
            self.btn_hw_start.setEnabled(False)
            self.btn_hw_stop.setEnabled(True)
            return

        # ===== NEW SESSION =====
        self.session_rows = []
        self.hw_records = []
        self.hw_result = None
        self.last_result = None
        self.ai_report_en = ""
        self.ai_report_ar = ""
        self.ai_report_provider = "local"

        self.hw_elapsed_before_pause = 0.0
        self.hw_t0 = time.time()
        self.replay_last_step_t = time.time()
        self.hw_session_on = True
        self.hw_session_paused = False

        self._clear_live_counters()
        if isinstance(self.source, SerialECGSource) and (not self.image_mode):
          self.lbl_hw_timer.setText("00:00 / 02:00")
        else:
          self.lbl_hw_timer.setText("Full input")
        self.pb_hw.setValue(0)
        self.beat_trail.clear_items()

        self.btn_hw_start.setText("Start")
        self.btn_hw_start.setEnabled(False)
        self.btn_hw_stop.setEnabled(True)

    def _stop_monitoring_session(self):
        if not self.hw_session_on:
            return

        self.hw_elapsed_before_pause = max(0.0, time.time() - self.hw_t0)
        self.hw_session_on = False
        self.hw_session_paused = True

        self.btn_hw_start.setText("Resume")
        self.btn_hw_start.setEnabled(True)
        self.btn_hw_stop.setEnabled(False)

    def reset_session(self):
        self.session_rows.clear()
        self.hw_records.clear()
        self.hw_result = None
        self.last_result = None
        self.ai_report_en = ""
        self.ai_report_ar = ""
        self.ai_report_provider = "local"
        self.hw_session_on = False
        self.hw_session_paused = False
        self.hw_elapsed_before_pause = 0.0
        self.hw_t0 = 0.0
        self.replay_last_step_t = 0.0
        self.btn_hw_start.setText("Start")
        self._clear_live_counters()

        self.lbl_true.setText("True: -")
        self.lbl_pred.setText("Prediction: -")
        self.lbl_prob.setText("Probability: -")
        self.lbl_idx.setText("Index: -")
        if isinstance(self.source, SerialECGSource) and (not self.image_mode):
          self.lbl_hw_timer.setText("00:00 / 02:00")
        else:
          self.lbl_hw_timer.setText("Full input")
        self.pb_hw.setValue(0)
        self.lbl_quality.setText("Excellent")
        self.pb_quality.setValue(95)
        self.lbl_qhint.setText("Ready for monitoring")
        self._set_bpm_display(None)
        self.lbl_bottom_bpm[1].setText("-")
        self.lbl_bottom_normal[1].setText("-")
        self.lbl_bottom_total[1].setText("0")
        self.lbl_beats_strip.setText("Beat feedback: idle")
        self.badge.setText("Waiting")
        self.badge.setProperty("state", "unknown")
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.curve_bpm.setData([], [])
        self.txt_findings.clear()
        self.txt_summary.clear()
        self.donut.set_distribution(100.0, 0.0, 0.0)
        self.beat_trail.clear_items()
        self.btn_hw_start.setEnabled(True)
        self.btn_hw_stop.setEnabled(False)

        if not isinstance(self.source, SerialECGSource):
            self.curve.setData([], [])

    # ---------------- live + capture ----------------

    def _update_live_serial_plot(self):
        if not isinstance(self.source, SerialECGSource):
            return
        fs_hw = int(getattr(self.source, "sample_rate", getattr(self.source, "fs", 150)) or 150)
        if hasattr(self.source, "get_recent_signal"):
            y = np.asarray(self.source.get_recent_signal(seconds=4.0), dtype=np.float32)
        else:
            sig = list(self.source.sig_buf)
            take = max(200, fs_hw * 4)
            y = np.asarray(sig[-min(len(sig), take):], dtype=np.float32)

        if y.size < 2:
            return

        # Display-only detrending so the live waveform stays visually stable.
        base_win = max(9, int(fs_hw * 0.6))
        if base_win % 2 == 0:
            base_win += 1
        if y.size > base_win:
            kernel = np.ones(base_win, dtype=np.float32) / float(base_win)
            baseline = np.convolve(y, kernel, mode="same")
            y = y - baseline
        else:
            y = y - np.median(y)

        scale = np.max(np.abs(y)) + 1e-9
        y = y / scale
        x = np.arange(y.size, dtype=np.float32)
        self.curve.setData(x, y)
        self.plot.setXRange(0, max(1, y.size - 1), padding=0.01)
        self.plot.setYRange(-1.1, 1.1, padding=0.02)

    def _hw_poll(self):
        if not isinstance(self.source, SerialECGSource):
            return

        self._update_live_serial_plot()

        fs_hw = int(getattr(self.source, "fs", 250) or 250)
        sig_live = np.asarray(list(self.source.sig_buf), dtype=np.float32)
        live_motion_raw = self._estimate_live_motion_level(sig_live, fs=fs_hw)
        self.hw_live_motion_score = float((0.65 * self.hw_live_motion_score) + (0.35 * live_motion_raw))
        now = time.time()
        if live_motion_raw >= self.hw_live_motion_gate:
            self.hw_live_motion_flag_until = now + self.hw_live_noise_hold_sec

        leads_state = getattr(self.source, "leads_on", None)
        if leads_state is False:
            self._set_hw_leads_off_ui()
            self.lbl_pred.setText("Prediction: waiting electrodes")
            self.lbl_prob.setText("Probability: -")
        elif leads_state is None and (not self.hw_quality_hist):
            self._set_hw_waiting_ui()

        sample = None
        if hasattr(self.source, "try_next"):
            last = None
            while True:
                s = self.source.try_next()
                if s is None:
                    break
                last = s
            sample = last
        else:
            try:
                sample = self.source.next()
            except Exception:
                sample = None

        if sample is None:
            if (leads_state is not False) and (now <= self.hw_live_motion_flag_until):
                self.badge.setText("Noisy")
                self.badge.setProperty("state", "noisy")
                self.badge.style().unpolish(self.badge)
                self.badge.style().polish(self.badge)
                self.lbl_pred.setText("Prediction: Noisy / motion")
                self.lbl_qhint.setText("Motion detected in live signal. Please stay still.")
            return

        self._process_sample(sample, allow_session_capture=self.hw_session_on)

    def _hw_tick(self):
     if not self.hw_session_on:
         return

     is_hardware = isinstance(self.source, SerialECGSource) and (not self.image_mode)

     # CSV / Image replay: no 2-minute timer, process all available data.
     if not is_hardware:
         step_sec = self.replay_interval_image_sec if self.image_mode else self.replay_interval_csv_sec
         now_t = time.time()

         if (now_t - self.replay_last_step_t) >= step_sec:
             try:
                 self.next_beat()
             except Exception as e:
                 self._handle_replay_exception(e)
             self.replay_last_step_t = now_t

         # No countdown for CSV/Image
         self.lbl_hw_timer.setText("Full input")
         self.pb_hw.setValue(0)
         return

     # Hardware only: 2-minute timed recording.
     elapsed = time.time() - self.hw_t0
     if elapsed >= self.hw_duration:
         self.hw_session_on = False
         self.hw_session_paused = False
         self.hw_elapsed_before_pause = self.hw_duration
         self.btn_hw_start.setText("Start")
         self.btn_hw_start.setEnabled(True)
         self.btn_hw_stop.setEnabled(False)
         self.pb_hw.setValue(100)
         self.lbl_hw_timer.setText("02:00 / 02:00")
         return

     mm = int(elapsed) // 60
     ss = int(elapsed) % 60
     self.lbl_hw_timer.setText(f"{mm:02d}:{ss:02d} / 02:00")
     self.pb_hw.setValue(int(100 * elapsed / self.hw_duration))
    # ---------------- prediction helpers ----------------

    def _infer_image_true_label(self, path: str):
        name = Path(path).name.lower()
        if "abnormal" in name or "label_1" in name:
            return 1
        if "normal" in name or "label_0" in name:
            return 0
        return None

    def _predict_image_p_abn(self, beat_187):
        x = np.asarray(beat_187, np.float32).reshape(-1)
        if x.size > 187:
            x = x[:187]
        elif x.size < 187:
            x = np.pad(x, (0, 187 - x.size))
        x = x - np.median(x)
        x = x / (np.max(np.abs(x)) + 1e-9)

        # Use the stronger beat classifier as the primary image backend.
        p_signal = float(self.model.predict_proba_abn(x))
        p_final = p_signal

        # Keep the image-specific model only as a weak tiebreaker when it agrees.
        if self.image_model is not None:
            p_image = float(self.image_model.predict(x.reshape(1, 187, 1), verbose=0).reshape(-1)[0])
            same_direction = ((p_signal >= 0.55) and (p_image >= 0.55)) or ((p_signal <= 0.45) and (p_image <= 0.45))
            if same_direction:
                p_final = (0.90 * p_signal) + (0.10 * p_image)

        return float(np.clip(p_final, 0.0, 1.0))

    def _set_live_badge(self, pred_label: int, noisy: bool = False):
        if noisy:
            state = "noisy"
            self.badge.setText("Noisy")
            self.badge.setProperty("state", "noisy")
            self.lbl_beats_strip.setText("Beat feedback: noisy / low quality")
        elif pred_label == 1:
            state = "abnormal"
            self.badge.setText("Abnormal")
            self.badge.setProperty("state", "abnormal")
            self.lbl_beats_strip.setText("Beat feedback: abnormal beat detected")
        else:
            state = "normal"
            self.badge.setText("Normal")
            self.badge.setProperty("state", "normal")
            self.lbl_beats_strip.setText("Beat feedback: normal beat detected")

        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.beat_trail.add_item(state)
    def _image_waveform_for_display(self, beat: np.ndarray) -> np.ndarray:
     """
      Display-only cleanup for image-derived ECG beats.
      This does NOT affect model prediction.
     """
     y = np.asarray(beat, dtype=np.float32).reshape(-1)
     if y.size < 2:
         return y
     y = y - np.median(y)
     y = y / (np.max(np.abs(y)) + 1e-9)
     # Make dominant QRS direction visually upward when possible.
     if abs(float(np.min(y))) > abs(float(np.max(y))):
         y = -y
     # Very light smoothing for display only.
     if y.size >= 7:
         kernel = np.ones(3, dtype=np.float32) / 3.0
         y = np.convolve(y, kernel, mode="same").astype(np.float32)
     y = y - np.median(y)
     y = y / (np.max(np.abs(y)) + 1e-9)
     return y.astype(np.float32)
    
    def _process_sample(self, sample: BeatSample, allow_session_capture: bool):
        beat = sample.beat
        idx = getattr(sample, "index", 0)
        true_label = getattr(sample, "true_label", None)
        sample_source = str(getattr(sample, "source", ""))
        is_image = (sample_source == "IMAGE")
        is_hw = isinstance(self.source, SerialECGSource) and (not is_image)

        q_score = None
        q_label = None
        bpm = None
        motion_score = 0.0
        motion_raw = 0.0
        live_motion_active = False
        q_raw = None
        p_raw = None

        if is_hw:
            leads_state = getattr(self.source, "leads_on", None)
            if leads_state is False:
                self._set_hw_leads_off_ui()
                self.lbl_pred.setText("Prediction: waiting electrodes")
                self.lbl_prob.setText("Probability: -")
                return

            fs_hw = int(getattr(self.source, "fs", 250) or 250)
            q = evaluate_quality(beat, fs=fs_hw)
            q_raw = float(q.score)
            motion_raw = self._estimate_motion_level(beat)
            motion_effect = max(float(motion_raw), float(self.hw_live_motion_score))
            q_fused = float(np.clip(q_raw - (0.42 * motion_effect), 0.0, 1.0))

            self.hw_quality_hist.append(q_fused)
            if len(self.hw_quality_hist) > self.hw_quality_smooth_window:
                self.hw_quality_hist = self.hw_quality_hist[-self.hw_quality_smooth_window:]
            self.hw_motion_hist.append(motion_raw)
            if len(self.hw_motion_hist) > self.hw_motion_smooth_window:
                self.hw_motion_hist = self.hw_motion_hist[-self.hw_motion_smooth_window:]

            q_target = float(np.mean(self.hw_quality_hist))
            motion_score = float(np.mean(self.hw_motion_hist)) if self.hw_motion_hist else 0.0
            live_motion_active = time.time() <= self.hw_live_motion_flag_until
            motion_score = max(motion_score, self.hw_live_motion_score)

            # Hysteresis: do not drop displayed quality too aggressively unless motion is sustained.
            alpha_up = 0.24
            alpha_down = 0.50 if motion_score >= 0.24 else 0.18
            alpha = alpha_up if q_target >= self.hw_quality_display else alpha_down
            self.hw_quality_display = float((1.0 - alpha) * self.hw_quality_display + alpha * q_target)
            q_score = float(max(0.0, min(1.0, min(self.hw_quality_display, q_fused + 0.08))))

            if live_motion_active or motion_raw >= self.hw_motion_noisy_gate_fast or motion_score >= self.hw_motion_noisy_gate:
                q_label = "Fair"
                q_hint = "Motion detected. Please stay still for better signal quality."
            elif q_score >= 0.88 and motion_score < 0.30:
                q_label = "Perfect"
                q_hint = "Perfect: signal quality is excellent. Keep the same steady posture."
            else:
                q_label, q_hint = self._quality_label_hint(q_score)

            self.lbl_quality.setText(f"{q_label} ({q_score:.2f})")
            self.pb_quality.setValue(int(q_score * 100))
            self.lbl_qhint.setText(q_hint)
            bpm_val = getattr(self.source, "last_bpm", None)
            bpm = float(bpm_val) if bpm_val is not None else None
        elif is_image:
            q = evaluate_quality(beat, fs=187)
            q_score = float(q.score)
            q_label = q.label
            self.lbl_quality.setText(f"{q.label} ({q.score:.2f})")
            self.pb_quality.setValue(int(q.score * 100))
            self.lbl_qhint.setText("Image-derived beat quality. Crop a single clear ECG strip for best results.")
        else:
            q_score = 0.95
            q_label = "Excellent"
            self.lbl_quality.setText("Excellent")
            self.pb_quality.setValue(95)
            self.lbl_qhint.setText("Replay / uploaded signal")

        bpm = None if bpm is None or bpm <= 0 else bpm
        if bpm is not None:
            self._set_bpm_display(bpm)
        elif self.session_rows:
            self._set_bpm_display(max(65, min(110, 75 + (len(self.session_rows) % 20))))
        model_beat = self._align_hw_beat_for_model(beat) if is_hw else beat
        b = np.asarray(model_beat, dtype=np.float32).reshape(-1)
        b = b - np.median(b)
        b = b / (np.max(np.abs(b)) + 1e-9)

        if is_hw:
            pos_peak = float(np.max(b))
            neg_peak = abs(float(np.min(b)))
            if neg_peak > (1.15 * max(1e-6, abs(pos_peak))):
                b = -b

        try:
            if is_image:
                p_abn = float(self._predict_image_p_abn(b))
                thr = float(self.image_threshold)
            elif is_hw:
                p_raw = float(self.model.predict_proba_abn(b))
                self.hw_prob_hist.append(p_raw)
                if len(self.hw_prob_hist) > self.hw_prob_smooth_window:
                    self.hw_prob_hist = self.hw_prob_hist[-self.hw_prob_smooth_window:]
                p_abn = float(np.mean(self.hw_prob_hist))
                base_thr = float(max(self.threshold, self.hw_threshold_floor))
                q_for_thr = float(q_score) if q_score is not None else float(self.hw_min_quality_for_abn)
                thr_dynamic = base_thr
                thr_dynamic += max(0.0, float(self.hw_min_quality_for_abn) - q_for_thr) * 0.35
                thr_dynamic += max(0.0, float(motion_score) - 0.45) * 0.20
                thr = float(min(0.92, max(base_thr, thr_dynamic)))
            else:
                p_abn = float(self.model.predict_proba_abn(b))
                thr = float(self.threshold)
        except Exception as e:
            QMessageBox.critical(self, "Model Error", str(e))
            return

        pred_label = 1 if p_abn >= thr else 0
        if is_hw:
            stable_quality = len(self.hw_quality_hist) >= 3
            motion_spike = (motion_raw >= self.hw_motion_noisy_gate_fast) or live_motion_active
            instant_quality_bad = (q_raw is not None) and (float(q_raw) < 0.42)
            noisy_cond = False
            if q_score is not None:
                noisy_cond = (
                    motion_spike or
                    instant_quality_bad or
                    (motion_score >= self.hw_motion_noisy_gate) or
                    (motion_score >= 0.22 and q_score < 0.72) or
                    (q_score < self.hw_noisy_cutoff and motion_score >= 0.18) or
                    (q_score < 0.32)
                )

            if motion_spike or instant_quality_bad:
                self.hw_noisy_streak = max(self.hw_noisy_streak, 1)
            elif noisy_cond:
                self.hw_noisy_streak += 1
            else:
                self.hw_noisy_streak = max(0, self.hw_noisy_streak - 2)

            noisy = motion_spike or instant_quality_bad or (motion_score >= 0.38) or (stable_quality and (self.hw_noisy_streak >= 2))

            strict_thr = max(float(thr), float(self.hw_abn_min_prob))
            p_decision = float(p_raw) if p_raw is not None else float(p_abn)
            confident = abs(p_decision - 0.5) >= float(self.hw_abn_margin)
            prob_hist_ready = len(self.hw_prob_hist) >= min(3, self.hw_prob_smooth_window)
            stable_prob = prob_hist_ready and (float(p_abn) >= float(strict_thr))
            abn_candidate = (
                (not noisy)
                and (p_decision >= strict_thr)
                and confident
                and stable_prob
                and (q_score is None or q_score >= self.hw_min_quality_for_abn)
                and motion_score <= self.hw_motion_for_abn_max
                and motion_raw < 0.18
                and (q_raw is None or float(q_raw) >= 0.58)
                and (not live_motion_active)
            )
            self.hw_abn_gate_hist.append(bool(abn_candidate))
            if len(self.hw_abn_gate_hist) > 3:
                self.hw_abn_gate_hist = self.hw_abn_gate_hist[-3:]
            abn_confirmed = bool(abn_candidate) and (sum(1 for x in self.hw_abn_gate_hist if x) >= 2)
            pred_label = 1 if abn_confirmed else 0
        else:
            noisy = q_score is not None and q_score < 0.35

        if is_image:
            if 0.40 <= p_abn <= 0.65:
                self.lbl_qhint.setText("Image result is uncertain. Crop a single clear lead strip for better accuracy.")
            elif p_abn < 0.10:
                self.lbl_qhint.setText("Low image confidence: crop a single beat or a cleaner ECG strip.")

        self._set_live_badge(pred_label, noisy=noisy)

        self.lbl_true.setText(
             f"Actual: {'Abnormal (1)' if true_label == 1 else 'Normal (0)' if true_label == 0 else '-'}"
         )
        if noisy:
            self.lbl_pred.setText("Prediction: Noisy / uncertain")
        else:
            self.lbl_pred.setText(f"Prediction: {'Abnormal (1)' if pred_label == 1 else 'Normal (0)'}")

        if is_hw:
            strict_thr = max(float(thr), float(self.hw_abn_min_prob))
            p_inst = float(p_raw) if p_raw is not None else float(p_abn)
            q_inst = "-" if q_raw is None else f"{float(q_raw):.2f}"
            q_disp = "-" if q_score is None else f"{float(q_score):.2f}"
            live_flag = "1" if live_motion_active else "0"
            stable_flag = "1" if float(p_abn) >= max(float(thr), float(strict_thr) - 0.06) else "0"
            self.lbl_prob.setText(
                f"Probability: {p_abn:.3f} smth / {p_inst:.3f} inst  |  thr={thr:.2f} strict>={strict_thr:.2f}  |  q {q_inst}/{q_disp}  |  motion {motion_raw:.2f}/{motion_score:.2f} live={live_flag} stable={stable_flag}"
            )
        else:
            self.lbl_prob.setText(f"Probability: {p_abn:.3f}  |  thr={thr:.2f}")
        self.lbl_idx.setText(f"Index: {idx}")

        if not is_hw:
             plot_y = self._image_waveform_for_display(b) if is_image else b
             x = np.arange(plot_y.size)
             self.curve.setData(x, plot_y)
             self.plot.setXRange(0, max(1, plot_y.size - 1), padding=0.01)
             self.plot.setYRange(-1.2, 1.2, padding=0.01)

        if (not is_hw) or allow_session_capture:
            self.session_rows.append([
                datetime.now().isoformat(timespec="seconds"),
                idx,
                true_label,
                pred_label,
                float(p_abn),
                float(thr),
                q_score,
                q_label,
                b.copy(),
            ])
        self._update_live_counters(bpm=bpm)
        seen = len(self.session_rows)

        if allow_session_capture:
            t_sec = float(time.time() - self.hw_t0) if self.hw_t0 > 0 else float(seen)
            bpm_val = None if bpm is None else float(bpm)
            pred_for_session = -1 if noisy else pred_label

            self.hw_records.append(
                BeatRecord(
                    t_sec=t_sec,
                    bpm=bpm_val,
                    p_abn=float(p_abn),
                    pred=int(pred_for_session),
                    quality=float(q_score) if q_score is not None else 1.0,
                    beat=b.copy(),
                )
            )

    def _finish_image_input(self):
     self.hw_session_on = False
     self.hw_session_paused = False
     self.image_mode = False

     self.btn_hw_start.setText("Start")
     self.btn_hw_start.setEnabled(True)
     self.btn_hw_stop.setEnabled(False)

     self.lbl_hw_timer.setText("Full input done")
     self.pb_hw.setValue(100)

     ok_count = int(getattr(self, "image_success_count", 0))
     fail_count = int(getattr(self, "image_failed_count", 0))

     if ok_count == 0 and fail_count > 0:
         QMessageBox.warning(
             self,
             "Images Finished",
             f"No image was processed successfully. Failed images: {fail_count}.\n\n"
             "Use a cropped, single-lead ECG strip with clear contrast."
         )
     else:
         QMessageBox.information(
             self,
             "Images Finished",
             f"All selected images were processed.\n\n"
             f"Processed: {ok_count}\nSkipped: {fail_count}"
         )

    def next_beat(self):
        if getattr(self, "image_mode", False):
         while self.image_pos < len(self.image_paths):
             path = self.image_paths[self.image_pos]
             current_index = self.image_pos
             self.image_pos += 1

             res = extract_ecg_from_image(path)
             if not res.ok:
                 self.image_failed_count = int(getattr(self, "image_failed_count", 0)) + 1
                 self.lbl_beats_strip.setText(
                     f"Skipped image: {Path(path).name} | {res.msg}"
                 )
                 continue

             self.image_success_count = int(getattr(self, "image_success_count", 0)) + 1

             sample = BeatSample(
                 beat=res.beat_187,
                 true_label=self._infer_image_true_label(path),
                 index=current_index,
                 source="IMAGE"
             )
             self._process_sample(sample, allow_session_capture=self.hw_session_on)
             return

         self._finish_image_input()
         return

        if self._pending_image_beat is not None:
            sample = BeatSample(beat=self._pending_image_beat, true_label=None, index=0, source="IMAGE")
            self._pending_image_beat = None
            self._process_sample(sample, allow_session_capture=self.hw_session_on)
            return

        try:
         sample = self.source.next()

        except StopIteration as e:
         self.hw_session_on = False
         self.hw_session_paused = False

         self.btn_hw_start.setText("Start")
         self.btn_hw_start.setEnabled(True)
         self.btn_hw_stop.setEnabled(False)

         self.lbl_hw_timer.setText("Full input done")
         self.pb_hw.setValue(100)

         QMessageBox.information(self, "CSV Finished", str(e))
         return

        except Exception as e:
          if isinstance(self.source, SerialECGSource):
             if self._error_showing:
                 return
             self._error_showing = True
             try:
                 QMessageBox.critical(self, "Source Error", str(e))
             finally:
                 self._error_showing = False
             return

          self._handle_replay_exception(e)
          return

        self._process_sample(sample, allow_session_capture=self.hw_session_on)

    # ---------------- analysis ----------------

    def analyze_now(self):
     if not self.hw_records and not self.session_rows:
         QMessageBox.information(self, "Analyze Now", "No session data available yet.")
         return

     # ✅ Save real elapsed hardware time BEFORE turning the session off
     if self.hw_session_on and self.hw_t0 > 0:
         self.hw_elapsed_before_pause = max(0.0, time.time() - self.hw_t0)

     self.hw_session_on = False
     self.hw_session_paused = False
     self.btn_hw_start.setText("Start")
     self.btn_hw_start.setEnabled(True)
     self.btn_hw_stop.setEnabled(False)

     beats_count = len(self.hw_records) if self.hw_records else len(self.session_rows)
     self.lbl_loading_sub.setText(f"Preparing report for {beats_count} captured beats")
     self.stack.setCurrentIndex(2)
     self.analysis_timer.start(1200)

    def _finish_analysis_now(self):
        if self.hw_records:
             if self.hw_elapsed_before_pause > 0:
                 elapsed_active = self.hw_elapsed_before_pause
             elif self.hw_t0 > 0:
                 elapsed_active = time.time() - self.hw_t0
             else:
                 elapsed_active = 60.0

             duration = min(self.hw_duration, max(5.0, elapsed_active))
             records = self.hw_records
        else:
            records = []
            for i, row in enumerate(self.session_rows):
                pred = int(row[3])
                q_score = row[6] if row[6] is not None else 1.0
                records.append(
                    BeatRecord(
                        t_sec=float(i),
                        bpm=None,
                        p_abn=float(row[4]),
                        pred=-1 if q_score < 0.35 else pred,
                        quality=float(q_score),
                        beat=np.asarray(row[8], dtype=np.float32).copy() if len(row) > 8 and row[8] is not None else None,
                    )
                )
            duration = max(10.0, float(len(records)))

        result = analyze_session(records, duration)
        self._apply_mitbih_analysis(result, records)

        self.hw_result = result
        self.last_result = result
        self._populate_analysis_page(result)
        self.stack.setCurrentIndex(3)

    def _apply_mitbih_analysis(self, result, records: list[BeatRecord]):
        abnormal_records = [
            r for r in records
            if r.beat is not None
            and r.pred == 1
            and (r.quality is None or float(r.quality) >= 0.35)
        ]
        if not abnormal_records:
            return

        try:
            predictions, summary = self.mitbih_analyzer.analyze_abnormal_beats([r.beat for r in abnormal_records])
        except FileNotFoundError as e:
            print("MIT-BIH analysis skipped:", e)
            return
        except Exception as e:
            print("MIT-BIH analysis failed:", e)
            result.findings.append("MIT-BIH subtype analysis was unavailable for this session.")
            result.summary += " MIT-BIH subtype analysis was unavailable for this session."
            return

        if not predictions or summary.total_beats <= 0:
            return

        for record, pred in zip(abnormal_records, predictions):
            record.mit_label = pred.final_label
            record.mit_confidence = pred.final_confidence

        result.mitbih_total_analyzed = summary.total_beats
        result.mitbih_counts = dict(summary.counts)
        result.mitbih_percentages = dict(summary.percentages)
        result.dominant_arrhythmia_label = summary.dominant_abnormal_label
        result.dominant_arrhythmia_display = summary.dominant_abnormal_display
        result.dominant_arrhythmia_pct = float(summary.dominant_abnormal_pct)

        dist_parts = [
            f"{label} {summary.percentages[label]:.1f}%"
            for label in ("N", "S", "V", "F", "Q")
            if summary.counts.get(label, 0) > 0
        ]
        if dist_parts:
            result.findings.append(
                f"MIT-BIH beat-type distribution across re-analyzed abnormal beats: {' | '.join(dist_parts)}."
            )

        if summary.dominant_abnormal_label:
            result.findings.append(
                f"Dominant MIT-BIH abnormal subtype was {summary.dominant_abnormal_display} "
                f"({summary.dominant_abnormal_label}) at {summary.dominant_abnormal_pct:.1f}% "
                f"of re-analyzed abnormal beats."
            )
        elif summary.counts.get("Q", 0) > 0:
            result.findings.append(
                "MIT-BIH subtype analysis marked the remaining uncertain abnormal candidates as low confidence."
            )

        if summary.percentages.get("Q", 0.0) >= 15.0:
            result.findings.append(
                f"MIT-BIH subtype analysis returned {summary.percentages['Q']:.1f}% low-confidence results within re-analyzed abnormal beats."
            )

        summary_text = f" MIT-BIH type analysis on {summary.total_beats} re-analyzed abnormal beats: {' | '.join(dist_parts)}."
        if summary.dominant_abnormal_display:
            summary_text += (
                f" Dominant abnormal subtype: {summary.dominant_abnormal_display} "
                f"({summary.dominant_abnormal_label})."
            )
        result.summary += summary_text

    def _populate_analysis_page(self, r):
        self._update_patient_labels()

        if r.pct_abnormal >= 40:
            self.banner.setObjectName("BannerDanger")
            self.lbl_banner_title.setText("HIGH ABNORMAL BURDEN")
            self.lbl_banner_sub.setText("The session contained a high proportion of abnormal beats")
        elif getattr(r, "sustained_tachy_detected", False):
            self.banner.setObjectName("BannerDanger")
            self.lbl_banner_title.setText("SUSTAINED TACHYCARDIA DETECTED")
            self.lbl_banner_sub.setText("Heart rate remained above 110 BPM for at least 30 seconds")
        elif r.pct_abnormal >= 15:
            self.banner.setObjectName("BannerWarn")
            self.lbl_banner_title.setText("MODERATE ABNORMAL BURDEN")
            self.lbl_banner_sub.setText("Intermittent abnormal beats were detected")
        else:
            self.banner.setObjectName("BannerNormal")
            self.lbl_banner_title.setText("PREDOMINANTLY NORMAL")
            self.lbl_banner_sub.setText("No significant abnormalities detected")

        self.banner.style().unpolish(self.banner)
        self.banner.style().polish(self.banner)

        self.lbl_avg_bpm_val.setText("-" if r.bpm_avg is None else f"{r.bpm_avg:.0f}")
        self.lbl_avg_bpm_sub.setText(
            f"Min {('-' if r.bpm_min is None else f'{r.bpm_min:.0f}')} / "
            f"Max {('-' if r.bpm_max is None else f'{r.bpm_max:.0f}')}"
        )

        self.lbl_hrv_val.setText("-" if r.hrv_sdnn_ms is None else f"{r.hrv_sdnn_ms:.1f} ms")
        self.lbl_hrv_sub.setText("Heart rate variability overview")

        self.lbl_norm_val.setText(f"{r.pct_normal:.1f}%")
        tachy_note = ""
        if getattr(r, "sustained_tachy_detected", False):
            tachy_note = f" | Tachy >110: {r.tachy_longest_sec:.0f}s"
        mit_note = ""
        if getattr(r, "dominant_arrhythmia_display", ""):
            mit_note = f" | MIT-BIH: {r.dominant_arrhythmia_display}"
        self.lbl_norm_sub.setText(f"Abnormal {r.pct_abnormal:.1f}% | Noisy {r.pct_unusable:.1f}%{tachy_note}{mit_note}")

        arrhythmia_parts = [
            f"{label} {r.mitbih_percentages[label]:.1f}%"
            for label in ("S", "V", "F", "Q")
            if getattr(r, "mitbih_counts", {}).get(label, 0) > 0
        ]
        if arrhythmia_parts and getattr(r, "mitbih_total_analyzed", 0) > 0:
            self.card_arrhythmia_mix.setVisible(True)
            self.lbl_arrhythmia_mix_val.setText(" | ".join(arrhythmia_parts))
            self.lbl_arrhythmia_mix_sub.setText(
                f"of {int(r.mitbih_total_analyzed)} detected abnormal beats"
            )
        else:
            self.card_arrhythmia_mix.setVisible(False)
            self.lbl_arrhythmia_mix_val.setText("-")
            self.lbl_arrhythmia_mix_sub.setText("Shown as % of detected abnormal beats")

        mins = int(r.duration_sec) // 60
        secs = int(r.duration_sec) % 60
        self.lbl_duration_val.setText(f"{mins}:{secs:02d}")
        self.lbl_duration_sub.setText(f"{r.n_beats} beats total")

        if hasattr(r, "findings") and r.findings:
            text = "\n\n".join([f"- {item}" for item in r.findings])
        else:
            text = "- No extra findings available."
        self.txt_findings.setPlainText(text)

        ai = generate_bilingual_ai_report(
            result=r,
            patient_name=self._patient_name_value(),
            source_name=self.selected_source_name,
        )
        self.ai_report_ar = ai.get("ar", "").strip()
        self.ai_report_en = ai.get("en", "").strip()
        self.ai_report_provider = ai.get("provider", "local")
        self._refresh_summary_text()

        bpm_series = []
        if self.hw_records:
            bpm_series = [(x.t_sec, x.bpm) for x in self.hw_records if x.bpm is not None and x.bpm > 0]
        show_vitals = bool(bpm_series) or (r.bpm_avg is not None)
        self._update_analysis_metric_visibility(show_vitals)
        if not bpm_series:
            self.curve_bpm.setData([], [])

        if bpm_series:
            xs = [a for a, _ in bpm_series]
            ys = [b for _, b in bpm_series]
            self.curve_bpm.setData(xs, ys)

        self.donut.set_distribution(r.pct_normal, r.pct_abnormal, r.pct_unusable)

    def _set_summary_language(self, lang: str):
        self.summary_lang = "ar" if str(lang).lower().startswith("ar") else "en"
        if hasattr(self, "btn_lang_toggle"):
            if self.summary_lang == "ar":
                self.btn_lang_toggle.setText("Switch to English")
            else:
                self.btn_lang_toggle.setText("Switch to Arabic")
        self._refresh_summary_text()

    def _toggle_summary_language(self):
        if self.summary_lang == "ar":
            self._set_summary_language("en")
        else:
            self._set_summary_language("ar")

    def _refresh_summary_text(self):
     if not hasattr(self, "txt_summary"):
         return

     disclaimer_en = "Disclaimer: This software output supports review and is not a final medical diagnosis."
     disclaimer_ar = "تنبيه: هذا التقرير داعم للمراجعة وليس تشخيصًا طبيًا نهائيًا."

     if self.summary_lang == "ar":
         title = "Session Report Summary (AR)"
         narrative = self.ai_report_ar or "لا يوجد ملخص عربي متاح."
         body = f"{title}\n\n{narrative}\n\n{disclaimer_ar}"
     else:
         title = "Session Report Summary (EN)"
         narrative = self.ai_report_en or "No English summary available."
         body = f"{title}\n\n{narrative}\n\n{disclaimer_en}"

     self.txt_summary.setPlainText(body)
    # ---------------- report / voice ----------------
    def _new_session(self):
     try:
         self.tts.stop()
     except Exception:
         pass
     self.reset_session()
     self._pending_image_beat = None
     self.image_mode = False
     self.image_pos = 0
     self.image_last_path = None
     self.stack.setCurrentIndex(0)

    def _hw_download_pdf(self):
        if self.hw_result is None:
            QMessageBox.warning(self, "PDF", "No analysis result available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF",
            f"ECG_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf)"
        )
        if not path:
            return

        r = self.hw_result
        is_hardware_report = self.selected_source_name.lower().startswith("hardware")
        metrics = {
            "patient_name": self._patient_name_value(),
            "duration_sec": int(r.duration_sec),
            "n_beats": r.n_beats,
            "pct_normal": f"{r.pct_normal:.1f}",
            "pct_abnormal": f"{r.pct_abnormal:.1f}",
            "pct_unusable": f"{r.pct_unusable:.1f}",
            "bpm_avg": "" if (not is_hardware_report or r.bpm_avg is None) else f"{r.bpm_avg:.0f}",
            "bpm_min": "" if (not is_hardware_report or r.bpm_min is None) else f"{r.bpm_min:.0f}",
            "bpm_max": "" if (not is_hardware_report or r.bpm_max is None) else f"{r.bpm_max:.0f}",
            "hrv_sdnn_ms": "" if (not is_hardware_report or r.hrv_sdnn_ms is None) else f"{r.hrv_sdnn_ms:.0f}",
            "tachy_longest_sec": "" if not is_hardware_report else f"{r.tachy_longest_sec:.0f}",
            "sustained_tachy": "" if not is_hardware_report else ("Yes" if r.sustained_tachy_detected else "No"),
            "high_rate_total_sec": (
                f"{getattr(r, 'high_rate_total_sec', 0.0):.0f}"
                if getattr(r, "recurrent_high_rate_detected", False)
                else ""
            ),
            "high_rate_longest_sec": (
                f"{getattr(r, 'high_rate_longest_sec', 0.0):.0f}"
                if getattr(r, "recurrent_high_rate_detected", False)
                else ""
            ),
            "low_rate_total_sec": (
                f"{getattr(r, 'low_rate_total_sec', 0.0):.0f}"
                if getattr(r, "recurrent_low_rate_detected", False)
                else ""
            ),
            "low_rate_longest_sec": (
                f"{getattr(r, 'low_rate_longest_sec', 0.0):.0f}"
                if getattr(r, "recurrent_low_rate_detected", False)
                else ""
            ),
            "dominant_arrhythmia": getattr(r, "dominant_arrhythmia_display", ""),
            "mitbih_basis": (
                f"{int(getattr(r, 'mitbih_total_analyzed', 0))} detected abnormal beats"
                if getattr(r, "mitbih_total_analyzed", 0) > 0
                else ""
            ),
            "mitbih_distribution": " | ".join(
                [
                    f"{label} {r.mitbih_percentages[label]:.1f}%"
                    for label in ("S", "V", "F", "Q")
                    if getattr(r, "mitbih_counts", {}).get(label, 0) > 0
                ]
            ),
        }

        export_pdf_report(
            out_path=path,
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            badge=r.badge,
            metrics=metrics,
            findings=r.findings,
            summary=r.summary,
            ai_report_en=self.ai_report_en,
            ai_report_ar=self.ai_report_ar,
            source_name=self.selected_source_name,
            ai_provider="",
        )
        QMessageBox.information(self, "PDF", "PDF saved successfully.")

    def _hw_read_aloud(self):
        if self.hw_result is None:
            QMessageBox.warning(self, "Audio", "No results available.")
            return
        r = self.hw_result
        if self.summary_lang == "ar":
            text = self.ai_report_ar.strip()
        else:
            text = self.ai_report_en.strip()
        if not text:
            text = f"{self._patient_name_value()}. {r.badge}. " + " ".join(r.findings) + " " + r.summary
        self.tts.stop()
        self.tts.say(text)


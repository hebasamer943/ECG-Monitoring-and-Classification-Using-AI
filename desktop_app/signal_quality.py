from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class QualityResult:
    score: float   # 0..1
    label: str     # Poor/Fair/Good/Excellent
    hint: str      # short guidance text


def _moving_average_safe(x: np.ndarray, w: int) -> np.ndarray:
    """
    Moving average that ALWAYS returns an array with the same length as x.
    Works safely even if window size > signal length.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n == 0:
        return x

    w = int(max(1, w))
    if w > n:
        w = n

    # Cumulative sum moving average -> length (n - w + 1)
    c = np.cumsum(np.insert(x, 0, 0.0)).astype(np.float32)
    y = (c[w:] - c[:-w]) / float(w)  # length = n - w + 1

    # Pad back to length n (centered)
    pad_left = (n - y.size) // 2
    pad_right = n - y.size - pad_left
    y = np.pad(y, (pad_left, pad_right), mode="edge")
    return y.astype(np.float32)


def _highpass_baseline_remove(x: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Baseline removal using a long moving average (~0.8s) as baseline estimate.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    w = max(3, int(0.8 * fs))
    base = _moving_average_safe(x, w)
    # Safety guard (should already match)
    if base.shape[0] != x.shape[0]:
        base = base[: x.shape[0]]
    return x - base


def _bandshape(x: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Simple band-shaping without scipy:
    - baseline removal (high-pass-like)
    - mild low-pass smoothing (~20ms)
    """
    y = _highpass_baseline_remove(x, fs=fs)
    w_lp = max(3, int(0.02 * fs))
    y = _moving_average_safe(y, w_lp)
    return y


def evaluate_quality(beat_187: np.ndarray, fs: int = 250) -> QualityResult:
    """
    Robust quality score for a single ECG beat segment (usually 187 samples).
    This does NOT require scipy and is designed to be stable (no shape mismatch).

    Main checks:
    - NaN/Inf
    - flatline / very low amplitude
    - clipping / quantization (too many near-constant steps)
    - SNR-like estimate using band-shaped signal vs high-frequency residue
    - baseline wander penalty
    - outlier penalty

    Returns QualityResult(score 0..1, label, hint).
    """
    x = np.asarray(beat_187, dtype=np.float32).reshape(-1)

    if x.size < 40:
        return QualityResult(0.0, "Poor", "Signal too short.")

    if not np.isfinite(x).all():
        return QualityResult(0.0, "Poor", "Signal contains NaN/Inf.")

    # Robust normalization (median + MAD)
    med = float(np.median(x))
    x0 = x - med
    mad = float(np.median(np.abs(x0))) + 1e-6
    xz = x0 / (1.4826 * mad)

    # Flatline / very low dynamic range
    dyn = float(np.percentile(xz, 95) - np.percentile(xz, 5))
    if dyn < 0.4:
        return QualityResult(0.05, "Poor", "Flatline / very low amplitude.")

    # Clipping / quantization detection
    dif = np.abs(np.diff(xz))
    near_const = float(np.mean(dif < 1e-3))
    if near_const > 0.25:
        return QualityResult(0.10, "Poor", "Clipping / quantization detected.")

    # Band-shaped signal vs residue (noise)
    y = _bandshape(xz, fs=fs)
    residue = xz - y

    sig_rms = float(np.sqrt(np.mean(y ** 2)) + 1e-6)
    noi_rms = float(np.sqrt(np.mean(residue ** 2)) + 1e-6)
    snr = sig_rms / noi_rms  # higher is better

    # Outlier penalty
    out = float(np.mean(np.abs(xz) > 6.0))  # fraction of extreme samples
    out_pen = max(0.0, 1.0 - 4.0 * out)

    # Baseline wander penalty (energy of long MA baseline)
    base = _moving_average_safe(xz, max(3, int(0.8 * fs)))
    base_energy = float(np.sqrt(np.mean(base ** 2)))
    base_pen = 1.0 / (1.0 + base_energy)  # smaller baseline energy -> closer to 1

    # Map SNR to [0,1]
    # Typical: snr ~ 0.8 (bad) to ~ 6.0 (very good)
    snr_norm = (snr - 0.8) / (6.0 - 0.8)
    snr_norm = float(np.clip(snr_norm, 0.0, 1.0))

    # Final score (weighted)
    score = (0.65 * snr_norm) + (0.20 * out_pen) + (0.15 * base_pen)
    score = float(np.clip(score, 0.0, 1.0))

    # Labels + hints
    if score >= 0.78:
        return QualityResult(score, "Excellent", "Stable signal. Good electrode contact.")
    if score >= 0.60:
        return QualityResult(score, "Good", "Usable signal. Minor noise is present.")
    if score >= 0.40:
        return QualityResult(score, "Fair", "Moderate noise. Reduce movement and recheck electrodes.")
    return QualityResult(score, "Poor", "High noise or unstable contact. Recheck electrodes and reduce movement.")
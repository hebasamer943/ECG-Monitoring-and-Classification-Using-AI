
"""
Experimental feature:
ECG image trace extraction was tested but is not included
in the final core workflow because reliability depends heavily
on image quality, cropping, grid noise, and lead visibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ExtractResult:
    beat_187: np.ndarray
    ok: bool
    msg: str


def _resample_to_187(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < 2:
        return np.zeros((187,), np.float32)
    x_old = np.linspace(0.0, 1.0, num=y.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=187, dtype=np.float32)
    return np.interp(x_new, x_old, y).astype(np.float32)


def _normalize_amp(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    amp = -(y - np.median(y))
    amp = amp / (np.max(np.abs(amp)) + 1e-9)
    return amp.astype(np.float32)


def _trace_score(amp: np.ndarray) -> float:
    a = np.asarray(amp, dtype=np.float32).reshape(-1)
    if a.size < 8:
        return -1e9
    dyn = float(np.std(a))
    jitter = float(np.mean(np.abs(np.diff(a))))
    return dyn - (0.65 * jitter)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return a
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    win = min(win, a.size if a.size % 2 == 1 else max(1, a.size - 1))
    if win <= 1:
        return a.copy()
    kernel = np.ones((win,), dtype=np.float32) / float(win)
    return np.convolve(a, kernel, mode="same").astype(np.float32)


def _crop_to_active_region(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    row_density = np.sum(mask > 0, axis=1)
    col_density = np.sum(mask > 0, axis=0)

    rows = np.where(row_density >= max(3, int(0.01 * w)))[0]
    cols = np.where(col_density >= max(2, int(0.01 * h)))[0]
    if rows.size < 12 or cols.size < 30:
        return mask

    pad_y = max(6, int(0.03 * h))
    pad_x = max(8, int(0.03 * w))
    y0 = max(0, int(rows[0]) - pad_y)
    y1 = min(h, int(rows[-1]) + pad_y + 1)
    x0 = max(0, int(cols[0]) - pad_x)
    x1 = min(w, int(cols[-1]) + pad_x + 1)
    return mask[y0:y1, x0:x1]


def _build_trace_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)

    _, bw_otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw_adapt = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    bw = cv2.bitwise_or(bw_otsu, bw_adapt)

    h, w = bw.shape
    k_h = max(15, (w // 18) | 1)
    k_v = max(15, (h // 18) | 1)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, k_h), np.uint8))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k_v, 1), np.uint8))
    grid = cv2.bitwise_or(horiz, vert)
    bw = cv2.bitwise_and(bw, cv2.bitwise_not(grid))

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    bw = _crop_to_active_region(bw)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return bw

    h, w = bw.shape
    keep = np.zeros_like(bw)
    for idx in range(1, num):
        x, y, ww, hh, area = stats[idx]
        if area < max(24, int(0.0003 * h * w)):
            continue
        if ww < max(20, int(0.10 * w)):
            continue
        if hh > int(0.60 * h):
            continue
        keep[labels == idx] = 255

    if int(np.count_nonzero(keep)) >= max(40, int(0.02 * h * w)):
        bw = keep
    return bw


def _pick_trace_row(idx: np.ndarray) -> float:
    idx = np.asarray(idx, dtype=np.int32).reshape(-1)
    if idx.size == 0:
        return float("nan")
    if idx.size <= 6:
        return float(np.median(idx))

    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    best = max(splits, key=lambda seg: seg.size)
    return float(np.mean(best))


def _extract_centerline(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    y = np.full((w,), np.nan, dtype=np.float32)

    for xi in range(w):
        idx = np.where(mask[:, xi] > 0)[0]
        if idx.size:
            y[xi] = _pick_trace_row(idx)

    finite = np.isfinite(y)
    if finite.sum() < max(30, int(0.12 * w)):
        return np.zeros((0,), np.float32)

    x = np.arange(w, dtype=np.float32)
    y = np.interp(x, x[finite], y[finite]).astype(np.float32)

    win = max(5, ((w // 90) * 2) + 1)
    y = _moving_average(y, win)
    return y.astype(np.float32)


def _detect_peaks(sig: np.ndarray, min_dist: int, min_height: float) -> np.ndarray:
    s = np.asarray(sig, dtype=np.float32).reshape(-1)
    n = s.size
    if n < 3:
        return np.zeros((0,), dtype=np.int32)

    peaks: list[int] = []
    for i in range(1, n - 1):
        if s[i] < min_height:
            continue
        if s[i] >= s[i - 1] and s[i] > s[i + 1]:
            if peaks and (i - peaks[-1]) < min_dist:
                if s[i] > s[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return np.asarray(peaks, dtype=np.int32)


def _extract_dominant_beat(amp: np.ndarray) -> np.ndarray:
    a = np.asarray(amp, dtype=np.float32).reshape(-1)
    if a.size < 50:
        return a

    smooth = np.convolve(a, np.ones((5,), dtype=np.float32) / 5.0, mode="same")
    energy = np.abs(smooth)
    min_dist = max(18, a.size // 10)
    min_height = max(float(np.percentile(energy, 78)), 0.35 * float(np.max(energy)))
    peaks = _detect_peaks(energy, min_dist=min_dist, min_height=min_height)

    search_lo = max(2, int(0.03 * a.size))
    search_hi = min(a.size - 2, max(search_lo + 1, int(0.92 * a.size)))
    if peaks.size:
        peak_idx = int(peaks[np.argmax(energy[peaks])])
    else:
        peak_idx = search_lo + int(np.argmax(energy[search_lo:search_hi]))

    if peaks.size >= 2:
        rr = int(np.median(np.diff(peaks)))
    else:
        rr = max(60, min(int(0.45 * a.size), 140))

    rr = max(60, min(rr, max(80, a.size)))
    pre = max(10, int(0.10 * rr))
    post = max(50, int(1.25 * rr))
    win = pre + post

    start = peak_idx - pre
    end = peak_idx + post
    if start < 0:
        end = min(a.size, end - start)
        start = 0
    if end > a.size:
        start = max(0, start - (end - a.size))
        end = a.size

    seg = a[start:end]
    if seg.size < max(60, win // 2):
        return a
    return seg.astype(np.float32)


def _is_ecg_like(sig: np.ndarray) -> bool:
    s = np.asarray(sig, dtype=np.float32).reshape(-1)
    if s.size < 60:
        return False

    s = s - np.median(s)
    s = s / (np.max(np.abs(s)) + 1e-9)
    smooth = _moving_average(s, 7)
    diff = np.diff(smooth)
    if diff.size < 8:
        return False

    slope = np.abs(diff)
    mean_slope = float(np.mean(slope)) + 1e-6
    peak_slope = float(np.max(slope))
    sharpness = peak_slope / mean_slope

    sign_changes = int(np.sum(np.diff(np.sign(diff)) != 0))
    energy = np.abs(smooth)
    peaks = _detect_peaks(
        energy,
        min_dist=max(12, s.size // 8),
        min_height=max(0.45, float(np.percentile(energy, 82))),
    )
    prominence = float(np.max(energy) - np.median(energy))

    return (
        sharpness >= 2.6
        and sign_changes >= 4
        and peaks.size >= 1
        and prominence >= 0.35
    )


def extract_ecg_from_image(image_path: str) -> ExtractResult:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return ExtractResult(np.zeros(187, np.float32), False, "Cannot read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 1600:
        scale = 1600.0 / float(gray.shape[1])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    mask = _build_trace_mask(gray)
    trace_y = _extract_centerline(mask)
    if trace_y.size < 40:
        return ExtractResult(
            np.zeros(187, np.float32),
            False,
            "Trace not detected clearly. Crop a single ECG strip with higher contrast.",
        )

    amp = _normalize_amp(trace_y)
    trend = _moving_average(amp, max(11, int(0.15 * amp.size)))
    amp = _normalize_amp(amp - trend)
    if _trace_score(amp) < 0.05:
        return ExtractResult(
            np.zeros(187, np.float32),
            False,
            "Extracted trace is too noisy. Crop a clearer ECG lead strip.",
        )

    beat = _extract_dominant_beat(amp)
    beat = _normalize_amp(beat)
    if not _is_ecg_like(beat):
        return ExtractResult(
            np.zeros(187, np.float32),
            False,
            "Image trace does not look like a clean ECG beat. Crop a single lead strip or one beat only.",
        )
    beat_187 = _resample_to_187(beat)
    return ExtractResult(beat_187, True, "OK")

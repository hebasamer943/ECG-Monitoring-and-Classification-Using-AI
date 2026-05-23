from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
import numpy as np


@dataclass
class BeatRecord:
    t_sec: float
    bpm: float | None
    p_abn: float
    pred: int
    quality: float | None = None
    beat: np.ndarray | None = None
    mit_label: str | None = None
    mit_confidence: float | None = None


@dataclass
class SessionResult:
    duration_sec: float
    n_beats: int
    n_usable: int
    n_abnormal: int
    n_normal: int
    n_unusable: int
    pct_abnormal: float
    pct_normal: float
    pct_unusable: float
    bpm_avg: float | None
    bpm_min: float | None
    bpm_max: float | None
    tachy_longest_sec: float
    sustained_tachy_detected: bool
    high_rate_longest_sec: float
    high_rate_total_sec: float
    recurrent_high_rate_detected: bool
    low_rate_longest_sec: float
    low_rate_total_sec: float
    recurrent_low_rate_detected: bool
    hrv_sdnn_ms: float | None
    badge: str
    findings: list[str] = field(default_factory=list)
    summary: str = ""
    mitbih_total_analyzed: int = 0
    mitbih_counts: dict[str, int] = field(default_factory=dict)
    mitbih_percentages: dict[str, float] = field(default_factory=dict)
    dominant_arrhythmia_label: str = ""
    dominant_arrhythmia_display: str = ""
    dominant_arrhythmia_pct: float = 0.0


QUALITY_MIN = 0.35
NORMAL_BPM_LOW = 60.0
NORMAL_BPM_HIGH = 100.0
TACHY_BPM_THRESHOLD = 110.0
TACHY_SUSTAINED_SEC = 30.0
TACHY_MAX_GAP_SEC = 3.0
RATE_OUT_OF_RANGE_REPORT_SEC = 15.0


def _safe_pct(x: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return 100.0 * float(x) / float(n)


def _calc_hrv_sdnn_ms(records: list[BeatRecord]) -> float | None:
    beat_times = [r.t_sec for r in records if r.bpm is not None]
    if len(beat_times) < 3:
        return None

    rr = np.diff(np.asarray(beat_times, dtype=np.float32)) * 1000.0
    if rr.size < 2:
        return None

    return float(np.std(rr, ddof=1))


def _longest_sustained_tachy_sec(
    records: list[BeatRecord],
    bpm_threshold: float = TACHY_BPM_THRESHOLD,
    max_gap_sec: float = TACHY_MAX_GAP_SEC,
) -> float:
    points = sorted(
        [(float(r.t_sec), float(r.bpm)) for r in records if r.bpm is not None and r.bpm > 0],
        key=lambda x: x[0],
    )
    if len(points) < 2:
        return 0.0

    longest = 0.0
    streak = 0.0

    prev_t, prev_bpm = points[0]
    prev_high = prev_bpm > bpm_threshold

    for t, bpm in points[1:]:
        dt = max(0.0, float(t - prev_t))
        high = bpm > bpm_threshold

        if dt > max_gap_sec:
            streak = 0.0
        elif prev_high and high:
            streak += dt
            if streak > longest:
                longest = streak
        elif high:
            streak = 0.0
        else:
            streak = 0.0

        prev_t = t
        prev_high = high

    return float(longest)


def _episode_duration_stats(
    records: list[BeatRecord],
    predicate,
    max_gap_sec: float = TACHY_MAX_GAP_SEC,
) -> tuple[float, float]:
    points = sorted(
        [(float(r.t_sec), float(r.bpm)) for r in records if r.bpm is not None and r.bpm > 0],
        key=lambda x: x[0],
    )
    if len(points) < 2:
        return 0.0, 0.0

    longest = 0.0
    total = 0.0
    streak = 0.0

    prev_t, prev_bpm = points[0]
    prev_match = bool(predicate(prev_bpm))

    for t, bpm in points[1:]:
        dt = max(0.0, float(t - prev_t))
        match = bool(predicate(bpm))

        if dt > max_gap_sec:
            streak = 0.0
        elif prev_match and match:
            streak += dt
            total += dt
            if streak > longest:
                longest = streak
        elif match:
            streak = 0.0
        else:
            streak = 0.0

        prev_t = t
        prev_match = match

    return float(longest), float(total)


def analyze_session(records: list[BeatRecord], duration_sec: float) -> SessionResult:
    usable: list[BeatRecord] = []
    unusable: list[BeatRecord] = []

    for r in records:
        if r.pred == -1:
            unusable.append(r)
        elif r.quality is not None and r.quality < QUALITY_MIN:
            unusable.append(r)
        else:
            usable.append(r)

    n_beats = len(records)
    n_usable = len(usable)
    n_unusable = len(unusable)

    n_abnormal = sum(1 for r in usable if r.pred == 1)
    n_normal = sum(1 for r in usable if r.pred == 0)

    # Use total captured beats as the public denominator so the displayed
    # Normal/Abnormal/Noisy distribution sums cleanly to 100%.
    pct_abnormal = _safe_pct(n_abnormal, n_beats)
    pct_normal = _safe_pct(n_normal, n_beats)
    pct_unusable = _safe_pct(n_unusable, n_beats)

    bpm_vals = [float(r.bpm) for r in records if r.bpm is not None and r.bpm > 0]
    bpm_avg = mean(bpm_vals) if bpm_vals else None
    bpm_min = min(bpm_vals) if bpm_vals else None
    bpm_max = max(bpm_vals) if bpm_vals else None
    tachy_longest_sec = _longest_sustained_tachy_sec(usable)
    sustained_tachy_detected = tachy_longest_sec >= TACHY_SUSTAINED_SEC
    high_rate_longest_sec, high_rate_total_sec = _episode_duration_stats(
        records, lambda bpm: bpm > NORMAL_BPM_HIGH
    )
    recurrent_high_rate_detected = (
        high_rate_longest_sec >= RATE_OUT_OF_RANGE_REPORT_SEC
        or high_rate_total_sec >= RATE_OUT_OF_RANGE_REPORT_SEC
    )
    low_rate_longest_sec, low_rate_total_sec = _episode_duration_stats(
        records, lambda bpm: bpm < NORMAL_BPM_LOW
    )
    recurrent_low_rate_detected = (
        low_rate_longest_sec >= RATE_OUT_OF_RANGE_REPORT_SEC
        or low_rate_total_sec >= RATE_OUT_OF_RANGE_REPORT_SEC
    )
    hrv_sdnn_ms = _calc_hrv_sdnn_ms(records)

    findings: list[str] = []

    if pct_unusable >= 25:
        findings.append("A large portion of the signal was noisy or unusable.")
    elif pct_unusable >= 10:
        findings.append("Some beats were excluded because of low signal quality.")

    if bpm_avg is not None:
        if bpm_avg > 100:
            findings.append(f"Average heart rate was elevated at about {bpm_avg:.0f} BPM.")
        elif bpm_avg < 60:
            findings.append(f"Average heart rate was low at about {bpm_avg:.0f} BPM.")
        else:
            findings.append(f"Average heart rate was about {bpm_avg:.0f} BPM.")

    if sustained_tachy_detected:
        findings.append(
            f"Sustained tachycardia pattern was detected: heart rate stayed above "
            f"{int(TACHY_BPM_THRESHOLD)} BPM for about {tachy_longest_sec:.0f} seconds."
        )
    elif recurrent_high_rate_detected:
        findings.append(
            f"Heart rate was repeatedly above the expected range (> {int(NORMAL_BPM_HIGH)} BPM), "
            f"totaling about {high_rate_total_sec:.0f} seconds with a longest continuous run of "
            f"{high_rate_longest_sec:.0f} seconds."
        )

    if recurrent_low_rate_detected:
        findings.append(
            f"Heart rate was repeatedly below the expected range (< {int(NORMAL_BPM_LOW)} BPM), "
            f"totaling about {low_rate_total_sec:.0f} seconds with a longest continuous run of "
            f"{low_rate_longest_sec:.0f} seconds."
        )

    if pct_abnormal >= 40:
        badge = "High abnormal burden"
        findings.append("The session contained a high proportion of abnormal beats across total captured beats.")
    elif sustained_tachy_detected:
        badge = "Sustained tachycardia episode"
        findings.append("Session is flagged abnormal because of sustained tachycardia.")
    elif pct_abnormal >= 15:
        badge = "Moderate abnormal burden"
        findings.append("The session contained intermittent abnormal beats across total captured beats.")
    else:
        badge = "Mostly normal session"
        findings.append("Most captured beats were classified as normal.")

    summary = (
        f"Session duration was {int(duration_sec)} seconds. "
        f"Total beats: {n_beats}. "
        f"Usable beats: {n_usable}. "
        f"Normal: {pct_normal:.1f}%. "
        f"Abnormal: {pct_abnormal:.1f}%. "
        f"Unusable: {pct_unusable:.1f}%."
    )
    if sustained_tachy_detected:
        summary += (
            f" Sustained tachycardia criterion met (> {int(TACHY_BPM_THRESHOLD)} BPM for >= "
            f"{int(TACHY_SUSTAINED_SEC)} sec; observed {tachy_longest_sec:.0f} sec)."
        )
    elif recurrent_high_rate_detected:
        summary += (
            f" Heart rate was outside the upper expected range (> {int(NORMAL_BPM_HIGH)} BPM) "
            f"for about {high_rate_total_sec:.0f} sec total (longest run {high_rate_longest_sec:.0f} sec)."
        )
    if recurrent_low_rate_detected:
        summary += (
            f" Heart rate was outside the lower expected range (< {int(NORMAL_BPM_LOW)} BPM) "
            f"for about {low_rate_total_sec:.0f} sec total (longest run {low_rate_longest_sec:.0f} sec)."
        )

    return SessionResult(
        duration_sec=float(duration_sec),
        n_beats=n_beats,
        n_usable=n_usable,
        n_abnormal=n_abnormal,
        n_normal=n_normal,
        n_unusable=n_unusable,
        pct_abnormal=pct_abnormal,
        pct_normal=pct_normal,
        pct_unusable=pct_unusable,
        bpm_avg=None if bpm_avg is None else float(bpm_avg),
        bpm_min=None if bpm_min is None else float(bpm_min),
        bpm_max=None if bpm_max is None else float(bpm_max),
        tachy_longest_sec=tachy_longest_sec,
        sustained_tachy_detected=sustained_tachy_detected,
        high_rate_longest_sec=high_rate_longest_sec,
        high_rate_total_sec=high_rate_total_sec,
        recurrent_high_rate_detected=recurrent_high_rate_detected,
        low_rate_longest_sec=low_rate_longest_sec,
        low_rate_total_sec=low_rate_total_sec,
        recurrent_low_rate_detected=recurrent_low_rate_detected,
        hrv_sdnn_ms=hrv_sdnn_ms,
        badge=badge,
        findings=findings,
        summary=summary,
    )

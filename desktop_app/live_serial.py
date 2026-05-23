from __future__ import annotations

import csv
import re
import time
import threading
from collections import deque
from queue import Empty, Queue
from typing import Deque, Optional

import numpy as np
import serial

from desktop_app.io_sources import BeatSample, ensure_187


class SerialECGSource:
    PATTERN_FULL = re.compile(
        r"DET:\s*([-+]?\d*\.?\d+)\s*,\s*DIS:\s*([-+]?\d*\.?\d+)\s*,\s*BPM:\s*([-+]?\d*\.?\d+)\s*,\s*BEAT:\s*([-+]?\d*\.?\d+)",
        re.IGNORECASE,
    )
    PATTERN_NO_BPM = re.compile(
        r"DET:\s*([-+]?\d*\.?\d+)\s*,\s*DIS:\s*([-+]?\d*\.?\d+)\s*,\s*BEAT:\s*([-+]?\d*\.?\d+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        fs: int = 150,
        beat_level: float = 400.0,
        refractory_ms: int = 300,
        max_buf: int = 6000,
    ):
        self.port = port
        self.baud = baud
        self.fs = int(fs)
        self.sample_rate = int(fs)
        self.beat_level = float(beat_level)
        self.refractory_ms = int(refractory_ms)
        self.max_buf = int(max_buf)

        # PTBDB windows often place the dominant beat early in the 187-sample span.
        self.PRE = 12
        self.POST = 175

        self.q: "Queue[BeatSample]" = Queue(maxsize=60)
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.ser: Optional[serial.Serial] = None

        self.sig_buf: list[float] = []
        self.base_idx = 0
        self.abs_i = 0

        self.pending_markers: Deque[int] = deque()
        self._prev_marker = False
        self._last_marker_i = -10**9
        self._prev_true_marker_i: Optional[int] = None
        self.index = 0
        self._lock = threading.Lock()

        self.leads_on: bool | None = None
        self.last_status: str = ""
        self.last_bpm: float | None = None

        self._rec_on = False
        self._rec_rows: list[list[object]] = []
        self._rec_t0 = 0.0

    def start(self):
        if self.thread:
            return
        self.stop_flag.clear()
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(1.0)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        self.thread = None
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def start_recording_raw(self):
        self._rec_on = True
        self._rec_rows = []
        self._rec_t0 = time.time()

    def stop_recording_raw(self):
        self._rec_on = False

    def save_recording_csv(self, path: str):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "DET", "DIS", "BPM", "BEAT", "STATUS_LAST"])
            writer.writerows(self._rec_rows)

    def _trim_buf(self):
        if len(self.sig_buf) <= self.max_buf:
            return
        keep = self.max_buf // 2
        drop = len(self.sig_buf) - keep
        if drop > 0:
            self.sig_buf = self.sig_buf[drop:]
            self.base_idx += drop

    def _parse_line(self, line: str):
        up = line.strip().upper()
        if up.startswith("STATUS:"):
            self.last_status = line.strip()
            if "LEADS_ON" in up:
                self.leads_on = True
            elif "LEADS_OFF" in up:
                self.leads_on = False
            return None

        m = self.PATTERN_FULL.search(line)
        if m:
            det = float(m.group(1))
            dis = float(m.group(2))
            bpm = float(m.group(3))
            beat = float(m.group(4))
            return det, dis, beat, bpm

        m2 = self.PATTERN_NO_BPM.search(line)
        if m2:
            det = float(m2.group(1))
            dis = float(m2.group(2))
            beat = float(m2.group(3))
            return det, dis, beat, None

        return None

    def _put_beat(self, beat: np.ndarray):
        beat = ensure_187(beat)
        sample = BeatSample(beat=beat, true_label=None, index=self.index, source="SERIAL")
        self.index += 1
        try:
            self.q.put_nowait(sample)
        except Exception:
            try:
                _ = self.q.get_nowait()
                self.q.put_nowait(sample)
            except Exception:
                pass

    def _update_bpm_from_arduino(self, bpm: Optional[float]):
        if bpm is None:
            return
        if 30.0 <= bpm <= 220.0:
            self._smooth_bpm(float(bpm))

    def _update_bpm_from_markers(self, marker_i: int):
        if self._prev_true_marker_i is None:
            self._prev_true_marker_i = marker_i
            return

        di = marker_i - self._prev_true_marker_i
        self._prev_true_marker_i = marker_i
        if di <= 0:
            return

        dt = di / float(max(self.sample_rate, 1))
        if dt <= 0:
            return

        bpm = 60.0 / dt
        if 30.0 <= bpm <= 220.0:
            self._smooth_bpm(float(bpm))

    def _smooth_bpm(self, bpm: float):
        if self.last_bpm is None or self.last_bpm <= 0:
            self.last_bpm = float(bpm)
            return

        diff = abs(float(bpm) - float(self.last_bpm))
        if diff >= 12.0:
            alpha = 0.55
        elif diff >= 6.0:
            alpha = 0.35
        else:
            alpha = 0.22

        self.last_bpm = float(((1.0 - alpha) * float(self.last_bpm)) + (alpha * float(bpm)))

    def _try_emit_pending(self):
        while self.pending_markers:
            marker = self.pending_markers[0]
            start = marker - self.PRE
            end = marker + self.POST

            if start < self.base_idx:
                self.pending_markers.popleft()
                continue

            if end > (self.base_idx + len(self.sig_buf)):
                break

            s = start - self.base_idx
            e = end - self.base_idx
            with self._lock:
                seg = np.asarray(self.sig_buf[s:e], dtype=np.float32)

            self._put_beat(seg)
            self.pending_markers.popleft()

    def _read_loop(self):
        while not self.stop_flag.is_set():
            try:
                line = self.ser.readline().decode(errors="ignore").strip()  # type: ignore[union-attr]
            except Exception:
                continue

            parsed = self._parse_line(line)
            if parsed is None:
                continue

            det, dis, beat, bpm = parsed
            cur_i = self.abs_i
            self.abs_i += 1

            with self._lock:
                self.sig_buf.append(float(det))
                self._trim_buf()

            self._update_bpm_from_arduino(bpm)

            if self._rec_on:
                tt = time.time() - self._rec_t0
                self._rec_rows.append([
                    f"{tt:.4f}",
                    det,
                    dis,
                    (bpm if bpm is not None else ""),
                    beat,
                    self.last_status,
                ])
                if tt >= 120.0:
                    self._rec_on = False

            is_marker = beat >= self.beat_level
            if is_marker and (not self._prev_marker):
                ref_samples = max(1, int((self.refractory_ms / 1000.0) * max(self.sample_rate, 1)))
                if (cur_i - self._last_marker_i) >= ref_samples:
                    self.pending_markers.append(cur_i)
                    self._last_marker_i = cur_i
                    if bpm is None:
                        self._update_bpm_from_markers(cur_i)

            self._prev_marker = is_marker
            self._try_emit_pending()

    def next(self, timeout: float = 6.0) -> BeatSample:
        if not self.thread:
            self.start()
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            raise RuntimeError("No BEAT received yet. Check electrodes / marker output / sample_rate.")

    def try_next(self) -> Optional[BeatSample]:
        if not self.thread:
            self.start()
        try:
            return self.q.get_nowait()
        except Empty:
            return None

    def get_recent_signal(self, seconds: float = 5.0) -> np.ndarray:
        sr = max(self.sample_rate, 1)
        n = int(max(200, sr * float(seconds)))
        with self._lock:
            if not self.sig_buf:
                return np.zeros(0, dtype=np.float32)
            seg = np.asarray(self.sig_buf[-min(n, len(self.sig_buf)):], dtype=np.float32)
        return seg

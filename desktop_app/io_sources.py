from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class BeatSample:
    beat: np.ndarray
    true_label: int | None
    index: int
    source: str  # "PTBDB" / "CSV" / "IMAGE" / "SERIAL"

def ensure_187(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size > 187:
        x = x[:187]
    if x.size < 187:
        x = np.pad(x, (0, 187 - x.size))
    return x

class PTBDBSource:
    def __init__(self, dataset_dir: Path, balanced_demo: bool = True):
        self.dataset_dir = dataset_dir
        self.balanced_demo = bool(balanced_demo)
        self.df: pd.DataFrame | None = None
        self.i = 0

    def load(self):
        n = self.dataset_dir / "ptbdb_normal.csv"
        a = self.dataset_dir / "ptbdb_abnormal.csv"
        if not n.exists() or not a.exists():
            raise FileNotFoundError("Missing Dataset/ptbdb_normal.csv or Dataset/ptbdb_abnormal.csv")

        df_n = pd.read_csv(n, header=None)
        df_a = pd.read_csv(a, header=None)

        def labelize(df: pd.DataFrame, file_label: int) -> pd.DataFrame:
            df = df.copy()
            if df.shape[1] >= 188:
                last = df.iloc[:, -1]
                if set(last.dropna().unique().tolist()).issubset({0, 1}):
                    df["__label__"] = last.astype(int)
                else:
                    df["__label__"] = file_label
            else:
                df["__label__"] = file_label
            return df

        df_n = labelize(df_n, 0)
        df_a = labelize(df_a, 1)

        # PTBDB is heavily skewed toward abnormal beats.
        # For demo playback, use a balanced subset so the UI does not look
        # "always abnormal" even when the model is behaving correctly.
        if self.balanced_demo:
            n_each = min(len(df_n), len(df_a))
            df_n = df_n.sample(n=n_each, random_state=42).reset_index(drop=True)
            df_a = df_a.sample(n=n_each, random_state=42).reset_index(drop=True)

        self.df = (
            pd.concat([df_n, df_a], ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        self.i = 0

    def next(self) -> BeatSample:
        if self.df is None:
            self.load()
        assert self.df is not None

        if self.i >= len(self.df):
            self.i = 0

        row = self.df.iloc[self.i]
        idx = self.i
        self.i += 1

        beat = ensure_187(row.iloc[:187].astype(float).to_numpy())
        true_label = int(row["__label__"]) if "__label__" in row else None
        return BeatSample(beat=beat, true_label=true_label, index=idx, source="PTBDB")

class CSVSource:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.i = 0

    def load(self):
        df = pd.read_csv(self.csv_path, header=None)

        # ✅ Features ALWAYS first 187 columns فقط
        X = df.iloc[:, :187].to_numpy(dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # ✅ Labels ONLY if there is a valid binary label column at index 187
        self.y = None
        if df.shape[1] >= 188:
            ycol = pd.to_numeric(df.iloc[:, 187], errors="coerce")
            uniq = set(ycol.dropna().unique().tolist())
            if uniq.issubset({0, 1}):
                self.y = ycol.astype(int).to_numpy()

        self.X = X
        self.i = 0

        print("CSV loaded shape:", df.shape, "| X:", self.X.shape,
              "| y:", None if self.y is None else (int(self.y.min()), int(self.y.max())))

    def next(self) -> BeatSample:
        if self.X is None:
            self.load()
        assert self.X is not None

        if self.i >= len(self.X):
         raise StopIteration("CSV finished. All uploaded beats were processed.")

        idx = self.i
        beat = ensure_187(self.X[self.i])

        true = None
        if self.y is not None and self.i < len(self.y):
            true = int(self.y[self.i])

        self.i += 1
        return BeatSample(beat=beat, true_label=true, index=idx, source="CSV")

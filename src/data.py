from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_summary: pd.DataFrame


def _read_cicddos_zip(zip_path: Path) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".parquet"):
                continue
            with archive.open(name) as handle:
                payload = io.BytesIO(handle.read())
            frame = pd.read_parquet(payload)
            frame["source_file"] = name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No Parquet files found inside {zip_path}")
    return pd.concat(frames, ignore_index=True)


def load_cicddos2019_binary(
    zip_path: str | Path,
    sample_size: int,
    test_size: float,
    validation_size: float,
    seed: int,
) -> DatasetBundle:
    zip_path = Path(zip_path)
    df = _read_cicddos_zip(zip_path)
    df["binary_label"] = (df["Label"].astype(str).str.lower() != "benign").astype(int)

    label_summary = (
        df.assign(binary=df["binary_label"].map({0: "Benign", 1: "DDoS"}))
        .groupby(["binary", "Label"], observed=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["binary", "rows"], ascending=[True, False])
    )

    if sample_size and sample_size < len(df):
        per_class = max(1, sample_size // 2)
        benign = df[df["binary_label"] == 0].sample(
            n=min(per_class, int((df["binary_label"] == 0).sum())), random_state=seed
        )
        attack = df[df["binary_label"] == 1].sample(
            n=min(sample_size - len(benign), int((df["binary_label"] == 1).sum())), random_state=seed
        )
        df = pd.concat([benign, attack], ignore_index=True).sample(frac=1.0, random_state=seed)

    y = df["binary_label"].to_numpy(dtype=np.int64)
    drop_cols = {"Label", "binary_label", "source_file"}
    feature_names = [col for col in df.columns if col not in drop_cols]
    x_df = df[feature_names].copy()
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_df = x_df.apply(pd.to_numeric, errors="coerce")

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x_df, y, test_size=test_size, stratify=y, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=validation_size,
        stratify=y_train_full,
        random_state=seed,
    )

    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians)
    x_val = x_val.fillna(medians)
    x_test = x_test.fillna(medians)

    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(x_train).astype(np.float32)
    x_val_np = scaler.transform(x_val).astype(np.float32)
    x_test_np = scaler.transform(x_test).astype(np.float32)

    return DatasetBundle(
        x_train=x_train_np,
        y_train=y_train.astype(np.int64),
        x_val=x_val_np,
        y_val=y_val.astype(np.int64),
        x_test=x_test_np,
        y_test=y_test.astype(np.int64),
        feature_names=feature_names,
        label_summary=label_summary,
    )


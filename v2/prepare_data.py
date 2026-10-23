"""Build the three benign-vs-DDoS flow datasets used by ShieldDFL v2.

For each dataset: drop identifier, address, port and timestamp columns (they identify a
capture rather than a behaviour and leak across splits), coerce to numeric, remove
exact duplicate rows BEFORE any split, and draw a class-balanced sample. Output:
v2/data/<name>.npz with X (float32, unscaled), y (0 benign, 1 DDoS) and feature names,
plus v2/data/summary.json with row counts before and after deduplication.

Run: python -I v2/prepare_data.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1]                     # folder holding the three zip archives
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
PER_CLASS = 20_000
SEED = 2026


def finish(name: str, df: pd.DataFrame, label: pd.Series, info: dict) -> None:
    x = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, x.notna().mean() > 0.5]                       # drop mostly-empty columns
    x = x.fillna(x.median(numeric_only=True))
    x = x.loc[:, x.nunique() > 1]                              # drop constant columns
    full = x.assign(_y=label.values)
    info["rows_raw"] = int(len(full))
    full = full.drop_duplicates()
    info["rows_dedup"] = int(len(full))
    info["duplicates_removed"] = info["rows_raw"] - info["rows_dedup"]
    parts = []
    for cls in (0, 1):
        sub = full[full["_y"] == cls]
        parts.append(sub.sample(n=min(PER_CLASS, len(sub)), random_state=SEED))
        info[f"available_class{cls}"] = int(len(sub))
    s = pd.concat(parts).sample(frac=1.0, random_state=SEED)
    X = s.drop(columns="_y").to_numpy(np.float32)
    y = s["_y"].to_numpy(np.int64)
    info.update(n=int(len(y)), features=int(X.shape[1]), benign=int((y == 0).sum()),
                ddos=int((y == 1).sum()))
    np.savez_compressed(OUT / f"{name}.npz", X=X, y=y,
                        features=np.array(list(s.drop(columns="_y").columns)))
    print(name, info, flush=True)
    return info


def cicddos2019() -> dict:
    frames = []
    with zipfile.ZipFile(RAW / "CICDDoS2019.zip") as z:
        for n in z.namelist():
            if n.endswith(".parquet"):
                frames.append(pd.read_parquet(io.BytesIO(z.read(n))))
    df = pd.concat(frames, ignore_index=True)
    y = (df["Label"].astype(str).str.lower() != "benign").astype(int)
    info = {"attack_types": sorted(df.loc[y == 1, "Label"].astype(str).unique().tolist())}
    return finish("cicddos2019", df.drop(columns=["Label"]), y, info)


def cicids2018() -> dict:
    # DoS/DDoS capture days; the 4 GB 02-20 file is not needed for a 20k-per-class sample.
    days = ["02-15-2018.csv", "02-16-2018.csv", "02-21-2018.csv"]
    drop = {"Timestamp", "Dst Port", "Flow ID", "Src IP", "Src Port", "Dst IP"}
    keep = []
    with zipfile.ZipFile(RAW / "CICIDS2018.zip") as z:
        for d in days:
            for chunk in pd.read_csv(z.open(d), chunksize=250_000, low_memory=False):
                chunk = chunk[chunk["Label"] != "Label"]                # repeated header rows
                lab = chunk["Label"].astype(str)
                chunk = chunk[(lab == "Benign") | lab.str.contains("dos", case=False)]
                chunk = chunk.sample(frac=0.08, random_state=SEED)
                lab = chunk["Label"].astype(str)
                num = chunk.drop(columns=[c for c in chunk.columns if c in drop | {"Label"}])
                num = num.apply(pd.to_numeric, errors="coerce").astype(np.float32)
                keep.append(num.assign(Label=lab.values))
    df = pd.concat(keep, ignore_index=True)
    y = (df["Label"].astype(str) != "Benign").astype(int)
    info = {"days": days, "attack_types": sorted(df.loc[y == 1, "Label"].unique().tolist())}
    return finish("cicids2018", df.drop(columns=[c for c in df.columns if c in drop | {"Label"}]),
                  y, info)


def edgeiiot() -> dict:
    with zipfile.ZipFile(RAW / "Edhe-IIoTset.zip") as z:
        df = pd.read_csv(z.open("Edge-IIoTset dataset/Selected dataset for ML and DL/"
                                "ML-EdgeIIoT-dataset.csv"), low_memory=False)
    df = df[(df["Attack_type"] == "Normal") | df["Attack_type"].str.startswith("DDoS")]
    y = (df["Attack_type"] != "Normal").astype(int)
    info = {"attack_types": sorted(df.loc[y == 1, "Attack_type"].unique().tolist())}
    ident = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
             "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
             "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options", "tcp.payload",
             "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg", "Attack_label", "Attack_type"]
    df = df.drop(columns=[c for c in ident if c in df.columns])
    for c in df.columns:                       # remaining protocol strings -> category codes
        if df[c].dtype == object:
            df[c] = df[c].astype("category").cat.codes
    return finish("edgeiiot", df, y, info)


if __name__ == "__main__":
    import sys
    sp = OUT / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    for fn in (cicddos2019, edgeiiot, cicids2018):
        if fn.__name__ in summary and (OUT / f"{fn.__name__}.npz").exists():
            continue
        summary[fn.__name__] = fn()
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

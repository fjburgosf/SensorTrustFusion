"""Dataset export and (optional) import.

Export
------
:func:`export_dataset` writes the scenario dataset generated internally::

    dataset.csv         time, true_<state>, input_<u>, <sensor>, fault_<sensor>,
                        truebias_<sensor>, truevar_<sensor>, clean_<sensor>
    dataset_meta.json   seed, fingerprint, model, sensors, faults, prior

Import (optional capability)
----------------------------
:func:`import_dataset` reads CSV, TXT (whitespace / delimiter separated) or
JSON (records or column mapping) files.  The user maps columns to

* ``time`` (required, uniformly sampled);
* one or more columns per sensor (required);
* ``ground_truth`` columns per state (optional);
* ``faults`` columns per sensor (optional, 0/1).

If no ground truth is supplied, every metric that requires it (errors, NEES,
detection) is disabled automatically.  No benchmark, example or test of the
software depends on external data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.errors import DatasetError


def export_dataset(scenario, directory: str | Path, stem: str = "dataset") -> dict[str, Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    csv = d / f"{stem}.csv"
    scenario.to_dataframe().to_csv(csv, index=False, float_format="%.10g")
    meta = d / f"{stem}_meta.json"
    from ..experiments.config import to_builtin
    meta.write_text(json.dumps(to_builtin(scenario.metadata()), indent=2, default=str), encoding="utf-8")
    return {"csv": csv, "meta": meta}


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset file not found: {p}")
    suf = p.suffix.lower()
    try:
        if suf == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "data" in raw:
                raw = raw["data"]
            df = pd.DataFrame(raw)
        elif suf == ".csv":
            df = pd.read_csv(p)
        elif suf in (".txt", ".dat", ".tsv"):
            df = pd.read_csv(p, sep=None, engine="python")
        else:
            raise DatasetError(f"Unsupported dataset format '{suf}' (use CSV, TXT or JSON).")
    except (ValueError, json.JSONDecodeError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Cannot read dataset {p.name}: {exc}") from None
    if df.empty:
        raise DatasetError(f"Dataset {p.name} is empty.")
    return df


@dataclass
class ImportedDataset:
    t: np.ndarray
    z: dict[str, np.ndarray]                  # sensor name -> (K, m)
    truth: np.ndarray | None                  # (K, n) or None
    faults: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = ""
    columns: dict = field(default_factory=dict)

    @property
    def fs(self) -> float:
        return 1.0 / float(np.median(np.diff(self.t)))

    @property
    def has_truth(self) -> bool:
        return self.truth is not None


def import_dataset(path: str | Path, time: str = "time", sensors: dict[str, list[str] | str] | None = None,
                   ground_truth: list[str] | None = None, faults: dict[str, str] | None = None,
                   rtol: float = 1e-3) -> ImportedDataset:
    """Import an external dataset with an explicit column mapping.

    ``sensors`` maps sensor names to one column (scalar sensor) or a list of
    columns (vector sensor).  If omitted, every column that is not time,
    ground truth or fault is treated as a scalar sensor.
    """
    df = read_table(path)
    if time not in df:
        raise DatasetError(f"Time column '{time}' not found. Columns: {', '.join(map(str, df.columns))}.")
    t = pd.to_numeric(df[time], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(t)):
        raise DatasetError("The time column contains non-numeric or missing values.")
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise DatasetError("The time column must be strictly increasing with at least 2 samples.")
    dt = np.diff(t)
    if np.max(np.abs(dt - np.median(dt))) > rtol * np.median(dt) + 1e-12:
        raise DatasetError("The time column is not uniformly sampled; resample the data before importing.")
    gt_cols = list(ground_truth or [])
    fault_map = dict(faults or {})
    if sensors is None:
        used = {time, *gt_cols, *fault_map.values()}
        sensors = {c: c for c in df.columns if c not in used and not str(c).startswith(("true_", "fault_",
                                                                                          "truebias_", "truevar_",
                                                                                          "clean_", "input_"))}
    if not sensors:
        raise DatasetError("No sensor columns were selected.")
    z = {}
    for name, cols in sensors.items():
        cols = [cols] if isinstance(cols, str) else list(cols)
        missing = [c for c in cols if c not in df]
        if missing:
            raise DatasetError(f"Sensor '{name}': column(s) not found: {missing}.")
        arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        z[name] = arr.reshape(t.size, len(cols))
    truth = None
    if gt_cols:
        missing = [c for c in gt_cols if c not in df]
        if missing:
            raise DatasetError(f"Ground-truth column(s) not found: {missing}.")
        truth = df[gt_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(truth)):
            raise DatasetError("Ground-truth columns contain missing or non-numeric values.")
    fl = {}
    for name, col in fault_map.items():
        if col not in df:
            raise DatasetError(f"Fault column '{col}' not found.")
        fl[name] = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy() > 0
    return ImportedDataset(t=t - t[0], z=z, truth=truth, faults=fl, source=str(path),
                           columns={"time": time, "sensors": sensors, "ground_truth": gt_cols, "faults": fault_map})

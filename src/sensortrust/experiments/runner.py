"""Shared machinery for multi-run studies (Monte Carlo, sweeps, robustness, sensitivity).

:func:`evaluate_point` generates **one** scenario for a configuration and a
seed and runs every requested method on it (fair comparison), returning one
tidy row per method.  :func:`parallel_map` executes independent points with
``concurrent.futures`` processes (``n_jobs = 0`` -> all CPU cores,
``n_jobs = 1`` -> sequential), preserving the input order of the results.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..fusion import OracleInfo, create_method
from ..metrics import evaluate, flatten
from ..simulation.scenario import generate_scenario
from ..utils.errors import MethodNotApplicableError
from ..utils.paths import set_path
from ..version import SOFTWARE_NAME, __version__
from .config import config_hash, normalize_config, to_builtin


def apply_parameters(cfg: dict, params: dict) -> dict:
    c = copy.deepcopy(cfg)
    for path, value in params.items():
        set_path(c, path, value)
    return c


def evaluate_point(cfg: dict, seed: int, params: dict | None = None, methods: list | None = None,
                   nominal_twin: bool = False, extra: dict | None = None) -> list[dict]:
    """Run all methods of ``cfg`` (or ``methods``) on the scenario defined by ``seed`` and ``params``."""
    c = normalize_config(apply_parameters(cfg, params or {}))
    c["experiment"]["seed"] = int(seed)
    sc = generate_scenario(c, normalized=True)
    twin = generate_scenario(c, include_degradations=False, normalized=True) if nominal_twin else None
    meas = sc.measurements()
    rows = []
    for spec in (methods if methods is not None else c["methods"]):
        m = create_method(spec)
        try:
            orc = OracleInfo(sc.truth.x, [d.true_var for d in sc.data], [d.true_bias for d in sc.data]) \
                if m.is_oracle else None
            res = m.run(meas, sc.model, orc)
            nom = None
            if twin is not None:
                orc2 = OracleInfo(twin.truth.x, [d.true_var for d in twin.data], [d.true_bias for d in twin.data]) \
                    if m.is_oracle else None
                rn = m.run(twin.measurements(), twin.model, orc2)
                nom = (evaluate(twin, rn, c["metrics"]).get("estimation") or {}).get("rmse")
            row = flatten(evaluate(sc, res, c["metrics"], nominal_rmse=nom))
        except MethodNotApplicableError:
            continue
        row["method_key"] = m.key
        row["seed"] = int(seed)
        row["dataset_sha256"] = sc.fingerprint()[:16]
        for k, v in (params or {}).items():
            row[f"param:{k}"] = v
        row.update(extra or {})
        rows.append(row)
    return rows


def _call(args):
    fn, a = args
    return fn(*a)


def resolve_jobs(n_jobs: int | None) -> int:
    if n_jobs is None or n_jobs <= 0:
        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def parallel_map(fn: Callable, arglist: list[tuple], n_jobs: int | None = 0,
                 progress: Callable[[str, float], None] | None = None, label: str = "run",
                 cancel: Callable[[], bool] | None = None) -> list:
    n = len(arglist)
    out: list = [None] * n
    jobs = resolve_jobs(n_jobs)
    if jobs == 1 or n <= 1:
        for i, a in enumerate(arglist):
            if cancel and cancel():
                raise InterruptedError("cancelled by user")
            out[i] = fn(*a)
            if progress:
                progress(f"{label} {i + 1}/{n}", (i + 1) / n)
        return out
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fn, *a): i for i, a in enumerate(arglist)}
        done = 0
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if progress:
                progress(f"{label} {done}/{n}", done / n)
            if cancel and cancel():
                for g in futs:
                    g.cancel()
                raise InterruptedError("cancelled by user")
    return out


def study_manifest(kind: str, eid: str | None, cfg: dict, extra: dict) -> dict:
    import platform
    import sys
    return to_builtin({
        "study_id": eid, "study_type": kind, "software": SOFTWARE_NAME, "version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "experiment_name": cfg["experiment"]["name"], "base_seed": cfg["experiment"]["seed"],
        "config_sha256": config_hash(cfg), "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(),
        "hardware_required": "none (virtual sensors, computational simulation only)", **extra,
    })


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(to_builtin(obj), indent=2, default=str), encoding="utf-8")


def iter_chunks(seq: Iterable, n: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

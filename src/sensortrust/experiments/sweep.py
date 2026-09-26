"""Parametric sweeps (full-factorial grids over 1 or 2+ parameters).

For every grid point and every repetition ``r`` a scenario is generated with
the repetition seed ``s_r`` (the same ``s_r`` at every grid point: common
random numbers, so differences between grid points are due to the parameters
and not to different noise draws).  Results are aggregated per grid point and
method (mean, median, 5 / 95 percentiles, success rate); 2-D sweeps produce
maps such as ``RMSE(bias, sigma)`` or ``DetectionDelay(bias, sigma)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.store import ResultStore, slug
from ..utils.errors import ConfigurationError
from ..utils.logging import file_log, get_logger
from ..utils.rng import derive_seeds
from .config import normalize_config, save_config
from .design import grid_design, grid_values
from .runner import evaluate_point, parallel_map, study_manifest, write_json

log = get_logger("sweep")


@dataclass
class SweepResult:
    study_id: str | None
    config: dict
    parameters: list[dict]
    runs: pd.DataFrame
    aggregate: pd.DataFrame
    manifest: dict
    output_dir: Path | None


def aggregate_runs(df: pd.DataFrame, keys: list[str], metrics: list[str], spec: float | None = None,
                   spec_metric: str = "rmse") -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys + ["method"], sort=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys + ["method"], k))
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce").dropna() if m in g else pd.Series(dtype=float)
            row[f"{m}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{m}_median"] = float(v.median()) if len(v) else np.nan
            row[f"{m}_p05"] = float(v.quantile(0.05)) if len(v) else np.nan
            row[f"{m}_p95"] = float(v.quantile(0.95)) if len(v) else np.nan
            row[f"{m}_n"] = int(len(v))
        ok = ~g["diverged"].astype(bool)
        if spec is not None:
            ok &= pd.to_numeric(g[spec_metric], errors="coerce") <= spec
        row["success_rate"] = float(ok.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_sweep(config: dict, save: bool = True, results_root: str | Path = "results", progress=None,
              n_jobs: int | None = None, figures: bool = True, cancel=None) -> SweepResult:
    cfg = normalize_config(config)
    an = dict(cfg.get("analysis") or {})
    params = list(an.get("parameters", []))
    if not params:
        raise ConfigurationError("analysis.parameters must list at least one parameter to sweep.")
    reps = int(an.get("repetitions", 5))
    metrics = list(an.get("metrics", ["rmse", "rmse_during_fault", "mean_detection_delay"]))
    spec = an.get("spec")
    master = int(an.get("master_seed", cfg["experiment"]["seed"]))
    jobs = n_jobs if n_jobs is not None else int(an.get("n_jobs", 0))
    points = grid_design(params)
    seeds = derive_seeds(master, reps, "sweep")
    args = [(cfg, seeds[r], pt, None, False, {"point": i, "repetition": r + 1})
            for i, pt in enumerate(points) for r in range(reps)]
    eid, out = (None, None)
    if save:
        eid, out = ResultStore(results_root).new_directory("ST_SWP")
    with (file_log(out / "run.log") if out else _Null()):
        log.info("Sweep %s: %d grid points x %d repetitions", eid, len(points), reps)
        chunks = parallel_map(evaluate_point, args, jobs, progress, "Sweep evaluation", cancel)
        df = pd.DataFrame([r for ch in chunks for r in ch])
        keys = [f"param:{p['path']}" for p in params]
        agg = aggregate_runs(df, keys, [m for m in metrics if m in df], spec)
        manifest = study_manifest("sweep", eid, cfg, {"parameters": params, "repetitions": reps,
                                                      "grid_points": len(points), "seeds": seeds, "spec": spec})
        res = SweepResult(eid, cfg, params, df, agg, manifest, out)
        if out is not None:
            save_config(cfg, out / "config.yaml")
            df.to_csv(out / "runs.csv", index=False, float_format="%.8g")
            agg.to_csv(out / "aggregate.csv", index=False, float_format="%.6g")
            if figures:
                manifest["figures"] = [f.name for f in save_sweep_figures(res, out / "figures")]
            manifest["files"] = sorted(p.name for p in out.iterdir())
            write_json(out / "manifest.json", manifest)
    return res


def sweep_grid(res: SweepResult, method: str, metric: str):
    px, py = res.parameters[0], res.parameters[1]
    xv, yv = grid_values(px), grid_values(py)
    a = res.aggregate[res.aggregate.method == method]
    G = np.full((len(yv), len(xv)), np.nan)
    for _, row in a.iterrows():
        i = int(np.argmin(np.abs(np.asarray(yv, float) - float(row[f"param:{py['path']}"]))))
        j = int(np.argmin(np.abs(np.asarray(xv, float) - float(row[f"param:{px['path']}"]))))
        G[i, j] = row[metric]
    return G, xv, yv


def save_sweep_figures(res: SweepResult, directory: Path, formats=("png",)) -> list[Path]:
    from ..visualization import plots
    from ..visualization.report import save_figure
    files = []
    if len(res.parameters) < 2:
        p = res.parameters[0]
        agg = res.aggregate.rename(columns={f"param:{p['path']}": "severity"})
        for m in [c[:-5] for c in agg.columns if c.endswith("_mean")]:
            env = {"table": agg, "metric": m, "spec_value": float(res.config.get("analysis", {}).get("spec") or
                                                                   np.nanmax(agg[f"{m}_mean"])),
                   "tolerable": {}, "parameter_label": p.get("label", p["path"])}
            files += save_figure(plots.plot_robustness_envelope(env), directory / f"sweep_{m}", formats)
        return files
    px, py = res.parameters[0], res.parameters[1]
    for method in dict.fromkeys(res.aggregate["method"]):
        for m in [c for c in res.aggregate.columns if c.endswith("_mean")]:
            G, xv, yv = sweep_grid(res, method, m)
            if not np.isfinite(G).any():
                continue
            fig = plots.plot_heatmap(G, xv, yv, px.get("label", px["path"]), py.get("label", py["path"]),
                                     f"{m.replace('_mean', '')} (mean) - {method}", cbar_label=m)
            files += save_figure(fig, directory / f"map_{slug(m)}_{slug(method)}", formats)
    return files


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False

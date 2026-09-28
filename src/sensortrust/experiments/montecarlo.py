"""Monte Carlo engine.

For run ``r = 1..N``:

1. a run seed ``s_r`` is derived from ``master_seed`` (SeedSequence);
2. the varied parameters are drawn from the design (random / LHS / Sobol,
   seeded by ``master_seed``);
3. **one** scenario is generated with ``(config + parameters, s_r)``;
4. every method is executed on that same scenario and evaluated.

Every run stores its seed and parameters, so any single run can be
re-executed exactly (``python -m sensortrust run config.yaml --seed s_r`` with
the listed parameters).  Output directory ``results/ST_MC_<year>_<n>/``::

    config.yaml, manifest.json, design.csv, runs.csv (tidy, one row per run x method),
    summary.csv (descriptive statistics + 95 % CI), paired_tests.csv (Wilcoxon vs the
    reference method, Holm-corrected), figures/
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..io.store import ResultStore
from ..statistics import pairwise_tests, summarize
from ..utils.errors import ConfigurationError
from ..utils.logging import file_log, get_logger
from ..utils.rng import derive_seeds, validate_seed
from .config import normalize_config, save_config
from .design import sample_design
from .runner import evaluate_point, parallel_map, study_manifest, write_json

log = get_logger("montecarlo")

MC_METRICS = ["rmse", "mae", "max_error", "rmse_pre_fault", "rmse_during_fault", "rmse_post_fault",
              "nis_normalized", "nees_normalized", "mean_detection_delay", "mean_f1", "mean_false_alarm_rate",
              "degraded_sensor_trust_min", "trust_response_time", "trust_recovery_time", "performance_loss",
              "runtime_s"]


@dataclass
class MonteCarloResult:
    study_id: str | None
    config: dict
    runs: pd.DataFrame
    design: pd.DataFrame
    summary: pd.DataFrame
    tests: pd.DataFrame
    success_rate: pd.DataFrame
    manifest: dict
    output_dir: Path | None


def run_montecarlo(config: dict, runs: int | None = None, n_jobs: int | None = None, save: bool = True,
                   results_root: str | Path = "results", progress: Callable[[str, float], None] | None = None,
                   figures: bool = True, cancel=None) -> MonteCarloResult:
    cfg = normalize_config(config)
    mc = dict(cfg.get("montecarlo") or {})
    n = int(runs if runs is not None else mc.get("runs", 100))
    if n < 1:
        raise ConfigurationError("Monte Carlo requires at least 1 run.")
    master = validate_seed(mc.get("master_seed", cfg["experiment"]["seed"]))
    sampler = str(mc.get("sampler", "random"))
    jobs = n_jobs if n_jobs is not None else int(mc.get("n_jobs", 0))
    params = list(mc.get("parameters", []))
    twin = bool(mc.get("paired_nominal", False))
    design = sample_design(params, n, sampler, master) if params else [{} for _ in range(n)]
    n = len(design)
    seeds = derive_seeds(master, n, "montecarlo")
    eid, out = (None, None)
    if save:
        eid, out = ResultStore(results_root).new_directory("ST_MC")
    ctx = file_log(out / "run.log") if out else _Null()
    with ctx:
        t0 = time.perf_counter()
        log.info("Monte Carlo %s: %d runs, sampler=%s, master seed=%d, %d varied parameters", eid, n, sampler,
                 master, len(params))
        args = [(cfg, seeds[i], design[i], None, twin, {"run": i + 1}) for i in range(n)]
        chunks = parallel_map(evaluate_point, args, jobs, progress, "Monte Carlo run", cancel)
        rows = [r for ch in chunks for r in ch]
        df = pd.DataFrame(rows)
        labels = {p["path"]: p.get("label", p["path"]) for p in params}
        dz = pd.DataFrame([{"run": i + 1, "seed": seeds[i], **{labels.get(k, k): v for k, v in d.items()}}
                           for i, d in enumerate(design)])
        metrics = [m for m in MC_METRICS if m in df and df[m].notna().any()]
        summary = summarize(df, metrics)
        ref_key = mc.get("reference_method", "sensortrust_kf")
        ref = df.loc[df.method_key == ref_key, "method"].iloc[0] if (df.method_key == ref_key).any() else None
        tests = pd.concat([pairwise_tests(df, m, ref) for m in ("rmse", "rmse_during_fault") if m in metrics],
                          ignore_index=True) if ref else pd.DataFrame()
        sr = df.groupby("method", sort=False).agg(success_rate=("success", "mean"),
                                                   divergence_rate=("diverged", "mean"), n=("run", "count"))
        sr = sr.reset_index()
        elapsed = time.perf_counter() - t0
        log.info("Monte Carlo finished in %.1f s (%d rows)", elapsed, len(df))
        manifest = study_manifest("montecarlo", eid, cfg, {
            "runs": n, "sampler": sampler, "master_seed": master, "run_seeds_first": seeds[:5],
            "varied_parameters": params, "reference_method": ref, "methods": list(dict.fromkeys(df["method"])),
            "paired_nominal": twin, "elapsed_s": elapsed, "n_jobs": jobs,
        })
        res = MonteCarloResult(eid, cfg, df, dz, summary, tests, sr, manifest, out)
        if out is not None:
            save_config(cfg, out / "config.yaml")
            dz.to_csv(out / "design.csv", index=False)
            df.to_csv(out / "runs.csv", index=False, float_format="%.8g")
            summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
            sr.to_csv(out / "success_rate.csv", index=False, float_format="%.6g")
            if not tests.empty:
                tests.to_csv(out / "paired_tests.csv", index=False, float_format="%.6g")
            if figures:
                files = save_montecarlo_figures(res, out / "figures")
                manifest["figures"] = [f.name for f in files]
            manifest["files"] = sorted(p.name for p in out.iterdir())
            write_json(out / "manifest.json", manifest)
    return res


def save_montecarlo_figures(res: MonteCarloResult, directory: Path, formats=("png",)) -> list[Path]:
    from ..visualization import plots
    from ..visualization.report import save_figure
    unit = ""
    files = []
    df = res.runs
    for metric in ("rmse", "rmse_during_fault", "mean_detection_delay", "degraded_sensor_trust_min"):
        if metric not in df or not df[metric].notna().any():
            continue
        d = df.dropna(subset=[metric])
        files += save_figure(plots.plot_mc_boxplot(d, metric, unit=unit, log=metric.startswith("rmse")),
                             directory / f"boxplot_{metric}", formats)
        files += save_figure(plots.plot_mc_ecdf(d, metric, unit=unit, log=metric.startswith("rmse")),
                             directory / f"ecdf_{metric}", formats)
    if "rmse" in df:
        files += save_figure(plots.plot_mc_histogram(df.dropna(subset=["rmse"]), "rmse"), directory / "histogram_rmse",
                             formats)
    return files


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class MonteCarloEngine:
    """Object-oriented wrapper (used by the GUI)."""

    def __init__(self, results_root: str | Path = "results"):
        self.results_root = results_root

    def run(self, config: dict, **kw) -> MonteCarloResult:
        return run_montecarlo(config, results_root=self.results_root, **kw)

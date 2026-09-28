r"""Sensitivity analysis: One-at-a-Time, Morris and Sobol.

The model under study is the map ``theta -> y = metric(method, scenario(theta, s))``
where ``theta`` are the selected parameters (degradation severity, window,
EWMA factor, noise, fault time, model parameters, ...) and ``y`` a scalar
performance metric (e.g. RMSE of the SensorTrust KF).

``seed_mode = "common"`` (default) evaluates every sample with the same
scenario seed (common random numbers), so that the indices measure the effect
of the parameters and not of different noise realisations; ``"random"``
uses a different derived seed per evaluation (the noise then contributes to
the unexplained variance).

* **OAT**: each parameter is varied over ``levels`` points in ``[low, high]``
  with the others at their ``nominal`` value (default: interval midpoint).
  Reported: output range ``max(y) - min(y)`` per parameter.
* **Morris** elementary effects (SALib, ``num_levels`` grid, ``N``
  trajectories -> ``N (D + 1)`` evaluations): ``mu*`` (mean absolute effect,
  importance) and ``sigma`` (nonlinearity / interactions).
Optionally ``output.transform: log10`` analyses ``log10(y)`` (recommended for
positive, right-skewed metrics such as the RMSE, where a few large values would
otherwise dominate the variance decomposition).

* **Sobol** variance-based indices (SALib Saltelli/Sobol, ``N`` base samples,
  ``N (D + 2)`` evaluations without second order): first-order ``S1`` and
  total-order ``ST`` with bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.store import ResultStore
from ..utils.errors import ConfigurationError
from ..utils.logging import file_log, get_logger
from ..utils.rng import derive_seeds
from .config import normalize_config, save_config
from .runner import evaluate_point, parallel_map, study_manifest, write_json

log = get_logger("sensitivity")
SENSITIVITY_METHODS = ("oat", "morris", "sobol")


@dataclass
class SensitivityResult:
    study_id: str | None
    config: dict
    method: str
    samples: pd.DataFrame
    result: dict
    manifest: dict
    output_dir: Path | None


def _resolve_method(cfg: dict, method_ref) -> dict:
    """Method specification taken from the (already modified) configuration.

    ``method_ref`` may be an index, a method name / label present in
    ``cfg['methods']`` (so that varied method parameters such as
    ``methods[5].trust.ewma_lambda`` are honoured) or a full specification.
    """
    if isinstance(method_ref, dict):
        return method_ref
    methods = cfg["methods"]
    if isinstance(method_ref, int):
        return methods[method_ref]
    for m in methods:
        if m.get("label") == method_ref:
            return m
    for m in methods:
        if m.get("name") == method_ref:
            return m
    return {"name": str(method_ref)}


def _eval(cfg, seed, point, method_ref, metric):
    from .runner import apply_parameters
    mod = apply_parameters(cfg, point)
    rows = evaluate_point(mod, seed, None, [_resolve_method(mod, method_ref)])
    if not rows:
        return np.nan
    v = rows[0].get(metric)
    return float(v) if v is not None else np.nan


def run_sensitivity(config: dict, save: bool = True, results_root: str | Path = "results", progress=None,
                    n_jobs: int | None = None, figures: bool = True, cancel=None) -> SensitivityResult:
    cfg = normalize_config(config)
    an = dict(cfg.get("analysis") or {})
    kind = str(an.get("method", "morris")).lower()
    if kind not in SENSITIVITY_METHODS:
        raise ConfigurationError(f"analysis.method must be one of {SENSITIVITY_METHODS}.")
    params = list(an.get("parameters", []))
    if not params:
        raise ConfigurationError("Sensitivity analysis requires analysis.parameters with path, low, high.")
    for p in params:
        if "path" not in p or "low" not in p or "high" not in p or float(p["low"]) >= float(p["high"]):
            raise ConfigurationError(f"Sensitivity parameter {p.get('path')}: requires low < high.")
    out_spec = an.get("output", {"method": "sensortrust_kf", "metric": "rmse"})
    method_ref = out_spec.get("method", "sensortrust_kf")
    transform = str(out_spec.get("transform", "none"))
    if transform not in ("none", "log10"):
        raise ConfigurationError("analysis.output.transform must be 'none' or 'log10'.")
    mspec = _resolve_method(cfg, method_ref)
    metric = out_spec.get("metric", "rmse")
    names = [p.get("label", p["path"]) for p in params]
    problem = {"num_vars": len(params), "names": names,
               "bounds": [[float(p["low"]), float(p["high"])] for p in params]}
    master = int(an.get("master_seed", cfg["experiment"]["seed"]))
    seed_mode = an.get("seed_mode", "common")
    jobs = n_jobs if n_jobs is not None else int(an.get("n_jobs", 0))
    N = int(an.get("samples", {"oat": 7, "morris": 10, "sobol": 64}[kind]))

    if kind == "oat":
        nominal = [float(p.get("nominal", 0.5 * (float(p["low"]) + float(p["high"])))) for p in params]
        X = []
        owner = []
        for j, p in enumerate(params):
            for v in np.linspace(float(p["low"]), float(p["high"]), N):
                x = list(nominal)
                x[j] = float(v)
                X.append(x)
                owner.append(j)
        X = np.array(X)
    elif kind == "morris":
        from SALib.sample import morris as ms
        X = ms.sample(problem, N, num_levels=int(an.get("num_levels", 4)), seed=master % (2**31))
    else:
        from SALib.sample import sobol as ss
        X = ss.sample(problem, N, calc_second_order=False, seed=master % (2**31))

    n_eval = X.shape[0]
    seeds = derive_seeds(master, n_eval, "sensitivity") if seed_mode == "random" else [master] * n_eval
    args = [(cfg, seeds[i], {p["path"]: float(X[i, j]) for j, p in enumerate(params)}, method_ref, metric)
            for i in range(n_eval)]
    eid, out = (None, None)
    if save:
        eid, out = ResultStore(results_root).new_directory("ST_SEN")
    with (file_log(out / "run.log") if out else _Null()):
        log.info("Sensitivity %s (%s): %d parameters, %d evaluations, output %s/%s", eid, kind, len(params), n_eval,
                 mspec["name"], metric)
        Y = np.array(parallel_map(_eval, args, jobs, progress, "Sensitivity evaluation", cancel), dtype=float)
        samples = pd.DataFrame(X, columns=names)
        samples["seed"] = seeds
        samples[metric] = Y
        if transform == "log10":
            Y = np.log10(np.where(Y > 0, Y, np.nan))
            samples[f"log10_{metric}"] = Y
        if np.any(~np.isfinite(Y)):
            log.warning("%d evaluations returned non-finite outputs; they are replaced by the mean for SALib.",
                        int(np.sum(~np.isfinite(Y))))
            Y = np.where(np.isfinite(Y), Y, np.nanmean(Y))
        label = f"{'log10 ' if transform == 'log10' else ''}{metric} ({mspec.get('label', mspec['name'])})"
        label_parts = {"metric": metric, "method": mspec.get("label", mspec["name"]), "log10": transform == "log10"}
        if kind == "oat":
            curves, rows = {}, []
            owner = np.array(owner)
            for j, nm in enumerate(names):
                sel = owner == j
                xv = X[sel, j]
                lo, hi = problem["bounds"][j]
                curves[nm] = {"values": xv, "relative": (xv - lo) / (hi - lo), "output": Y[sel]}
                rows.append({"parameter": nm, "output_min": float(Y[sel].min()), "output_max": float(Y[sel].max()),
                             "output_range": float(Y[sel].max() - Y[sel].min())})
            idx = pd.DataFrame(rows).sort_values("output_range", ascending=False)
            result = {"indices": idx, "curves": curves, "output_label": label, "output_parts": label_parts,
                      "n_evaluations": n_eval}
        elif kind == "morris":
            from SALib.analyze import morris as ma
            r = ma.analyze(problem, X, Y, num_levels=int(an.get("num_levels", 4)), seed=master % (2**31))
            idx = pd.DataFrame({"parameter": names, "mu": r["mu"], "mu_star": r["mu_star"], "sigma": r["sigma"],
                                "mu_star_conf": r["mu_star_conf"]}).sort_values("mu_star", ascending=False)
            result = {"indices": idx, "output_label": label, "output_parts": label_parts,
                      "n_evaluations": n_eval}
        else:
            from SALib.analyze import sobol as sa
            r = sa.analyze(problem, Y, calc_second_order=False, seed=master % (2**31))
            idx = pd.DataFrame({"parameter": names, "S1": r["S1"], "S1_conf": r["S1_conf"], "ST": r["ST"],
                                "ST_conf": r["ST_conf"]}).sort_values("ST", ascending=False)
            result = {"indices": idx, "output_label": label, "output_parts": label_parts,
                      "n_evaluations": n_eval}
        manifest = study_manifest("sensitivity", eid, cfg, {
            "method": kind, "parameters": params, "output_method": mspec, "metric": metric, "samples": N,
            "evaluations": n_eval, "seed_mode": seed_mode, "master_seed": master,
            "ranking": list(result["indices"]["parameter"])})
        res = SensitivityResult(eid, cfg, kind, samples, result, manifest, out)
        if out is not None:
            save_config(cfg, out / "config.yaml")
            samples.to_csv(out / "samples.csv", index=False, float_format="%.8g")
            result["indices"].to_csv(out / "indices.csv", index=False, float_format="%.6g")
            if figures:
                from ..visualization import plots
                from ..visualization.report import save_figure
                fn = {"oat": plots.plot_oat, "morris": plots.plot_morris, "sobol": plots.plot_sobol}[kind]
                manifest["figures"] = [f.name for f in save_figure(fn(result), out / "figures" / f"sensitivity_{kind}")]
            manifest["files"] = sorted(p.name for p in out.iterdir())
            write_json(out / "manifest.json", manifest)
    return res


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False

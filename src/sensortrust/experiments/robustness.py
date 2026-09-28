r"""Robustness envelope and robustness maps.

Robustness envelope (one degradation parameter ``b``)
-----------------------------------------------------
For each tested value ``b_j`` (sorted increasingly) and repetition ``r`` the
scenario is generated with the repetition seed ``s_r`` (common random numbers
across ``b_j``).  For every method the statistic ``g(b_j)`` is

* ``criterion = "mean"``: mean of the metric over repetitions, pass if
  ``g(b_j) <= spec``;
* ``criterion = "success_rate"``: fraction of repetitions with
  ``metric <= spec`` and no divergence, pass if ``g(b_j) >= required_rate``.

The tolerable severity is

.. math:: b^\star = \sup\{b_j : \text{all } b_i \le b_j \text{ pass}\},

refined by linear interpolation of ``g`` between the last passing value and
the first failing one.  If every tested value passes, ``b*`` is reported as
``>= max(b)`` (flag ``reached = False``); if the smallest value already fails,
``b*`` is ``None``.

Robustness map (two parameters)
-------------------------------
Success rate of every method on a grid ``(b_x, b_y)``.
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
from .runner import evaluate_point, parallel_map, study_manifest, write_json
from .sweep import aggregate_runs

log = get_logger("robustness")


@dataclass
class RobustnessResult:
    study_id: str | None
    config: dict
    envelope: dict | None
    map: dict | None
    runs: pd.DataFrame
    manifest: dict
    output_dir: Path | None


def tolerable_limit(values, stat, passes, criterion: str, spec: float, required: float):
    """Return ``(b_star, reached)`` for sorted ``values`` with statistic ``stat`` and pass flags."""
    values = np.asarray(values, float)
    if len(values) == 0 or not passes[0]:
        return None, True
    last = 0
    for j in range(1, len(values)):
        if passes[j]:
            last = j
        else:
            break
    else:
        return float(values[-1]), False
    j = last + 1
    g0, g1 = float(stat[last]), float(stat[j])
    target = spec if criterion == "mean" else required
    if np.isfinite(g0) and np.isfinite(g1) and g1 != g0:
        frac = (target - g0) / (g1 - g0)
        frac = min(max(frac, 0.0), 1.0)
        return float(values[last] + frac * (values[j] - values[last])), True
    return float(values[last]), True


def robustness_envelope(cfg: dict, save_dir: Path | None = None, progress=None, n_jobs=0, cancel=None) -> dict:
    an = cfg["analysis"]
    par = an["parameter"]
    values = sorted(float(v) for v in par["values"])
    reps = int(an.get("repetitions", 10))
    metric = an.get("metric", "rmse")
    spec = float(an["spec"])
    criterion = an.get("criterion", "mean")
    required = float(an.get("required_success_rate", 0.95))
    if criterion not in ("mean", "success_rate"):
        raise ConfigurationError("analysis.criterion must be 'mean' or 'success_rate'.")
    seeds = derive_seeds(int(an.get("master_seed", cfg["experiment"]["seed"])), reps, "robustness")
    args = [(cfg, seeds[r], {par["path"]: v}, None, False, {"repetition": r + 1})
            for v in values for r in range(reps)]
    chunks = parallel_map(evaluate_point, args, n_jobs, progress, "Robustness evaluation", cancel)
    df = pd.DataFrame([r for ch in chunks for r in ch])
    key = f"param:{par['path']}"
    agg = aggregate_runs(df, [key], [metric], spec, metric).rename(columns={key: "severity"})
    agg = agg.sort_values(["method", "severity"])
    tol, reached = {}, {}
    for m, g in agg.groupby("method", sort=False):
        stat = g[f"{metric}_mean"].values if criterion == "mean" else g["success_rate"].values
        passes = stat <= spec if criterion == "mean" else stat >= required
        tol[m], reached[m] = tolerable_limit(g["severity"].values, stat, passes, criterion, spec, required)
    return {"table": agg, "runs": df, "metric": metric, "spec_value": spec, "criterion": criterion,
            "required_success_rate": required, "tolerable": tol, "limit_reached": reached,
            "parameter_label": par.get("label", par["path"]), "parameter_path": par["path"],
            "repetitions": reps, "values": values}


def robustness_map(cfg: dict, progress=None, n_jobs=0, cancel=None) -> dict:
    an = cfg["analysis"]
    mp = an["map"]
    xs, ys = [float(v) for v in mp["values_x"]], [float(v) for v in mp["values_y"]]
    reps = int(mp.get("repetitions", an.get("repetitions", 5)))
    metric = an.get("metric", "rmse")
    spec = float(an["spec"])
    seeds = derive_seeds(int(an.get("master_seed", cfg["experiment"]["seed"])), reps, "robustness_map")
    args = [(cfg, seeds[r], {mp["path_x"]: x, mp["path_y"]: y}, None, False, {"repetition": r + 1})
            for y in ys for x in xs for r in range(reps)]
    chunks = parallel_map(evaluate_point, args, n_jobs, progress, "Robustness map evaluation", cancel)
    df = pd.DataFrame([r for ch in chunks for r in ch])
    kx, ky = f"param:{mp['path_x']}", f"param:{mp['path_y']}"
    agg = aggregate_runs(df, [kx, ky], [metric], spec, metric)
    grids = {}
    for m, g in agg.groupby("method", sort=False):
        G = np.full((len(ys), len(xs)), np.nan)
        for _, row in g.iterrows():
            G[ys.index(float(row[ky])), xs.index(float(row[kx]))] = row["success_rate"]
        grids[m] = G
    return {"x": xs, "y": ys, "label_x": mp.get("label_x", mp["path_x"]), "label_y": mp.get("label_y", mp["path_y"]),
            "grids": grids, "aggregate": agg, "runs": df, "spec_value": spec, "metric": metric, "repetitions": reps}


def run_robustness(config: dict, save: bool = True, results_root: str | Path = "results", progress=None,
                   n_jobs: int | None = None, include_map: bool = True, figures: bool = True,
                   cancel=None) -> RobustnessResult:
    cfg = normalize_config(config)
    an = cfg.get("analysis") or {}
    if an.get("type", "robustness") != "robustness" or "parameter" not in an or "spec" not in an:
        raise ConfigurationError("Robustness analysis requires analysis: {type: robustness, parameter: {...}, "
                                 "spec: value}.")
    jobs = n_jobs if n_jobs is not None else int(an.get("n_jobs", 0))
    eid, out = (None, None)
    if save:
        eid, out = ResultStore(results_root).new_directory("ST_ROB")
    with (file_log(out / "run.log") if out else _Null()):
        prog1 = (lambda msg, f: progress(msg, 0.6 * f)) if progress else None
        env = robustness_envelope(cfg, None, prog1, jobs, cancel)
        mp = None
        if include_map and an.get("map"):
            prog2 = (lambda msg, f: progress(msg, 0.6 + 0.4 * f)) if progress else None
            mp = robustness_map(cfg, prog2, jobs, cancel)
        for m, b in env["tolerable"].items():
            log.info("Tolerable %s for %s: %s%s", env["parameter_label"], m, b,
                     "" if env["limit_reached"][m] else " (not reached: all tested values pass)")
        manifest = study_manifest("robustness", eid, cfg, {
            "parameter": an["parameter"], "metric": env["metric"], "spec": env["spec_value"],
            "criterion": env["criterion"], "repetitions": env["repetitions"],
            "tolerable_severity": env["tolerable"], "limit_reached": env["limit_reached"]})
        runs = env["runs"] if mp is None else pd.concat([env["runs"].assign(study="envelope"),
                                                          mp["runs"].assign(study="map")], ignore_index=True)
        res = RobustnessResult(eid, cfg, env, mp, runs, manifest, out)
        if out is not None:
            save_config(cfg, out / "config.yaml")
            env["table"].to_csv(out / "envelope.csv", index=False, float_format="%.6g")
            pd.DataFrame([{"method": m, "tolerable_severity": b, "limit_reached": env["limit_reached"][m]}
                          for m, b in env["tolerable"].items()]).to_csv(out / "tolerable_severity.csv", index=False)
            runs.to_csv(out / "runs.csv", index=False, float_format="%.8g")
            if mp is not None:
                mp["aggregate"].to_csv(out / "map.csv", index=False, float_format="%.6g")
            if figures:
                manifest["figures"] = [f.name for f in save_robustness_figures(res, out / "figures")]
            manifest["files"] = sorted(p.name for p in out.iterdir())
            write_json(out / "manifest.json", manifest)
    return res


def save_robustness_figures(res: RobustnessResult, directory: Path, formats=("png",)) -> list[Path]:
    from ..i18n import tr
    from ..visualization import plots
    from ..visualization.report import save_figure
    files = save_figure(plots.plot_robustness_envelope(res.envelope), directory / "robustness_envelope", formats)
    if res.map is not None:
        mp = res.map
        for m, G in mp["grids"].items():
            fig = plots.plot_heatmap(G, mp["x"], mp["y"], mp["label_x"], mp["label_y"],
                                     f"{tr('Robustness map')}: P({mp['metric']} <= {mp['spec_value']:g}) - {m}",
                                     cbar_label=tr("Success rate"), contour_level=0.95, fmt="{:.2f}")
            files += save_figure(fig, directory / f"robustness_map_{slug(m)}", formats)
    return files


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False

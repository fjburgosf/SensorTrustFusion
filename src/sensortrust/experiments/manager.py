"""Experiment Manager: the single entry point used by the GUI, the CLI and scripts.

``ExperimentManager.run(config)`` executes, in this order:

1. configuration validation and experiment identifier;
2. ground-truth generation;
3. virtual sensors, noise and degradation (scenario dataset);
4. every selected fusion method on **exactly the same** scenario;
5. metrics (and the nominal twin for relative performance loss);
6. figures;
7. storage of the results directory and the reproducibility manifest.

Oracle methods receive the ground truth through :class:`OracleInfo`; all
other methods only receive the :class:`MeasurementSet`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..fusion import OracleInfo, create_method
from ..fusion.base import MethodResult
from ..io.dataset import ImportedDataset, export_dataset
from ..io.store import ResultStore, slug
from ..metrics import evaluate, flatten
from ..simulation.ground_truth import GroundTruth
from ..simulation.scenario import Scenario, _prior, build_model, generate_scenario
from ..sensors.sensor import SensorData, VirtualSensor
from ..utils.errors import ConfigurationError, MethodNotApplicableError, SensorTrustError
from ..utils.logging import file_log, get_logger
from ..utils.rng import seed_global
from ..version import SOFTWARE_NAME, SUBTITLE_EN, __version__
from .config import config_hash, normalize_config, save_config, to_builtin

log = get_logger("experiments")

Progress = Callable[[str, float], None]


@dataclass
class ExperimentResult:
    experiment_id: str | None
    config: dict
    scenario: Scenario
    results: dict[str, MethodResult]
    metrics: dict[str, dict]
    skipped: dict[str, str]
    table: pd.DataFrame
    manifest: dict
    output_dir: Path | None = None
    nominal_results: dict[str, MethodResult] = field(default_factory=dict)

    def method(self, label: str) -> MethodResult:
        return self.results[label]

    @property
    def labels(self) -> list[str]:
        return list(self.results)


def _environment() -> dict:
    import matplotlib
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()}


def _fault_text(meta: dict) -> str:
    prof = meta.get("profile", {})
    txt = f"{meta['sensor']}: {meta['type']} (severity {meta['severity']:g}, profile {prof.get('type')}, " \
          f"onset {meta['onset_time']:g} s"
    if meta.get("recovery_time") is not None:
        txt += f", recovery from {meta['recovery_time']:g} s"
    return txt + ")"


class ExperimentManager:
    def __init__(self, results_root: str | Path = "results"):
        self.store = ResultStore(results_root)

    # ------------------------------------------------------------------
    def run(self, config: dict, save: bool = True, figures: bool = True, nominal_twin: bool = True,
            progress: Progress | None = None, experiment_id: str | None = None,
            scenario: Scenario | None = None) -> ExperimentResult:
        cfg = normalize_config(config)
        prog = progress or (lambda msg, frac: None)
        eid, out = (None, None)
        if save:
            eid, out = self.store.new_directory("ST_EXP") if experiment_id is None else (
                experiment_id, self.store.root / experiment_id)
            out.mkdir(parents=True, exist_ok=True)
        ctx = file_log(out / "run.log") if out is not None else _null()
        with ctx:
            t_start = time.perf_counter()
            log.info("%s v%s - experiment %s started", SOFTWARE_NAME, __version__, eid or "(not saved)")
            log.info("Configuration '%s' (hash %s), seed %d", cfg["experiment"]["name"], config_hash(cfg)[:12],
                     cfg["experiment"]["seed"])
            seed_global(cfg["experiment"]["seed"])
            prog("Generating ground truth, sensors and degradations", 0.05)
            if scenario is None:
                scenario = generate_scenario(cfg, normalized=True)
            log.info("Scenario generated: K=%d samples, %d sensors, %d degradations, fingerprint %s",
                     scenario.K, scenario.N, len(scenario.fault_meta()), scenario.fingerprint()[:16])
            has_faults = scenario.fault_windows().any() and getattr(scenario, "has_truth", True)
            twin = None
            if nominal_twin and has_faults and not getattr(scenario, "external", False):
                twin = generate_scenario(cfg, include_degradations=False, normalized=True)
            results, metrics, skipped, nominal_res = {}, {}, {}, {}
            meas = scenario.measurements()
            oracle = OracleInfo(scenario.truth.x, [d.true_var for d in scenario.data],
                                [d.true_bias for d in scenario.data])
            n_m = len(cfg["methods"])
            for mi, mspec in enumerate(cfg["methods"]):
                label = mspec.get("label") or mspec["name"]
                prog(f"Running {label}", 0.1 + 0.7 * mi / n_m)
                try:
                    method = create_method(mspec)
                    label = method.label
                    if method.is_oracle and not getattr(scenario, "has_truth", True):
                        raise MethodNotApplicableError("oracle methods require ground truth")
                    res = method.run(meas, scenario.model, oracle if method.is_oracle else None)
                    nom_rmse = None
                    if twin is not None:
                        rn = method.run(twin.measurements(), twin.model,
                                        OracleInfo(twin.truth.x, [d.true_var for d in twin.data],
                                                   [d.true_bias for d in twin.data]) if method.is_oracle else None)
                        nominal_res[label] = rn
                        mn = evaluate(twin, rn, cfg["metrics"])
                        nom_rmse = (mn.get("estimation") or {}).get("rmse")
                    results[label] = res
                    metrics[label] = evaluate(scenario, res, cfg["metrics"], nominal_rmse=nom_rmse)
                    est = metrics[label].get("estimation") or {}
                    log.info("Method %-40s RMSE=%s runtime=%.2f s", label,
                             f"{est['rmse']:.5g}" if est.get("rmse") is not None else "n/a", res.runtime_s)
                    if res.diverged:
                        log.warning("Method %s diverged: %s", label, "; ".join(res.notes))
                except MethodNotApplicableError as exc:
                    skipped[label] = str(exc)
                    log.warning("Method %s not applicable: %s", label, exc)
                except ConfigurationError:
                    raise
                except Exception as exc:  # keep the other methods running, but record the failure
                    skipped[label] = f"error: {exc}"
                    log.error("Method %s failed: %s\n%s", label, exc, traceback.format_exc())
            if not results:
                raise SensorTrustError("No fusion method could be executed: " +
                                       "; ".join(f"{k}: {v}" for k, v in skipped.items()))
            table = pd.DataFrame([flatten(m) for m in metrics.values()])
            manifest = self._manifest(eid, cfg, scenario, metrics, skipped, time.perf_counter() - t_start)
            result = ExperimentResult(eid, cfg, scenario, results, metrics, skipped, table, manifest, out,
                                      nominal_res)
            if out is not None:
                prog("Saving results", 0.85)
                self.save(result, figures=figures, progress=prog)
            prog("Finished", 1.0)
            log.info("Experiment %s finished in %.2f s", eid or "(not saved)", time.perf_counter() - t_start)
        return result

    # ------------------------------------------------------------------
    def _manifest(self, eid, cfg, scenario, metrics, skipped, elapsed) -> dict:
        model = scenario.model
        trust_methods = sorted({str((m.get("trust") or {}).get("method", "first_two_moments"))
                                for m in cfg["methods"] if "sensortrust" in m["name"] or m.get("trust")})
        summary = {}
        for label, m in metrics.items():
            est = m.get("estimation") or {}
            ds = m.get("detection_summary") or {}
            summary[label] = {"RMSE": est.get("rmse"), "MAE": est.get("mae"),
                              "detection_delay_s": ds.get("mean_detection_delay"),
                              "success": (m.get("robustness") or {}).get("success")}
        return to_builtin({
            "experiment_id": eid,
            "software": SOFTWARE_NAME,
            "version": __version__,
            "subtitle": SUBTITLE_EN,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "experiment_name": cfg["experiment"]["name"],
            "description": cfg["experiment"].get("description", ""),
            "model": model.label,
            "model_key": model.key,
            "duration_s": cfg["simulation"]["duration"],
            "sampling_hz": cfg["simulation"]["fs"],
            "n_sensors": scenario.N,
            "sensors": [f"{s.name}: measures {s.measured}, rate {s.rate:g} Hz, noise {s.noise.type_name} "
                        f"(std {', '.join(f'{v:g}' for v in s.noise.nominal_std)})" for s in scenario.sensors],
            "faults": [_fault_text(mt) for mt in scenario.fault_meta()] or ["none (nominal)"],
            "fusion_methods": list(metrics),
            "skipped_methods": skipped,
            "trust_estimators": trust_methods,
            "seed": cfg["experiment"]["seed"],
            "config_sha256": config_hash(cfg),
            "dataset_sha256": scenario.fingerprint(),
            "metrics_summary": summary,
            "elapsed_s": elapsed,
            "environment": _environment(),
            "hardware_required": "none (virtual sensors, computational simulation only)",
        })

    # ------------------------------------------------------------------
    def save(self, result: ExperimentResult, figures: bool = True, progress: Progress | None = None) -> Path:
        out = result.output_dir
        assert out is not None
        sc = result.scenario
        save_config(result.config, out / "config.yaml")
        if result.config.get("output", {}).get("save_dataset", True):
            export_dataset(sc, out)
        t = sc.t
        # estimates
        cols = {"time": t}
        for j, nm in enumerate(sc.model.state_names):
            cols[f"true_{nm}"] = sc.truth.x[:, j]
        for label, r in result.results.items():
            sl = slug(label)
            for j in r.estimated_states:
                nm = sc.model.state_names[j]
                cols[f"{sl}__{nm}"] = r.x[:, j]
                if r.P is not None:
                    cols[f"{sl}__std_{nm}"] = np.sqrt(np.clip(r.P[:, j, j], 0, None))
        pd.DataFrame(cols).to_csv(out / "estimates.csv", index=False, float_format="%.8g")
        # trust / weights / alarms / moments
        cols = {"time": t}
        for label, r in result.results.items():
            sl = slug(label)
            for i, s in enumerate(sc.sensors):
                if r.trust is not None:
                    cols[f"{sl}__trust_{s.name}"] = r.trust[:, i]
                if r.weights is not None:
                    cols[f"{sl}__weight_{s.name}"] = r.weights[:, i]
                if r.alarms is not None:
                    cols[f"{sl}__alarm_{s.name}"] = r.alarms[:, i].astype(int)
                if r.moments:
                    for key in ("mean", "second_moment", "variance"):
                        if key in r.moments:
                            cols[f"{sl}__innov_{key}_{s.name}"] = r.moments[key][:, i]
                if r.r_eff is not None:
                    cols[f"{sl}__Reff_{s.name}"] = r.r_eff[:, i]
                if r.r_hat is not None:
                    cols[f"{sl}__Rhat_{s.name}"] = r.r_hat[:, i]
        pd.DataFrame(cols).to_csv(out / "trust.csv", index=False, float_format="%.8g")
        # innovations
        cols = {"time": t}
        for label, r in result.results.items():
            if r.innovation is None:
                continue
            sl = slug(label)
            for i, s in enumerate(sc.sensors):
                for c in range(s.m):
                    suf = "" if s.m == 1 else f"_{c + 1}"
                    cols[f"{sl}__innovation_{s.name}{suf}"] = r.innovation[i][:, c]
                    cols[f"{sl}__innovation_std_{s.name}{suf}"] = r.innovation_std[i][:, c]
                cols[f"{sl}__nis_{s.name}"] = r.nis_sensor[:, i]
            if r.nis is not None:
                cols[f"{sl}__nis_total"] = r.nis
                cols[f"{sl}__nis_dof"] = r.nis_dof
        pd.DataFrame(cols).to_csv(out / "innovations.csv", index=False, float_format="%.8g")
        (out / "metrics.json").write_text(json.dumps(to_builtin(
            {"experiment_id": result.experiment_id, "methods": result.metrics, "skipped": result.skipped}),
            indent=2, default=str), encoding="utf-8")
        result.table.to_csv(out / "metrics.csv", index=False, float_format="%.6g")
        if figures:
            if progress:
                progress("Generating figures", 0.9)
            from ..visualization.report import save_experiment_figures
            try:
                files = save_experiment_figures(result, out / "figures",
                                                formats=result.config["output"].get("figures", ["png"]),
                                                dpi=int(result.config["output"].get("figure_dpi", 150)))
                result.manifest["figures"] = [f.name for f in files]
                log.info("Saved %d figure files", len(files))
            except Exception as exc:
                log.error("Figure generation failed: %s\n%s", exc, traceback.format_exc())
        result.manifest["files"] = sorted(p.name for p in out.iterdir())
        (out / "manifest.json").write_text(json.dumps(to_builtin(result.manifest), indent=2, default=str),
                                           encoding="utf-8")
        log.info("Results exported to %s", out)
        return out

    # ------------------------------------------------------------------
    def run_external(self, config: dict, dataset: ImportedDataset, **kw) -> ExperimentResult:
        """Run fusion methods on an imported dataset (optional capability)."""
        cfg = dict(config)
        cfg.setdefault("simulation", {})
        cfg["simulation"] = {**cfg["simulation"], "fs": float(dataset.fs),
                             "duration": float(dataset.t.size / dataset.fs)}
        cfg = normalize_config(cfg)
        sc = scenario_from_dataset(cfg, dataset)
        kw.setdefault("nominal_twin", False)
        return self.run(cfg, scenario=sc, **kw)


def scenario_from_dataset(cfg: dict, ds: ImportedDataset) -> Scenario:
    model = build_model(cfg)
    x0, P0, xe, Pe = _prior(cfg, model)
    fs = float(cfg["simulation"]["fs"])
    K = ds.t.size
    sensors = [VirtualSensor(s, i, model, fs) for i, s in enumerate(cfg["sensors"])]
    data = []
    for s in sensors:
        if s.name not in ds.z:
            raise ConfigurationError(f"Sensor '{s.name}' of the configuration has no column in the dataset.")
        z = ds.z[s.name]
        if z.shape[1] != s.m:
            raise ConfigurationError(f"Sensor '{s.name}': dataset has {z.shape[1]} columns, model expects {s.m}.")
        nan = np.full((K, s.m), np.nan)
        fault = ds.faults.get(s.name, np.zeros(K, dtype=bool))
        meta = []
        if fault.any():
            idx = np.flatnonzero(fault)
            meta = [{"sensor": s.name, "type": "external_label", "severity": float("nan"), "profile": {},
                     "onset_time": float(ds.t[idx[0]]), "first_active_time": float(ds.t[idx[0]]),
                     "last_active_time": float(ds.t[idx[-1]]), "recovery_time": None, "end_time": None}]
        data.append(SensorData(z=z.copy(), clean=nan.copy(), true_bias=nan.copy(), true_var=nan.copy(),
                               noise=nan.copy(), available=~np.any(np.isnan(z), axis=1), fault_active=fault,
                               fault_masks=[fault] if fault.any() else [], fault_meta=meta))
    if ds.truth is not None:
        if ds.truth.shape[1] != model.n:
            raise ConfigurationError(f"Ground truth has {ds.truth.shape[1]} columns; model has {model.n} states.")
        x = ds.truth
    else:
        x = np.full((K, model.n), np.nan)
    truth = GroundTruth(t=ds.t.copy(), x=x, u=model.inputs(ds.t).reshape(K, model.nu), x0=xe, w=np.zeros((K, model.n)))
    sc = Scenario(config=cfg, seed=int(cfg["experiment"]["seed"]), model=model, truth=truth, sensors=sensors,
                  data=data, x0_prior=xe, P0_prior=Pe, include_degradations=True)
    sc.has_truth = ds.truth is not None  # type: ignore[attr-defined]
    sc.external = True  # type: ignore[attr-defined]
    return sc


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def run_experiment(config: dict, results_root: str | Path = "results", **kw) -> ExperimentResult:
    return ExperimentManager(results_root).run(config, **kw)

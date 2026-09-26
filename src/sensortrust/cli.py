"""Command-line interface.

``python -m sensortrust``                       opens the graphical interface
``python -m sensortrust gui``                   idem
``python -m sensortrust run CONFIG [--seed S]`` runs one experiment
``python -m sensortrust benchmark ID|all``      runs benchmarks (ST-BENCH-01 ...)
``python -m sensortrust montecarlo CONFIG``     Monte Carlo study (section 'montecarlo')
``python -m sensortrust robustness CONFIG``     robustness envelope / map (section 'analysis')
``python -m sensortrust sweep CONFIG``          parametric sweep (section 'analysis')
``python -m sensortrust sensitivity CONFIG``    OAT / Morris / Sobol (section 'analysis')
``python -m sensortrust generate-dataset CONFIG -o DIR``   synthetic dataset only
``python -m sensortrust import-dataset CONFIG DATA [...]`` fusion on external data
``python -m sensortrust verify RESULT_DIR``     re-generates and compares the dataset hash
``python -m sensortrust list``                  models, sensors noise, degradations, methods
``python -m sensortrust export-benchmarks DIR`` writes the benchmark YAML files
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .version import FULL_NAME, SUBTITLE_EN


def _load(path: str) -> dict:
    from .experiments.config import load_config
    return load_config(path)


def _progress(msg: str, frac: float) -> None:
    logging.getLogger("sensortrust.cli").info("[%3.0f%%] %s", 100 * frac, msg)


def cmd_run(a):
    from .experiments.manager import ExperimentManager
    cfg = _load(a.config)
    if a.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = a.seed
    r = ExperimentManager(a.results).run(cfg, figures=not a.no_figures, progress=_progress)
    _print_table(r)
    print(f"\nResults: {r.output_dir}")


def _print_table(r):
    cols = ["method", "rmse", "mae", "rmse_during_fault", "mean_detection_delay", "success"]
    tab = r.table[[c for c in cols if c in r.table]]
    with_pd = tab.to_string(index=False, float_format=lambda v: f"{v:.5g}")
    print(f"\n{r.config['experiment']['name']}  (seed {r.config['experiment']['seed']}, "
          f"dataset {r.scenario.fingerprint()[:12]})")
    print(with_pd)
    if r.skipped:
        print("Skipped:", json.dumps(r.skipped, indent=1))


def cmd_benchmark(a):
    from .experiments.benchmarks import BENCHMARKS, get_benchmark
    from .experiments.manager import ExperimentManager
    ids = list(BENCHMARKS) if a.id.lower() == "all" else [a.id]
    mgr = ExperimentManager(a.results)
    for bid in ids:
        r = mgr.run(get_benchmark(bid), figures=not a.no_figures, progress=_progress)
        _print_table(r)
        print(f"Results: {r.output_dir}")


def cmd_montecarlo(a):
    from .experiments.montecarlo import run_montecarlo
    r = run_montecarlo(_load(a.config), runs=a.runs, n_jobs=a.jobs, results_root=a.results, progress=_progress)
    print(r.summary[r.summary.metric.isin(["rmse", "rmse_during_fault", "mean_detection_delay"])]
          [["method", "metric", "n", "mean", "median", "std", "ci_low", "ci_high"]].to_string(index=False))
    print(r.success_rate.to_string(index=False))
    print(f"\nResults: {r.output_dir}")


def cmd_robustness(a):
    from .experiments.robustness import run_robustness
    r = run_robustness(_load(a.config), n_jobs=a.jobs, results_root=a.results, progress=_progress,
                       include_map=not a.no_map)
    env = r.envelope
    print(f"Tolerable {env['parameter_label']} for {env['metric']} spec {env['spec_value']} ({env['criterion']}):")
    for m, b in env["tolerable"].items():
        print(f"  {m:45s} {b if b is not None else 'none (fails at the smallest value)'}"
              f"{'' if env['limit_reached'][m] else ' (not reached: all tested values pass)'}")
    print(f"\nResults: {r.output_dir}")


def cmd_sweep(a):
    from .experiments.sweep import run_sweep
    r = run_sweep(_load(a.config), n_jobs=a.jobs, results_root=a.results, progress=_progress)
    print(r.aggregate.head(40).to_string(index=False))
    print(f"\nResults: {r.output_dir}")


def cmd_sensitivity(a):
    from .experiments.sensitivity import run_sensitivity
    r = run_sensitivity(_load(a.config), n_jobs=a.jobs, results_root=a.results, progress=_progress)
    print(r.result["indices"].to_string(index=False))
    print(f"\nResults: {r.output_dir}")


def cmd_generate(a):
    from .io.dataset import export_dataset
    from .simulation import generate_scenario
    cfg = _load(a.config)
    sc = generate_scenario(cfg, seed=a.seed)
    files = export_dataset(sc, a.output, a.name)
    print(f"Dataset written: {files['csv']} (fingerprint {sc.fingerprint()})")


def cmd_import(a):
    from .experiments.manager import ExperimentManager
    from .io.dataset import import_dataset
    cfg = _load(a.config)
    sensors = None
    if a.sensor:
        sensors = {}
        for item in a.sensor:
            name, _, cols = item.partition("=")
            sensors[name] = cols.split(",") if "," in cols else (cols or name)
    ds = import_dataset(a.data, time=a.time, sensors=sensors, ground_truth=a.truth)
    r = ExperimentManager(a.results).run_external(cfg, ds, progress=_progress)
    _print_table(r)
    print(f"\nResults: {r.output_dir}")


def cmd_verify(a):
    from .experiments.config import config_hash, load_config, normalize_config
    from .simulation import generate_scenario
    from .utils.errors import SensorTrustError
    d = Path(a.directory)
    if not (d / "manifest.json").is_file() or not (d / "config.yaml").is_file():
        raise SensorTrustError(f"Results directory {d} not found or incomplete (manifest.json and config.yaml "
                               f"are required).")
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if "dataset_sha256" not in man or "config_sha256" not in man:
        raise SensorTrustError(f"The manifest of {d} has no reproducibility fingerprints.")
    cfg = normalize_config(load_config(d / "config.yaml"))
    sc = generate_scenario(cfg, normalized=True)
    ok_cfg = config_hash(cfg) == man["config_sha256"]
    ok_ds = sc.fingerprint() == man["dataset_sha256"]
    print(f"configuration hash : {'OK' if ok_cfg else 'MISMATCH'}")
    print(f"dataset fingerprint: {'OK' if ok_ds else 'MISMATCH'} ({sc.fingerprint()[:16]})")
    sys.exit(0 if (ok_cfg and ok_ds) else 1)


def cmd_list(a):
    from .degradation import PROFILE_TYPES, list_degradations
    from .experiments.benchmarks import list_benchmarks
    from .fusion import list_methods
    from .models import list_models
    from .sensors import NOISE_REGISTRY
    from .trust import list_trust_methods
    print(f"{FULL_NAME} - {SUBTITLE_EN}\n\nModels:")
    for m in list_models():
        print(f"  {m['key']:24s} {m['label']}")
    print("\nNoise models:\n  " + ", ".join(NOISE_REGISTRY))
    print("\nDegradations:")
    for d in list_degradations():
        print(f"  {d['type']:20s} {d['description']}")
    print("\nTemporal profiles:\n  " + ", ".join(PROFILE_TYPES))
    print("\nTrust estimators:")
    for t in list_trust_methods():
        print(f"  {t['key']:22s} {t['label']}")
    print("\nFusion methods:")
    for m in list_methods():
        print(f"  {m['name']:26s} {m['label']}{'  [ORACLE]' if m['oracle'] else ''}")
    print("\nBenchmarks:")
    for b in list_benchmarks():
        print(f"  {b['name']}")


FRIENDLY = {
    "ST-BENCH-01": "nominal", "ST-BENCH-02": "abrupt_bias", "ST-BENCH-03": "progressive_bias",
    "ST-BENCH-04": "variance_degradation", "ST-BENCH-05": "drift", "ST-BENCH-06": "outliers",
    "ST-BENCH-07": "stuck_sensor", "ST-BENCH-08": "dropout", "ST-BENCH-09": "recovery",
    "ST-BENCH-10": "two_simultaneous_faults", "ST-BENCH-11": "bias_plus_noise",
    "ST-BENCH-12": "multiple_degradation_profiles", "ST-BENCH-13": "pendulum_ekf_ukf",
    "ST-BENCH-14": "thermal_drift", "ST-BENCH-15": "oscillator_stuck", "ST-BENCH-16": "dominant_sensor_bias",
}


def cmd_export_benchmarks(a):
    from .experiments.benchmarks import BENCHMARKS, examples, studies
    from .experiments.config import save_config
    out = Path(a.directory)
    for bid, cfg in BENCHMARKS.items():
        p = save_config(cfg, out / f"{bid}.yaml")
        print(p)
    if a.configs:
        cdir = Path(a.configs)
        for bid, cfg in BENCHMARKS.items():
            print(save_config(cfg, cdir / f"{FRIENDLY[bid]}.yaml"))
        for name, cfg in studies().items():
            print(save_config(cfg, cdir / f"{name}.yaml"))
    if a.examples:
        from .io.store import slug
        exdir = Path(a.examples)
        for name, cfg in examples().items():
            if name.startswith("Example"):
                print(save_config(cfg, exdir / f"{slug(name).lower()}.yaml"))


def cmd_gui(a):
    from .gui.app import main as gui_main
    gui_main(lang=a.lang)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sensortrust", description=f"{FULL_NAME} - {SUBTITLE_EN}")
    p.add_argument("--version", action="version", version=FULL_NAME)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd")

    def common(sp, jobs=False):
        sp.add_argument("--results", default="results", help="results root directory (default: results)")
        if jobs:
            sp.add_argument("--jobs", type=int, default=None, help="worker processes (0 = all cores)")

    s = sub.add_parser("gui", help="open the graphical interface")
    s.add_argument("--lang", choices=["en", "es"], default=None)
    s.set_defaults(fn=cmd_gui)
    s = sub.add_parser("run", help="run one experiment from a YAML/JSON configuration")
    s.add_argument("config")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--no-figures", action="store_true")
    common(s)
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("benchmark", help="run a benchmark (e.g. ST-BENCH-02) or 'all'")
    s.add_argument("id")
    s.add_argument("--no-figures", action="store_true")
    common(s)
    s.set_defaults(fn=cmd_benchmark)
    s = sub.add_parser("montecarlo", help="Monte Carlo study")
    s.add_argument("config")
    s.add_argument("--runs", type=int, default=None)
    common(s, True)
    s.set_defaults(fn=cmd_montecarlo)
    s = sub.add_parser("robustness", help="robustness envelope and map")
    s.add_argument("config")
    s.add_argument("--no-map", action="store_true")
    common(s, True)
    s.set_defaults(fn=cmd_robustness)
    s = sub.add_parser("sweep", help="parametric sweep")
    s.add_argument("config")
    common(s, True)
    s.set_defaults(fn=cmd_sweep)
    s = sub.add_parser("sensitivity", help="sensitivity analysis (OAT / Morris / Sobol)")
    s.add_argument("config")
    common(s, True)
    s.set_defaults(fn=cmd_sensitivity)
    s = sub.add_parser("generate-dataset", help="generate and export a synthetic dataset")
    s.add_argument("config")
    s.add_argument("-o", "--output", default="datasets")
    s.add_argument("--name", default="dataset")
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(fn=cmd_generate)
    s = sub.add_parser("import-dataset", help="run fusion methods on an external CSV/TXT/JSON dataset")
    s.add_argument("config", help="configuration (model, sensors with nominal noise, methods)")
    s.add_argument("data")
    s.add_argument("--time", default="time")
    s.add_argument("--sensor", action="append", help="NAME=COLUMN[,COLUMN...] (repeatable)")
    s.add_argument("--truth", nargs="*", default=None, help="ground-truth columns, one per state (optional)")
    common(s)
    s.set_defaults(fn=cmd_import)
    s = sub.add_parser("verify", help="verify the reproducibility of a stored experiment")
    s.add_argument("directory")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("list", help="list models, noise, degradations, trust and fusion methods, benchmarks")
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("export-benchmarks", help="write benchmark YAML files")
    s.add_argument("directory", nargs="?", default="benchmarks")
    s.add_argument("--examples", default=None, help="also write the 12 example configurations to this directory")
    s.add_argument("--configs", default=None, help="also write named benchmark and study configurations here")
    s.set_defaults(fn=cmd_export_benchmarks)
    return p


def main(argv=None) -> None:
    from .utils.errors import SensorTrustError
    from .utils.logging import configure_logging
    parser = build_parser()
    a = parser.parse_args(argv)
    configure_logging(logging.DEBUG if a.verbose else logging.INFO)
    if a.cmd is None:
        cmd_gui(argparse.Namespace(lang=None))
        return
    log = logging.getLogger("sensortrust.cli")
    try:
        a.fn(a)
    except SensorTrustError as exc:
        log.error("%s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        log.error("Interrupted by the user.")
        sys.exit(130)
    except (OSError, ValueError) as exc:  # file system problems, invalid values not caught by the validation
        log.error("%s: %s", type(exc).__name__, exc)
        if a.verbose:
            log.exception("details")
        sys.exit(2)


if __name__ == "__main__":
    main()

"""Reproducible benchmark library (ST-BENCH-01 ... ST-BENCH-16).

Every benchmark is a complete experiment configuration.  Given its
configuration and seed, the scenario dataset is reproduced bit-for-bit
(checked by ``python -m sensortrust verify``).  ST-BENCH-01..12 share the same
base: constant-velocity target (``q = 0.05 m^2/s^3``), 60 s at 100 Hz, three
position sensors with nominal standard deviations 0.10, 0.15 and 0.20 m
(ST-BENCH-12 adds a fourth one), seed 42.  ST-BENCH-13..15 exercise the
nonlinear pendulum (EKF/UKF), the thermal system and the oscillator;
ST-BENCH-16 applies a progressive bias to the most precise (dominant) sensor.

The YAML files in ``benchmarks/`` are exported from this module
(``python -m sensortrust export-benchmarks``) and are equivalent.
"""

from __future__ import annotations

import copy

from ..utils.errors import ConfigurationError

BASE_METHODS = [
    {"name": "simple_average"},
    {"name": "inverse_variance"},
    {"name": "kf"},
    {"name": "adaptive_kf"},
    {"name": "kf_nis_gate"},
    {"name": "sensortrust_kf"},
    {"name": "kf_true_r"},
]


def _sensor(name, std, degradations=None, **kw):
    s = {"name": name, "measures": "position", "rate": 100.0, "noise": {"type": "gaussian", "std": std},
         "nominal_reliability": 1.0, "degradations": degradations or []}
    s.update(kw)
    return s


def _base(bid, title, description, sensors, methods=None, seed=42, duration=60.0, fs=100.0):
    return {
        "experiment": {"name": f"{bid} - {title}", "description": description, "seed": seed,
                       "tags": ["benchmark", bid]},
        "model": {"type": "constant_velocity", "params": {"q": 0.05}, "x0": [0.0, 1.0], "P0": [0.01, 0.01],
                  "sample_initial_state": True},
        "simulation": {"duration": duration, "fs": fs},
        "sensors": sensors,
        "estimation": {"x0": [0.0, 1.0], "P0": [1.0, 1.0]},
        "methods": copy.deepcopy(methods or BASE_METHODS),
        "metrics": {"target_state": "position", "warmup": 1.0, "rmse_max": 0.05, "trust_threshold": 0.5},
        "output": {"figures": ["png"], "figure_dpi": 150, "save_dataset": True},
    }


def _three(s2_deg=None, s3_deg=None, s1_deg=None):
    return [_sensor("S1", 0.10, s1_deg), _sensor("S2", 0.15, s2_deg), _sensor("S3", 0.20, s3_deg)]


def _build() -> dict[str, dict]:
    B: dict[str, dict] = {}
    B["ST-BENCH-01"] = _base(
        "ST-BENCH-01", "Nominal", "All three sensors operate nominally (reference case, false-alarm baseline).",
        _three(), methods=BASE_METHODS + [{"name": "single_sensor_kf", "label": "KF (S1 only)", "use_sensors": ["S1"]},
                                          {"name": "single_sensor_kf", "label": "KF (S2 only)", "use_sensors": ["S2"]},
                                          {"name": "single_sensor_kf", "label": "KF (S3 only)", "use_sensors": ["S3"]}])
    B["ST-BENCH-02"] = _base(
        "ST-BENCH-02", "Abrupt bias", "Sensor S2 receives an abrupt bias of 1.0 m at t = 20 s.",
        _three([{"type": "bias", "severity": 1.0, "profile": {"type": "step", "start": 20.0}}]))
    B["ST-BENCH-03"] = _base(
        "ST-BENCH-03", "Progressive bias", "Sensor S2 bias grows linearly at 0.05 m/s from t = 20 s (b = 2 m at 60 s).",
        _three([{"type": "progressive_bias", "severity": 0.05, "b0": 0.0, "profile": {"type": "step", "start": 20.0}}]))
    B["ST-BENCH-04"] = _base(
        "ST-BENCH-04", "Increasing variance",
        "Noise variance of S2 grows linearly from R0 to 50 R0 between t = 20 s and t = 40 s.",
        _three([{"type": "variance_increase", "severity": 50.0,
                 "profile": {"type": "ramp", "start": 20.0, "rise_time": 20.0}}]))
    B["ST-BENCH-05"] = _base(
        "ST-BENCH-05", "Drift",
        "Sensor S2 develops a cumulative drift (rate 0.04 m/s, exponential onset) plus a random-walk drift "
        "(0.05 m/sqrt(s)) from t = 20 s.",
        _three([{"type": "combined", "name": "drift", "profile": {"type": "exponential", "start": 20.0, "rise_time": 10.0},
                 "components": [{"type": "drift", "severity": 0.04}, {"type": "random_walk_drift", "severity": 0.05}]}]))
    B["ST-BENCH-06"] = _base(
        "ST-BENCH-06", "Outliers",
        "Sensor S2 produces impulsive outliers (amplitude 3 m x U(0.5, 1.5), probability 5 %) between 20 and 45 s.",
        _three([{"type": "outliers", "severity": 3.0, "probability": 0.05,
                 "profile": {"type": "step", "start": 20.0, "duration": 25.0}}]))
    B["ST-BENCH-07"] = _base(
        "ST-BENCH-07", "Sensor stuck", "Sensor S2 freezes at its t = 25 s output until the end of the run.",
        _three([{"type": "stuck", "severity": 1.0, "profile": {"type": "step", "start": 25.0}}]))
    B["ST-BENCH-08"] = _base(
        "ST-BENCH-08", "Dropout",
        "S2 loses 60 % of its samples between 20 and 40 s; S3 suffers bursty packet loss (Gilbert-Elliott, "
        "P(good->bad) = 0.02, mean burst 50 samples) from 30 s.",
        _three([{"type": "dropout", "severity": 0.6, "profile": {"type": "step", "start": 20.0, "duration": 20.0}}],
               [{"type": "packet_loss", "severity": 0.02, "burst_length": 50,
                 "profile": {"type": "step", "start": 30.0}}]))
    B["ST-BENCH-09"] = _base(
        "ST-BENCH-09", "Recovery",
        "Sensor S2 receives a 1.5 m bias at t = 20 s that disappears linearly between 35 and 40 s.",
        _three([{"type": "bias", "severity": 1.5,
                 "profile": {"type": "recovery", "start": 20.0, "recovery_time": 35.0, "recovery_duration": 5.0}}]))
    B["ST-BENCH-10"] = _base(
        "ST-BENCH-10", "Two simultaneous faults",
        "At t = 20 s, S2 receives a 1.0 m bias and the noise variance of S3 is multiplied by 30.",
        _three([{"type": "bias", "severity": 1.0, "profile": {"type": "step", "start": 20.0}}],
               [{"type": "variance_increase", "severity": 30.0, "profile": {"type": "step", "start": 20.0}}]))
    B["ST-BENCH-11"] = _base(
        "ST-BENCH-11", "Bias + noise",
        "Combined failure of S2 from t = 20 s: bias ramping to 0.8 m in 5 s and noise variance ramping to 20 R0.",
        _three([{"type": "combined", "name": "bias+noise", "profile": {"type": "ramp", "start": 20.0, "rise_time": 5.0},
                 "components": [{"type": "bias", "severity": 0.8}, {"type": "variance_increase", "severity": 20.0}]}]))
    B["ST-BENCH-12"] = _base(
        "ST-BENCH-12", "Multiple degradation profiles",
        "Four sensors: S1 nominal; S2 sigmoid bias (1.0 m, 8 s transition) from 15 s; S3 periodic bias "
        "(0.8 m, period 8 s) from 25 s; S4 intermittent bias (1.2 m, mean on 1.5 s / off 3 s) from 35 s.",
        [_sensor("S1", 0.10),
         _sensor("S2", 0.15, [{"type": "bias", "severity": 1.0,
                               "profile": {"type": "sigmoid", "start": 15.0, "rise_time": 8.0}}]),
         _sensor("S3", 0.20, [{"type": "periodic", "severity": 0.8, "period": 8.0,
                               "profile": {"type": "step", "start": 25.0}}]),
         _sensor("S4", 0.12, [{"type": "intermittent", "severity": 1.2, "mean_on": 1.5, "mean_off": 3.0,
                               "profile": {"type": "step", "start": 35.0}}])])
    # ----- other models -------------------------------------------------
    pend = {
        "experiment": {"name": "ST-BENCH-13 - Nonlinear pendulum", "seed": 42, "tags": ["benchmark", "ST-BENCH-13"],
                       "description": "Nonlinear pendulum observed by an angle encoder and two horizontal-position "
                                      "sensors h(x) = L sin(theta); sensor P2 receives a 0.15 m bias at 15 s. "
                                      "Justifies EKF / UKF (nonlinear dynamics and measurement)."},
        "model": {"type": "pendulum", "params": {"length": 1.0, "damping": 0.1, "q": 0.01}, "x0": [0.8, 0.0],
                  "P0": [0.001, 0.001]},
        "simulation": {"duration": 40.0, "fs": 100.0},
        "sensors": [
            {"name": "ANG", "function": "angle", "noise": {"type": "gaussian", "std": 0.02}},
            {"name": "P1", "function": "horizontal_position", "noise": {"type": "gaussian", "std": 0.03}},
            {"name": "P2", "function": "horizontal_position", "noise": {"type": "gaussian", "std": 0.03},
             "degradations": [{"type": "bias", "severity": 0.15, "profile": {"type": "step", "start": 15.0}}]},
        ],
        "estimation": {"x0": [0.7, 0.0], "P0": [0.05, 0.05]},
        "methods": [{"name": "ekf"}, {"name": "ukf"}, {"name": "adaptive_ekf"}, {"name": "sensortrust_ekf"},
                    {"name": "sensortrust_ukf"}],
        "metrics": {"target_state": "angle", "warmup": 1.0, "rmse_max": 0.02},
    }
    B["ST-BENCH-13"] = pend
    thermal = {
        "experiment": {"name": "ST-BENCH-14 - Thermal drift", "seed": 42, "tags": ["benchmark", "ST-BENCH-14"],
                       "description": "Two-node thermal system; three object-temperature sensors with 0.1 degC "
                                      "resolution; sensor T2 drifts at 0.02 degC/s from 60 s."},
        "model": {"type": "thermal", "params": {"q": 0.01}, "P0": [0.1, 0.1]},
        "simulation": {"duration": 180.0, "fs": 10.0},
        "sensors": [
            {"name": "T1", "measures": "object_temperature", "noise": 0.10, "resolution": 0.1},
            {"name": "T2", "measures": "object_temperature", "noise": 0.15, "resolution": 0.1,
             "degradations": [{"type": "drift", "severity": 0.02, "profile": {"type": "step", "start": 60.0}}]},
            {"name": "T3", "measures": "object_temperature", "noise": 0.20, "resolution": 0.1},
        ],
        "methods": [{"name": "simple_average"}, {"name": "kf"}, {"name": "adaptive_kf"},
                    {"name": "sensortrust_kf", "trust": {"ewma_lambda": 0.9}}],
        "metrics": {"target_state": "object_temperature", "warmup": 5.0, "rmse_max": 0.1},
    }
    B["ST-BENCH-14"] = thermal
    osc = {
        "experiment": {"name": "ST-BENCH-15 - Oscillator stuck sensor", "seed": 42,
                       "tags": ["benchmark", "ST-BENCH-15"],
                       "description": "Forced mass-spring-damper; position sensors (100 Hz) and a velocity sensor "
                                      "(50 Hz); position sensor X2 gets stuck at 20 s."},
        "model": {"type": "mass_spring_damper", "params": {"mass": 1.0, "stiffness": 4.0, "damping": 0.4, "q": 0.01}},
        "simulation": {"duration": 60.0, "fs": 100.0},
        "sensors": [
            {"name": "X1", "measures": "position", "noise": 0.02},
            {"name": "X2", "measures": "position", "noise": 0.02,
             "degradations": [{"type": "stuck", "severity": 1.0, "profile": {"type": "step", "start": 20.0}}]},
            {"name": "V1", "measures": "velocity", "noise": 0.05, "rate": 50.0},
        ],
        "methods": [{"name": "kf"}, {"name": "adaptive_kf"}, {"name": "sensortrust_kf"}, {"name": "ci_fusion"},
                    {"name": "sensortrust_ci"}],
        "metrics": {"target_state": "position", "warmup": 1.0, "rmse_max": 0.02},
    }
    B["ST-BENCH-15"] = osc
    dom = _base(
        "ST-BENCH-16", "Dominant sensor progressive bias",
        "The most precise sensor S2 (0.05 m, dominating the fused estimate) develops a progressive bias of "
        "0.02 m/s from t = 15 s. Includes the SensorTrust ablation without the consensus reference of the first "
        "moment.",
        [_sensor("S1", 0.10), _sensor("S2", 0.05, [{"type": "progressive_bias", "severity": 0.02,
                                                     "profile": {"type": "step", "start": 15.0}}]),
         _sensor("S3", 0.20)],
        methods=[{"name": "kf"}, {"name": "adaptive_kf"}, {"name": "kf_nis_gate"}, {"name": "sensortrust_kf"},
                 {"name": "sensortrust_kf", "label": "SensorTrust KF (no consensus, ablation)",
                  "trust": {"consensus": False}}])
    B["ST-BENCH-16"] = dom
    return B


BENCHMARKS: dict[str, dict] = _build()


def get_benchmark(bid: str) -> dict:
    key = bid.upper()
    if not key.startswith("ST-BENCH-"):
        key = f"ST-BENCH-{int(bid):02d}"
    try:
        return copy.deepcopy(BENCHMARKS[key])
    except KeyError:
        raise ConfigurationError(f"Unknown benchmark {bid!r}. Available: {', '.join(BENCHMARKS)}.") from None


def list_benchmarks() -> list[dict]:
    return [{"id": k, "name": v["experiment"]["name"], "description": v["experiment"]["description"]}
            for k, v in BENCHMARKS.items()]


# --------------------------------------------------------------------------
# Examples of the graphical interface (12 scientific examples)
# --------------------------------------------------------------------------

def _example_comparison():
    cfg = get_benchmark("ST-BENCH-11")
    cfg["experiment"]["name"] = "Example 10 - Method comparison"
    cfg["experiment"]["description"] = ("All fusion methods on the same Bias + noise scenario (ST-BENCH-11), "
                                        "including the first-/second-moment ablations and estimate-level fusion.")
    cfg["methods"] = [{"name": n} for n in ("simple_average", "inverse_variance", "kf", "adaptive_kf", "kf_nis_gate",
                                            "sensortrust_kf_mean_only", "sensortrust_kf_m2_only", "sensortrust_kf",
                                            "ci_fusion", "sensortrust_ci", "kf_true_r", "best_sensor_oracle")]
    return cfg


def _example_montecarlo():
    cfg = get_benchmark("ST-BENCH-02")
    cfg["experiment"]["name"] = "Example 11 - Monte Carlo abrupt bias"
    cfg["experiment"]["description"] = ("100 Monte Carlo runs of an abrupt bias on S2 with random onset "
                                        "t_f ~ U(10, 30) s, bias b ~ U(0.5, 3) m and S2 noise std ~ U(0.05, 0.5) m.")
    cfg["simulation"]["duration"] = 40.0
    cfg["methods"] = [{"name": n} for n in ("inverse_variance", "kf", "adaptive_kf", "kf_nis_gate", "sensortrust_kf")]
    cfg["montecarlo"] = {
        "runs": 100, "master_seed": 2026, "sampler": "random", "n_jobs": 0, "reference_method": "sensortrust_kf",
        "parameters": [
            {"path": "sensors[1].degradations[0].profile.start", "distribution": "uniform", "low": 10.0,
             "high": 30.0, "label": "fault onset t_f [s]"},
            {"path": "sensors[1].degradations[0].severity", "distribution": "uniform", "low": 0.5, "high": 3.0,
             "label": "bias b [m]"},
            {"path": "sensors[1].noise.std", "distribution": "uniform", "low": 0.05, "high": 0.5,
             "label": "S2 noise std sigma [m]"},
        ],
    }
    return cfg


def _example_robustness():
    cfg = get_benchmark("ST-BENCH-02")
    cfg["experiment"]["name"] = "Example 12 - Robustness envelope (abrupt bias)"
    cfg["experiment"]["description"] = ("Largest tolerable abrupt bias b* on S2 such that the mean RMSE "
                                        "over 20 repetitions stays <= 0.05 m.")
    cfg["simulation"]["duration"] = 40.0
    cfg["methods"] = [{"name": n} for n in ("inverse_variance", "kf", "adaptive_kf", "sensortrust_kf")]
    cfg["analysis"] = {
        "type": "robustness", "metric": "rmse", "spec": 0.05, "criterion": "mean", "repetitions": 20,
        "master_seed": 7, "n_jobs": 0,
        "parameter": {"path": "sensors[1].degradations[0].severity", "label": "bias b [m]",
                      "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]},
        "map": {"path_x": "sensors[1].degradations[0].severity", "label_x": "bias b [m]",
                "values_x": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
                "path_y": "sensors[1].noise.std", "label_y": "S2 noise std sigma [m]",
                "values_y": [0.05, 0.1, 0.2, 0.4, 0.8], "repetitions": 5},
    }
    return cfg


def examples() -> dict[str, dict]:
    ex = {}
    names = [("Example 1 - Nominal fusion", "ST-BENCH-01"), ("Example 2 - Abrupt bias", "ST-BENCH-02"),
             ("Example 3 - Progressive bias", "ST-BENCH-03"), ("Example 4 - Increasing noise", "ST-BENCH-04"),
             ("Example 5 - Drift", "ST-BENCH-05"), ("Example 6 - Outliers", "ST-BENCH-06"),
             ("Example 7 - Stuck sensor", "ST-BENCH-07"), ("Example 8 - Sensor recovery", "ST-BENCH-09"),
             ("Example 9 - Multiple sensor degradation", "ST-BENCH-10")]
    for title, bid in names:
        c = get_benchmark(bid)
        c["experiment"]["name"] = f"{title} ({bid})"
        ex[title] = c
    ex["Example 10 - Method comparison"] = _example_comparison()
    ex["Example 11 - Monte Carlo"] = _example_montecarlo()
    ex["Example 12 - Robustness envelope"] = _example_robustness()
    for bid in ("ST-BENCH-08", "ST-BENCH-11", "ST-BENCH-12", "ST-BENCH-13", "ST-BENCH-14", "ST-BENCH-15",
                "ST-BENCH-16"):
        c = get_benchmark(bid)
        ex[c["experiment"]["name"]] = c
    return ex


# --------------------------------------------------------------------------
# Named multi-run studies (Monte Carlo, robustness, sweep, sensitivity)
# --------------------------------------------------------------------------

def studies() -> dict[str, dict]:
    """Named multi-run study configurations (Monte Carlo, robustness, sweep, sensitivity)."""
    out: dict[str, dict] = {}
    out["montecarlo_abrupt_bias"] = _example_montecarlo()
    out["robustness_abrupt_bias"] = _example_robustness()
    abl = get_benchmark("ST-BENCH-11")
    abl["experiment"]["name"] = "Monte Carlo ablation - first vs second moment"
    abl["experiment"]["description"] = ("100 runs of the bias + noise failure of S2 with random bias "
                                        "b ~ U(0, 1.5) m and variance factor F ~ U(1, 40): contribution of each "
                                        "moment of the innovation.")
    abl["simulation"]["duration"] = 40.0
    abl["methods"] = [{"name": n} for n in ("kf", "adaptive_kf", "sensortrust_kf_mean_only",
                                            "sensortrust_kf_m2_only", "sensortrust_kf")]
    abl["montecarlo"] = {"runs": 100, "master_seed": 11, "sampler": "lhs", "n_jobs": 0,
                         "reference_method": "sensortrust_kf", "parameters": [
                             {"path": "sensors[1].degradations[0].components[0].severity", "distribution": "uniform",
                              "low": 0.0, "high": 1.5, "label": "bias b [m]"},
                             {"path": "sensors[1].degradations[0].components[1].severity", "distribution": "uniform",
                              "low": 1.0, "high": 40.0, "label": "variance factor F [-]"}]}
    out["montecarlo_ablation_moments"] = abl
    sw = get_benchmark("ST-BENCH-02")
    sw["experiment"]["name"] = "Parametric sweep - RMSE(bias, sigma)"
    sw["experiment"]["description"] = "Maps of RMSE and detection delay over the bias of S2 and its noise std."
    sw["simulation"]["duration"] = 40.0
    sw["methods"] = [{"name": n} for n in ("kf", "adaptive_kf", "sensortrust_kf")]
    sw["analysis"] = {"type": "sweep", "repetitions": 3, "master_seed": 5, "n_jobs": 0, "spec": 0.05,
                      "metrics": ["rmse", "mean_detection_delay"], "parameters": [
                          {"path": "sensors[1].degradations[0].severity", "label": "bias b [m]",
                           "values": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]},
                          {"path": "sensors[1].noise.std", "label": "S2 noise std sigma [m]",
                           "values": [0.05, 0.1, 0.2, 0.4, 0.8]}]}
    out["sweep_bias_sigma"] = sw
    sens_params = [
        {"path": "sensors[1].degradations[0].severity", "low": 0.1, "high": 3.0, "label": "bias b [m]"},
        {"path": "sensors[1].noise.std", "low": 0.05, "high": 0.4, "label": "S2 noise std [m]"},
        {"path": "sensors[1].degradations[0].profile.start", "low": 5.0, "high": 25.0, "label": "fault onset [s]"},
        {"path": "methods[0].trust.ewma_lambda", "low": 0.9, "high": 0.995, "label": "EWMA lambda"},
        {"path": "methods[0].trust.mean_deadzone", "low": 1.0, "high": 5.0, "label": "dead zone gamma1"},
        {"path": "methods[0].trust.mean_scale", "low": 0.1, "high": 2.0, "label": "trust scale beta1"},
    ]
    for kind, n in (("morris", 20), ("sobol", 256)):
        c = get_benchmark("ST-BENCH-02")
        c["experiment"]["name"] = f"Sensitivity ({kind}) - SensorTrust KF RMSE, abrupt bias"
        c["experiment"]["description"] = ("Which parameters (fault severity, noise, onset, estimator settings) "
                                          "explain the RMSE of the SensorTrust KF under an abrupt bias?")
        c["simulation"]["duration"] = 40.0
        c["methods"] = [{"name": "sensortrust_kf", "trust": {"ewma_lambda": 0.98, "mean_deadzone": 3.0,
                                                               "mean_scale": 0.5}}]
        c["analysis"] = {"type": "sensitivity", "method": kind, "samples": n, "seed_mode": "common",
                         "master_seed": 42, "n_jobs": 0,
                         "output": {"method": 0, "metric": "rmse", "transform": "log10"},
                         "parameters": sens_params}
        out[f"sensitivity_{kind}"] = c
    return out

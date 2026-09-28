"""Scientific figures.

Every function draws into a :class:`matplotlib.figure.Figure` (created if not
given) and returns it; no global pyplot state is used, so the same functions
serve the GUI canvas, batch export and scripts.  Texts follow the active
language of :mod:`sensortrust.i18n`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..i18n import tr
from .style import FAULT_FACE, INK, INK_2, MUTED, SEQ_CMAP, apply_style, series_style

apply_style()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _fig(fig: Figure | None, nrows: int = 1, ncols: int = 1, size=(9, 4.8), sharex=True):
    if fig is None:
        fig = Figure(figsize=size, layout="constrained")
    else:
        fig.clear()
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            pass
    axes = fig.subplots(nrows, ncols, sharex=sharex, squeeze=False)
    return fig, axes


def _markevery(K: int) -> int:
    return max(1, K // 12)


def _shade_faults(ax, t, mask, label=True):
    """Hatched neutral band on intervals where ``mask`` is true."""
    if mask is None or not np.any(mask):
        return
    m = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    first = True
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[min(b, t.size - 1)], facecolor=FAULT_FACE, alpha=0.45, hatch="//",
                   edgecolor=MUTED, linewidth=0, label=tr("Fault period") if (label and first) else None, zorder=0)
        first = False


def _line(ax, t, y, i, label, lw=1.6, markers=True, **kw):
    st = series_style(i)
    ax.plot(t, y, color=st["color"], linestyle=st["linestyle"], marker=st["marker"] if markers else None,
            markevery=_markevery(len(t)), markersize=5, lw=lw, label=label, **kw)


def _state_label(sc, j):
    from ..labels import state_label
    return f"{state_label(sc.model.state_names[j])} [{sc.model.state_units[j]}]"


def _sens_label(res):
    parts = res.get("output_parts")
    if not parts:
        return res["output_label"]
    return f"{'log10 ' if parts['log10'] else ''}{_mlabel(parts['metric'])} ({tr(parts['method'])})"


def _mlabel(metric):
    from ..labels import metric_label
    return metric_label(metric)


def _target(result):
    from ..metrics import target_index
    return target_index(result.scenario, result.config["metrics"])


def _methods(result, methods):
    labels = list(result.results) if methods is None else [m for m in methods if m in result.results]
    return labels


def _first_with(result, attr, method=None, key=None):
    """Method that has the diagnostic ``attr`` (and, for dictionaries, the entry ``key``)."""
    def has(r):
        v = getattr(r, attr)
        return v is not None and (key is None or key in v)
    if method is not None:
        return method if has(result.results[method]) else None
    for label, r in result.results.items():
        if has(r) and "SensorTrust" in label:
            return label
    for label, r in result.results.items():
        if has(r):
            return label
    return None


def _not_available(ax, title):
    """Empty panel with an explanation when the selected method does not produce this diagnostic."""
    ax.set_title(f"{title} - n/a")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, tr("Not available for the selected method (it does not compute this quantity)."),
            ha="center", va="center", transform=ax.transAxes, color=INK_2, wrap=True)


def _sensor_fault(sc, i):
    return sc.data[i].fault_active


def _legend(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    if len(h) >= 2:
        ax.legend(loc=kw.pop("loc", "best"), **kw)


# ----------------------------------------------------------------------------
# single-experiment figures
# ----------------------------------------------------------------------------

def plot_truth_vs_measurements(result, fig=None, sensors=None):
    sc = result.scenario
    j = _target(result)
    fig, ax = _fig(fig)
    ax = ax[0, 0]
    t = sc.t
    _shade_faults(ax, t, sc.fault_windows())
    for i, (s, d) in enumerate(zip(sc.sensors, sc.data)):
        if sensors is not None and s.name not in sensors:
            continue
        if s.target_index != j and s.m == 1 and s.target_index is not None:
            continue
        st = series_style(i)
        z = d.z[:, 0]
        ok = np.isfinite(z)
        ax.plot(t[ok], z[ok], linestyle="none", marker=st["marker"], markersize=2.2, color=st["color"],
                alpha=0.55, label=f"{s.name}")
    if getattr(sc, "has_truth", True):
        ax.plot(t, sc.truth.x[:, j], color=INK, lw=2.0, label=tr("Ground truth"))
    ax.set_xlabel(tr("Time [s]"))
    ax.set_ylabel(_state_label(sc, j))
    ax.set_title(tr("Ground truth vs measurements"))
    _legend(ax, markerscale=3)
    return fig


def plot_truth_vs_estimates(result, fig=None, methods=None):
    sc = result.scenario
    j = _target(result)
    fig, ax = _fig(fig)
    ax = ax[0, 0]
    t = sc.t
    _shade_faults(ax, t, sc.fault_windows())
    if getattr(sc, "has_truth", True):
        ax.plot(t, sc.truth.x[:, j], color=INK, lw=2.2, label=tr("Ground truth"))
    for i, label in enumerate(_methods(result, methods)):
        r = result.results[label]
        if j in r.estimated_states:
            _line(ax, t, r.x[:, j], i, tr(label) + (f" ({tr('oracle')})" if r.is_oracle and "oracle" not in label else ""))
    ax.set_xlabel(tr("Time [s]"))
    ax.set_ylabel(_state_label(sc, j))
    ax.set_title(tr("Ground truth vs fused estimates"))
    _legend(ax)
    return fig


def plot_estimation_error(result, fig=None, methods=None):
    sc = result.scenario
    j = _target(result)
    fig, ax = _fig(fig)
    ax = ax[0, 0]
    t = sc.t
    _shade_faults(ax, t, sc.fault_windows())
    labels = _methods(result, methods)
    for i, label in enumerate(labels):
        r = result.results[label]
        if j not in r.estimated_states:
            continue
        e = r.x[:, j] - sc.truth.x[:, j]
        _line(ax, t, e, i, tr(label))
        if r.P is not None and len(labels) == 1:
            s2 = 2 * np.sqrt(np.clip(r.P[:, j, j], 0, None))
            ax.fill_between(t, -s2, s2, color=series_style(i)["color"], alpha=0.12, label=tr("±2 std (P)"))
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel(tr("Time [s]"))
    ax.set_ylabel(f"{tr('Estimation error')} [{sc.model.state_units[j]}]")
    ax.set_title(tr("Estimation error"))
    _legend(ax)
    return fig


def _per_sensor_axes(fig, sc, sensors):
    idx = [i for i, s in enumerate(sc.sensors) if sensors is None or s.name in sensors]
    fig, axes = _fig(fig, len(idx), 1, size=(9, 1.9 * len(idx) + 1.2))
    return fig, [axes[r, 0] for r in range(len(idx))], idx


def plot_innovation(result, fig=None, method=None, sensors=None):
    sc = result.scenario
    method = _first_with(result, "innovation", method)
    fig, axs, idx = _per_sensor_axes(fig, sc, sensors)
    if method is None:
        _not_available(axs[0], tr("Innovation vs time"))
        return fig
    r = result.results[method]
    t = sc.t
    for ax, i in zip(axs, idx):
        s = sc.sensors[i]
        _shade_faults(ax, t, _sensor_fault(sc, i))
        nu = r.innovation[i][:, 0]
        sd = r.innovation_std[i][:, 0]
        ok = np.isfinite(nu)
        st = series_style(i)
        ax.plot(t[ok], nu[ok], color=st["color"], lw=0.9, linestyle="-", label=f"nu ({s.name})")
        ax.plot(t[ok], 2 * sd[ok], color=INK_2, lw=1.0, linestyle="--", label=tr("±2 std (innovation)"))
        ax.plot(t[ok], -2 * sd[ok], color=INK_2, lw=1.0, linestyle="--")
        ax.set_ylabel(f"{s.name} [{s.unit}]")
        _legend(ax, loc="upper left")
    axs[-1].set_xlabel(tr("Time [s]"))
    axs[0].set_title(f"{tr('Innovation vs time')} - {tr(method)}")
    return fig


def plot_innovation_mean(result, fig=None, method=None):
    sc = result.scenario
    method = _first_with(result, "moments", method, key="mean")
    fig, ax = _fig(fig)
    ax = ax[0, 0]
    if method is None:
        _not_available(ax, tr("Innovation mean (first moment)"))
        return fig
    r = result.results[method]
    t = sc.t
    _shade_faults(ax, t, sc.fault_windows())
    for i, s in enumerate(sc.sensors):
        _line(ax, t, r.moments["mean"][:, i], i, s.name)
    ne = np.nanmedian(r.moments["n_eff"])
    g1 = float(r.params.get("trust", {}).get("mean_deadzone", 3.0)) if isinstance(r.params.get("trust"), dict) else 3.0
    dz = g1 / np.sqrt(ne)
    ax.axhline(dz, color=INK_2, lw=1.0, linestyle=":", label=f"{tr('dead zone')} ±{dz:.2f}")
    ax.axhline(-dz, color=INK_2, lw=1.0, linestyle=":")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel(tr("Time [s]"))
    ax.set_ylabel(tr("Standardised innovation mean [-]"))
    ax.set_title(f"{tr('Innovation mean (first moment)')} - {tr(method)}")
    _legend(ax)
    return fig


def plot_innovation_second_moment(result, fig=None, method=None):
    sc = result.scenario
    method = _first_with(result, "moments", method, key="second_moment")
    fig, ax = _fig(fig)
    ax = ax[0, 0]
    if method is None:
        _not_available(ax, tr("Innovation second moment and variance"))
        return fig
    r = result.results[method]
    t = sc.t
    _shade_faults(ax, t, sc.fault_windows())
    for i, s in enumerate(sc.sensors):
        st = series_style(i)
        ax.plot(t, r.moments["second_moment"][:, i], color=st["color"], linestyle="-", marker=st["marker"],
                markevery=_markevery(t.size), markersize=5, label=f"{s.name}: {tr('second moment')}")
        ax.plot(t, r.moments["variance"][:, i], color=st["color"], linestyle=":", lw=1.4,
                label=f"{s.name}: {tr('variance')}")
    ax.axhline(1.0, color=INK_2, lw=1.0, linestyle="--", label=tr("nominal (=1)"))
    ax.set_yscale("log")
    ax.set_xlabel(tr("Time [s]"))
    ax.set_ylabel(tr("Second moment / variance [-]"))
    ax.set_title(f"{tr('Innovation second moment and variance')} - {tr(method)}")
    _legend(ax, ncol=2)
    return fig


def plot_trust(result, fig=None, method=None):
    sc = result.scenario
    method = _first_with(result, "trust", method)
    fig, ax = _fig(fig)
    ax = ax[0, 0]
    if method is None:
        _not_available(ax, tr("Sensor trust"))
        return fig
    r = result.results[method]
    t = sc.t
    _shade_faults(ax, t, sc.fault_windows())
    for i, s in enumerate(sc.sensors):
        _line(ax, t, r.trust[:, i], i, s.name)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(tr("Time [s]"))
    ax.set_ylabel(tr("Trust T_i [-]"))
    ax.set_title(f"{tr('Sensor trust')} - {tr(method)}")
    _legend(ax, loc="lower left")
    return fig


def plot_weights(result, fig=None, method=None):
    sc = result.scenario
    method = _first_with(result, "weights", method)
    fig, ax = _fig(fig)
    ax = ax[0, 0]
    if method is None:
        _not_available(ax, tr("Sensor weights"))
        return fig
    r = result.results[method]
    t = sc.t
    _shade_faults(ax, t, sc.fault_windows())
    for i, s in enumerate(sc.sensors):
        w = r.weights[:, i]
        ok = np.isfinite(w)
        st = series_style(i)
        ax.plot(t[ok], w[ok], color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                markevery=_markevery(int(ok.sum())), markersize=5, label=s.name)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(tr("Time [s]"))
    ax.set_ylabel(tr("Weight w_i [-]"))
    ax.set_title(f"{tr('Sensor weights')} - {tr(method)}")
    _legend(ax, loc="upper left")
    return fig


def plot_noise_variance(result, fig=None, sensors=None, method=None):
    sc = result.scenario
    fig, axs, idx = _per_sensor_axes(fig, sc, sensors)
    t = sc.t
    adapt = next((l for l, r in result.results.items() if r.r_hat is not None), None)
    if method is not None and result.results[method].r_eff is not None and method != adapt:
        trustm = method
    else:
        trustm = next((l for l, r in result.results.items() if r.r_eff is not None and r.key.startswith("sensortrust")),
                      None)
    for ax, i in zip(axs, idx):
        s = sc.sensors[i]
        _shade_faults(ax, t, _sensor_fault(sc, i))
        ax.plot(t, sc.data[i].true_var[:, 0], color=INK, lw=2.0, label=tr("true R"))
        ax.axhline(float(s.R_nominal[0, 0]), color=INK_2, lw=1.0, linestyle="--", label=tr("nominal R"))
        if adapt:
            y = result.results[adapt].r_hat[:, i]
            ok = np.isfinite(y)
            ax.plot(t[ok], y[ok], color=series_style(0)["color"], lw=1.3, linestyle="-.",
                    label=tr("estimated R (adaptive KF)"))
        if trustm:
            y = result.results[trustm].r_eff[:, i]
            ok = np.isfinite(y)
            ax.plot(t[ok], y[ok], color=series_style(1)["color"], lw=1.3, linestyle=":",
                    label=f"{tr('effective R')} ({tr(trustm)})")
        ax.set_yscale("log")
        ax.set_ylabel(f"{s.name} [{s.unit}^2]")
        _legend(ax, loc="upper left", fontsize=7)
    axs[-1].set_xlabel(tr("Time [s]"))
    axs[0].set_title(tr("Noise variance vs time"))
    return fig


def plot_fault_timeline(result, fig=None, method=None):
    sc = result.scenario
    method = _first_with(result, "alarms", method)
    fig, ax = _fig(fig, size=(9, 1.0 + 0.7 * sc.N))
    ax = ax[0, 0]
    t = sc.t
    Ts = sc.model.Ts
    yt = []
    for i, (s, d) in enumerate(zip(sc.sensors, sc.data)):
        y = sc.N - 1 - i
        yt.append(y)
        m = d.fault_active
        edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
        spans = [(t[a], (b - a) * Ts) for a, b in zip(edges[::2], edges[1::2])]
        if spans:
            ax.broken_barh(spans, (y + 0.05, 0.4), facecolors=FAULT_FACE, edgecolors=MUTED, hatch="//",
                           label=tr("fault (ground truth)") if i == 0 or not ax.get_legend_handles_labels()[0] else None)
        if method is not None:
            a = result.results[method].alarms[:, i]
            e2 = np.flatnonzero(np.diff(np.concatenate([[0], a.astype(int), [0]])))
            sp2 = [(t[p], (q - p) * Ts) for p, q in zip(e2[::2], e2[1::2])]
            if sp2:
                handles = ax.get_legend_handles_labels()[1]
                ax.broken_barh(sp2, (y - 0.4, 0.35), facecolors=series_style(0)["color"],
                               label=tr("alarm D_i") if tr("alarm D_i") not in handles else None)
        for meta in d.fault_meta:
            from ..labels import degradation_label
            ax.annotate(degradation_label(meta["type"]), (meta["onset_time"], y + 0.25), fontsize=7, color=INK, va="center",
                        xytext=(4, 0), textcoords="offset points",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.set_yticks(yt)
    ax.set_yticklabels([s.name for s in sc.sensors])
    ax.set_ylim(-0.6, sc.N)
    ax.set_xlim(t[0], t[-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(tr("Time [s]"))
    ax.set_title(tr("Fault timeline") + (f" - {tr(method)}" if method else ""))
    _legend(ax, loc="upper left", fontsize=7)
    return fig


def plot_rmse_comparison(result, fig=None):
    tab = result.table.dropna(subset=["rmse"]).copy()
    fig, ax = _fig(fig, size=(9, 0.5 * max(3, len(tab)) + 1.5))
    ax = ax[0, 0]
    if tab.empty:
        _not_available(ax, tr("RMSE comparison"))
        return fig
    tab = tab.sort_values("rmse", ascending=False)
    y = np.arange(len(tab))
    has_fault = tab["rmse_during_fault"].notna().any()
    h = 0.36 if has_fault else 0.6
    ax.barh(y + (h / 2 if has_fault else 0), tab["rmse"], height=h, color=series_style(0)["color"],
            label=tr("RMSE (whole run)"))
    if has_fault:
        ax.barh(y - h / 2, tab["rmse_during_fault"], height=h, color=series_style(1)["color"], hatch="//",
                edgecolor="white", label=tr("RMSE during fault"))
    for yy, v in zip(y, tab["rmse"]):
        ax.annotate(f"{v:.3g}", (v, yy + (h / 2 if has_fault else 0)), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    names = [tr(m) + (f" ({tr('oracle')})" if o and "oracle" not in m.lower() else "") for m, o in
             zip(tab["method"], tab["oracle"])]
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    unit = next(iter(result.metrics.values()))["target_unit"]
    ax.set_xlabel(f"RMSE [{unit}] (log)")
    ax.set_title(tr("RMSE comparison"))
    _legend(ax, loc="upper right")
    return fig


def plot_detection_delay(result, fig=None):
    rows = []
    for label, m in result.metrics.items():
        for sname, sm in (m.get("sensors") or {}).items():
            det = sm.get("detection")
            if det is None or not sm["degraded"]:
                continue
            for k, dly in enumerate(det["detection_delays"]):
                rows.append((label, f"{sname}#{k + 1}" if len(det["detection_delays"]) > 1 else sname, dly))
    fig, ax = _fig(fig, size=(9, 4.2))
    ax = ax[0, 0]
    if not rows:
        _not_available(ax, tr("Detection delay"))
        ax.text(0.5, 0.5, "No method with explicit fault detection or no fault in this scenario",
                ha="center", va="center", transform=ax.transAxes, color=INK_2)
        return fig
    df = pd.DataFrame(rows, columns=["method", "fault", "delay"])
    methods = list(dict.fromkeys(df["method"]))
    faults = list(dict.fromkeys(df["fault"]))
    w = 0.8 / len(methods)
    x = np.arange(len(faults))
    ymax = np.nanmax(df["delay"].astype(float).values) if df["delay"].notna().any() else 1.0
    for i, m in enumerate(methods):
        st = series_style(i)
        for jx, f in enumerate(faults):
            v = df[(df.method == m) & (df.fault == f)]["delay"]
            if v.empty:
                continue
            v = v.iloc[0]
            xx = x[jx] - 0.4 + w * (i + 0.5)
            if v is None or not np.isfinite(v):
                ax.annotate(tr("not detected"), (xx, 0), rotation=90, fontsize=7, ha="center", va="bottom",
                            color=INK_2)
            else:
                ax.bar(xx, v, width=w * 0.9, color=st["color"], hatch=["", "//", "..", "xx"][i % 4],
                       edgecolor="white", label=tr(m) if jx == 0 else None)
                ax.annotate(f"{v:.2f} s", (xx, v), xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(faults)
    ax.set_ylim(0, ymax * 1.25 + 1e-9)
    ax.set_ylabel(tr("Detection delay [s]"))
    ax.set_title(tr("Detection delay"))
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(loc="upper left")
    return fig


# ----------------------------------------------------------------------------
# multi-run figures
# ----------------------------------------------------------------------------

def plot_mc_boxplot(df: pd.DataFrame, metric: str = "rmse", fig=None, unit: str = "", log: bool = True):
    fig, ax = _fig(fig, size=(9, 4.8))
    ax = ax[0, 0]
    methods = list(dict.fromkeys(df["method"]))
    data = [pd.to_numeric(df[df.method == m][metric], errors="coerce").dropna().values for m in methods]
    kw = {"patch_artist": True, "widths": 0.5, "showfliers": True,
          "flierprops": {"marker": ".", "markersize": 3, "markeredgecolor": MUTED}}
    try:
        bp = ax.boxplot(data, orientation="horizontal", **kw)   # matplotlib >= 3.10
    except TypeError:
        bp = ax.boxplot(data, vert=False, **kw)
    for i, (b, med) in enumerate(zip(bp["boxes"], bp["medians"])):
        st = series_style(i)
        b.set_facecolor(st["color"])
        b.set_alpha(0.55)
        b.set_hatch(["", "//", "..", "xx", "\\\\", "++", "--", "oo"][i % 8])
        b.set_edgecolor(INK_2)
        med.set_color(INK)
        med.set_linewidth(2)
    ax.set_yticks(np.arange(1, len(methods) + 1))
    ax.set_yticklabels([tr(m) for m in methods])
    if log and all(np.all(d > 0) for d in data if d.size):
        ax.set_xscale("log")
    ax.set_xlabel(f"{_mlabel(metric)}{f' [{unit}]' if unit else ''}")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{tr('Monte Carlo distribution')}: {_mlabel(metric)} (N = {df['run'].nunique() if 'run' in df else len(df)})")
    return fig


def plot_mc_ecdf(df: pd.DataFrame, metric: str = "rmse", fig=None, unit: str = "", log: bool = True):
    from ..statistics import ecdf
    fig, ax = _fig(fig, size=(9, 4.8))
    ax = ax[0, 0]
    allpos = True
    for i, m in enumerate(dict.fromkeys(df["method"])):
        x, y = ecdf(pd.to_numeric(df[df.method == m][metric], errors="coerce").values)
        if x.size == 0:
            continue
        allpos &= bool(np.all(x > 0))
        st = series_style(i)
        ax.step(x, y, where="post", color=st["color"], linestyle=st["linestyle"], lw=1.8, label=tr(m))
        ax.plot(x[::max(1, x.size // 8)], y[::max(1, x.size // 8)], linestyle="none", marker=st["marker"],
                color=st["color"], markersize=5)
    if log and allpos:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{_mlabel(metric)}{f' [{unit}]' if unit else ''}")
    ax.set_ylabel(tr("Probability"))
    ax.set_title(f"{tr('Empirical CDF')}: {_mlabel(metric)}")
    _legend(ax, loc="lower right")
    return fig


def plot_mc_histogram(df: pd.DataFrame, metric: str = "rmse", fig=None, unit: str = "", bins: int = 30):
    fig, ax = _fig(fig, size=(9, 4.8))
    ax = ax[0, 0]
    vals = pd.to_numeric(df[metric], errors="coerce").dropna()
    if vals.empty:
        return fig
    edges = np.histogram_bin_edges(vals, bins=bins)
    for i, m in enumerate(dict.fromkeys(df["method"])):
        st = series_style(i)
        v = pd.to_numeric(df[df.method == m][metric], errors="coerce").dropna()
        ax.hist(v, bins=edges, histtype="step", lw=1.8, color=st["color"], linestyle=st["linestyle"], label=tr(m))
    ax.set_xlabel(f"{_mlabel(metric)}{f' [{unit}]' if unit else ''}")
    ax.set_ylabel(tr("Count"))
    ax.set_title(f"{tr('Histogram')}: {_mlabel(metric)}")
    _legend(ax)
    return fig


def plot_heatmap(grid: np.ndarray, xvals, yvals, xlabel: str, ylabel: str, title: str, fig=None,
                 cbar_label: str = "", contour_level: float | None = None, fmt: str = "{:.3g}", annotate=True):
    fig, ax = _fig(fig, size=(7.5, 5.5))
    ax = ax[0, 0]
    G = np.asarray(grid, dtype=float)
    im = ax.imshow(G, origin="lower", aspect="auto", cmap=SEQ_CMAP,
                   extent=(-0.5, len(xvals) - 0.5, -0.5, len(yvals) - 0.5))
    ax.set_xticks(range(len(xvals)))
    ax.set_xticklabels([f"{v:.3g}" for v in xvals], rotation=45 if len(xvals) > 8 else 0)
    ax.set_yticks(range(len(yvals)))
    ax.set_yticklabels([f"{v:.3g}" for v in yvals])
    ax.grid(False)
    if annotate and G.size <= 150:
        vmax, vmin = np.nanmax(G), np.nanmin(G)
        mid = (vmax + vmin) / 2
        for yy in range(G.shape[0]):
            for xx in range(G.shape[1]):
                v = G[yy, xx]
                if np.isfinite(v):
                    ax.text(xx, yy, fmt.format(v), ha="center", va="center", fontsize=7,
                            color="white" if v > mid else INK)
    if contour_level is not None and np.nanmin(G) < contour_level < np.nanmax(G):
        cs = ax.contour(np.arange(len(xvals)), np.arange(len(yvals)), G, levels=[contour_level], colors=[INK],
                        linewidths=2, linestyles="--")
        ax.clabel(cs, fmt={contour_level: f"{contour_level:g}"}, fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def plot_robustness_envelope(env: dict, fig=None):
    fig, ax = _fig(fig, size=(9, 4.8))
    ax = ax[0, 0]
    df = env["table"]
    spec = env["spec_value"]
    metric = env["metric"]
    for i, m in enumerate(dict.fromkeys(df["method"])):
        d = df[df.method == m]
        st = series_style(i)
        ax.plot(d["severity"], d[f"{metric}_mean"], color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=6, lw=1.8, label=tr(m))
        if f"{metric}_p95" in d:
            ax.fill_between(d["severity"], d[f"{metric}_p05"], d[f"{metric}_p95"], color=st["color"], alpha=0.10)
        star = env["tolerable"].get(m)
        if star is not None and np.isfinite(star):
            reached = env.get("limit_reached", {}).get(m, True)
            ax.axvline(star, color=st["color"], linestyle=":", lw=1.2)
            short = tr(m).split(" (")[0]
            txt = f"{short}: b* {'=' if reached else '>='} {star:.3g}"
            ax.annotate(txt, (star, spec), rotation=90, fontsize=7, color=INK_2,
                        xytext=(-10 * i - 3, 6), textcoords="offset points", va="bottom", ha="right")
    ax.axhline(spec, color=INK, linestyle="--", lw=1.4, label=f"{tr('Specification')} = {spec:g}")
    ax.set_yscale("log")
    ax.set_xlabel(f"{tr('Severity')}: {env['parameter_label']}")
    ax.set_ylabel(f"{_mlabel(metric)} ({tr('mean, 5-95 %')})")
    ax.set_title(tr("Robustness envelope"))
    _legend(ax, loc="upper left")
    return fig


def plot_oat(res: dict, fig=None):
    fig, ax = _fig(fig, size=(9, 4.8))
    ax = ax[0, 0]
    for i, (p, d) in enumerate(res["curves"].items()):
        st = series_style(i)
        ax.plot(d["relative"], d["output"], color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                lw=1.8, label=p)
    ax.set_xlabel(tr("Parameter value (normalised to its range, 0-1)"))
    ax.set_ylabel(_sens_label(res))
    ax.set_title(tr("One-at-a-time sensitivity"))
    _legend(ax)
    return fig


def plot_morris(res: dict, fig=None):
    fig, ax = _fig(fig, size=(8, 5.5))
    ax = ax[0, 0]
    df = res["indices"]
    for i, row in df.reset_index(drop=True).iterrows():
        st = series_style(i)
        ax.plot(row["mu_star"], row["sigma"], linestyle="none", marker=st["marker"], color=st["color"],
                markersize=9, label=row["parameter"])
        ax.annotate(row["parameter"], (row["mu_star"], row["sigma"]), xytext=(5, 3), textcoords="offset points",
                    fontsize=7.5, color=INK)
    ax.set_xlabel(tr("Mean elementary effect |mu*|"))
    ax.set_ylabel(tr("Std of elementary effects sigma"))
    ax.set_title(f"{tr('Morris screening')}: {_sens_label(res)}")
    _legend(ax, fontsize=7)
    return fig


def plot_sobol(res: dict, fig=None):
    fig, ax = _fig(fig, size=(9, 4.8))
    ax = ax[0, 0]
    df = res["indices"]
    y = np.arange(len(df))
    h = 0.38
    ax.barh(y + h / 2, df["S1"], height=h, xerr=df["S1_conf"], color=series_style(0)["color"],
            label=tr("First-order S1"), error_kw={"ecolor": INK_2, "lw": 1})
    ax.barh(y - h / 2, df["ST"], height=h, xerr=df["ST_conf"], color=series_style(1)["color"], hatch="//",
            edgecolor="white", label=tr("Total-order ST"), error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{tr('Sobol indices')}: {_sens_label(res)} (N = {res['n_evaluations']})")
    _legend(ax, loc="lower right")
    return fig


SINGLE_FIGURES = {
    "truth_vs_measurements": plot_truth_vs_measurements,
    "truth_vs_estimates": plot_truth_vs_estimates,
    "estimation_error": plot_estimation_error,
    "innovation": plot_innovation,
    "innovation_mean": plot_innovation_mean,
    "innovation_second_moment": plot_innovation_second_moment,
    "trust": plot_trust,
    "weights": plot_weights,
    "noise_variance": plot_noise_variance,
    "fault_timeline": plot_fault_timeline,
    "rmse_comparison": plot_rmse_comparison,
    "detection_delay": plot_detection_delay,
}

FIGURE_TITLES = {
    "truth_vs_measurements": "Ground truth vs measurements",
    "truth_vs_estimates": "Ground truth vs fused estimates",
    "estimation_error": "Estimation error",
    "innovation": "Innovation vs time",
    "innovation_mean": "Innovation mean (first moment)",
    "innovation_second_moment": "Innovation second moment and variance",
    "trust": "Sensor trust",
    "weights": "Sensor weights",
    "noise_variance": "Noise variance vs time",
    "fault_timeline": "Fault timeline",
    "rmse_comparison": "RMSE comparison",
    "detection_delay": "Detection delay",
}

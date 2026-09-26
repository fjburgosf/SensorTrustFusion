"""Execution and result tabs: Results, Compare, Monte Carlo, Analysis, Export.

Every "Run" button delegates to the scientific engine
(:class:`ExperimentManager`, :func:`run_montecarlo`, :func:`run_robustness`,
:func:`run_sensitivity`, :func:`run_sweep`) in a background thread.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..fusion.registry import METHOD_PRESETS, create_method
from ..i18n_errors import translate_error
from ..i18n import get_language, tr
from ..visualization import plots
from ..visualization.plots import FIGURE_TITLES, SINGLE_FIGURES
from .config_tabs import scroll, tbutton, tcheck, tgroup, tlabel
from ..labels import metric_label, path_label, value_label
from .widgets import DataTable, LabeledCombo, PlotCanvas

METHOD_FIGS = {"innovation", "innovation_mean", "innovation_second_moment", "trust", "weights", "fault_timeline",
               "noise_variance"}
SENSOR_FIGS = {"truth_vs_measurements", "innovation", "noise_variance"}
TABLE_COLS = ["method", "rmse", "mae", "nrmse", "max_error", "bias", "rmse_pre_fault", "rmse_during_fault",
              "rmse_post_fault", "nis_normalized", "nees_normalized", "mean_detection_delay", "mean_f1",
              "mean_false_alarm_rate", "degraded_sensor_trust_min", "trust_response_time", "trust_recovery_time",
              "performance_loss", "success", "runtime_s"]


def pretty_path(path: str, cfg: dict) -> str:
    """Readable label of a parameter path (e.g. ``sensors[1].noise.std`` -> ``S2: noise std``)."""
    return path_label(path, cfg)


def parameter_paths(cfg: dict) -> list[str]:
    """Suggested parameter paths of a configuration (Monte Carlo / analyses)."""
    out = ["simulation.duration", "model.params.q"]
    for i, s in enumerate(cfg.get("sensors", [])):
        out.append(f"sensors[{i}].noise.std")
        out.append(f"sensors[{i}].bias")
        for j, d in enumerate(s.get("degradations", [])):
            out.append(f"sensors[{i}].degradations[{j}].severity")
            out.append(f"sensors[{i}].degradations[{j}].profile.start")
            if "rise_time" in d.get("profile", {}):
                out.append(f"sensors[{i}].degradations[{j}].profile.rise_time")
    for k, m in enumerate(cfg.get("methods", [])):
        if isinstance(m.get("trust"), dict) and m["trust"].get("method", "first_two_moments") != "nis_gate":
            for key in ("ewma_lambda", "mean_deadzone", "dispersion_deadzone", "mean_scale", "dispersion_scale",
                        "trust_decrease_rate", "trust_recovery_rate"):
                out.append(f"methods[{k}].trust.{key}")
    return out


class ResultsTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.result = None
        split = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        bar = QHBoxLayout()
        self.fig_combo = QComboBox()
        for k in SINGLE_FIGURES:
            self.fig_combo.addItem("", k)
        self.method_combo = QComboBox()
        self.sensor_combo = QComboBox()
        self.l_fig, self.l_m, self.l_s = tlabel("Figure"), tlabel("Method"), tlabel("Sensor filter")
        for w in (self.l_fig, self.fig_combo, self.l_m, self.method_combo, self.l_s, self.sensor_combo):
            bar.addWidget(w)
        self.fig_combo.setMinimumWidth(260)
        self.method_combo.setMinimumWidth(260)
        bar.addStretch(1)
        tl.addLayout(bar)
        self.canvas = PlotCanvas(size=(10, 5))
        tl.addWidget(self.canvas, 1)
        from PySide6.QtWidgets import QLabel
        self.skipped_label = QLabel()
        self.skipped_label.setWordWrap(True)
        self.skipped_label.setStyleSheet("color:#8a4b00")
        self.skipped_label.setVisible(False)
        tl.addWidget(self.skipped_label)
        split.addWidget(top)
        tabs = QTabWidget()
        self.table = DataTable()
        self.sensor_table = DataTable()
        tabs.addTab(self.table, "")
        tabs.addTab(self.sensor_table, "")
        self.subtabs = tabs
        split.addWidget(tabs)
        split.setSizes([560, 230])
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self.fig_combo.currentIndexChanged.connect(self.redraw)
        self.method_combo.currentIndexChanged.connect(self._on_method)
        self.sensor_combo.currentIndexChanged.connect(self.redraw)
        self.canvas.message(tr("No results yet. Configure an experiment and press Run Experiment."))

    def _show_skipped(self):
        res = self.result
        skipped = getattr(res, "skipped", None) or {}
        if not skipped:
            self.skipped_label.setVisible(False)
            return
        items = "; ".join(f"{tr(k)}: {translate_error(v)}" for k, v in skipped.items())
        self.skipped_label.setText(tr("Methods not applicable to this scenario (not executed):") + " " + items)
        self.skipped_label.setVisible(True)

    def set_result(self, res):
        self.result = res
        self._show_skipped()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        pref = 0
        for i, label in enumerate(res.results):
            self.method_combo.addItem(tr(label), label)
            if "SensorTrust" in label and pref == 0:
                pref = i
        self.method_combo.setCurrentIndex(pref)
        self.method_combo.blockSignals(False)
        self.sensor_combo.blockSignals(True)
        self.sensor_combo.clear()
        self.sensor_combo.addItem(tr("All sensors"), None)
        for s in res.scenario.sensors:
            self.sensor_combo.addItem(s.name, s.name)
        self.sensor_combo.blockSignals(False)
        self.table.set_frame(res.table, TABLE_COLS)
        self._on_method()

    def _on_method(self):
        self._sensor_table()
        self.redraw()

    def _sensor_table(self):
        res = self.result
        if res is None:
            return
        label = self.method_combo.currentData()
        m = res.metrics.get(label, {})
        rows = []
        for sname, sm in (m.get("sensors") or {}).items():
            row = {"sensor": sname, "degraded": sm.get("degraded")}
            for sect in ("trust", "weights", "detection"):
                for k, v in (sm.get(sect) or {}).items():
                    if k in ("detection_delays",):
                        continue
                    row[k] = v
            rows.append(row)
        self.sensor_table.set_frame(pd.DataFrame(rows))

    def select_figure(self, key: str, method: str | None = None, sensor: str | None = None):
        i = self.fig_combo.findData(key)
        if i >= 0:
            self.fig_combo.setCurrentIndex(i)
        if method is not None:
            j = self.method_combo.findData(method)
            if j >= 0:
                self.method_combo.setCurrentIndex(j)
        k = self.sensor_combo.findData(sensor)
        if k >= 0:
            self.sensor_combo.setCurrentIndex(k)
        self.redraw()

    def redraw(self):
        if self.result is None:
            return
        key = self.fig_combo.currentData()
        fn = SINGLE_FIGURES[key]
        kw = {}
        self.method_combo.setEnabled(key in METHOD_FIGS)
        self.sensor_combo.setEnabled(key in SENSOR_FIGS)
        if key in METHOD_FIGS:
            kw["method"] = self.method_combo.currentData()
        if key in SENSOR_FIGS and self.sensor_combo.currentData():
            kw["sensors"] = [self.sensor_combo.currentData()]
        self.canvas.draw(fn, self.result, **kw)

    def retranslate(self):
        for i in range(self.fig_combo.count()):
            self.fig_combo.setItemText(i, tr(FIGURE_TITLES[self.fig_combo.itemData(i)]))
        if self.sensor_combo.count():
            self.sensor_combo.setItemText(0, tr("All sensors"))
        self.subtabs.setTabText(0, tr("Metrics table"))
        if self.result is not None:
            self._show_skipped()
        self.subtabs.setTabText(1, tr("Per-sensor metrics"))
        if self.result is not None:
            self.redraw()
        else:
            self.canvas.message(tr("No results yet. Configure an experiment and press Run Experiment."))


class CompareTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        self.head = tlabel("Methods to compare (executed on exactly the same measurements)")
        ll.addWidget(self.head)
        self.list = QListWidget()
        ll.addWidget(self.list, 1)
        self.b_run = tbutton("Run comparison")
        self.b_run.setObjectName("primary")
        ll.addWidget(self.b_run)
        split.addWidget(left)
        right = QWidget()
        rl = QVBoxLayout(right)
        bar = QHBoxLayout()
        self.fig = QComboBox()
        for k in ("rmse_comparison", "truth_vs_estimates", "estimation_error", "detection_delay"):
            self.fig.addItem("", k)
        self.l_fig = tlabel("Figure")
        bar.addWidget(self.l_fig)
        bar.addWidget(self.fig)
        bar.addStretch(1)
        rl.addLayout(bar)
        self.canvas = PlotCanvas(size=(9, 4.5))
        rl.addWidget(self.canvas, 1)
        self.l_tab = tlabel("Comparison table")
        rl.addWidget(self.l_tab)
        self.table = DataTable()
        rl.addWidget(self.table, 1)
        split.addWidget(right)
        split.setSizes([330, 820])
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self.result = None
        self.b_run.clicked.connect(self.run)
        self.fig.currentIndexChanged.connect(self.redraw)

    def populate(self, specs_selected: list[dict]):
        self.list.clear()
        taken = {s["name"] for s in specs_selected if "label" not in s and "use_sensors" not in s}
        rows = list(specs_selected) + [{"name": n} for n in METHOD_PRESETS if n not in taken]
        for r, s in enumerate(rows):
            try:
                label = create_method(s if s["name"] != "fixed_weights" else {**s, "weights": s.get("weights", [1])}).label
            except Exception:
                continue
            it = QListWidgetItem(tr(label))
            it.setData(Qt.UserRole, s)
            it.setData(Qt.UserRole + 1, label)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if r < len(specs_selected) else Qt.Unchecked)
            self.list.addItem(it)

    def checked(self) -> list[dict]:
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(copy.deepcopy(it.data(Qt.UserRole)))
        return out

    def run(self):
        specs = self.checked()
        if not specs:
            QMessageBox.information(self, tr("Compare"), tr("Check at least one method to compare."))
            return
        cfg = self.mw.collect()
        if cfg is None:
            return
        n = len(cfg["sensors"])
        for s in specs:
            if s["name"] == "fixed_weights" and len(s.get("weights", [])) != n:
                s["weights"] = [1.0] * n
        cfg["methods"] = specs
        self.mw.start_experiment(cfg, on_done=self.show_result)

    def show_result(self, res):
        self.result = res
        self.table.set_frame(res.table, TABLE_COLS)
        self.redraw()

    def redraw(self):
        if self.result is not None:
            self.canvas.draw(SINGLE_FIGURES[self.fig.currentData()], self.result)

    def retranslate(self):
        for i in range(self.fig.count()):
            self.fig.setItemText(i, tr(FIGURE_TITLES[self.fig.itemData(i)]))
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole + 1):
                it.setText(tr(it.data(Qt.UserRole + 1)))
        if self.result is not None:
            self.table.set_frame(self.result.table, TABLE_COLS)
        self.redraw()


class ParamTable(QTableWidget):
    """Editable table of varied parameters (path, label, distribution, low, high)."""

    def __init__(self, with_distribution=True):
        super().__init__(0, 5 if with_distribution else 4)
        self.with_dist = with_distribution
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setMinimumSectionSize(70)
        self.verticalHeader().setVisible(False)
        self.paths: list[str] = []

    def set_paths(self, paths, cfg=None):
        self.paths = paths
        self.cfg = cfg

    def add_row(self, path="", label="", dist="uniform", low=0.0, high=1.0):
        r = self.rowCount()
        self.insertRow(r)
        cb = QComboBox()
        cb.setEditable(True)
        for pth in self.paths:
            cb.addItem(path_label(pth, getattr(self, 'cfg', None)), pth)
            cb.setItemData(cb.count() - 1, pth, Qt.ToolTipRole)
        i = cb.findData(path)
        if i >= 0:
            cb.setCurrentIndex(i)
        else:
            cb.setEditText(path)
        cb.lineEdit().setCursorPosition(0)
        cb.setToolTip(tr("Parameter varied in the study (its configuration path is shown when hovering over the "
                         "list). A different path of the configuration can also be typed."))
        cb.setMinimumContentsLength(18)
        self.setCellWidget(r, 0, cb)
        self.setItem(r, 1, QTableWidgetItem(tr(label) if label else label))
        c = 2
        if self.with_dist:
            d = LabeledCombo(["uniform", "loguniform", "normal", "integer"], value_label)
            d.set_value(dist)
            d.setToolTip(tr("Probability distribution of the parameter (for Normal: mean and standard deviation)."))
            self.setCellWidget(r, 2, d)
            c = 3
        for k, v in ((c, low), (c + 1, high)):
            sp = QDoubleSpinBox()
            sp.setRange(-1e9, 1e9)
            sp.setDecimals(5)
            sp.setValue(float(v))
            self.setCellWidget(r, k, sp)

    def rows(self) -> list[dict]:
        out = []
        for r in range(self.rowCount()):
            cb = self.cellWidget(r, 0)
            txt = cb.currentText().strip()
            path = cb.currentData() if cb.currentIndex() >= 0 and txt == cb.itemText(cb.currentIndex()) else txt
            path = path.strip()
            if not path:
                continue
            label = (self.item(r, 1).text() if self.item(r, 1) else "") or path_label(path, getattr(self, "cfg", None))
            if self.with_dist:
                dist = self.cellWidget(r, 2).value()
                a, b = self.cellWidget(r, 3).value(), self.cellWidget(r, 4).value()
                row = {"path": path, "label": label, "distribution": dist}
                if dist == "normal":
                    row.update({"mean": a, "std": b})
                else:
                    row.update({"low": a, "high": b})
            else:
                row = {"path": path, "label": label, "low": self.cellWidget(r, 2).value(),
                       "high": self.cellWidget(r, 3).value()}
            out.append(row)
        return out

    def retranslate(self):
        h = [tr("Parameter"), tr("Label")]
        if self.with_dist:
            h.append(tr("Distribution"))
        h += [tr("Low / mean"), tr("High / std")]
        self.setHorizontalHeaderLabels(h)
        for r in range(self.rowCount()):
            cb = self.cellWidget(r, 0)
            if cb is not None:
                for i in range(cb.count()):
                    cb.setItemText(i, path_label(cb.itemData(i), getattr(self, 'cfg', None)))
            if self.with_dist and isinstance(self.cellWidget(r, 2), LabeledCombo):
                self.cellWidget(r, 2).retranslate()


class MonteCarloTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.result = None
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        g = tgroup("Monte Carlo configuration")
        f = QFormLayout(g)
        self.runs = QSpinBox()
        self.runs.setRange(1, 1_000_000)
        self.runs.setValue(100)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(2026)
        self.sampler = LabeledCombo(["random", "lhs", "sobol"], value_label)
        self.jobs = QSpinBox()
        self.jobs.setRange(0, 256)
        self.ref = QComboBox()
        self.ls = [tlabel("Number of runs"), tlabel("Master seed"), tlabel("Sampling design"),
                   tlabel("Parallel processes (0 = all)"), tlabel("Reference method for paired tests")]
        for lab, w in zip(self.ls, (self.runs, self.seed, self.sampler, self.jobs, self.ref)):
            f.addRow(lab, w)
        ll.addWidget(g)
        pg = tgroup("Randomised parameters")
        pl = QVBoxLayout(pg)
        self.params = ParamTable(True)
        pl.addWidget(self.params)
        bl = QHBoxLayout()
        self.b_add, self.b_del = tbutton("Add parameter"), tbutton("Remove parameter")
        bl.addWidget(self.b_add)
        bl.addWidget(self.b_del)
        pl.addLayout(bl)
        ll.addWidget(pg, 1)
        self.b_run = tbutton("Run Monte Carlo")
        self.b_run.setObjectName("primary")
        ll.addWidget(self.b_run)
        split.addWidget(left)
        right = QWidget()
        rl = QVBoxLayout(right)
        bar = QHBoxLayout()
        self.fig = QComboBox()
        self.fig.addItems(["Boxplot", "ECDF", "Histogram"])
        self.metric = LabeledCombo(["rmse", "rmse_during_fault", "mean_detection_delay", "degraded_sensor_trust_min",
                                    "trust_response_time", "mean_false_alarm_rate", "nis_normalized"], metric_label)
        self.l_fig, self.l_met = tlabel("Figure"), tlabel("Metric")
        for w in (self.l_fig, self.fig, self.l_met, self.metric):
            bar.addWidget(w)
        bar.addStretch(1)
        rl.addLayout(bar)
        self.canvas = PlotCanvas(size=(8, 4.5))
        rl.addWidget(self.canvas, 2)
        self.sub = QTabWidget()
        self.summary = DataTable()
        self.tests = DataTable()
        self.sub.addTab(self.summary, "")
        self.sub.addTab(self.tests, "")
        rl.addWidget(self.sub, 1)
        split.addWidget(right)
        split.setSizes([560, 700])
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self.b_add.clicked.connect(lambda: self.params.add_row(self.params.paths[0] if self.params.paths else ""))
        self.b_del.clicked.connect(lambda: self.params.removeRow(self.params.currentRow())
                                   if self.params.currentRow() >= 0 else None)
        self.b_run.clicked.connect(self.run)
        self.fig.currentIndexChanged.connect(self.redraw)
        self.metric.currentIndexChanged.connect(self.redraw)

    def load(self, cfg):
        self.params.set_paths(parameter_paths(cfg), cfg)
        self.params.setRowCount(0)
        mc = cfg.get("montecarlo") or {}
        self.runs.setValue(int(mc.get("runs", 100)))
        self.seed.setValue(int(mc.get("master_seed", 2026)))
        self.sampler.set_value(mc.get("sampler", "random"))
        self.jobs.setValue(int(mc.get("n_jobs", 0)))
        for p in mc.get("parameters", []):
            if p.get("distribution", "uniform") == "normal":
                self.params.add_row(p["path"], p.get("label", ""), "normal", p["mean"], p["std"])
            else:
                self.params.add_row(p["path"], p.get("label", ""), p.get("distribution", "uniform"), p.get("low", 0),
                                    p.get("high", 1))
        self.ref.clear()
        for m in cfg["methods"]:
            try:
                shown = m.get("label") or create_method(m).label
            except Exception:
                shown = m.get("label") or m["name"]
            self.ref.addItem(tr(shown), m["name"])
            self.ref.setItemData(self.ref.count() - 1, shown, Qt.UserRole + 1)
        i = self.ref.findData(mc.get("reference_method", "sensortrust_kf"))
        self.ref.setCurrentIndex(max(i, 0))

    def store(self, cfg):
        cfg["montecarlo"] = {"runs": self.runs.value(), "master_seed": self.seed.value(),
                             "sampler": self.sampler.value(), "n_jobs": self.jobs.value(),
                             "reference_method": self.ref.currentData() or "sensortrust_kf",
                             "parameters": self.params.rows()}

    def run(self):
        cfg = self.mw.collect()
        if cfg is None:
            return
        from ..experiments.montecarlo import run_montecarlo
        root = self.mw.results_root()
        if root is None:
            return
        self.mw.start_task(run_montecarlo, self.show_result, cfg, results_root=root, cancellable=True)

    def show_result(self, res):
        self.result = res
        self.mw.last_study = res
        s = res.summary
        self.summary.set_frame(s, ["method", "metric", "n", "mean", "median", "std", "min", "max", "p05", "p95",
                                   "ci_low", "ci_high"])
        self.tests.set_frame(res.tests, ["reference", "method", "metric", "n", "mean_diff", "median_diff",
                                         "win_rate_a", "p_value", "p_value_holm", "ci_low", "ci_high"])
        self.redraw()
        self.mw.export_tab.refresh()

    def redraw(self):
        if self.result is None:
            return
        df = self.result.runs
        m = self.metric.value()
        if m not in df or not df[m].notna().any():
            self.canvas.message(f"{metric_label(m)}: " + tr("not available for this study"))
            return
        d = df.dropna(subset=[m])
        fn = {0: plots.plot_mc_boxplot, 1: plots.plot_mc_ecdf, 2: plots.plot_mc_histogram}[self.fig.currentIndex()]
        kw = {"log": m.startswith("rmse")} if self.fig.currentIndex() < 2 else {}
        self.canvas.draw(fn, d, m, **kw)

    def retranslate(self):
        for i, k in enumerate(["Boxplot", "ECDF", "Histogram"]):
            self.fig.setItemText(i, tr(k))
        self.sampler.retranslate()
        self.metric.retranslate()
        for i in range(self.ref.count()):
            self.ref.setItemText(i, tr(self.ref.itemData(i, Qt.UserRole + 1) or self.ref.itemText(i)))
        self.sub.setTabText(0, tr("Statistical summary (mean, median, std, percentiles, 95 % CI)"))
        self.sub.setTabText(1, tr("Paired tests vs reference (Wilcoxon, Holm)"))
        self.params.retranslate()
        self.redraw()


class AnalysisTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.result = None
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        top = QFormLayout()
        self.study = QComboBox()
        self.study.addItem("", "robustness")
        self.study.addItem("", "sensitivity")
        self.study.addItem("", "sweep")
        self.l_study = tlabel("Study")
        top.addRow(self.l_study, self.study)
        ll.addLayout(top)
        # robustness / sweep
        self.rob = tgroup("Robustness envelope")
        rf = QFormLayout(self.rob)
        self.r_path = LabeledCombo([], lambda p: path_label(p, self._cfg))
        self._cfg = None
        self.r_label = QLineEdit()
        self.r_values = QLineEdit("0, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5")
        self.r_metric = LabeledCombo(["rmse", "rmse_during_fault", "max_error"], metric_label)
        self.r_spec = QDoubleSpinBox()
        self.r_spec.setDecimals(5)
        self.r_spec.setRange(0, 1e9)
        self.r_spec.setValue(0.05)
        self.r_crit = LabeledCombo(["mean", "success_rate"], value_label)
        self.r_reps = QSpinBox()
        self.r_reps.setRange(1, 10000)
        self.r_reps.setValue(10)
        self.r_map = tcheck("Also compute 2-D robustness map")
        self.r_path2 = LabeledCombo([], lambda p: path_label(p, self._cfg))
        self.r_values2 = QLineEdit("0.05, 0.1, 0.2, 0.4")
        self.rl = [tlabel("Degradation parameter"), tlabel("Label"), tlabel("Tested values (comma separated)"),
                   tlabel("Metric"), tlabel("Specification (metric <= value)"), tlabel("Criterion"),
                   tlabel("Repetitions per value"), QWidget(), tlabel("Second parameter (map / sweep)"),
                   tlabel("Second parameter values")]
        for lab, w in zip(self.rl, (self.r_path, self.r_label, self.r_values, self.r_metric, self.r_spec, self.r_crit,
                                    self.r_reps, self.r_map, self.r_path2, self.r_values2)):
            rf.addRow(lab, w)
        ll.addWidget(self.rob)
        # sensitivity
        self.sen = tgroup("Sensitivity analysis")
        sf = QFormLayout(self.sen)
        self.s_method = LabeledCombo(["oat", "morris", "sobol"],
                                     lambda v: value_label("sobol_indices" if v == "sobol" else v))
        self.s_samples = QSpinBox()
        self.s_samples.setRange(2, 100000)
        self.s_samples.setValue(8)
        self.s_out = QComboBox()
        self.s_metric = LabeledCombo(["rmse", "rmse_during_fault", "mean_detection_delay", "trust_response_time"],
                                     metric_label)
        self.s_seed = LabeledCombo(["common", "random"], value_label)
        self.s_params = ParamTable(False)
        self.s_add = tbutton("Add parameter")
        self.sl = [tlabel("Sensitivity method"), tlabel("Samples (OAT levels / Morris trajectories / Sobol N)"),
                   tlabel("Output method"), tlabel("Output metric"), tlabel("Seed mode")]
        for lab, w in zip(self.sl, (self.s_method, self.s_samples, self.s_out, self.s_metric, self.s_seed)):
            sf.addRow(lab, w)
        self.l_sp = tlabel("Parameters (path, low, high)")
        sf.addRow(self.l_sp)
        sf.addRow(self.s_params)
        sf.addRow(self.s_add)
        ll.addWidget(self.sen)
        jf = QFormLayout()
        self.jobs = QSpinBox()
        self.jobs.setRange(0, 256)
        self.l_jobs = tlabel("Parallel processes (0 = all)")
        jf.addRow(self.l_jobs, self.jobs)
        ll.addLayout(jf)
        self.b_run = tbutton("Run analysis")
        self.b_run.setObjectName("primary")
        ll.addWidget(self.b_run)
        ll.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rl = QVBoxLayout(right)
        bar = QHBoxLayout()
        self.fig = QComboBox()
        self.l_fig = tlabel("Figure")
        bar.addWidget(self.l_fig)
        bar.addWidget(self.fig, 1)
        rl.addLayout(bar)
        self.canvas = PlotCanvas(size=(8, 5))
        rl.addWidget(self.canvas, 2)
        self.table = DataTable()
        rl.addWidget(self.table, 1)
        split.addWidget(right)
        split.setSizes([560, 700])
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self.study.currentIndexChanged.connect(self._vis)
        self.b_run.clicked.connect(self.run)
        self.s_add.clicked.connect(lambda: self.s_params.add_row(self.s_params.paths[0] if self.s_params.paths else ""))
        self.fig.currentIndexChanged.connect(self.redraw)
        self._vis()

    def _vis(self):
        st = self.study.currentData()
        self.rob.setVisible(st in ("robustness", "sweep"))
        self.sen.setVisible(st == "sensitivity")
        self.r_map.setVisible(st == "robustness")
        self.rl[7].setVisible(st == "robustness")

    def load(self, cfg):
        paths = parameter_paths(cfg)
        self._cfg = cfg
        for cb in (self.r_path, self.r_path2):
            cb.set_values(paths, keep=False)
        sev = next((p for p in paths if p.endswith("].severity")), paths[0])
        noise = next((p for p in paths if "degradations" not in p and p.endswith("noise.std")
                      and sev.split(".")[0] in p), paths[0])
        self.r_path.set_value(sev)
        self.r_path2.set_value(noise)
        self.s_params.set_paths(paths, cfg)
        self.s_out.clear()
        for m in cfg["methods"]:
            self.s_out.addItem(tr(m.get("label") or m["name"]), m.get("label") or m["name"])
        i = self.s_out.findData("sensortrust_kf")
        self.s_out.setCurrentIndex(max(i, 0))
        an = cfg.get("analysis") or {}
        t = an.get("type")
        if t:
            self.study.setCurrentIndex(max(0, self.study.findData(t)))
        if t == "robustness":
            p = an["parameter"]
            self.r_path.set_value(p["path"])
            self.r_label.setText(tr(p.get("label", "")) if p.get("label") else "")
            self.r_values.setText(", ".join(f"{v:g}" for v in p["values"]))
            self.r_metric.set_value(an.get("metric", "rmse"))
            self.r_spec.setValue(float(an["spec"]))
            self.r_crit.set_value(an.get("criterion", "mean"))
            self.r_reps.setValue(int(an.get("repetitions", 10)))
            mp = an.get("map")
            self.r_map.setChecked(bool(mp))
            if mp:
                self.r_path2.set_value(mp["path_y"])
                self.r_values2.setText(", ".join(f"{v:g}" for v in mp["values_y"]))
                self._map_x = mp.get("values_x")
        self.jobs.setValue(int(an.get("n_jobs", 0)))
        if t == "sensitivity":
            self.s_method.set_value(an.get("method", "morris"))
            self.s_samples.setValue(int(an.get("samples", 8)))
            self.s_params.setRowCount(0)
            for p in an.get("parameters", []):
                self.s_params.add_row(p["path"], p.get("label", ""), low=p["low"], high=p["high"])

    @staticmethod
    def _floats(text):
        try:
            return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid list of numbers: {text!r}") from None

    def build(self, cfg) -> dict:
        st = self.study.currentData()
        if st == "robustness":
            vals = self._floats(self.r_values.text())
            an = {"type": "robustness", "metric": self.r_metric.value(), "spec": self.r_spec.value(),
                  "criterion": self.r_crit.value(), "repetitions": self.r_reps.value(),
                  "n_jobs": self.jobs.value(), "master_seed": cfg["experiment"]["seed"],
                  "parameter": {"path": self.r_path.value(), "label": self.r_label.text() or
                                pretty_path(self.r_path.value(), cfg), "values": vals}}
            if self.r_map.isChecked():
                xs = getattr(self, "_map_x", None) or vals[:: max(1, len(vals) // 6)]
                an["map"] = {"path_x": self.r_path.value(), "label_x": an["parameter"]["label"],
                             "values_x": xs, "path_y": self.r_path2.value(),
                             "label_y": pretty_path(self.r_path2.value(), cfg),
                             "values_y": self._floats(self.r_values2.text()),
                             "repetitions": max(1, self.r_reps.value() // 3)}
        elif st == "sweep":
            an = {"type": "sweep", "repetitions": self.r_reps.value(), "n_jobs": self.jobs.value(),
                  "spec": self.r_spec.value(), "metrics": [self.r_metric.value(), "mean_detection_delay"],
                  "parameters": [{"path": self.r_path.value(), "label": self.r_label.text() or
                                  pretty_path(self.r_path.value(), cfg), "values": self._floats(self.r_values.text())},
                                 {"path": self.r_path2.value(), "label": pretty_path(self.r_path2.value(), cfg),
                                  "values": self._floats(self.r_values2.text())}]}
        else:
            an = {"type": "sensitivity", "method": self.s_method.value(), "samples": self.s_samples.value(),
                  "seed_mode": self.s_seed.value(), "n_jobs": self.jobs.value(),
                  "output": {"method": self.s_out.currentData(), "metric": self.s_metric.value()},
                  "parameters": self.s_params.rows()}
        return an

    def run(self):
        cfg = self.mw.collect()
        if cfg is None:
            return
        if self.study.currentData() == "sensitivity" and not self.s_params.rows():
            QMessageBox.information(self, tr("Sensitivity analysis"),
                                    tr("Add at least one parameter (path, minimum, maximum) to the sensitivity analysis."))
            return
        try:
            cfg["analysis"] = self.build(cfg)
        except ValueError as exc:
            QMessageBox.warning(self, tr("Configuration error"), translate_error(exc))
            return
        root = self.mw.results_root()
        if root is None:
            return
        st = cfg["analysis"]["type"]
        if st == "robustness":
            from ..experiments.robustness import run_robustness as fn
        elif st == "sweep":
            from ..experiments.sweep import run_sweep as fn
        else:
            from ..experiments.sensitivity import run_sensitivity as fn
        self.mw.start_task(fn, self.show_result, cfg, results_root=root, cancellable=True)

    def show_result(self, res):
        self.result = res
        self.mw.last_study = res
        self.fig.blockSignals(True)
        self.fig.clear()
        kind = type(res).__name__
        if kind == "RobustnessResult":
            self.fig.addItem(tr("Robustness envelope"), ("envelope", None))
            if res.map:
                for m in res.map["grids"]:
                    self.fig.addItem(f"{tr('Robustness map')} - {tr(m)}", ("map", m))
            tol = pd.DataFrame([{"method": m, "tolerable_severity": b, "limit_reached": res.envelope["limit_reached"][m]}
                                for m, b in res.envelope["tolerable"].items()])
            self.table.set_frame(tol)
        elif kind == "SensitivityResult":
            self.fig.addItem({"oat": tr("One-at-a-time sensitivity"), "morris": tr("Morris screening"),
                              "sobol": tr("Sobol indices")}[res.method], ("sens", None))
            self.table.set_frame(res.result["indices"])
        else:
            for m in dict.fromkeys(res.aggregate["method"]):
                for c in [c for c in res.aggregate.columns if c.endswith("_mean")]:
                    self.fig.addItem(f"{metric_label(c[:-5])} ({tr('mean')}) - {tr(m)}", ("sweep", (m, c)))
            self.table.set_frame(res.aggregate)
        self.fig.blockSignals(False)
        self.redraw()
        self.mw.export_tab.refresh()

    def redraw(self):
        res = self.result
        if res is None or self.fig.currentIndex() < 0:
            return
        kind, arg = self.fig.currentData()
        if kind == "envelope":
            self.canvas.draw(plots.plot_robustness_envelope, res.envelope)
        elif kind == "map":
            mp = res.map
            self.canvas.draw(plots.plot_heatmap, mp["grids"][arg], mp["x"], mp["y"], mp["label_x"], mp["label_y"],
                             f"{tr('Robustness map')}: P({metric_label(mp['metric'])} <= {mp['spec_value']:g}) - {tr(arg)}",
                             cbar_label=tr("Success rate"), contour_level=0.95, fmt="{:.2f}")
        elif kind == "sens":
            fn = {"oat": plots.plot_oat, "morris": plots.plot_morris, "sobol": plots.plot_sobol}[res.method]
            self.canvas.draw(fn, res.result)
        else:
            from ..experiments.sweep import sweep_grid
            m, c = arg
            G, xv, yv = sweep_grid(res, m, c)
            px, py = res.parameters[0], res.parameters[1]
            self.canvas.draw(plots.plot_heatmap, G, xv, yv, px.get("label", px["path"]), py.get("label", py["path"]),
                             f"{metric_label(c[:-5])} ({tr('mean')}) - {tr(m)}", cbar_label=metric_label(c[:-5]))

    def retranslate(self):
        for i, k in enumerate(["Robustness envelope", "Sensitivity analysis", "Parametric sweep (2-D map)"]):
            self.study.setItemText(i, tr(k))
        for cb in (self.r_path, self.r_path2, self.r_metric, self.r_crit, self.s_method, self.s_metric, self.s_seed):
            cb.retranslate()
        for i in range(self.s_out.count()):
            self.s_out.setItemText(i, tr(self.s_out.itemData(i)))
        self.s_params.retranslate()
        self.redraw()


class ExportTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        lay = QVBoxLayout(self)
        g = tgroup("Last experiment")
        gl = QVBoxLayout(g)
        self.path = QLineEdit()
        self.path.setReadOnly(True)
        gl.addWidget(self.path)
        bl = QHBoxLayout()
        self.b_open = tbutton("Open results folder")
        self.b_figs = tbutton("Export all figures")
        self.b_cfg = tbutton("Export configuration")
        self.b_ds = tbutton("Export dataset")
        self.b_tab = tbutton("Export metrics table")
        self.b_ver = tbutton("Verify reproducibility")
        for b in (self.b_open, self.b_figs, self.b_cfg, self.b_ds, self.b_tab, self.b_ver):
            bl.addWidget(b)
        gl.addLayout(bl)
        fl = QHBoxLayout()
        self.f_png, self.f_svg, self.f_pdf = QCheckBox("PNG"), QCheckBox("SVG"), QCheckBox("PDF")
        self.f_png.setChecked(True)
        self.f_svg.setChecked(True)
        self.f_pdf.setChecked(True)
        for c in (self.f_png, self.f_svg, self.f_pdf):
            fl.addWidget(c)
        fl.addStretch(1)
        gl.addLayout(fl)
        lay.addWidget(g)
        mg = tgroup("Reproducibility manifest")
        ml = QVBoxLayout(mg)
        self.manifest = QPlainTextEdit()
        self.manifest.setReadOnly(True)
        self.manifest.setStyleSheet("font-family: monospace; font-size: 9pt")
        ml.addWidget(self.manifest)
        lay.addWidget(mg, 2)
        ig = tgroup("Import external dataset (optional)")
        il = QFormLayout(ig)
        self.i_time = QLineEdit("time")
        self.i_truth = QLineEdit("")
        self.i_btn = tbutton("Import and run...")
        self.il = [tlabel("Time column"), tlabel("Ground-truth columns (optional)")]
        il.addRow(self.il[0], self.i_time)
        il.addRow(self.il[1], self.i_truth)
        il.addRow(self.i_btn)
        lay.addWidget(ig)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        lay.addWidget(self.log)
        self.b_open.clicked.connect(self.open_folder)
        self.b_figs.clicked.connect(self.export_figures)
        self.b_cfg.clicked.connect(self.export_config)
        self.b_ds.clicked.connect(self.export_dataset)
        self.b_tab.clicked.connect(self.export_table)
        self.b_ver.clicked.connect(self.verify)
        self.i_btn.clicked.connect(self.import_dataset)

    def _dir(self) -> Path | None:
        obj = self.mw.last_study or self.mw.last_result
        d = getattr(obj, "output_dir", None)
        return Path(d) if d else None

    def refresh(self):
        d = self._dir()
        self.path.setText(str(d.resolve()) if d else "")
        man = None
        obj = self.mw.last_study or self.mw.last_result
        if obj is not None:
            man = obj.manifest
        self.manifest.setPlainText(json.dumps(man, indent=2, default=str) if man else "")

    def _say(self, text):
        self.log.appendPlainText(text)

    def _result(self):
        """Last experiment result, or an explanatory message if there is none yet."""
        res = self.mw.last_result
        if res is None:
            msg = tr("No experiment has been run yet. Configure an experiment and press Run Experiment.")
            self._say(msg)
            QMessageBox.information(self, tr("Export"), msg)
        return res

    def open_folder(self):
        d = self._dir()
        if d is None:
            if self._result() is None:
                return
            d = self._dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d.resolve())))
        self._say(str(d.resolve()))

    def export_figures(self, target: str | None = None):
        res = self._result()
        if res is None:
            return
        from ..visualization.report import save_experiment_figures
        d = target or QFileDialog.getExistingDirectory(self, tr("Export all figures"), str(res.output_dir or "."))
        if not d:
            return
        fmts = [f for f, c in (("png", self.f_png), ("svg", self.f_svg), ("pdf", self.f_pdf)) if c.isChecked()]
        files = save_experiment_figures(res, Path(d), fmts or ["png"])
        self._say(f"{tr('Files written:')} {len(files)} -> {d}")

    def export_config(self):
        cfg = self.mw.collect(validate=False)
        p, _ = QFileDialog.getSaveFileName(self, tr("Export configuration"), "experiment.yaml",
                                           "YAML (*.yaml);;JSON (*.json)")
        if p:
            from ..experiments.config import save_config
            save_config(cfg, p)
            self._say(f"{tr('Files written:')} {p}")

    def export_dataset(self):
        res = self._result()
        if res is None:
            return
        d = QFileDialog.getExistingDirectory(self, tr("Export dataset"), ".")
        if d:
            from ..io.dataset import export_dataset
            files = export_dataset(res.scenario, d)
            self._say(f"{tr('Files written:')} {files['csv']}, {files['meta']}")

    def export_table(self):
        res = self._result()
        if res is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, tr("Export metrics table"), "metrics.csv", "CSV (*.csv)")
        if p:
            res.table.to_csv(p, index=False)
            self._say(f"{tr('Files written:')} {p}")

    def verify(self):
        res = self._result()
        if res is None:
            return
        from ..experiments.config import config_hash
        from ..simulation import generate_scenario
        man = res.manifest or {}
        if getattr(res.scenario, "external", False) or "dataset_sha256" not in man:
            self._say(tr("Imported datasets cannot be regenerated from a seed; only simulated experiments "
                         "can be verified."))
            return None
        sc = generate_scenario(res.config, normalized=True)
        ok = sc.fingerprint() == man["dataset_sha256"] and config_hash(res.config) == man.get("config_sha256")
        msg = tr("Reproducibility verified: identical configuration hash and dataset fingerprint.") if ok else \
            tr("Reproducibility check FAILED.")
        self._say(f"{msg} ({sc.fingerprint()[:16]})")
        return ok

    def import_dataset(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Import external dataset (optional)"), ".",
                                           "Data (*.csv *.txt *.json)")
        if not p:
            return
        cfg = self.mw.collect()
        if cfg is None:
            return
        from ..io.dataset import import_dataset
        truth = [c.strip() for c in self.i_truth.text().split(",") if c.strip()] or None
        try:
            ds = import_dataset(p, time=self.i_time.text().strip() or "time",
                                sensors={s["name"]: s["name"] for s in cfg["sensors"]}, ground_truth=truth)
        except Exception as exc:
            QMessageBox.warning(self, tr("Error"), translate_error(exc))
            return
        from ..experiments.manager import ExperimentManager
        root = self.mw.results_root()
        if root is None:
            return

        def job(progress=None):
            return ExperimentManager(root).run_external(cfg, ds, progress=progress)
        self.mw.start_task(job, self.mw.on_experiment_done)

    def retranslate(self):
        pass

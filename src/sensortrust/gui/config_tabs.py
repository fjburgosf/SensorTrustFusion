"""Configuration tabs: System, Sensors, Degradation, Fusion, Trust, Experiment.

These tabs only edit the experiment configuration (a plain mapping); they
contain no scientific computation.  The degradation preview calls the real
degradation engine through :class:`~sensortrust.sensors.sensor.VirtualSensor`.
"""

from __future__ import annotations

import copy

import numpy as np
import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QFormLayout,
)

from ..fusion.registry import METHOD_PRESETS, create_method
from ..i18n_errors import translate_error
from ..labels import degradation_label, model_description, state_label
from ..i18n import get_language, tr
from ..models import create_model
from ..utils.errors import SensorTrustError
from . import schema as S
from .widgets import ParamForm, PlotCanvas


def tlabel(key: str) -> QLabel:
    w = QLabel(tr(key))
    w.setProperty("tr_key", key)
    w.setWordWrap(True)
    return w


def tgroup(key: str) -> QGroupBox:
    g = QGroupBox(tr(key))
    g.setProperty("tr_key", key)
    return g


def tbutton(key: str) -> QPushButton:
    b = QPushButton(tr(key))
    b.setProperty("tr_key", key)
    return b


def tcheck(key: str) -> QCheckBox:
    c = QCheckBox(tr(key))
    c.setProperty("tr_key", key)
    return c


def scroll(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(widget)
    return sa


MEAS_FN_LABELS = {
    "horizontal_position": ("horizontal bob position L sin(theta)", "posición horizontal de la masa L sen(theta)"),
    "vertical_position": ("vertical bob position -L cos(theta)", "posición vertical de la masa -L cos(theta)"),
}


def _lab(pair) -> str:
    return pair[1] if get_language() == "es" else pair[0]


def model_for(cfg: dict):
    try:
        return create_model(cfg["model"]["type"], 1.0 / float(cfg["simulation"]["fs"]), cfg["model"].get("params", {}))
    except SensorTrustError:
        return create_model("constant_velocity", 0.01, {})


# ----------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------
class SystemTab(QWidget):
    model_changed = Signal()

    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        inner = QWidget()
        lay = QVBoxLayout(inner)
        g = tgroup("Dynamic model")
        gl = QVBoxLayout(g)
        self.model_combo = QComboBox()
        for key in S.MODEL_LABELS:
            self.model_combo.addItem("", key)
        gl.addWidget(self.model_combo)
        self.model_desc = QLabel()
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet("color:#52514e")
        gl.addWidget(self.model_desc)
        lay.addWidget(g)
        self.params_group = tgroup("Model parameters")
        self.params_lay = QVBoxLayout(self.params_group)
        self.param_form: ParamForm | None = None
        self.custom_label = tlabel("Custom model parameters (YAML: A, B, Q or continuous Ac, Bc, Qc; state_names, state_units, inputs, x0)")
        self.custom_edit = QPlainTextEdit()
        self.custom_edit.setMinimumHeight(140)
        self.params_lay.addWidget(self.custom_label)
        self.params_lay.addWidget(self.custom_edit)
        lay.addWidget(self.params_group)

        h = tgroup("Simulation horizon")
        hf = QFormLayout(h)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.01, 1e6)
        self.duration.setDecimals(3)
        self.duration.setSuffix(" s")
        self.fs = QDoubleSpinBox()
        self.fs.setRange(0.01, 1e5)
        self.fs.setDecimals(3)
        self.fs.setSuffix(" Hz")
        self.l_dur, self.l_fs = tlabel("Duration"), tlabel("Sampling frequency")
        hf.addRow(self.l_dur, self.duration)
        hf.addRow(self.l_fs, self.fs)
        self.ts_label = QLabel()
        hf.addRow("Ts:", self.ts_label)
        lay.addWidget(h)

        ig = tgroup("Initial state and uncertainty")
        il = QVBoxLayout(ig)
        self.x0_table = QTableWidget(0, 6)
        self.x0_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.x0_table.setMinimumHeight(120)
        il.addWidget(self.x0_table)
        self.sample_x0 = tcheck("Sample the true initial state from N(x0, P0)")
        il.addWidget(self.sample_x0)
        self.q_label = tlabel("Process noise covariance Q (discrete, computed from the model)")
        self.q_text = QLabel()
        self.q_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.q_text.setStyleSheet("font-family: monospace; color:#52514e")
        il.addWidget(self.q_label)
        il.addWidget(self.q_text)
        lay.addWidget(ig)

        eg = tgroup("Evaluation")
        ef = QFormLayout(eg)
        self.target = QComboBox()
        self.l_target = tlabel("Evaluated state (metrics)")
        ef.addRow(self.l_target, self.target)
        self.metric_form = ParamForm(S.METRIC_PARAMS)
        ef.addRow(self.metric_form)
        lay.addWidget(eg)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll(inner))

        self.model_combo.currentIndexChanged.connect(self._on_model)
        self.custom_edit.textChanged.connect(self._on_custom_text)
        self.fs.valueChanged.connect(self._refresh_derived)
        self._current_params: dict = {}

    # -- helpers ------------------------------------------------------------
    def _build_param_form(self, key):
        if self.param_form is not None:
            self.params_lay.removeWidget(self.param_form)
            self.param_form.hide()
            self.param_form.setParent(None)
            self.param_form.deleteLater()
        self.param_form = ParamForm(S.MODEL_PARAMS.get(key, []))
        self.param_form.changed.connect(self._refresh_derived)
        self.params_lay.insertWidget(0, self.param_form)
        custom = key == "custom_linear"
        self.custom_label.setVisible(custom)
        self.custom_edit.setVisible(custom)

    CUSTOM_TEMPLATE = ("# continuous-time linear model dx/dt = Ac x + w,  E[w w^T] = Qc\n"
                       "continuous: true\n"
                       "Ac: [[0.0, 1.0], [0.0, 0.0]]\n"
                       "Qc: [[0.0, 0.0], [0.0, 0.05]]\n"
                       "state_names: [position, velocity]\n"
                       "state_units: [m, m/s]\n"
                       "x0: [0.0, 1.0]\n")

    def _on_model(self):
        key = self.model_combo.currentData()
        self._build_param_form(key)
        self._current_params = {}
        if key == "custom_linear" and not self.custom_edit.toPlainText().strip():
            self.custom_edit.setPlainText(self.CUSTOM_TEMPLATE)
        m = create_model(key, 1.0 / self.fs.value(), {}) if key != "custom_linear" else self.safe_model()
        if m is not None:
            self._fill_states(m, m.default_x0, np.full(m.n, 0.1), m.default_x0, np.full(m.n, 1.0))
        self._refresh_derived()
        self.model_changed.emit()

    def _on_custom_text(self):
        if self.model_combo.currentData() == "custom_linear":
            self._refresh_derived()
            if self.safe_model() is not None:
                self.model_changed.emit()

    def _fill_states(self, model, x0, sd0, xe, sde):
        n = model.n
        self.x0_table.setRowCount(n)
        for i in range(n):
            vals = [state_label(model.state_names[i]), model.state_units[i], x0[i], sd0[i], xe[i], sde[i]]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v) if c < 2 else f"{float(v):.6g}")
                if c < 2:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.x0_table.setItem(i, c, it)
        cur = self.target.currentData()
        self.target.blockSignals(True)
        self.target.clear()
        for nm, u in zip(model.state_names, model.state_units):
            self.target.addItem(f"{state_label(nm)} [{u}]", nm)
        i = self.target.findData(cur)
        self.target.setCurrentIndex(max(i, 0))
        self.target.blockSignals(False)
        self.metric_form.set_unit(model.state_units[max(self.target.currentIndex(), 0)])

    def model_params(self) -> dict:
        key = self.model_combo.currentData()
        if key == "custom_linear":
            try:
                p = yaml.safe_load(self.custom_edit.toPlainText()) or {}
            except yaml.YAMLError as exc:
                from ..experiments.config import syntax_error_position
                line, col = syntax_error_position(exc)
                raise SensorTrustError(f"Custom model YAML: syntax error at line {line}, column {col}.") from None
            return p if isinstance(p, dict) else {}
        out = copy.deepcopy(self._current_params)
        for k, v in (self.param_form.values() if self.param_form else {}).items():
            if "." in k:
                a, b = k.split(".", 1)
                sub = out.setdefault(a, {"type": {"force": "sine", "torque": "sine", "power": "square"}.get(a, "sine")})
                sub[b] = v
            else:
                out[k] = v
        return out

    def current_model(self):
        return create_model(self.model_combo.currentData(), 1.0 / self.fs.value(), self.model_params())

    def safe_model(self):
        """Current model, or None while its parameters are invalid (e.g. a custom model being edited)."""
        try:
            return self.current_model()
        except Exception:
            return None

    def _refresh_derived(self):
        self.ts_label.setText(f"{1.0 / self.fs.value():.6g} s")
        try:
            m = self.current_model()
            self.q_text.setText(np.array2string(m.Q, precision=3, max_line_width=120))
            if self.x0_table.rowCount() != m.n:
                self._fill_states(m, m.default_x0, np.full(m.n, 0.1), m.default_x0, np.full(m.n, 1.0))
        except Exception as exc:
            self.q_text.setText(translate_error(exc))

    # -- config I/O ----------------------------------------------------------
    def load(self, cfg: dict):
        key = cfg["model"]["type"]
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(key)))
        self.model_combo.blockSignals(False)
        self.fs.blockSignals(True)
        self.fs.setValue(float(cfg["simulation"]["fs"]))
        self.fs.blockSignals(False)
        self.duration.setValue(float(cfg["simulation"]["duration"]))
        self._build_param_form(key)
        params = copy.deepcopy(cfg["model"].get("params", {}) or {})
        self._current_params = params
        if key == "custom_linear":
            self.custom_edit.setPlainText(yaml.safe_dump(params, sort_keys=False))
        else:
            flat = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}.{kk}"] = vv
                else:
                    flat[k] = v
            self.param_form.set_values(flat)
        m = model_for(cfg)
        x0 = np.asarray(cfg["model"].get("x0") if cfg["model"].get("x0") is not None else m.default_x0, float)
        P0 = cfg["model"].get("P0", 0.01)
        sd0 = np.sqrt(np.diag(np.atleast_2d(P0)) if np.ndim(P0) == 2 else np.broadcast_to(P0, (m.n,)))
        est = cfg.get("estimation") or {}
        xe = np.asarray(est.get("x0") if est.get("x0") is not None else x0, float)
        Pe = est.get("P0") if est.get("P0") is not None else P0
        sde = np.sqrt(np.diag(np.atleast_2d(Pe)) if np.ndim(Pe) == 2 else np.broadcast_to(Pe, (m.n,)))
        self._fill_states(m, x0, sd0, xe, sde)
        self.sample_x0.setChecked(bool(cfg["model"].get("sample_initial_state", True)))
        met = cfg.get("metrics", {})
        ts = met.get("target_state", 0)
        idx = self.target.findData(ts if isinstance(ts, str) else m.state_names[int(ts)])
        self.target.setCurrentIndex(max(idx, 0))
        self.metric_form.set_unit(m.state_units[max(self.target.currentIndex(), 0)])
        self.metric_form.set_values({**met, "rmse_max": met.get("rmse_max") or 0.0})
        self._refresh_derived()

    def store(self, cfg: dict):
        key = self.model_combo.currentData()
        cfg["model"]["type"] = key
        cfg["model"]["params"] = self.model_params()
        cols = [[float(self.x0_table.item(r, c).text()) for r in range(self.x0_table.rowCount())] for c in range(2, 6)]
        cfg["model"]["x0"] = cols[0]
        cfg["model"]["P0"] = [v**2 for v in cols[1]]
        cfg["model"]["sample_initial_state"] = self.sample_x0.isChecked()
        cfg.setdefault("estimation", {})
        cfg["estimation"]["x0"] = cols[2]
        cfg["estimation"]["P0"] = [max(v, 1e-9) ** 2 for v in cols[3]]
        cfg["simulation"]["duration"] = float(self.duration.value())
        cfg["simulation"]["fs"] = float(self.fs.value())
        met = cfg.setdefault("metrics", {})
        met["target_state"] = self.target.currentData()
        vals = self.metric_form.values()
        vals["rmse_max"] = vals["rmse_max"] or None
        met.update(vals)

    def retranslate(self):
        lang = get_language()
        for i in range(self.model_combo.count()):
            self.model_combo.setItemText(i, _lab(S.MODEL_LABELS[self.model_combo.itemData(i)]))
        self.x0_table.setHorizontalHeaderLabels([tr("State"), tr("Unit"), tr("True initial mean x0"),
                                                 tr("Initial std sqrt(P0)"), tr("Estimator prior mean"),
                                                 tr("Estimator prior std")])
        try:
            m = self.current_model()
            self.model_desc.setText(model_description(m.key, m.description))
        except Exception:
            self.model_desc.setText("")
        if self.param_form:
            self.param_form.retranslate()
        self.metric_form.retranslate()
        m = self.safe_model()
        if m is not None:
            for i in range(min(self.x0_table.rowCount(), m.n)):
                it = self.x0_table.item(i, 0)
                if it is not None:
                    it.setText(state_label(m.state_names[i]))
            for i in range(self.target.count()):
                j = m.state_names.index(self.target.itemData(i)) if self.target.itemData(i) in m.state_names else None
                if j is not None:
                    self.target.setItemText(i, f"{state_label(m.state_names[j])} [{m.state_units[j]}]")
        _ = lang


# ----------------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------------
SENSOR_CHAR = [
    S.P("bias", "Nominal (uncompensated) bias", "Sesgo nominal (no compensado)", 0.0, unit="{u}", decimals=5,
        help_en="Constant offset present from the start and unknown to the estimators (not a fault).",
        help_es="Desplazamiento constante presente desde el inicio y desconocido por los estimadores (no es un fallo)."),
    S.P("resolution", "Resolution (quantization step, 0 = none)", "Resolución (paso de cuantización, 0 = ninguna)",
        0.0, unit="{u}", lo=0, decimals=5,
        help_en="Measurements are rounded to multiples of this step (0 disables quantization).",
        help_es="Las mediciones se redondean a múltiplos de este paso (0 desactiva la cuantización)."),
    S.P("range_enabled", "Limit the measurement range (saturation)", "Limitar el rango de medición (saturación)",
        False, kind="bool", help_en="If checked, measurements are clipped to the range below.",
        help_es="Si se marca, las mediciones se recortan al rango indicado abajo."),
    S.P("range_low", "Measurement range lower limit", "Límite inferior del rango de medición", -100.0, unit="{u}",
        help_en="Lowest value the sensor can report.", help_es="Valor mínimo que puede reportar el sensor."),
    S.P("range_high", "Measurement range upper limit", "Límite superior del rango de medición", 100.0, unit="{u}",
        help_en="Highest value the sensor can report.", help_es="Valor máximo que puede reportar el sensor."),
    S.P("delay", "Transport delay", "Retardo de transporte", 0.0, unit="s", lo=0, decimals=4,
        help_en="Time delay of the measurements (simulated; the estimators do not compensate it).",
        help_es="Retardo de las mediciones (simulado; los estimadores no lo compensan)."),
    S.P("dropout", "Nominal dropout probability", "Probabilidad nominal de pérdida", 0.0, lo=0, hi=1,
        help_en="Probability that a sample is missing even without faults.",
        help_es="Probabilidad de que falte una muestra incluso sin fallos."),
    S.P("nominal_reliability", "Nominal reliability rho0", "Confiabilidad nominal rho0", 1.0, lo=0.001, hi=1,
        help_en="Maximum trust of the sensor (a priori reliability, 1 = fully reliable).",
        help_es="Confianza máxima del sensor (confiabilidad a priori, 1 = totalmente confiable)."),
    S.P("assumed_std", "Std assumed by estimators (0 = true nominal std)",
        "Desv. supuesta por los estimadores (0 = desv. nominal)", 0.0, unit="{u}", lo=0, decimals=5,
        help_en="Noise standard deviation believed by the estimators; different from the true one to study "
                "model mismatch (0 = use the true nominal value).",
        help_es="Desviación del ruido que suponen los estimadores; distinta de la real para estudiar el desajuste "
                "del modelo (0 = usar el valor nominal real)."),
]


class SensorsTab(QWidget):
    sensors_changed = Signal()

    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.sensors: list[dict] = []
        self._loading = False
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        g = tgroup("Virtual sensors")
        gl = QVBoxLayout(g)
        self.table = QTableWidget(0, 5)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        gl.addWidget(self.table)
        bl = QHBoxLayout()
        self.b_add, self.b_dup, self.b_del = tbutton("Add sensor"), tbutton("Duplicate sensor"), tbutton("Remove sensor")
        for b in (self.b_add, self.b_dup, self.b_del):
            bl.addWidget(b)
        gl.addLayout(bl)
        ll.addWidget(g)
        hint = tlabel("No hardware required: all measurements come from virtual sensors.")
        hint.setStyleSheet("color:#52514e")
        ll.addWidget(hint)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.detail = tgroup("Selected sensor")
        df = QFormLayout(self.detail)
        self.name = QLineEdit()
        self.measured = QComboBox()
        self.rate = QDoubleSpinBox()
        self.rate.setRange(0.001, 1e5)
        self.rate.setDecimals(3)
        self.rate.setSuffix(" Hz")
        self.noise_type = QComboBox()
        for k in S.NOISE_PARAMS:
            self.noise_type.addItem("", k)
        self.l_name, self.l_meas = tlabel("Name"), tlabel("Measured variable")
        self.l_rate, self.l_noise = tlabel("Rate [Hz]"), tlabel("Noise model")
        df.addRow(self.l_name, self.name)
        df.addRow(self.l_meas, self.measured)
        df.addRow(self.l_rate, self.rate)
        df.addRow(self.l_noise, self.noise_type)
        rl.addWidget(self.detail)
        self.noise_group = tgroup("Noise parameters")
        self.noise_lay = QVBoxLayout(self.noise_group)
        self.noise_form: ParamForm | None = None
        rl.addWidget(self.noise_group)
        cg = tgroup("Sensor characteristics")
        cl = QVBoxLayout(cg)
        self.char_form = ParamForm(SENSOR_CHAR)
        cl.addWidget(self.char_form)
        rl.addWidget(cg)
        rl.addStretch(1)
        split.addWidget(scroll(right))
        split.setSizes([520, 560])
        outer = QVBoxLayout(self)
        outer.addWidget(split)

        self.table.itemSelectionChanged.connect(self._on_select)
        self.b_add.clicked.connect(self._add)
        self.b_dup.clicked.connect(self._dup)
        self.b_del.clicked.connect(self._del)
        self.name.editingFinished.connect(self._commit)
        self.measured.currentIndexChanged.connect(self._commit)
        self.rate.valueChanged.connect(self._commit)
        self.noise_type.currentIndexChanged.connect(self._on_noise_type)
        self.char_form.changed.connect(self._commit)
        self._build_noise_form("gaussian")

    # ------------------------------------------------------------------
    def _measure_options(self):
        m = self.mw.system_tab.safe_model() if hasattr(self.mw, "system_tab") else None
        opts = []
        if m is not None:
            for nm, u in zip(m.state_names, m.state_units):
                opts.append((f"{state_label(nm)} [{u}]", ("measures", nm), u))
            for fn, mf in m.measurement_functions.items():
                if not mf.linear:
                    opts.append((f"h(x) = {_lab(MEAS_FN_LABELS.get(fn, (mf.description, mf.description)))} "
                                 f"[{mf.unit}]", ("function", fn), mf.unit))
        return opts

    def refresh_measure_options(self):
        cur = self.measured.currentData()
        self.measured.blockSignals(True)
        self.measured.clear()
        for text, data, unit in self._measure_options():
            self.measured.addItem(text, data)
        i = self.measured.findData(cur)
        self.measured.setCurrentIndex(max(i, 0))
        self.measured.blockSignals(False)
        self._refresh_table()

    def _unit_of(self, s: dict) -> str:
        for text, data, unit in self._measure_options():
            if ("function" in s and data == ("function", s["function"])) or \
                    (data == ("measures", s.get("measures"))):
                return unit
        return "u"

    def _build_noise_form(self, typ):
        if self.noise_form is not None:
            self.noise_lay.removeWidget(self.noise_form)
            self.noise_form.hide()
            self.noise_form.setParent(None)
            self.noise_form.deleteLater()
        self.noise_form = ParamForm(S.NOISE_PARAMS[typ])
        self.noise_form.changed.connect(self._commit)
        self.noise_lay.addWidget(self.noise_form)

    def _on_noise_type(self):
        if self._loading:
            return
        self._build_noise_form(self.noise_type.currentData())
        self._commit()

    def _row(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else -1

    def _refresh_table(self):
        self.table.blockSignals(True)
        sel = self._row()
        self.table.setRowCount(len(self.sensors))
        for r, s in enumerate(self.sensors):
            meas = s.get("function") or s.get("measures")
            nz = s.get("noise", {})
            if not isinstance(nz, dict):
                nz = {"type": "gaussian", "std": nz}
            ndeg = len(s.get("degradations", []))
            ntype = nz.get("type", "gaussian")
            vals = [s["name"], state_label(str(meas)), f"{float(s.get('rate', self.mw.system_tab.fs.value())):g}",
                    f"{_lab(S.NOISE_LABELS.get(ntype, (ntype, ntype)))} (sigma = {nz.get('std', 0.1)})", str(ndeg)]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        if 0 <= sel < len(self.sensors):
            self.table.selectRow(sel)
        self.table.blockSignals(False)

    def _on_select(self):
        r = self._row()
        if r < 0 or r >= len(self.sensors):
            return
        s = self.sensors[r]
        self._loading = True
        self.name.setText(s["name"])
        key = ("function", s["function"]) if s.get("function") else ("measures", s.get("measures", 0))
        m = self.mw.system_tab.safe_model()
        if key[0] == "measures" and not isinstance(key[1], str) and m is not None:
            key = ("measures", m.state_names[int(key[1])])
        i = self.measured.findData(key)
        self.measured.setCurrentIndex(max(i, 0))
        self.rate.setValue(float(s.get("rate", self.mw.system_tab.fs.value())))
        nz = s.get("noise", {"type": "gaussian", "std": 0.1})
        if not isinstance(nz, dict):
            nz = {"type": "gaussian", "std": float(nz)}
        self.noise_type.setCurrentIndex(max(0, self.noise_type.findData(nz.get("type", "gaussian"))))
        self._build_noise_form(nz.get("type", "gaussian"))
        unit = self._unit_of(s)
        self.noise_form.set_unit(unit)
        self.noise_form.set_values(nz)
        rng = s.get("range")
        self.char_form.set_unit(unit)
        self.char_form.set_values({
            "bias": float(np.atleast_1d(s.get("bias", 0.0))[0]), "resolution": s.get("resolution") or 0.0,
            "range_enabled": rng is not None, "range_low": rng[0] if rng else -100.0,
            "range_high": rng[1] if rng else 100.0, "delay": s.get("delay", 0.0), "dropout": s.get("dropout", 0.0),
            "nominal_reliability": s.get("nominal_reliability", 1.0),
            "assumed_std": s.get("assumed_std") or 0.0})
        self._loading = False

    def _commit(self):
        if self._loading:
            return
        r = self._row()
        if r < 0 or r >= len(self.sensors):
            return
        s = self.sensors[r]
        s["name"] = self.name.text().strip() or s["name"]
        data = self.measured.currentData()
        s.pop("measures", None)
        s.pop("function", None)
        s.pop("H", None)
        if data:
            s[data[0]] = data[1]
        s["rate"] = float(self.rate.value())
        s["noise"] = {"type": self.noise_type.currentData(), **self.noise_form.values()}
        c = self.char_form.values()
        s["bias"] = c["bias"]
        s["resolution"] = c["resolution"] or None
        s["range"] = [c["range_low"], c["range_high"]] if c["range_enabled"] else None
        s["delay"] = c["delay"]
        s["dropout"] = c["dropout"]
        s["nominal_reliability"] = c["nominal_reliability"]
        s["assumed_std"] = c["assumed_std"] or None
        unit = self._unit_of(s)
        self.noise_form.set_unit(unit)
        self.char_form.set_unit(unit)
        self._refresh_table()
        self.sensors_changed.emit()

    def _add(self):
        names = {s["name"] for s in self.sensors}
        k = len(self.sensors) + 1
        while f"S{k}" in names:
            k += 1
        m = self.mw.system_tab.safe_model()
        self.sensors.append({"name": f"S{k}", "measures": m.state_names[0] if m is not None else 0, "rate": self.mw.system_tab.fs.value(),
                             "noise": {"type": "gaussian", "std": 0.1}, "degradations": []})
        self._refresh_table()
        self.table.selectRow(len(self.sensors) - 1)
        self.sensors_changed.emit()

    def _dup(self):
        r = self._row()
        if r < 0:
            return
        s = copy.deepcopy(self.sensors[r])
        names = {x["name"] for x in self.sensors}
        base = s["name"]
        k = 2
        while f"{base}_{k}" in names:
            k += 1
        s["name"] = f"{base}_{k}"
        self.sensors.append(s)
        self._refresh_table()
        self.table.selectRow(len(self.sensors) - 1)
        self.sensors_changed.emit()

    def _del(self):
        r = self._row()
        if len(self.sensors) <= 1:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, tr("Remove sensor"), tr("At least one sensor is required."))
            return
        if r < 0:
            return
        del self.sensors[r]
        self._refresh_table()
        self.table.selectRow(min(r, len(self.sensors) - 1))
        self.sensors_changed.emit()

    def load(self, cfg):
        self.sensors = copy.deepcopy(cfg["sensors"])
        for s in self.sensors:
            s.setdefault("degradations", [])
        self.refresh_measure_options()
        if self.sensors:
            self.table.selectRow(0)
            self._on_select()

    def store(self, cfg):
        cfg["sensors"] = copy.deepcopy(self.sensors)

    def retranslate(self):
        self.table.setHorizontalHeaderLabels([tr("Name"), tr("Measured variable"), tr("Rate [Hz]"), tr("Noise model"),
                                              tr("Degradation")])
        for i in range(self.noise_type.count()):
            self.noise_type.setItemText(i, _lab(S.NOISE_LABELS[self.noise_type.itemData(i)]))
        self.char_form.retranslate()
        self._refresh_table()
        self.refresh_measure_options()
        if self.noise_form:
            self.noise_form.retranslate()


# ----------------------------------------------------------------------------
# Degradation
# ----------------------------------------------------------------------------
SEVERITY_HIDDEN = {"saturation", "stuck"}


class DegradationTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self._loading = False
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        top = QFormLayout()
        self.sensor = QComboBox()
        self.l_sensor = tlabel("Sensor")
        top.addRow(self.l_sensor, self.sensor)
        ll.addLayout(top)
        g = tgroup("Degradations of the selected sensor")
        gl = QVBoxLayout(g)
        self.list = QListWidget()
        self.list.setMaximumHeight(120)
        gl.addWidget(self.list)
        bl = QHBoxLayout()
        self.b_add, self.b_del = tbutton("Add degradation"), tbutton("Remove degradation")
        bl.addWidget(self.b_add)
        bl.addWidget(self.b_del)
        gl.addLayout(bl)
        ll.addWidget(g)
        eg = tgroup("Degradation parameters")
        el = QFormLayout(eg)
        self.type = QComboBox()
        for k in S.DEG_SEVERITY:
            self.type.addItem("", k)
        self.l_type = tlabel("Degradation type")
        el.addRow(self.l_type, self.type)
        self.sev_holder = QVBoxLayout()
        el.addRow(self.sev_holder)
        ll.addWidget(eg)
        pg = tgroup("Temporal profile")
        pl = QFormLayout(pg)
        self.ptype = QComboBox()
        for k in S.PROFILE_LABELS:
            self.ptype.addItem("", k)
        self.l_ptype = tlabel("Profile type")
        pl.addRow(self.l_ptype, self.ptype)
        self.pform = ParamForm(S.PROFILE_PARAMS)
        pl.addRow(self.pform)
        ll.addWidget(pg)
        ll.addStretch(1)
        split.addWidget(scroll(left))
        right = QWidget()
        rl = QVBoxLayout(right)
        self.prev_label = tlabel("Preview (computed with the real degradation engine, noise-free signal)")
        rl.addWidget(self.prev_label)
        self.canvas = PlotCanvas(size=(7, 6))
        rl.addWidget(self.canvas, 1)
        self.b_prev = tbutton("Update preview")
        rl.addWidget(self.b_prev)
        split.addWidget(right)
        split.setSizes([470, 650])
        outer = QVBoxLayout(self)
        outer.addWidget(split)
        self.sev_form: ParamForm | None = None

        self.sensor.currentIndexChanged.connect(self._on_sensor)
        self.list.currentRowChanged.connect(self._on_item)
        self.b_add.clicked.connect(self._add)
        self.b_del.clicked.connect(self._del)
        self.type.currentIndexChanged.connect(self._on_type)
        self.ptype.currentIndexChanged.connect(self._on_ptype)
        self.pform.changed.connect(self._commit)
        self.b_prev.clicked.connect(self.preview)

    @property
    def sensors(self):
        return self.mw.sensors_tab.sensors

    def _cur_sensor(self):
        i = self.sensor.currentIndex()
        return self.sensors[i] if 0 <= i < len(self.sensors) else None

    def _unit(self):
        s = self._cur_sensor()
        return self.mw.sensors_tab._unit_of(s) if s else "u"

    def refresh_sensors(self):
        cur = self.sensor.currentIndex()
        self.sensor.blockSignals(True)
        self.sensor.clear()
        for s in self.sensors:
            self.sensor.addItem(s["name"])
        self.sensor.setCurrentIndex(min(max(cur, 0), len(self.sensors) - 1))
        self.sensor.blockSignals(False)
        self._on_sensor()

    def _deg_text(self, d):
        prof = d.get("profile", {})
        lab = _lab(S.DEG_LABELS.get(d["type"], (d["type"], d["type"])))
        if d.get("type") == "combined":
            lab = tr("combined") + ": " + " + ".join(
                _lab(S.DEG_LABELS.get(c["type"], (c["type"], c["type"]))) for c in d.get("components", []))
        ptype = prof.get("type", "step")
        plab = _lab(S.PROFILE_LABELS.get(ptype, (ptype, ptype)))
        sev = d.get("severity", "")
        return f"{lab} | {tr('severity')} = {sev} | {plab}, {tr('onset')} {prof.get('start', 0)} s"

    def _on_sensor(self):
        s = self._cur_sensor()
        self.list.blockSignals(True)
        self.list.clear()
        if s:
            for d in s.get("degradations", []):
                self.list.addItem(self._deg_text(d))
        self.list.blockSignals(False)
        if self.list.count():
            self.list.setCurrentRow(0)
            self._on_item(0)
        else:
            self._set_editor_enabled(False)
            self._update_profile_visibility()
        self.preview()

    def _set_editor_enabled(self, on: bool):
        for w in (self.type, self.ptype, self.pform):
            w.setEnabled(on)
        if self.sev_form:
            self.sev_form.setEnabled(on)

    def _build_sev(self, typ):
        if self.sev_form is not None:
            self.sev_holder.removeWidget(self.sev_form)
            self.sev_form.hide()
            self.sev_form.setParent(None)
            self.sev_form.deleteLater()
        params = [] if typ in SEVERITY_HIDDEN else [S.DEG_SEVERITY[typ]]
        params += S.DEG_EXTRA.get(typ, [])
        self.sev_form = ParamForm(params, self._unit())
        self.sev_form.changed.connect(self._commit)
        self.sev_holder.addWidget(self.sev_form)

    def _deg(self):
        s = self._cur_sensor()
        r = self.list.currentRow()
        if s is None or r < 0 or r >= len(s.get("degradations", [])):
            return None
        return s["degradations"][r]

    def _on_item(self, r):
        d = self._deg()
        if d is None:
            return
        self._loading = True
        self._set_editor_enabled(d.get("type") != "combined")
        typ = d["type"] if d.get("type") in S.DEG_SEVERITY else "bias"
        self.type.setCurrentIndex(max(0, self.type.findData(typ)))
        self._build_sev(typ)
        self.sev_form.set_values(d)
        prof = d.get("profile", {"type": "step", "start": 0.0})
        self.ptype.setCurrentIndex(max(0, self.ptype.findData(prof.get("type", "step"))))
        vals = dict(prof)
        rec = prof.get("recovery_time") is not None or prof.get("duration") is not None
        if prof.get("duration") is not None and prof.get("recovery_time") is None:
            vals["recovery_time"] = float(prof.get("start", 0)) + float(prof["duration"])
            vals.setdefault("recovery_duration", 0.0)
        vals["recovery"] = rec
        self.pform.set_values(vals)
        self._update_profile_visibility()
        self._loading = False

    def _update_profile_visibility(self):
        typ = self.ptype.currentData()
        uses = S.PROFILE_USES.get(typ, {"start"})
        rec = self.pform.values()["recovery"] or typ == "recovery"
        for p in S.PROFILE_PARAMS:
            if p.key in ("recovery",):
                self.pform.set_visible(p.key, typ != "recovery")
            elif p.key in ("recovery_time", "recovery_duration", "recovery_shape"):
                self.pform.set_visible(p.key, rec)
            else:
                self.pform.set_visible(p.key, p.key in uses)

    def _on_type(self):
        if self._loading:
            return
        typ = self.type.currentData()
        self._build_sev(typ)
        sev = S.DEG_SEVERITY[typ]
        self.sev_form.set_values({"severity": sev.default, **{p.key: p.default for p in S.DEG_EXTRA.get(typ, [])}})
        self._commit()

    def _on_ptype(self):
        self._update_profile_visibility()
        self._commit()

    def _commit(self):
        if self._loading:
            return
        d = self._deg()
        if d is None or d.get("type") == "combined":
            return
        typ = self.type.currentData()
        d.clear()
        d["type"] = typ
        d.update(self.sev_form.values())
        d.setdefault("severity", 1.0)
        pv = self.pform.values()
        ptyp = self.ptype.currentData()
        uses = S.PROFILE_USES.get(ptyp, {"start"})
        prof = {"type": ptyp}
        for k in uses:
            prof[k] = pv[k]
        if pv["recovery"] or ptyp == "recovery":
            prof["recovery_time"] = max(pv["recovery_time"], pv["start"])
            prof["recovery_duration"] = pv["recovery_duration"]
            prof["recovery_shape"] = pv["recovery_shape"]
        d["profile"] = prof
        self._update_profile_visibility()
        self.list.blockSignals(True)
        self.list.currentItem().setText(self._deg_text(d))
        self.list.blockSignals(False)
        self.mw.sensors_tab._refresh_table()

    def _add(self):
        s = self._cur_sensor()
        if s is None:
            return
        start = round(0.33 * self.mw.system_tab.duration.value(), 3)
        s.setdefault("degradations", []).append({"type": "bias", "severity": 1.0,
                                                 "profile": {"type": "step", "start": start}})
        self._on_sensor()
        self.list.setCurrentRow(self.list.count() - 1)
        self.mw.sensors_tab._refresh_table()

    def _del(self):
        s = self._cur_sensor()
        r = self.list.currentRow()
        if s is None or r < 0:
            return
        del s["degradations"][r]
        self._on_sensor()
        self.mw.sensors_tab._refresh_table()

    def preview(self):
        s = self._cur_sensor()
        if s is None:
            return
        from ..i18n import tr as _tr
        from ..simulation.ground_truth import GroundTruthSimulator
        from ..sensors.sensor import VirtualSensor
        from ..visualization.style import INK, series_style
        try:
            cfg = self.mw.collect(validate=False)
            m = self.mw.system_tab.current_model()
            dur, fs = cfg["simulation"]["duration"], cfg["simulation"]["fs"]
            truth = GroundTruthSimulator(m, dur, fs).run(np.asarray(cfg["model"]["x0"]), np.zeros((m.n, m.n)),
                                                          cfg["experiment"]["seed"], False)
            sc = dict(s)
            nz = dict(sc.get("noise", {}))
            nz["std"] = 0.0 if not isinstance(sc.get("noise"), (int, float)) else 0.0
            sc["noise"] = {**nz, "type": "gaussian", "std": 0.0}
            vs = VirtualSensor(sc, 0, m, fs)
            d0 = vs.generate(truth.t, truth.x, cfg["experiment"]["seed"])
            true_sd = VirtualSensor(s, 0, m, fs).generate(truth.t, truth.x, cfg["experiment"]["seed"]).true_var
        except Exception as exc:
            self.canvas.message(f"{type(exc).__name__}: {exc}")
            return
        fig = self.canvas.figure
        fig.clear()
        axs = fig.subplots(3, 1, sharex=True)
        t = truth.t
        axs[0].plot(t, d0.clean[:, 0], color=INK, lw=1.8, label=_tr("Ground truth"))
        st = series_style(1)
        ok = np.isfinite(d0.z[:, 0])
        axs[0].plot(t[ok], d0.z[ok, 0], color=st["color"], lw=1.2, linestyle="--", label=s["name"])
        lost = ~ok & ((np.arange(t.size) % vs.period_steps) == 0)
        if lost.any():
            axs[0].plot(t[lost], d0.clean[lost, 0], "x", color="#898781", markersize=4, label=_tr("sample lost"))
        axs[0].legend(loc="best", fontsize=7)
        for j, g in enumerate(d0.activation):
            stj = series_style(j)
            axs[1].plot(t, g, color=stj["color"], linestyle=stj["linestyle"],
                        label=degradation_label(d0.fault_meta[j]["type"]))
        axs[1].set_ylim(-0.05, 1.05)
        axs[1].set_ylabel(_tr("activation g(t)"), fontsize=8)
        if d0.activation:
            axs[1].legend(loc="best", fontsize=7)
        axs[2].plot(t, d0.true_bias[:, 0], color=series_style(0)["color"], label=_tr("systematic error"))
        axs[2].plot(t, np.sqrt(true_sd[:, 0]), color=series_style(2)["color"], linestyle="--", label=_tr("noise std"))
        axs[2].legend(loc="best", fontsize=7)
        axs[2].set_xlabel(_tr("Time [s]"))
        axs[0].set_title(f"{_tr('Degradation preview')} - {s['name']}")
        self.canvas.canvas.draw_idle()

    def load(self, cfg):
        self.refresh_sensors()

    def store(self, cfg):
        pass  # degradations live inside the sensor dictionaries

    def retranslate(self):
        for i in range(self.type.count()):
            self.type.setItemText(i, _lab(S.DEG_LABELS[self.type.itemData(i)]))
        for i in range(self.ptype.count()):
            self.ptype.setItemText(i, _lab(S.PROFILE_LABELS[self.ptype.itemData(i)]))
        self.pform.retranslate()
        if self.sev_form:
            self.sev_form.retranslate()
        s = self._cur_sensor()
        if s:
            for r, d in enumerate(s.get("degradations", [])):
                if self.list.item(r):
                    self.list.item(r).setText(self._deg_text(d))


# ----------------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------------
METHOD_DESC_ES = {
    "simple_average": "Media no ponderada de las mediciones disponibles.",
    "fixed_weights": "Media ponderada con pesos constantes definidos por el usuario.",
    "inverse_variance": "Media ponderada con w_i proporcional a 1/sigma_i^2 (varianzas nominales).",
    "best_sensor_oracle": "ORÁCULO (referencia offline, no realizable): salida del mejor sensor según el valor verdadero.",
    "kf": "Filtro de Kalman centralizado con R nominal fija (caso A).",
    "kf_true_r": "ORÁCULO: KF centralizado con la varianza verdadera variante en el tiempo (caso B).",
    "adaptive_kf": "KF centralizado con estimación en línea de R_i por emparejamiento de covarianza Sage-Husa (caso C).",
    "sensortrust_kf": "KF centralizado con R_i escalada por la confianza relativa de los dos primeros momentos (caso D).",
    "sensortrust_kf_mean_only": "Ablación: confianza solo con el primer momento de la innovación.",
    "sensortrust_kf_m2_only": "Ablación: confianza solo con el segundo momento (central).",
    "kf_nis_gate": "KF centralizado que rechaza muestras fuera de la compuerta chi-cuadrado.",
    "single_sensor_kf": "Filtro de Kalman que usa solo los sensores indicados (referencia de sensor individual).",
    "ekf": "Filtro de Kalman extendido con R nominal.", "adaptive_ekf": "EKF con R adaptativa Sage-Husa.",
    "sensortrust_ekf": "EKF con confianza SensorTrust.", "ukf": "Filtro de Kalman unscented con R nominal.",
    "sensortrust_ukf": "UKF con confianza SensorTrust.",
    "ci_fusion": "KF local por sensor fusionado por intersección de covarianzas con pesos iguales.",
    "sensortrust_ci": "KF local por sensor fusionado por intersección de covarianzas con pesos de confianza.",
}
RECOMMENDED = ("simple_average", "inverse_variance", "kf", "adaptive_kf", "kf_nis_gate", "sensortrust_kf", "kf_true_r")


class FusionTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        lay = QVBoxLayout(self)
        self.head = tlabel("Fusion and estimation methods (all run on the same scenario dataset)")
        f = self.head.font()
        f.setBold(True)
        self.head.setFont(f)
        lay.addWidget(self.head)
        self.table = QTableWidget(0, 4)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        lay.addWidget(self.table, 1)
        bl = QHBoxLayout()
        self.b_all, self.b_none, self.b_rec = tbutton("Select all"), tbutton("Select none"), tbutton("Recommended set")
        for b in (self.b_all, self.b_none, self.b_rec):
            bl.addWidget(b)
        bl.addStretch(1)
        lay.addLayout(bl)
        wg = QFormLayout()
        self.weights = QLineEdit("1, 1, 1")
        self.l_weights = tlabel("Fixed weights (comma separated, one per sensor)")
        wg.addRow(self.l_weights, self.weights)
        lay.addLayout(wg)
        ag = tgroup("Adaptive Kalman filter (case C)")
        al = QVBoxLayout(ag)
        self.adapt_form = ParamForm(S.ADAPTIVE_PARAMS)
        al.addWidget(self.adapt_form)
        lay.addWidget(ag)
        self.specs: list[dict] = []
        self.b_all.clicked.connect(lambda: self._check(lambda s: True))
        self.b_none.clicked.connect(lambda: self._check(lambda s: False))
        self.b_rec.clicked.connect(lambda: self._check(lambda s: s["name"] in RECOMMENDED and "label" not in s))

    def _check(self, pred):
        for r, s in enumerate(self.specs):
            self.table.item(r, 0).setCheckState(Qt.Checked if pred(s) else Qt.Unchecked)

    def load(self, cfg):
        chosen = copy.deepcopy(cfg["methods"])
        rows = list(chosen)
        self.weights.setText(", ".join(["1"] * max(1, len(cfg.get("sensors", [])))))
        present = {(m["name"]) for m in chosen if "label" not in m and "use_sensors" not in m}
        for name in METHOD_PRESETS:
            if name not in present:
                rows.append({"name": name})
        self.specs = rows
        self.table.setRowCount(len(rows))
        for r, s in enumerate(rows):
            m = create_method(s) if s["name"] != "fixed_weights" else create_method({**s, "weights": s.get("weights", [1])})
            it = QTableWidgetItem(tr(m.label))
            it.setData(Qt.UserRole, m.label)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if r < len(chosen) else Qt.Unchecked)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(m.category))
            self.table.setItem(r, 2, QTableWidgetItem(s["name"]))
            self.table.setItem(r, 3, QTableWidgetItem(""))
            if s["name"] == "fixed_weights" and s.get("weights"):
                self.weights.setText(", ".join(str(w) for w in s["weights"]))
            if s.get("adaptive"):
                self.adapt_form.set_values(s["adaptive"])
        self.retranslate()

    def selected_specs(self) -> list[dict]:
        out = []
        for r, s in enumerate(self.specs):
            if self.table.item(r, 0).checkState() == Qt.Checked:
                out.append(copy.deepcopy(s))
        return out

    def store(self, cfg):
        specs = self.selected_specs()
        trust = self.mw.trust_tab.values()
        n = len(cfg.get("sensors", []))
        for s in specs:
            preset = METHOD_PRESETS.get(s["name"], {})
            if s["name"] == "fixed_weights":
                try:
                    w = [float(x) for x in self.weights.text().replace(";", ",").split(",") if x.strip()]
                except ValueError:
                    raise SensorTrustError("Fixed weights must be numbers separated by commas.") from None
                if len(w) != n:
                    raise SensorTrustError(f"Fixed weights: {len(w)} values given but there are {n} sensors.")
                s["weights"] = w
            if preset.get("r_strategy") == "adaptive":
                s["adaptive"] = self.adapt_form.values()
            ptrust = (preset.get("trust") or {}).get("method")
            uses_ftm = ptrust in ("first_two_moments", "first_moment_only", "second_moment_only") or \
                preset.get("weighting") == "trust"
            if uses_ftm:
                tp = dict(trust["trust"])
                if ptrust in ("first_moment_only", "second_moment_only"):
                    tp.pop("use_first_moment", None)
                    tp.pop("use_second_moment", None)
                s["trust"] = {**(s.get("trust") or {}), **({"method": ptrust} if ptrust else {}), **tp}
                s["detector"] = trust["detector"]
                if preset.get("class") == "centralized":
                    s["innovation_gate"] = trust["innovation_gate"]
        cfg["methods"] = specs

    def retranslate(self):
        self.table.setHorizontalHeaderLabels([tr("Select"), tr("Category"), tr("Key"), tr("Description")])
        es = get_language() == "es"
        for r, s in enumerate(self.specs):
            it0 = self.table.item(r, 0)
            if it0 is not None and it0.data(Qt.UserRole):
                it0.setText(tr(it0.data(Qt.UserRole)))
            it1 = self.table.item(r, 1)
            if it1 is not None:
                cat = it1.data(Qt.UserRole) or it1.text()
                it1.setData(Qt.UserRole, cat)
                it1.setText(tr(cat))
            try:
                m = create_method(s if s["name"] != "fixed_weights" else {**s, "weights": s.get("weights", [1])})
                desc = METHOD_DESC_ES.get(s["name"], m.description) if es else m.description
                if s.get("use_sensors"):
                    desc += f" [{', '.join(s['use_sensors'])}]"
            except Exception:
                desc = ""
            self.table.item(r, 3).setText(desc)
        self.adapt_form.retranslate()


# ----------------------------------------------------------------------------
# Trust
# ----------------------------------------------------------------------------
TRUST_HELP_EN = """<b>SensorTrust - trust from the first two moments of the innovation</b><br>
For every sensor the standardised innovation eps = L<sup>-1</sup> nu (S = L L<sup>T</sup>, nominal R) is
tracked by an online estimator of its first moment (mean mu) and second moment (m2 = E[eps<sup>2</sup>]).
The variance s<sup>2</sup> = m2 - mu<sup>2</sup> separates noise growth from bias.<br><br>
d1 = max(0, |mu<sup>c</sup>| - gamma1/sqrt(N<sub>eff</sub>)),&nbsp;&nbsp;
d2 = max(0, s<sup>2</sup> - 1 - gamma2 sqrt(2/N<sub>eff</sub>))<br>
T<sub>raw</sub> = rho0 exp(-d1/beta1 - d2/beta2),&nbsp; T(k) = T(k-1) + eta [T<sub>raw</sub> - T(k-1)]<br><br>
mu<sup>c</sup> is the mean referenced to the median of the trusted redundant sensors (consensus), which
prevents a dominant faulty sensor from dragging the estimate. R<sub>i</sub><sup>eff</sup> =
R<sub>i</sub> / (T<sub>i</sub> / max<sub>j</sub> T<sub>j</sub>). The detector declares D<sub>i</sub> = 1 with
chi-square tests on the same moments (hysteresis in samples). Detection is independent from weighting."""
TRUST_HELP_ES = """<b>SensorTrust - confianza a partir de los dos primeros momentos de la innovación</b><br>
Para cada sensor la innovación estandarizada eps = L<sup>-1</sup> nu (S = L L<sup>T</sup>, R nominal) se
sigue con un estimador en línea de su primer momento (media mu) y su segundo momento (m2 = E[eps<sup>2</sup>]).
La varianza s<sup>2</sup> = m2 - mu<sup>2</sup> separa el aumento de ruido del sesgo.<br><br>
d1 = max(0, |mu<sup>c</sup>| - gamma1/sqrt(N<sub>eff</sub>)),&nbsp;&nbsp;
d2 = max(0, s<sup>2</sup> - 1 - gamma2 sqrt(2/N<sub>eff</sub>))<br>
T<sub>raw</sub> = rho0 exp(-d1/beta1 - d2/beta2),&nbsp; T(k) = T(k-1) + eta [T<sub>raw</sub> - T(k-1)]<br><br>
mu<sup>c</sup> es la media referida a la mediana de los sensores redundantes confiables (consenso), lo que
evita que un sensor dominante defectuoso arrastre la estimación. R<sub>i</sub><sup>eff</sup> =
R<sub>i</sub> / (T<sub>i</sub> / max<sub>j</sub> T<sub>j</sub>). El detector declara D<sub>i</sub> = 1 con
pruebas chi-cuadrado sobre los mismos momentos (histéresis en muestras). La detección es independiente de la
ponderación."""


class TrustTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        split = QSplitter(Qt.Horizontal)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        g = tgroup("First-Two-Moments trust estimator (SensorTrust)")
        gl = QVBoxLayout(g)
        self.form = ParamForm(S.TRUST_PARAMS)
        gl.addWidget(self.form)
        lay.addWidget(g)
        dg = tgroup("Degradation detector")
        dl = QVBoxLayout(dg)
        self.det_form = ParamForm(S.DETECTOR_PARAMS)
        dl.addWidget(self.det_form)
        lay.addWidget(dg)
        gg = tgroup("Innovation gate")
        gll = QVBoxLayout(gg)
        self.gate_form = ParamForm([S.GATE_PARAM])
        gll.addWidget(self.gate_form)
        lay.addWidget(gg)
        self.b_def = tbutton("Restore defaults")
        lay.addWidget(self.b_def)
        lay.addStretch(1)
        split.addWidget(scroll(inner))
        hg = tgroup("How it works")
        hl = QVBoxLayout(hg)
        self.help = QTextBrowser()
        hl.addWidget(self.help)
        split.addWidget(hg)
        split.setSizes([620, 480])
        outer = QVBoxLayout(self)
        outer.addWidget(split)
        self.b_def.clicked.connect(self._defaults)
        self.form.changed.connect(self._vis)
        self.det_form.changed.connect(self._vis)
        self._vis()

    def _defaults(self):
        self.form.reset_defaults()
        self.det_form.reset_defaults()
        self.gate_form.reset_defaults()
        self._vis()

    def _vis(self):
        v = self.form.values()
        self.form.set_visible("ewma_lambda", v["estimator"] == "ewma")
        self.form.set_visible("window_length", v["estimator"] == "window")
        d = self.det_form.values()["method"]
        for k in ("significance",):
            self.det_form.set_visible(k, d == "moment_test")
        for k in ("alarm_trust_below", "clear_trust_above"):
            self.det_form.set_visible(k, d == "trust_threshold")

    def values(self) -> dict:
        t = self.form.values()
        d = self.det_form.values()
        method = d.pop("method")
        if method == "none":
            det = None
        elif method == "moment_test":
            det = {"method": method, "significance": d["significance"], "alarm_on_samples": d["alarm_on_samples"],
                   "alarm_off_samples": d["alarm_off_samples"]}
        else:
            det = {"method": method, "alarm_trust_below": d["alarm_trust_below"],
                   "clear_trust_above": d["clear_trust_above"], "alarm_on_samples": d["alarm_on_samples"],
                   "alarm_off_samples": d["alarm_off_samples"]}
        gate = self.gate_form.values()["innovation_gate"]
        return {"trust": t, "detector": det, "innovation_gate": gate if gate > 0 else None}

    def load(self, cfg):
        self._defaults()
        for m in cfg["methods"]:
            tr_ = m.get("trust") or {}
            if m["name"].startswith("sensortrust") or tr_.get("method") == "first_two_moments":
                vals = {k: v for k, v in tr_.items() if k != "method"}
                self.form.set_values(vals)
                det = m.get("detector")
                if det is None and "detector" in m:
                    self.det_form.set_values({"method": "none"})
                elif isinstance(det, dict):
                    self.det_form.set_values(det)
                if "innovation_gate" in m:
                    self.gate_form.set_values({"innovation_gate": m["innovation_gate"] or 0.0})
                break
        self._vis()

    def store(self, cfg):
        pass  # applied to trust-based methods by FusionTab.store

    def retranslate(self):
        self.form.retranslate()
        self.det_form.retranslate()
        self.gate_form.retranslate()
        self.help.setHtml(TRUST_HELP_ES if get_language() == "es" else TRUST_HELP_EN)


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
class ExperimentTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        lay = QVBoxLayout(self)
        g = tgroup("Experiment definition")
        f = QFormLayout(g)
        self.name = QLineEdit()
        self.desc = QPlainTextEdit()
        self.desc.setMaximumHeight(70)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.fmt_png, self.fmt_svg, self.fmt_pdf = QCheckBox("PNG"), QCheckBox("SVG"), QCheckBox("PDF")
        fl = QHBoxLayout()
        for c in (self.fmt_png, self.fmt_svg, self.fmt_pdf):
            fl.addWidget(c)
        fl.addStretch(1)
        self.save_ds = tcheck("Save dataset (dataset.csv)")
        from ..utils.paths import default_results_dir
        self.res_dir = QLineEdit(default_results_dir())
        self.b_browse = tbutton("Browse...")
        self.b_browse.clicked.connect(self._browse)
        dl = QHBoxLayout()
        dl.addWidget(self.res_dir, 1)
        dl.addWidget(self.b_browse)
        self.l_name, self.l_desc, self.l_seed = tlabel("Experiment name"), tlabel("Description text"), tlabel("Random seed")
        self.l_fmt, self.l_dir = tlabel("Figure formats"), tlabel("Results directory")
        f.addRow(self.l_name, self.name)
        f.addRow(self.l_desc, self.desc)
        f.addRow(self.l_seed, self.seed)
        f.addRow(self.l_fmt, fl)
        f.addRow("", self.save_ds)
        f.addRow(self.l_dir, dl)
        lay.addWidget(g)
        bl = QHBoxLayout()
        self.b_validate = tbutton("Validate configuration")
        self.b_run = tbutton("Run Experiment")
        self.b_run.setObjectName("primary")
        bl.addWidget(self.b_validate)
        bl.addWidget(self.b_run)
        bl.addStretch(1)
        lay.addLayout(bl)
        sg = tgroup("Configuration summary")
        sl = QVBoxLayout(sg)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setStyleSheet("font-family: monospace; font-size: 9pt")
        sl.addWidget(self.summary)
        lay.addWidget(sg, 1)
        self.b_validate.clicked.connect(self.mw.validate)
        self.b_run.clicked.connect(self.mw.run_experiment)

    def _browse(self):
        from PySide6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, tr("Results directory"), self.res_dir.text() or ".")
        if d:
            self.res_dir.setText(d)

    def load(self, cfg):
        e = cfg["experiment"]
        self.name.setText(e.get("name", ""))
        self.desc.setPlainText(e.get("description", ""))
        self.seed.setValue(int(e.get("seed", 42)))
        figs = cfg.get("output", {}).get("figures", ["png"])
        self.fmt_png.setChecked("png" in figs)
        self.fmt_svg.setChecked("svg" in figs)
        self.fmt_pdf.setChecked("pdf" in figs)
        self.save_ds.setChecked(bool(cfg.get("output", {}).get("save_dataset", True)))

    def store(self, cfg):
        e = cfg.setdefault("experiment", {})
        e["name"] = self.name.text().strip() or "Unnamed experiment"
        e["description"] = self.desc.toPlainText().strip()
        e["seed"] = int(self.seed.value())
        figs = [f for f, c in (("png", self.fmt_png), ("svg", self.fmt_svg), ("pdf", self.fmt_pdf)) if c.isChecked()]
        cfg.setdefault("output", {})["figures"] = figs or ["png"]
        cfg["output"]["save_dataset"] = self.save_ds.isChecked()

    def show_summary(self, cfg):
        self.summary.setPlainText(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    def retranslate(self):
        pass

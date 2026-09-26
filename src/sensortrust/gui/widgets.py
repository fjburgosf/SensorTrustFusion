"""Reusable widgets: schema-driven forms, plot canvas, data tables, background worker."""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..i18n import get_language, tr
from .schema import P, choice_label


class ParamForm(QWidget):
    """Form generated from a list of :class:`P` descriptors."""

    changed = Signal()

    def __init__(self, params: list[P], unit_sub: str = "", parent=None):
        super().__init__(parent)
        self.params = list(params)
        self.unit_sub = unit_sub
        self.form = QFormLayout(self)
        self.form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.widgets: dict[str, QWidget] = {}
        self.labels: dict[str, QLabel] = {}
        for p in self.params:
            w = self._make(p)
            lab = QLabel()
            self.labels[p.key] = lab
            self.widgets[p.key] = w
            self.form.addRow(lab, w)
        self.retranslate()

    def _make(self, p: P) -> QWidget:
        if p.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(p.decimals)
            w.setRange(p.lo, p.hi)
            w.setSingleStep(p.step or (10 ** -(min(p.decimals, 3) - 1) if p.decimals > 1 else 1.0))
            w.setValue(float(p.default))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda *_: self.changed.emit())
        elif p.kind == "int":
            w = QSpinBox()
            w.setRange(int(max(p.lo, -2**31 + 1)), int(min(p.hi, 2**31 - 1)))
            w.setValue(int(p.default))
            w.valueChanged.connect(lambda *_: self.changed.emit())
        elif p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(p.default))
            w.toggled.connect(lambda *_: self.changed.emit())
        elif p.kind == "choice":
            w = QComboBox()
            for c in p.choices:
                w.addItem(choice_label(c, get_language()), c)
            w.setCurrentIndex(max(0, list(p.choices).index(p.default)) if p.default in p.choices else 0)
            w.currentIndexChanged.connect(lambda *_: self.changed.emit())
        else:
            w = QLineEdit(str(p.default))
            w.editingFinished.connect(lambda *_: self.changed.emit())
        return w

    def retranslate(self):
        lang = get_language()
        for p in self.params:
            self.labels[p.key].setText(p.label(lang, self.unit_sub) + ":")
            if p.kind == "choice":
                cb = self.widgets[p.key]
                for i in range(cb.count()):
                    cb.setItemText(i, choice_label(cb.itemData(i), lang))
            h = p.help(lang)
            self.labels[p.key].setToolTip(h)
            self.widgets[p.key].setToolTip(h)

    def set_unit(self, unit: str):
        self.unit_sub = unit
        self.retranslate()

    def set_visible(self, key: str, visible: bool):
        self.labels[key].setVisible(visible)
        self.widgets[key].setVisible(visible)

    def values(self) -> dict:
        out = {}
        for p in self.params:
            w = self.widgets[p.key]
            if p.kind == "float":
                out[p.key] = float(w.value())
            elif p.kind == "int":
                out[p.key] = int(w.value())
            elif p.kind == "bool":
                out[p.key] = bool(w.isChecked())
            elif p.kind == "choice":
                out[p.key] = w.currentData()
            else:
                out[p.key] = w.text()
        return out

    def set_values(self, values: dict):
        for p in self.params:
            if p.key not in values or values[p.key] is None:
                continue
            w = self.widgets[p.key]
            v = values[p.key]
            w.blockSignals(True)
            try:
                if p.kind == "float":
                    w.setValue(float(v))
                elif p.kind == "int":
                    w.setValue(int(v))
                elif p.kind == "bool":
                    w.setChecked(bool(v))
                elif p.kind == "choice":
                    i = w.findData(v)
                    if i >= 0:
                        w.setCurrentIndex(i)
                else:
                    w.setText(str(v))
            finally:
                w.blockSignals(False)

    def reset_defaults(self):
        self.set_values({p.key: p.default for p in self.params})


NAV_TIPS_ES = {
    "Reset original view": "Restablecer la vista original",
    "Back to previous view": "Volver a la vista anterior",
    "Forward to next view": "Avanzar a la vista siguiente",
    "Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect":
        "Botón izquierdo: desplazar; botón derecho: acercar\nx/y fija un eje, CTRL fija la proporción",
    "Zoom to rectangle\nx/y fixes axis": "Acercar a un rectángulo\nx/y fija un eje",
    "Configure subplots": "Configurar márgenes de los ejes",
    "Edit axis, curve and image parameters": "Editar parámetros de ejes, curvas e imágenes",
    "Save the figure": "Guardar la figura (PNG, SVG, PDF...)",
}


NAV_TEXT_ES = {"Home": "Inicio", "Back": "Atrás", "Forward": "Adelante", "Pan": "Desplazar", "Zoom": "Acercar",
               "Subplots": "Márgenes", "Customize": "Personalizar", "Save": "Guardar"}


class PlotCanvas(QWidget):
    """Matplotlib figure + navigation toolbar."""

    def __init__(self, parent=None, size=(9, 5)):
        super().__init__(parent)
        self.figure = Figure(figsize=size, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        for a in self.toolbar.actions():
            a.setProperty("tip_en", a.toolTip())
            a.setProperty("text_en", a.text())
        self.retranslate()

    def retranslate(self):
        es = get_language() == "es"
        for a in self.toolbar.actions():
            en = a.property("tip_en") or ""
            if en:
                a.setToolTip(NAV_TIPS_ES.get(en, en) if es else en)
            txt = a.property("text_en") or ""
            if txt:
                a.setText(NAV_TEXT_ES.get(txt, txt) if es else txt)

    def draw(self, fn, *args, **kw):
        try:
            fn(*args, fig=self.figure, **kw)
        except Exception as exc:  # show the error inside the canvas instead of crashing the GUI
            import logging
            logging.getLogger("sensortrust.gui").exception("figure %s could not be drawn", getattr(fn, "__name__", fn))
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, f"{type(exc).__name__}: {exc}", ha="center", va="center", wrap=True)
        self.canvas.draw_idle()

    def message(self, text: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color="#52514e", wrap=True)
        self.canvas.draw_idle()


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return ("sí" if get_language() == "es" else "yes") if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        a = abs(float(v))
        if a != 0 and (a < 1e-3 or a >= 1e5):
            return f"{float(v):.3e}"
        return f"{float(v):.4g}"
    return str(v)


class DataTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)

    def set_frame(self, df: pd.DataFrame | None, columns: list[str] | None = None, headers: dict | None = None):
        self.clear()
        if df is None or df.empty:
            self.setRowCount(0)
            self.setColumnCount(0)
            return
        cols = [c for c in (columns or list(df.columns)) if c in df.columns]
        self.setColumnCount(len(cols))
        self.setRowCount(len(df))
        from ..labels import METRICS, metric_label
        self.setHorizontalHeaderLabels([(headers or {}).get(c, metric_label(c) if c in METRICS else tr(c))
                                        for c in cols])
        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(cols):
                val = row[col]
                if col in ("method", "reference", "category") and isinstance(val, str):
                    val = tr(val)
                elif col == "metric" and isinstance(val, str):
                    val = metric_label(val)
                it = QTableWidgetItem(_fmt(val))
                if isinstance(row[col], (int, float, np.integer, np.floating)):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(r, c, it)


class Worker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda m, f: self.progress.emit(m, float(f)), **self.kwargs)
            self.finished.emit(res)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())


def start_worker(owner, worker: Worker) -> QThread:
    th = QThread(owner)
    worker.moveToThread(th)
    th.started.connect(worker.run)
    worker.finished.connect(th.quit)
    worker.failed.connect(th.quit)
    th.finished.connect(th.deleteLater)
    th.start()
    return th


class LabeledCombo(QComboBox):
    """Combo box that stores internal values and shows translated labels (``label_fn(value)``)."""

    def __init__(self, values=(), label_fn=str, parent=None):
        super().__init__(parent)
        self.label_fn = label_fn
        self.set_values(values)

    def set_values(self, values, keep: bool = True):
        cur = self.currentData() if keep else None
        self.blockSignals(True)
        self.clear()
        for v in values:
            self.addItem(self.label_fn(v), v)
        i = self.findData(cur)
        self.setCurrentIndex(max(i, 0))
        self.blockSignals(False)

    def value(self):
        return self.currentData()

    def set_value(self, v):
        i = self.findData(v)
        if i >= 0:
            self.setCurrentIndex(i)

    def retranslate(self):
        for i in range(self.count()):
            self.setItemText(i, self.label_fn(self.itemData(i)))


def label(text_key: str, bold: bool = False) -> QLabel:
    lab = QLabel(tr(text_key))
    lab.setProperty("tr_key", text_key)
    if bold:
        f = lab.font()
        f.setBold(True)
        lab.setFont(f)
    return lab

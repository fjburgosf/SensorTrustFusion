"""Main window of the SensorTrust Fusion graphical interface.

The window keeps the experiment configuration (a plain mapping), lets the
tabs edit it and delegates every computation to the scientific engine::

    run_experiment()  ->  ExperimentManager(results).run(config)
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QToolBar,
    QWidget,
    QSizePolicy,
)

from ..experiments.benchmarks import examples
from ..experiments.config import load_config, normalize_config
from ..i18n_errors import translate_error
from ..i18n import get_language, set_language, tr
from ..utils.errors import SensorTrustError
from ..version import FULL_NAME, SUBTITLE_EN, SUBTITLE_ES, __version__
from .config_tabs import DegradationTab, ExperimentTab, FusionTab, SensorsTab, SystemTab, TrustTab
from .run_tabs import AnalysisTab, CompareTab, ExportTab, MonteCarloTab, ResultsTab
from .widgets import Worker, start_worker

EXAMPLE_ES = {
    "Example 1 - Nominal fusion": "Ejemplo 1 - Fusión nominal",
    "Example 2 - Abrupt bias": "Ejemplo 2 - Sesgo abrupto",
    "Example 3 - Progressive bias": "Ejemplo 3 - Sesgo progresivo",
    "Example 4 - Increasing noise": "Ejemplo 4 - Ruido creciente",
    "Example 5 - Drift": "Ejemplo 5 - Deriva",
    "Example 6 - Outliers": "Ejemplo 6 - Valores atípicos",
    "Example 7 - Stuck sensor": "Ejemplo 7 - Sensor congelado",
    "Example 8 - Sensor recovery": "Ejemplo 8 - Recuperación del sensor",
    "Example 9 - Multiple sensor degradation": "Ejemplo 9 - Degradación de múltiples sensores",
    "Example 10 - Method comparison": "Ejemplo 10 - Comparación de métodos",
    "Example 11 - Monte Carlo": "Ejemplo 11 - Monte Carlo",
    "Example 12 - Robustness envelope": "Ejemplo 12 - Envolvente de robustez",
    "ST-BENCH-08 - Dropout": "ST-BENCH-08 - Pérdida de muestras",
    "ST-BENCH-11 - Bias + noise": "ST-BENCH-11 - Sesgo + ruido",
    "ST-BENCH-12 - Multiple degradation profiles": "ST-BENCH-12 - Múltiples perfiles de degradación",
    "ST-BENCH-13 - Nonlinear pendulum": "ST-BENCH-13 - Péndulo no lineal",
    "ST-BENCH-14 - Thermal drift": "ST-BENCH-14 - Deriva térmica",
    "ST-BENCH-15 - Oscillator stuck sensor": "ST-BENCH-15 - Oscilador con sensor congelado",
    "ST-BENCH-16 - Dominant sensor progressive bias": "ST-BENCH-16 - Sesgo progresivo en el sensor dominante",
}

EXAMPLE_TEXT_ES = {  # example -> (experiment name, description) in Spanish
    "Example 1 - Nominal fusion": ("Ejemplo 1 - Fusión nominal (ST-BENCH-01)",
                                   "Los tres sensores operan nominalmente (caso de referencia y línea base de falsas "
                                   "alarmas)."),
    "Example 2 - Abrupt bias": ("Ejemplo 2 - Sesgo abrupto (ST-BENCH-02)",
                                "El sensor S2 recibe un sesgo abrupto de 1,0 m en t = 20 s."),
    "Example 3 - Progressive bias": ("Ejemplo 3 - Sesgo progresivo (ST-BENCH-03)",
                                     "El sesgo de S2 crece linealmente a 0,05 m/s desde t = 20 s (b = 2 m a los 60 s)."),
    "Example 4 - Increasing noise": ("Ejemplo 4 - Ruido creciente (ST-BENCH-04)",
                                     "La varianza del ruido de S2 crece linealmente de R0 a 50 R0 entre t = 20 s y "
                                     "t = 40 s."),
    "Example 5 - Drift": ("Ejemplo 5 - Deriva (ST-BENCH-05)",
                          "S2 desarrolla una deriva acumulativa (0,04 m/s, inicio exponencial) más una deriva de "
                          "caminata aleatoria (0,05 m/sqrt(s)) desde t = 20 s."),
    "Example 6 - Outliers": ("Ejemplo 6 - Valores atípicos (ST-BENCH-06)",
                             "S2 produce valores atípicos impulsivos (amplitud 3 m x U(0,5, 1,5), probabilidad 5 %) "
                             "entre 20 y 45 s."),
    "Example 7 - Stuck sensor": ("Ejemplo 7 - Sensor congelado (ST-BENCH-07)",
                                 "S2 se congela en su valor de t = 25 s hasta el final de la corrida."),
    "Example 8 - Sensor recovery": ("Ejemplo 8 - Recuperación del sensor (ST-BENCH-09)",
                                    "S2 recibe un sesgo de 1,5 m en t = 20 s que desaparece linealmente entre 35 y "
                                    "40 s."),
    "Example 9 - Multiple sensor degradation": ("Ejemplo 9 - Degradación de múltiples sensores (ST-BENCH-10)",
                                                "En t = 20 s, S2 recibe un sesgo de 1,0 m y la varianza del ruido de "
                                                "S3 se multiplica por 30."),
    "Example 10 - Method comparison": ("Ejemplo 10 - Comparación de métodos",
                                       "Todos los métodos de fusión sobre el mismo escenario de sesgo + ruido "
                                       "(ST-BENCH-11), incluidas las ablaciones de primer y segundo momento y la "
                                       "fusión a nivel de estimaciones."),
    "Example 11 - Monte Carlo": ("Ejemplo 11 - Monte Carlo de sesgo abrupto",
                                 "100 corridas Monte Carlo de un sesgo abrupto en S2 con inicio t_f ~ U(10, 30) s, "
                                 "sesgo b ~ U(0,5, 3) m y desviación del ruido de S2 ~ U(0,05, 0,5) m."),
    "Example 12 - Robustness envelope": ("Ejemplo 12 - Envolvente de robustez (sesgo abrupto)",
                                         "Mayor sesgo abrupto tolerable b* en S2 tal que el RMSE medio de 20 "
                                         "repeticiones se mantenga <= 0,05 m."),
    "ST-BENCH-08 - Dropout": ("ST-BENCH-08 - Pérdida de muestras",
                              "S2 pierde el 60 % de sus muestras entre 20 y 40 s; S3 sufre pérdida de paquetes en "
                              "ráfagas (Gilbert-Elliott, P(bueno->malo) = 0,02, ráfaga media de 50 muestras) desde "
                              "30 s."),
    "ST-BENCH-11 - Bias + noise": ("ST-BENCH-11 - Sesgo + ruido",
                                   "Fallo combinado de S2 desde t = 20 s: sesgo en rampa hasta 0,8 m en 5 s y varianza "
                                   "del ruido en rampa hasta 20 R0."),
    "ST-BENCH-12 - Multiple degradation profiles": ("ST-BENCH-12 - Múltiples perfiles de degradación",
                                                    "Cuatro sensores: S1 nominal; S2 sesgo sigmoide (1,0 m, transición "
                                                    "de 8 s) desde 15 s; S3 sesgo periódico (0,8 m, periodo 8 s) desde "
                                                    "25 s; S4 sesgo intermitente (1,2 m, activo 1,5 s / inactivo 3 s "
                                                    "en media) desde 35 s."),
    "ST-BENCH-13 - Nonlinear pendulum": ("ST-BENCH-13 - Péndulo no lineal",
                                         "Péndulo no lineal observado por un codificador de ángulo y dos sensores de "
                                         "posición horizontal h(x) = L sen(theta); el sensor P2 recibe un sesgo de "
                                         "0,15 m a los 15 s. Justifica el uso de EKF y UKF (dinámica y medición no "
                                         "lineales)."),
    "ST-BENCH-14 - Thermal drift": ("ST-BENCH-14 - Deriva térmica",
                                    "Sistema térmico de dos nodos; tres sensores de temperatura del objeto con "
                                    "resolución de 0,1 degC; el sensor T2 deriva a 0,02 degC/s desde 60 s."),
    "ST-BENCH-15 - Oscillator stuck sensor": ("ST-BENCH-15 - Oscilador con sensor congelado",
                                              "Oscilador masa-resorte-amortiguador forzado; sensores de posición "
                                              "(100 Hz) y un sensor de velocidad (50 Hz); el sensor de posición X2 se "
                                              "congela a los 20 s."),
    "ST-BENCH-16 - Dominant sensor progressive bias": ("ST-BENCH-16 - Sesgo progresivo en el sensor dominante",
                                                       "El sensor más preciso, S2 (0,05 m, que domina la estimación "
                                                       "fusionada), desarrolla un sesgo progresivo de 0,02 m/s desde "
                                                       "t = 15 s. Incluye la ablación de SensorTrust sin la referencia "
                                                       "de consenso del primer momento."),
}


def localized_example(name: str, cfg: dict, lang: str) -> dict:
    """Example configuration with its name and description in the interface language."""
    cfg = copy.deepcopy(cfg)
    if lang == "es" and name in EXAMPLE_TEXT_ES:
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"], cfg["experiment"]["description"] = EXAMPLE_TEXT_ES[name]
    return cfg


TAB_KEYS = ["System", "Sensors", "Degradation", "Fusion", "Trust", "Experiment", "Monte Carlo", "Analysis",
            "Results", "Compare", "Export"]


class MainWindow(QMainWindow):
    def __init__(self, lang: str = "en"):
        super().__init__()
        set_language(lang)
        self.examples = examples()
        self.last_result = None
        self.last_study = None
        self._thread = None
        self._worker = None
        self.resize(1360, 860)
        self._build_toolbar()
        self.tabs = QTabWidget()
        self.system_tab = SystemTab(self)
        self.sensors_tab = SensorsTab(self)
        self.degradation_tab = DegradationTab(self)
        self.fusion_tab = FusionTab(self)
        self.trust_tab = TrustTab(self)
        self.experiment_tab = ExperimentTab(self)
        self.mc_tab = MonteCarloTab(self)
        self.analysis_tab = AnalysisTab(self)
        self.results_tab = ResultsTab(self)
        self.compare_tab = CompareTab(self)
        self.export_tab = ExportTab(self)
        self.all_tabs = [self.system_tab, self.sensors_tab, self.degradation_tab, self.fusion_tab, self.trust_tab,
                         self.experiment_tab, self.mc_tab, self.analysis_tab, self.results_tab, self.compare_tab,
                         self.export_tab]
        for t in self.all_tabs:
            self.tabs.addTab(t, "")
        from PySide6.QtWidgets import QBoxLayout, QLabel as _QLabel
        for name in ("system_tab", "sensors_tab", "degradation_tab", "fusion_tab", "trust_tab", "experiment_tab",
                     "mc_tab", "analysis_tab", "results_tab", "compare_tab", "export_tab"):
            tab = getattr(self, name)
            lay = tab.layout()
            if isinstance(lay, QBoxLayout):
                lab = _QLabel()
                lab.setWordWrap(True)
                from PySide6.QtWidgets import QSizePolicy as _SP
                lab.setSizePolicy(_SP.Preferred, _SP.Maximum)
                lab.setStyleSheet("background:#eef4fc; border:1px solid #c9dcf5; border-radius:4px; padding:6px; "
                                  "color:#0d366b")
                lay.insertWidget(0, lab, 0)
                for k in range(1, lay.count()):
                    if lay.stretch(k) == 0 and lay.itemAt(k).widget() is not None:
                        lay.setStretch(k, 1)
                        break
                tab._help_label = lab
        self.setCentralWidget(self.tabs)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self.version_label = QLabel(f"{FULL_NAME}")
        self.statusBar().addPermanentWidget(self.version_label)
        self.system_tab.model_changed.connect(self._on_model_changed)
        self.sensors_tab.sensors_changed.connect(self.degradation_tab.refresh_sensors)
        self.tabs.currentChanged.connect(self._on_tab)
        self.current_example = "Example 2 - Abrupt bias"
        self.load_config(localized_example(self.current_example, self.examples[self.current_example], lang))
        self.retranslate()

    # ------------------------------------------------------------------
    def _build_toolbar(self):
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        self.title_label = QLabel()
        self.title_label.setObjectName("title")
        tb.addWidget(self.title_label)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.ex_label = QLabel()
        tb.addWidget(self.ex_label)
        self.ex_combo = QComboBox()
        self.ex_combo.setMinimumWidth(330)
        tb.addWidget(self.ex_combo)
        self.b_load = QPushButton()
        tb.addWidget(self.b_load)
        tb.addSeparator()
        self.b_run = QPushButton()
        self.b_run.setObjectName("primary")
        tb.addWidget(self.b_run)
        self.b_cancel = QPushButton()
        self.b_cancel.setEnabled(False)
        tb.addWidget(self.b_cancel)
        tb.addSeparator()
        self.b_lang = QPushButton()
        self.b_lang.setObjectName("lang")
        self.b_lang.setToolTip("Español / English")
        tb.addWidget(self.b_lang)
        mb = self.menuBar()
        self.m_file = mb.addMenu("")
        self.a_open = self.m_file.addAction("")
        self.a_save = self.m_file.addAction("")
        self.m_file.addSeparator()
        self.a_quit = self.m_file.addAction("")
        self.m_help = mb.addMenu("")
        self.a_guide = self.m_help.addAction("")
        self.a_user_manual = self.m_help.addAction("")
        self.a_tech_manual = self.m_help.addAction("")
        self.m_help.addSeparator()
        self.a_about = self.m_help.addAction("")
        self.a_guide.triggered.connect(self._quick_guide)
        self.a_user_manual.triggered.connect(lambda: self._open_manual("Manual_de_Usuario_SensorTrust_Fusion.docx"))
        self.a_tech_manual.triggered.connect(lambda: self._open_manual("Manual_Tecnico_SensorTrust_Fusion.docx"))
        self.a_open.triggered.connect(self._open_config)
        self.a_save.triggered.connect(self._save_config)
        self.a_quit.triggered.connect(self.close)
        self.a_about.triggered.connect(self._about)
        self.b_load.clicked.connect(self._load_example)
        self.b_run.clicked.connect(self.run_experiment)
        self.b_cancel.clicked.connect(self._cancel)
        self.b_lang.clicked.connect(self.toggle_language)

    # ------------------------------------------------------------------
    def load_config(self, cfg: dict):
        cfg = normalize_config(copy.deepcopy(cfg))
        self.cfg = cfg
        self.system_tab.load(cfg)
        self.sensors_tab.load(cfg)
        self.degradation_tab.load(cfg)
        self.trust_tab.load(cfg)
        self.fusion_tab.load(cfg)
        self.experiment_tab.load(cfg)
        self.mc_tab.load(cfg)
        self.analysis_tab.load(cfg)
        self.compare_tab.populate(cfg["methods"])
        self.experiment_tab.show_summary(cfg)
        self._status(tr("Ready"))

    def collect(self, validate: bool = True) -> dict | None:
        """Gather the configuration from all tabs (and validate it)."""
        cfg = copy.deepcopy(self.cfg)
        try:
            self.system_tab.store(cfg)
            self.sensors_tab.store(cfg)
            self.experiment_tab.store(cfg)
            self.fusion_tab.store(cfg)
            self.mc_tab.store(cfg)
            if not cfg["methods"]:
                raise SensorTrustError("Select at least one fusion method.")
            if validate:
                cfg = normalize_config(cfg)
                from ..simulation.scenario import build_model, _prior
                from ..sensors.sensor import VirtualSensor
                m = build_model(cfg)
                _prior(cfg, m)
                for i, s in enumerate(cfg["sensors"]):
                    VirtualSensor(s, i, m, cfg["simulation"]["fs"])
                from ..fusion import create_method
                for spec in cfg["methods"]:
                    create_method(spec)
        except (SensorTrustError, ValueError, KeyError, TypeError) as exc:
            if not validate:
                return cfg
            QMessageBox.warning(self, tr("Configuration error"), translate_error(exc))
            return None
        self.cfg = cfg
        self.experiment_tab.show_summary(cfg)
        return cfg

    def validate(self):
        if self.collect() is not None:
            QMessageBox.information(self, FULL_NAME, tr("Configuration is valid."))

    # ------------------------------------------------------------------
    def start_task(self, fn, on_done, *args, cancellable: bool = False, **kwargs):
        if self._thread is not None:
            return
        self._worker = Worker(fn, *args, **kwargs)
        if cancellable:
            self._worker.kwargs["cancel"] = self._worker.is_cancelled
        self._on_done = on_done
        # bound methods of this (GUI-thread) QObject -> queued connections, handlers run in the GUI thread
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._task_done)
        self._worker.failed.connect(self._task_failed)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.b_run.setEnabled(False)
        self.b_cancel.setEnabled(cancellable)
        self._status(tr("Running..."))
        self._thread = start_worker(self, self._worker)

    @Slot(str, float)
    def _on_progress(self, msg, frac):
        self.progress.setValue(int(100 * frac))
        self._status(msg)

    @Slot(object)
    def _task_done(self, res):
        on_done = self._on_done
        self._reset_task()
        on_done(res)
        if not self.status_label.text().startswith("ST_"):
            self._status(tr("Finished"))

    @Slot(str, str)
    def _task_failed(self, msg, tb):
        self._reset_task()
        if msg.startswith("InterruptedError"):  # the user pressed Cancel
            self._status(tr("Cancelled by the user"))
            return
        self._status(tr("Error"))
        box = QMessageBox(QMessageBox.Critical, tr("Error"), translate_error(msg), parent=self)
        box.setDetailedText(tb)
        box.exec()

    def _reset_task(self):
        self._thread = None
        self._worker = None
        self.progress.setVisible(False)
        self.b_run.setEnabled(True)
        self.b_cancel.setEnabled(False)

    def closeEvent(self, event):
        """Closing while a computation runs: ask, cancel the task and wait for it to stop cleanly."""
        if self._thread is not None:
            ans = QMessageBox.question(self, tr("Exit"), tr("A computation is running. Stop it and exit?"),
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                event.ignore()
                return
            if self._worker is not None:
                self._worker.cancel()
            self._status(tr("Stopping..."))
            th = self._thread
            try:
                th.quit()
                th.wait(180000)
            except RuntimeError:  # thread object already deleted
                pass
        event.accept()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def results_root(self) -> str | None:
        """Results folder of the Experiment tab, created and checked for writing (None + message if impossible)."""
        import tempfile
        from pathlib import Path
        root = self.experiment_tab.res_dir.text().strip() or "results"
        try:
            Path(root).mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=root, prefix=".st_write_check_"):
                pass
        except OSError as exc:
            QMessageBox.warning(self, tr("Error"), translate_error(
                f"Cannot write to the results directory {root}: {exc}"))
            return None
        return root

    def start_experiment(self, cfg, on_done=None):
        from ..experiments.manager import ExperimentManager
        root = self.results_root()
        if root is None:
            return

        def done(res):
            self.on_experiment_done(res)
            if on_done:
                on_done(res)
        self.start_task(ExperimentManager(root).run, done, cfg)

    def run_experiment(self):
        cfg = self.collect()
        if cfg is not None:
            self.start_experiment(cfg)

    def on_experiment_done(self, res):
        self.last_result = res
        self.last_study = None
        self.results_tab.set_result(res)
        self.export_tab.refresh()
        if self.tabs.currentWidget() is not self.compare_tab:
            self.tabs.setCurrentWidget(self.results_tab)
        self._status(f"{res.experiment_id}: {res.output_dir}")

    # ------------------------------------------------------------------
    def _on_model_changed(self):
        self.sensors_tab.refresh_measure_options()

    def _on_tab(self, i):
        w = self.tabs.widget(i)
        if w is self.experiment_tab:
            self.collect(validate=False)
            self.experiment_tab.show_summary(self.cfg)
        elif w is self.degradation_tab:
            self.degradation_tab.refresh_sensors()
        elif w in (self.mc_tab, self.analysis_tab, self.compare_tab):
            cfg = self.collect(validate=False)
            if w is self.compare_tab:
                self.compare_tab.populate(cfg["methods"])
            else:
                paths = __import__("sensortrust.gui.run_tabs", fromlist=["parameter_paths"]).parameter_paths(cfg)
                self.mc_tab.params.set_paths(paths)
                self.analysis_tab.s_params.set_paths(paths)

    def _load_example(self):
        name = self.ex_combo.currentData()
        if name:
            self.current_example = name
            self.load_config(localized_example(name, self.examples[name], get_language()))
            self.tabs.setCurrentWidget(self.system_tab)

    def _open_config(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Open configuration..."), ".", "YAML/JSON (*.yaml *.yml *.json)")
        if p:
            try:
                self.load_config(load_config(p))
            except SensorTrustError as exc:
                QMessageBox.warning(self, tr("Configuration error"), translate_error(exc))

    def _save_config(self):
        cfg = self.collect()
        if cfg is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, tr("Save configuration..."), "experiment.yaml",
                                           "YAML (*.yaml);;JSON (*.json)")
        if p:
            from ..experiments.config import save_config
            save_config(cfg, p)

    def _quick_guide(self):
        es = get_language() == "es"
        steps_es = ["<b>Sistema</b>: elija el modelo dinámico, la duración y la frecuencia de muestreo.",
                    "<b>Sensores</b>: defina los sensores virtuales, qué miden y su ruido nominal.",
                    "<b>Degradación</b>: agregue fallos a los sensores y su perfil temporal.",
                    "<b>Fusión</b>: marque los métodos que desea comparar.",
                    "<b>Confianza</b>: ajuste el estimador SensorTrust (opcional).",
                    "<b>Experimento</b>: fije la semilla y pulse <b>Ejecutar experimento</b>.",
                    "<b>Resultados</b>: revise figuras y métricas; <b>Comparar</b> contrasta métodos.",
                    "<b>Monte Carlo</b> y <b>Análisis</b>: estudios estadísticos, de robustez y de sensibilidad.",
                    "<b>Exportar</b>: guarde figuras, datos y configuración y verifique la reproducibilidad."]
        steps_en = ["<b>System</b>: choose the dynamic model, the duration and the sampling frequency.",
                    "<b>Sensors</b>: define the virtual sensors, what they measure and their nominal noise.",
                    "<b>Degradation</b>: add faults to the sensors and their temporal profile.",
                    "<b>Fusion</b>: check the methods you want to compare.",
                    "<b>Trust</b>: tune the SensorTrust estimator (optional).",
                    "<b>Experiment</b>: set the seed and press <b>Run Experiment</b>.",
                    "<b>Results</b>: inspect figures and metrics; <b>Compare</b> contrasts methods.",
                    "<b>Monte Carlo</b> and <b>Analysis</b>: statistical, robustness and sensitivity studies.",
                    "<b>Export</b>: save figures, data and configuration and verify reproducibility."]
        tip = ("Consejo: cargue un ejemplo científico desde la barra superior para empezar. Pase el cursor sobre "
               "cualquier control para ver su explicación." if es else
               "Tip: load a scientific example from the toolbar to start. Hover over any control to read its "
               "explanation.")
        html = "<ol>" + "".join(f"<li>{x}</li>" for x in (steps_es if es else steps_en)) + f"</ol><p>{tip}</p>"
        QMessageBox.information(self, tr("Quick guide"), html)

    def _open_manual(self, filename: str):
        import sys
        from pathlib import Path

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        roots = [Path(getattr(sys, "_MEIPASS", "")) / "Manuales", Path(__file__).resolve().parents[3] / "Manuales",
                 Path.cwd() / "Manuales"]
        for r in roots:
            f = r / filename
            if f.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(f)))
                self._status(str(f))
                return
        QMessageBox.information(self, tr("About"), tr("The manual is in the Manuales folder of the repository:") +
                                "\nhttps://github.com/fjburgosf/SensorTrustFusion/tree/main/Manuales")

    def _about(self):
        es = get_language() == "es"
        txt = (f"<h3>{FULL_NAME}</h3><p><i>{SUBTITLE_EN}</i><br><i>{SUBTITLE_ES}</i></p>"
               f"<p>{'Versión' if es else 'Version'} {__version__}</p>"
               f"<p>{'Autor' if es else 'Author'}: Francisco Javier Burgos Flórez<br>"
               f"{'Contacto' if es else 'Contact'}: fjburgosf@gmail.com</p>"
               f"<p>{tr('No hardware required: all measurements come from virtual sensors.')}</p>")
        QMessageBox.about(self, tr("About"), txt)

    def toggle_language(self):
        new = "en" if get_language() == "es" else "es"
        # the name and description of an unmodified example follow the language of the interface
        ex = getattr(self, "current_example", None)
        if ex in self.examples:
            et = self.experiment_tab
            cur_lang = get_language()
            now = localized_example(ex, self.examples[ex], cur_lang)["experiment"]
            if et.name.text() == now.get("name") and et.desc.toPlainText() == now.get("description", ""):
                nxt = localized_example(ex, self.examples[ex], new)["experiment"]
                et.name.setText(nxt.get("name", ""))
                et.desc.setPlainText(nxt.get("description", ""))
                self.cfg.setdefault("experiment", {}).update({"name": nxt.get("name", ""),
                                                             "description": nxt.get("description", "")})
        set_language(new)
        self.retranslate()

    def _status(self, text):
        from ..i18n import ES
        inv = {v: k for k, v in ES.items()}
        key = inv.get(text, text)
        self._status_key = key if key in ES else None
        self.status_label.setText(text)

    # ------------------------------------------------------------------
    def retranslate(self):
        es = get_language() == "es"
        self.setWindowTitle(f"{FULL_NAME} - {SUBTITLE_ES if es else SUBTITLE_EN}")
        self.title_label.setText(f"<b>SensorTrust Fusion</b> <span style='color:#52514e'>v{__version__}</span>")
        self.ex_label.setText(tr("Scientific example:") + " ")
        cur = self.ex_combo.currentData() or getattr(self, "current_example", None)
        self.ex_combo.blockSignals(True)
        self.ex_combo.clear()
        for name in self.examples:
            self.ex_combo.addItem(EXAMPLE_ES.get(name, name) if es else name, name)
        i = self.ex_combo.findData(cur)
        self.ex_combo.setCurrentIndex(max(i, 0))
        self.ex_combo.blockSignals(False)
        self.b_load.setText(tr("Load"))
        self.m_file.setTitle("&Archivo" if es else "&File")
        self.m_help.setTitle("A&yuda" if es else "&Help")
        self.a_open.setText(tr("Open configuration..."))
        self.a_save.setText(tr("Save configuration..."))
        self.a_quit.setText("Salir" if es else "Exit")
        self.a_about.setText(tr("About"))
        self.a_guide.setText(tr("Quick guide"))
        self.a_user_manual.setText(tr("User manual"))
        self.a_tech_manual.setText(tr("Technical manual"))
        self.b_run.setText(tr("Run Experiment"))
        self.b_cancel.setText(tr("Cancel"))
        self.b_lang.setText("EN  |  [ES]" if es else "[EN]  |  ES")
        self.b_lang.setToolTip("Switch to English" if es else "Cambiar a español")
        for i, k in enumerate(TAB_KEYS):
            self.tabs.setTabText(i, tr(k))
        for w in self.findChildren(QWidget):
            key = w.property("tr_key")
            if key:
                if hasattr(w, "setTitle"):
                    w.setTitle(tr(key))
                elif hasattr(w, "setText"):
                    w.setText(tr(key))
        for t in self.all_tabs:
            t.retranslate()
        from .widgets import PlotCanvas
        for c in self.findChildren(PlotCanvas):
            c.retranslate()
        from .tooltips import apply_tooltips
        apply_tooltips(self, get_language())
        self._status(tr(self._status_key) if getattr(self, "_status_key", None) else self.status_label.text())

"""Explanations of the interface: a tooltip for every control and a short description of every tab.

Tooltips are keyed by ``<tab attribute>.<widget attribute>`` of the main
window; :func:`apply_tooltips` is called whenever the language changes.
"""

from __future__ import annotations

TIPS: dict[str, tuple[str, str]] = {
    # toolbar
    "ex_combo": ("Scientific examples and benchmarks that can be loaded as a starting configuration.",
                 "Ejemplos científicos y benchmarks que pueden cargarse como configuración inicial."),
    "b_load": ("Load the selected example into all tabs (replaces the current configuration).",
               "Cargar el ejemplo seleccionado en todas las pestañas (reemplaza la configuración actual)."),
    "b_run": ("Simulate the scenario and run every selected method on the same measurements.",
              "Simular el escenario y ejecutar todos los métodos seleccionados sobre las mismas mediciones."),
    "b_cancel": ("Stop the running Monte Carlo study or analysis.", "Detener el estudio Monte Carlo o el análisis en curso."),
    # system
    "system_tab.model_combo": ("Dynamic model that generates the true state (ground truth).",
                               "Modelo dinámico que genera el estado verdadero (valor verdadero)."),
    "system_tab.custom_edit": ("Matrices of your own linear model in YAML (discrete A, B, Q or continuous Ac, Bc, Qc).",
                               "Matrices de su propio modelo lineal en YAML (discretas A, B, Q o continuas Ac, Bc, Qc)."),
    "system_tab.duration": ("Length of the simulated experiment.", "Duración del experimento simulado."),
    "system_tab.fs": ("Sampling frequency of the simulation (the fastest sensor rate).",
                      "Frecuencia de muestreo de la simulación (la frecuencia del sensor más rápido)."),
    "system_tab.x0_table": ("True initial state and its uncertainty, and the initial estimate (prior) of the filters.",
                            "Estado inicial verdadero y su incertidumbre, y estimación inicial (a priori) de los filtros."),
    "system_tab.sample_x0": ("If checked, the true initial state is drawn at random from N(x0, P0) (seeded).",
                             "Si se marca, el estado inicial verdadero se sortea de N(x0, P0) (con la semilla)."),
    "system_tab.target": ("State variable on which the estimation metrics (RMSE, MAE...) are computed.",
                          "Variable de estado sobre la que se calculan las métricas de estimación (RMSE, MAE...)."),
    # sensors
    "sensors_tab.table": ("Virtual sensors of the experiment; select a row to edit it below.",
                          "Sensores virtuales del experimento; seleccione una fila para editarla abajo."),
    "sensors_tab.b_add": ("Add a new virtual sensor with Gaussian noise.", "Agregar un nuevo sensor virtual con ruido gaussiano."),
    "sensors_tab.b_dup": ("Duplicate the selected sensor (redundant sensor).", "Duplicar el sensor seleccionado (sensor redundante)."),
    "sensors_tab.b_del": ("Remove the selected sensor.", "Eliminar el sensor seleccionado."),
    "sensors_tab.name": ("Unique name of the sensor (used in tables, figures and files).",
                         "Nombre único del sensor (usado en tablas, figuras y archivos)."),
    "sensors_tab.measured": ("State variable or nonlinear function h(x) measured by the sensor.",
                             "Variable de estado o función no lineal h(x) que mide el sensor."),
    "sensors_tab.rate": ("Sampling rate of the sensor (must not exceed the simulation frequency).",
                         "Frecuencia de muestreo del sensor (no puede superar la de la simulación)."),
    "sensors_tab.noise_type": ("Statistical model of the nominal measurement noise.",
                               "Modelo estadístico del ruido nominal de medición."),
    # degradation
    "degradation_tab.sensor": ("Sensor whose degradations are edited.", "Sensor cuyas degradaciones se editan."),
    "degradation_tab.list": ("Degradations of the selected sensor; several can be combined.",
                             "Degradaciones del sensor seleccionado; pueden combinarse varias."),
    "degradation_tab.b_add": ("Add a degradation (bias by default) to the selected sensor.",
                              "Agregar una degradación (sesgo por defecto) al sensor seleccionado."),
    "degradation_tab.b_del": ("Remove the selected degradation.", "Eliminar la degradación seleccionada."),
    "degradation_tab.type": ("Type of degradation or fault.", "Tipo de degradación o fallo."),
    "degradation_tab.ptype": ("Temporal profile: how the degradation appears (and disappears) in time.",
                              "Perfil temporal: cómo aparece (y desaparece) la degradación en el tiempo."),
    "degradation_tab.b_prev": ("Recompute the preview with the real degradation engine (noise-free signal).",
                               "Recalcular la vista previa con el motor real de degradación (señal sin ruido)."),
    # fusion
    "fusion_tab.table": ("Check the methods to run; all of them process exactly the same measurements.",
                         "Marque los métodos a ejecutar; todos procesan exactamente las mismas mediciones."),
    "fusion_tab.b_all": ("Select every available method.", "Seleccionar todos los métodos disponibles."),
    "fusion_tab.b_none": ("Clear the selection.", "Quitar la selección."),
    "fusion_tab.b_rec": ("Select the recommended comparison set (static, KF, adaptive KF, NIS gate, SensorTrust, oracle).",
                         "Seleccionar el conjunto de comparación recomendado (estáticos, KF, KF adaptativo, compuerta "
                         "NIS, SensorTrust, oráculo)."),
    "fusion_tab.weights": ("Weights of the fixed-weight average, one per sensor, separated by commas.",
                           "Pesos del promedio con pesos fijos, uno por sensor, separados por coma."),
    # trust
    "trust_tab.b_def": ("Restore the reference values of all trust, detector and gate parameters.",
                        "Restaurar los valores de referencia de todos los parámetros de confianza, detector y compuerta."),
    # experiment
    "experiment_tab.name": ("Name of the experiment (stored in the manifest).", "Nombre del experimento (se guarda en el manifiesto)."),
    "experiment_tab.desc": ("Free description of the experiment.", "Descripción libre del experimento."),
    "experiment_tab.seed": ("Random seed: the same configuration and seed reproduce exactly the same results.",
                            "Semilla aleatoria: la misma configuración y semilla reproducen exactamente los mismos resultados."),
    "experiment_tab.fmt_png": ("Save the figures as PNG images.", "Guardar las figuras como imágenes PNG."),
    "experiment_tab.fmt_svg": ("Save the figures as SVG vector graphics.", "Guardar las figuras como gráficos vectoriales SVG."),
    "experiment_tab.fmt_pdf": ("Save the figures as PDF files.", "Guardar las figuras como archivos PDF."),
    "experiment_tab.save_ds": ("Save the complete simulated dataset (truth, measurements, fault labels) as CSV.",
                               "Guardar el conjunto de datos simulado completo (valor verdadero, mediciones, etiquetas de "
                               "fallo) en CSV."),
    "experiment_tab.res_dir": ("Folder where each experiment creates its results directory ST_EXP_<year>_<number>.",
                               "Carpeta donde cada experimento crea su directorio de resultados ST_EXP_<año>_<número>."),
    "experiment_tab.b_browse": ("Choose the results folder.", "Elegir la carpeta de resultados."),
    "experiment_tab.b_validate": ("Check the complete configuration without running it.",
                                  "Comprobar la configuración completa sin ejecutarla."),
    "experiment_tab.b_run": ("Simulate the scenario and run every selected method on the same measurements.",
                             "Simular el escenario y ejecutar todos los métodos seleccionados sobre las mismas mediciones."),
    "experiment_tab.summary": ("Complete configuration (YAML) that will be executed and saved as config.yaml.",
                               "Configuración completa (YAML) que se ejecutará y se guardará como config.yaml."),
    # monte carlo
    "mc_tab.runs": ("Number of independent Monte Carlo runs.", "Número de corridas Monte Carlo independientes."),
    "mc_tab.seed": ("Master seed from which the seed of every run is derived.",
                    "Semilla maestra de la que se deriva la semilla de cada corrida."),
    "mc_tab.sampler": ("How the randomized parameters are sampled: random, Latin hypercube or Sobol sequence.",
                       "Cómo se muestrean los parámetros aleatorizados: aleatorio, hipercubo latino o secuencia de Sobol."),
    "mc_tab.jobs": ("Worker processes used in parallel (0 = all processor cores); results do not depend on it.",
                    "Procesos usados en paralelo (0 = todos los núcleos); los resultados no dependen de este valor."),
    "mc_tab.ref": ("Method against which every other method is compared with paired Wilcoxon tests.",
                   "Método contra el que se compara cada método con pruebas pareadas de Wilcoxon."),
    "mc_tab.params": ("Parameters that change randomly from run to run, with their distribution and range.",
                      "Parámetros que cambian aleatoriamente de una corrida a otra, con su distribución y rango."),
    "mc_tab.b_add": ("Add a randomized parameter.", "Agregar un parámetro aleatorizado."),
    "mc_tab.b_del": ("Remove the selected randomized parameter.", "Eliminar el parámetro aleatorizado seleccionado."),
    "mc_tab.b_run": ("Run the Monte Carlo study in the background (can be cancelled).",
                     "Ejecutar el estudio Monte Carlo en segundo plano (puede cancelarse)."),
    "mc_tab.fig": ("Type of statistical figure.", "Tipo de figura estadística."),
    "mc_tab.metric": ("Metric shown in the figure.", "Métrica mostrada en la figura."),
    "mc_tab.summary": ("Descriptive statistics of every metric and method over all runs.",
                       "Estadística descriptiva de cada métrica y método sobre todas las corridas."),
    "mc_tab.tests": ("Paired Wilcoxon signed-rank tests against the reference method, with Holm correction.",
                     "Pruebas pareadas de rangos con signo de Wilcoxon frente al método de referencia, con corrección de Holm."),
    # analysis
    "analysis_tab.study": ("Type of study: robustness envelope, global sensitivity analysis or 2-D parametric sweep.",
                           "Tipo de estudio: envolvente de robustez, análisis de sensibilidad global o barrido paramétrico 2-D."),
    "analysis_tab.r_path": ("Parameter whose severity is increased (usually a degradation magnitude).",
                            "Parámetro cuya severidad se aumenta (normalmente la magnitud de una degradación)."),
    "analysis_tab.r_label": ("Axis label of the parameter in the figures (optional).",
                             "Etiqueta del parámetro en los ejes de las figuras (opcional)."),
    "analysis_tab.r_values": ("Values of the parameter to evaluate, separated by commas.",
                              "Valores del parámetro a evaluar, separados por coma."),
    "analysis_tab.r_metric": ("Metric compared with the specification.", "Métrica que se compara con la especificación."),
    "analysis_tab.r_spec": ("Specification: the method complies while the metric is below this value.",
                            "Especificación: el método cumple mientras la métrica sea menor que este valor."),
    "analysis_tab.r_crit": ("Compliance criterion: mean of the repetitions or success rate.",
                            "Criterio de cumplimiento: media de las repeticiones o tasa de éxito."),
    "analysis_tab.r_reps": ("Independent repetitions (different seeds) per tested value.",
                            "Repeticiones independientes (semillas distintas) por valor probado."),
    "analysis_tab.r_map": ("Also compute the success rate over a grid of two parameters (robustness map).",
                           "Calcular además la tasa de éxito sobre una rejilla de dos parámetros (mapa de robustez)."),
    "analysis_tab.r_path2": ("Second parameter of the robustness map or of the parametric sweep.",
                             "Segundo parámetro del mapa de robustez o del barrido paramétrico."),
    "analysis_tab.r_values2": ("Values of the second parameter, separated by commas.",
                               "Valores del segundo parámetro, separados por coma."),
    "analysis_tab.s_method": ("Sensitivity method: one-at-a-time, Morris screening or Sobol variance-based indices.",
                              "Método de sensibilidad: uno a la vez, cribado de Morris o índices de Sobol basados en varianza."),
    "analysis_tab.s_samples": ("OAT: levels per parameter; Morris: trajectories; Sobol: base sample size N.",
                               "OAT: niveles por parámetro; Morris: trayectorias; Sobol: tamaño base de muestra N."),
    "analysis_tab.s_out": ("Method whose metric is the output of the sensitivity analysis.",
                           "Método cuya métrica es la salida del análisis de sensibilidad."),
    "analysis_tab.s_metric": ("Metric used as output of the sensitivity analysis.",
                              "Métrica usada como salida del análisis de sensibilidad."),
    "analysis_tab.s_seed": ("Common random numbers (same noise for every evaluation) or a new seed per evaluation.",
                            "Números aleatorios comunes (el mismo ruido en cada evaluación) o una semilla nueva por evaluación."),
    "analysis_tab.s_params": ("Parameters of the sensitivity analysis with their ranges.",
                              "Parámetros del análisis de sensibilidad con sus rangos."),
    "analysis_tab.s_add": ("Add a parameter to the sensitivity analysis.", "Agregar un parámetro al análisis de sensibilidad."),
    "analysis_tab.jobs": ("Worker processes used in parallel (0 = all processor cores).",
                          "Procesos usados en paralelo (0 = todos los núcleos)."),
    "analysis_tab.b_run": ("Run the selected study in the background (can be cancelled).",
                           "Ejecutar el estudio seleccionado en segundo plano (puede cancelarse)."),
    "analysis_tab.fig": ("Figure of the study.", "Figura del estudio."),
    "analysis_tab.table": ("Numerical results of the study.", "Resultados numéricos del estudio."),
    # results
    "results_tab.fig_combo": ("Scientific figure to display.", "Figura científica a mostrar."),
    "results_tab.method_combo": ("Method shown in method-specific figures and in the per-sensor table.",
                                 "Método mostrado en las figuras por método y en la tabla por sensor."),
    "results_tab.sensor_combo": ("Show all sensors or only one in sensor-specific figures.",
                                 "Mostrar todos los sensores o solo uno en las figuras por sensor."),
    "results_tab.table": ("Metrics of every method (one row per method).", "Métricas de cada método (una fila por método)."),
    "results_tab.sensor_table": ("Trust, weight and detection metrics of each sensor for the selected method.",
                                 "Métricas de confianza, pesos y detección de cada sensor para el método seleccionado."),
    # compare
    "compare_tab.list": ("Check the methods to compare; they are run on exactly the same measurements.",
                         "Marque los métodos a comparar; se ejecutan exactamente sobre las mismas mediciones."),
    "compare_tab.b_run": ("Run the checked methods on the current scenario and compare them.",
                          "Ejecutar los métodos marcados sobre el escenario actual y compararlos."),
    "compare_tab.fig": ("Comparison figure.", "Figura de comparación."),
    "compare_tab.table": ("Metrics of the compared methods.", "Métricas de los métodos comparados."),
    # export
    "export_tab.path": ("Folder with all the files of the last experiment or study.",
                        "Carpeta con todos los archivos del último experimento o estudio."),
    "export_tab.b_open": ("Open the results folder in the file explorer.", "Abrir la carpeta de resultados en el explorador de archivos."),
    "export_tab.b_figs": ("Save all figures of the last experiment in the formats checked below.",
                          "Guardar todas las figuras del último experimento en los formatos marcados abajo."),
    "export_tab.b_cfg": ("Save the current configuration as YAML or JSON (it can be opened again later).",
                         "Guardar la configuración actual en YAML o JSON (puede abrirse de nuevo después)."),
    "export_tab.b_ds": ("Save the simulated dataset (CSV and metadata) to reuse it in other tools.",
                        "Guardar el conjunto de datos simulado (CSV y metadatos) para reutilizarlo en otras herramientas."),
    "export_tab.b_tab": ("Save the metrics table as CSV.", "Guardar la tabla de métricas en CSV."),
    "export_tab.b_ver": ("Regenerate the scenario from the configuration and seed and compare its fingerprint (SHA-256).",
                         "Regenerar el escenario a partir de la configuración y la semilla y comparar su huella (SHA-256)."),
    "export_tab.f_png": ("Export figures as PNG.", "Exportar figuras en PNG."),
    "export_tab.f_svg": ("Export figures as SVG.", "Exportar figuras en SVG."),
    "export_tab.f_pdf": ("Export figures as PDF.", "Exportar figuras en PDF."),
    "export_tab.manifest": ("Reproducibility manifest: version, seed, SHA-256 fingerprints and metric summary.",
                            "Manifiesto de reproducibilidad: versión, semilla, huellas SHA-256 y resumen de métricas."),
    "export_tab.i_time": ("Name of the time column of the external file.", "Nombre de la columna de tiempo del archivo externo."),
    "export_tab.i_truth": ("Ground-truth columns (one per state), optional; without them only consistency metrics are computed.",
                           "Columnas de valor verdadero (una por estado), opcionales; sin ellas solo se calculan métricas "
                           "de consistencia."),
    "export_tab.i_btn": ("Import a CSV/TXT/JSON file whose sensor columns are named like the configured sensors and "
                         "run the selected methods on it.",
                         "Importar un archivo CSV/TXT/JSON cuyas columnas de sensores se llamen como los sensores "
                         "configurados y ejecutar sobre él los métodos seleccionados."),
    "export_tab.log": ("Messages of the export and verification actions.", "Mensajes de las acciones de exportación y verificación."),
}

TAB_HELP: dict[str, tuple[str, str]] = {
    "system_tab": ("Choose the dynamic system that generates the true state, the simulated time horizon, the initial "
                   "state and the variable on which the estimation error is measured.",
                   "Elija el sistema dinámico que genera el estado verdadero, el horizonte de simulación, el estado "
                   "inicial y la variable sobre la que se mide el error de estimación."),
    "sensors_tab": ("Define any number of virtual sensors: what they measure, their rate, nominal noise and "
                    "characteristics. The nominal values are what the estimators assume.",
                    "Defina cualquier número de sensores virtuales: qué miden, su frecuencia, su ruido nominal y sus "
                    "características. Los valores nominales son los que suponen los estimadores."),
    "degradation_tab": ("Add degradations to each sensor (bias, drift, noise growth, stuck sensor...) and shape them in "
                        "time with a profile. The preview is computed by the real degradation engine.",
                        "Agregue degradaciones a cada sensor (sesgo, deriva, aumento de ruido, congelamiento...) y "
                        "deles forma en el tiempo con un perfil. La vista previa la calcula el motor real de degradación."),
    "fusion_tab": ("Select the fusion and estimation methods to compare. All of them process exactly the same "
                   "simulated measurements; oracle methods use information unavailable in practice.",
                   "Seleccione los métodos de fusión y estimación a comparar. Todos procesan exactamente las mismas "
                   "mediciones simuladas; los métodos oráculo usan información no disponible en la práctica."),
    "trust_tab": ("Parameters of the SensorTrust estimator (first two moments of the innovation), of the degradation "
                  "detector and of the innovation gate. They apply to every SensorTrust method.",
                  "Parámetros del estimador SensorTrust (dos primeros momentos de la innovación), del detector de "
                  "degradación y de la compuerta de innovación. Se aplican a todos los métodos SensorTrust."),
    "experiment_tab": ("Name the experiment, set the random seed and the outputs, validate the configuration and run it.",
                       "Nombre el experimento, fije la semilla aleatoria y las salidas, valide la configuración y ejecútela."),
    "mc_tab": ("Repeat the experiment many times with randomized parameters to obtain distributions, confidence "
               "intervals, success rates and paired statistical tests.",
               "Repita el experimento muchas veces con parámetros aleatorizados para obtener distribuciones, "
               "intervalos de confianza, tasas de éxito y pruebas estadísticas pareadas."),
    "analysis_tab": ("Robustness envelope (largest tolerable degradation), global sensitivity analysis (which parameters "
                     "matter most) and 2-D parametric sweeps.",
                     "Envolvente de robustez (mayor degradación tolerable), análisis de sensibilidad global (qué "
                     "parámetros influyen más) y barridos paramétricos 2-D."),
    "results_tab": ("Figures and metrics of the last experiment. Choose the figure, the method and the sensor.",
                    "Figuras y métricas del último experimento. Elija la figura, el método y el sensor."),
    "compare_tab": ("Compare any subset of methods on the current scenario with the same measurements.",
                    "Compare cualquier subconjunto de métodos sobre el escenario actual con las mismas mediciones."),
    "export_tab": ("Export figures, configuration, dataset and metrics, verify the reproducibility of the last "
                   "experiment and optionally import external data.",
                   "Exporte figuras, configuración, conjunto de datos y métricas, verifique la reproducibilidad del "
                   "último experimento y, opcionalmente, importe datos externos."),
}


def apply_tooltips(mw, lang: str) -> None:
    i = 1 if lang == "es" else 0
    for key, texts in TIPS.items():
        obj = mw
        try:
            for part in key.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        obj.setToolTip(texts[i])
        if hasattr(obj, "setWhatsThis"):
            obj.setWhatsThis(texts[i])
    for tab, texts in TAB_HELP.items():
        w = getattr(mw, tab, None)
        lab = getattr(w, "_help_label", None)
        if lab is not None:
            lab.setText("ⓘ  " + texts[i])

"""Spanish translations of the interface texts (merged into :mod:`sensortrust.i18n`)."""

from .. import i18n

GUI_ES = {
    # window / toolbar
    "Scientific example:": "Ejemplo científico:",
    "Load": "Cargar",
    "Run Experiment": "Ejecutar experimento",
    "Open configuration...": "Abrir configuración...",
    "Save configuration...": "Guardar configuración...",
    "Language": "Idioma",
    "About": "Acerca de",
    "Ready": "Listo",
    "Running...": "Ejecutando...",
    "Cancel": "Cancelar",
    "Finished": "Finalizado",
    "Error": "Error",
    "Configuration error": "Error de configuración",
    "No hardware required: all measurements come from virtual sensors.":
        "No requiere hardware: todas las mediciones provienen de sensores virtuales.",
    # tabs
    "System": "Sistema", "Sensors": "Sensores", "Degradation": "Degradación", "Fusion": "Fusión",
    "Trust": "Confianza", "Experiment": "Experimento", "Monte Carlo": "Monte Carlo", "Analysis": "Análisis",
    "Results": "Resultados", "Compare": "Comparar", "Export": "Exportar",
    # system tab
    "Dynamic model": "Modelo dinámico",
    "Model parameters": "Parámetros del modelo",
    "Custom model parameters (YAML: A, B, Q or continuous Ac, Bc, Qc; state_names, state_units, inputs, x0)":
        "Parámetros del modelo personalizado (YAML: A, B, Q o continuas Ac, Bc, Qc; state_names, state_units, "
        "inputs, x0)",
    "Simulation horizon": "Horizonte de simulación",
    "Duration": "Duración", "Sampling frequency": "Frecuencia de muestreo",
    "Initial state and uncertainty": "Estado inicial e incertidumbre",
    "State": "Estado", "Unit": "Unidad",
    "True initial mean x0": "Media inicial verdadera x0",
    "Initial std sqrt(P0)": "Desv. inicial sqrt(P0)",
    "Estimator prior mean": "Media a priori del estimador",
    "Estimator prior std": "Desv. a priori del estimador",
    "Sample the true initial state from N(x0, P0)": "Muestrear el estado inicial verdadero de N(x0, P0)",
    "Process noise covariance Q (discrete, computed from the model)":
        "Covarianza del ruido de proceso Q (discreta, calculada del modelo)",
    "Evaluation": "Evaluación",
    "Evaluated state (metrics)": "Estado evaluado (métricas)",
    # sensors tab
    "Virtual sensors": "Sensores virtuales",
    "Add sensor": "Agregar sensor", "Remove sensor": "Eliminar sensor", "Duplicate sensor": "Duplicar sensor",
    "Name": "Nombre", "Measured variable": "Variable medida", "Rate [Hz]": "Frecuencia [Hz]",
    "Noise model": "Modelo de ruido", "Selected sensor": "Sensor seleccionado",
    "Noise parameters": "Parámetros del ruido",
    "Sensor characteristics": "Características del sensor",
    "Nominal (uncompensated) bias": "Sesgo nominal (no compensado)",
    "Resolution (quantization step, 0 = none)": "Resolución (paso de cuantización, 0 = ninguna)",
    "Measurement range lower limit": "Límite inferior del rango de medición",
    "Measurement range upper limit": "Límite superior del rango de medición",
    "Limit the measurement range (saturation)": "Limitar el rango de medición (saturación)",
    "Transport delay": "Retardo de transporte",
    "Nominal dropout probability": "Probabilidad nominal de pérdida",
    "Nominal reliability rho0": "Confiabilidad nominal rho0",
    "Std assumed by estimators (0 = true nominal std)": "Desv. supuesta por los estimadores (0 = desv. nominal)",
    # degradation
    "Sensor": "Sensor",
    "Degradations of the selected sensor": "Degradaciones del sensor seleccionado",
    "Add degradation": "Agregar degradación", "Remove degradation": "Eliminar degradación",
    "Degradation type": "Tipo de degradación", "Degradation parameters": "Parámetros de la degradación",
    "Temporal profile": "Perfil temporal", "Profile type": "Tipo de perfil",
    "Preview (computed with the real degradation engine, noise-free signal)":
        "Vista previa (calculada con el motor real de degradación, señal sin ruido)",
    "Update preview": "Actualizar vista previa",
    "activation g(t)": "activación g(t)", "systematic error": "error sistemático",
    "noise std": "desviación del ruido", "sample lost": "muestra perdida",
    "Degradation preview": "Vista previa de la degradación",
    # fusion
    "Fusion and estimation methods (all run on the same scenario dataset)":
        "Métodos de fusión y estimación (todos se ejecutan sobre el mismo conjunto de datos)",
    "Select": "Seleccionar", "Category": "Categoría", "Description": "Descripción",
    "Fixed weights (comma separated, one per sensor)": "Pesos fijos (separados por coma, uno por sensor)",
    "Adaptive Kalman filter (case C)": "Filtro de Kalman adaptativo (caso C)",
    "Select all": "Seleccionar todo", "Select none": "Ninguno", "Recommended set": "Conjunto recomendado",
    # trust
    "First-Two-Moments trust estimator (SensorTrust)": "Estimador de confianza de los dos primeros momentos (SensorTrust)",
    "Degradation detector": "Detector de degradación",
    "Innovation gate": "Compuerta de innovación",
    "Restore defaults": "Restaurar valores por defecto",
    "How it works": "Cómo funciona",
    # experiment
    "Experiment definition": "Definición del experimento",
    "Experiment name": "Nombre del experimento", "Description text": "Descripción",
    "Random seed": "Semilla aleatoria",
    "Figure formats": "Formatos de figura",
    "Save dataset (dataset.csv)": "Guardar conjunto de datos (dataset.csv)",
    "Results directory": "Directorio de resultados", "Browse...": "Examinar...",
    "Configuration summary": "Resumen de la configuración",
    "Validate configuration": "Validar configuración",
    "Configuration is valid.": "La configuración es válida.",
    # monte carlo
    "Monte Carlo configuration": "Configuración Monte Carlo",
    "Number of runs": "Número de corridas", "Master seed": "Semilla maestra",
    "Sampling design": "Diseño de muestreo", "Worker processes (0 = all cores)": "Procesos (0 = todos los núcleos)",
    "Reference method for paired tests": "Método de referencia para pruebas pareadas",
    "Randomised parameters": "Parámetros aleatorizados",
    "Parameter path": "Ruta del parámetro", "Label": "Etiqueta", "Distribution": "Distribución",
    "Low / mean": "Mín. / media", "High / std": "Máx. / desv.", "Parameter": "Parámetro",
    "Add parameter": "Agregar parámetro", "Remove parameter": "Eliminar parámetro",
    "Run Monte Carlo": "Ejecutar Monte Carlo",
    "Figure": "Figura", "Metric": "Métrica",
    "Statistical summary (mean, median, std, percentiles, 95 % CI)":
        "Resumen estadístico (media, mediana, desv., percentiles, IC 95 %)",
    "Paired tests vs reference (Wilcoxon, Holm)": "Pruebas pareadas vs referencia (Wilcoxon, Holm)",
    "Boxplot": "Diagrama de caja", "ECDF": "FDA empírica", "Histogram": "Histograma",
    # analysis
    "Study": "Estudio", "Robustness envelope": "Envolvente de robustez",
    "Sensitivity analysis": "Análisis de sensibilidad", "Parametric sweep (2-D map)": "Barrido paramétrico (mapa 2-D)",
    "Degradation parameter": "Parámetro de degradación",
    "Tested values (comma separated)": "Valores probados (separados por coma)",
    "Specification (metric <= value)": "Especificación (métrica <= valor)",
    "Criterion": "Criterio", "Repetitions per value": "Repeticiones por valor",
    "Also compute 2-D robustness map": "Calcular también mapa de robustez 2-D",
    "Second parameter (map / sweep)": "Segundo parámetro (mapa / barrido)",
    "Second parameter values": "Valores del segundo parámetro",
    "Sensitivity method": "Método de sensibilidad",
    "Samples (OAT levels / Morris trajectories / Sobol N)":
        "Muestras (niveles OAT / trayectorias Morris / N Sobol)",
    "Output method": "Método de salida", "Output metric": "Métrica de salida",
    "Seed mode": "Modo de semilla",
    "Parameters (path, low, high)": "Parámetros (ruta, mínimo, máximo)",
    "Run analysis": "Ejecutar análisis",
    "Tolerable severity b*": "Severidad tolerable b*",
    # results
    "Method": "Método", "Sensor filter": "Filtro de sensores", "All sensors": "Todos los sensores",
    "Metrics table": "Tabla de métricas", "Per-sensor metrics": "Métricas por sensor",
    "No results yet. Configure an experiment and press Run Experiment.":
        "Aún no hay resultados. Configure un experimento y presione Ejecutar experimento.",
    # compare
    "Methods to compare (executed on exactly the same measurements)":
        "Métodos a comparar (ejecutados exactamente sobre las mismas mediciones)",
    "Run comparison": "Ejecutar comparación",
    "Comparison table": "Tabla comparativa",
    # export
    "Last experiment": "Último experimento", "Open results folder": "Abrir carpeta de resultados",
    "Export all figures": "Exportar todas las figuras", "Export configuration": "Exportar configuración",
    "Export dataset": "Exportar conjunto de datos", "Export metrics table": "Exportar tabla de métricas",
    "Verify reproducibility": "Verificar reproducibilidad",
    "Reproducibility manifest": "Manifiesto de reproducibilidad",
    "Import external dataset (optional)": "Importar conjunto de datos externo (opcional)",
    "Import and run...": "Importar y ejecutar...",
    "Files written:": "Archivos escritos:",
    "Reproducibility verified: identical configuration hash and dataset fingerprint.":
        "Reproducibilidad verificada: hash de configuración y huella del conjunto de datos idénticos.",
    "Reproducibility check FAILED.": "La verificación de reproducibilidad FALLÓ.",
    "Time column": "Columna de tiempo", "Ground-truth columns (optional)": "Columnas de valor verdadero (opcional)",
    "Sensor columns": "Columnas de sensores",
    "Parallel processes (0 = all)": "Procesos (0 = todos)",
    "No experiment has been run yet. Configure an experiment and press Run Experiment.":
        "Aún no se ha ejecutado ningún experimento. Configure un experimento y presione Ejecutar experimento.",
    "Imported datasets cannot be regenerated from a seed; only simulated experiments can be verified.":
        "Los conjuntos de datos importados no pueden regenerarse a partir de una semilla; solo pueden verificarse "
        "los experimentos simulados.",
    "Unexpected error": "Error inesperado",
    "Cancelled by the user": "Cancelado por el usuario",
    "Parameter value (normalised to its range, 0-1)": "Valor del parámetro (normalizado a su rango, 0-1)",
    "Parameter varied in the study (its configuration path is shown when hovering over the list). A different "
    "path of the configuration can also be typed.": "Parámetro variado en el estudio (su ruta en la configuración "
                                                     "se muestra al pasar el cursor sobre la lista). También puede "
                                                     "escribirse otra ruta de la configuración.",
    "Probability distribution of the parameter (for Normal: mean and standard deviation).":
        "Distribución de probabilidad del parámetro (para Normal: media y desviación estándar).",
    "mean, 5-95 %": "media, 5-95 %",
    "not available for this study": "no disponible para este estudio",
    "±2 std (P)": "±2 desv. (P)",
    "Check at least one method to compare.": "Marque al menos un método para comparar.",
    "Add at least one parameter (path, minimum, maximum) to the sensitivity analysis.":
        "Agregue al menos un parámetro (ruta, mínimo, máximo) al análisis de sensibilidad.",
    "combined": "combinada", "severity": "severidad", "onset": "inicio",
    "Exit": "Salir", "A computation is running. Stop it and exit?": "Hay un cálculo en curso. ¿Detenerlo y salir?",
    "Stopping...": "Deteniendo...", "At least one sensor is required.": "Se requiere al menos un sensor.",
    "Quick guide": "Guía rápida", "User manual": "Manual de usuario", "Technical manual": "Manual técnico",
    "The manual is in the Manuales folder of the repository:":
        "El manual se encuentra en la carpeta Manuales del repositorio:",
    "Methods not applicable to this scenario (not executed):":
        "Métodos no aplicables a este escenario (no se ejecutaron):",
    "SensorTrust EKF": "SensorTrust EKF", "SensorTrust UKF": "SensorTrust UKF",
    # parameter labels of the example studies
    "EWMA lambda": "lambda EWMA", "S2 noise std [m]": "desviación del ruido de S2 [m]",
    "S2 noise std sigma [m]": "desviación del ruido de S2 sigma [m]", "bias b [m]": "sesgo b [m]",
    "dead zone gamma1": "zona muerta gamma1", "fault onset [s]": "inicio del fallo [s]",
    "fault onset t_f [s]": "inicio del fallo t_f [s]", "trust scale beta1": "escala de confianza beta1",
    "variance factor F [-]": "factor de varianza F [-]",
    "Not available for the selected method (it does not compute this quantity).":
        "No disponible para el método seleccionado (no calcula esta magnitud).",
    "The operation could not be completed. The application remains open; details were written to the log file:":
        "La operación no pudo completarse. La aplicación sigue abierta; los detalles se guardaron en el registro:",
    # table headers (metric identifiers such as rmse, mae, nis_normalized are kept)
    "method": "Método", "metric": "Métrica", "category": "Categoría", "mean": "media", "median": "mediana",
    "std": "desv.", "min": "mín.", "max": "máx.", "ci_low": "IC95 inf.", "ci_high": "IC95 sup.",
    "reference": "Referencia", "success": "éxito", "success_rate": "tasa de éxito",
    "divergence_rate": "tasa de divergencia", "runtime_s": "tiempo [s]", "parameter": "Parámetro",
    "sensor": "Sensor", "degraded": "degradado", "tolerable_severity": "severidad tolerable",
    "limit_reached": "límite alcanzado", "win_rate_a": "fracción ganada (ref.)", "p_value": "p",
    "p_value_holm": "p (Holm)", "mean_diff": "dif. media", "median_diff": "dif. mediana",
}

i18n.ES.update(GUI_ES)

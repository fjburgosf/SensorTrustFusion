"""Spanish versions of the error messages of the scientific engine.

The engine raises its errors in English (they are also used by the command
line and the log files).  The graphical interface shows them in the language
selected by the user: :func:`translate_error` matches an English message
against the templates below (``{}`` = variable part) and returns the Spanish
text with the same variable parts.  A test checks that every error message
raised by the engine has a translation.
"""

from __future__ import annotations

import re

ERROR_TEMPLATES_ES: list[tuple[str, str]] = [
    ("{}: 'severity' must be finite.", "{0}: la severidad ('severity') debe ser un número finito."),
    ("{}: '{}' must be finite{}.", "{0}: el parámetro '{1}' debe ser un número finito{2}."),
    ("periodic: 'period' must be > 0.", "Degradación periódica: el periodo ('period') debe ser > 0."),
    ("intermittent: mean_on and mean_off must be > 0.",
     "Degradación intermitente: las duraciones medias activa e inactiva (mean_on, mean_off) deben ser > 0."),
    ("variance_increase: 'severity' is a variance factor and must be >= 1.",
     "Varianza creciente: la severidad es un factor de varianza y debe ser >= 1."),
    ("burst_fault: 'burst_duration' must be > 0.", "Fallo en ráfagas: la duración de la ráfaga debe ser > 0."),
    ("outliers: 'probability' must be <= 1.", "Valores atípicos: la probabilidad debe ser <= 1."),
    ("sensitivity_loss: 'severity' must be in [0, 1].", "Pérdida de sensibilidad: la severidad debe estar en [0, 1]."),
    ("saturation: 'low' must be < 'high'.", "Saturación: el límite inferior debe ser menor que el superior."),
    ("quantization: 'severity' (quantum) must be > 0.", "Cuantización: el paso de cuantización (severidad) debe ser > 0."),
    ("dropout: 'severity' is a probability in [0, 1].",
     "Pérdida de muestras: la severidad es una probabilidad y debe estar en [0, 1]."),
    ("packet_loss: 'severity' is a probability in [0, 1].",
     "Pérdida de paquetes: la severidad es una probabilidad y debe estar en [0, 1]."),
    ("packet_loss: 'burst_length' must be >= 1 sample.",
     "Pérdida de paquetes: la longitud de ráfaga debe ser de al menos 1 muestra."),
    ("Each degradation must be a mapping with a 'type'.", "Cada degradación debe definirse con un tipo ('type')."),
    ("Unknown degradation type {}. Available: {}, combined.",
     "Tipo de degradación desconocido {0}. Disponibles: {1}, combined."),
    ("{}: 'severity' must be numeric.", "{0}: la severidad ('severity') debe ser numérica."),
    ("{}: '{}' must be numeric.", "{0}: el parámetro '{1}' debe ser numérico."),
    ("'combined' degradation requires a non-empty 'components' list.",
     "La degradación combinada requiere una lista de componentes no vacía."),
    ("Profile parameter '{}' must be finite.", "El parámetro del perfil '{0}' debe ser un número finito."),
    ("Profile parameter '{}' must be > 0, got {}.", "El parámetro del perfil '{0}' debe ser > 0 (valor recibido: {1})."),
    ("Profile parameter '{}' must be >= 0, got {}.", "El parámetro del perfil '{0}' debe ser >= 0 (valor recibido: {1})."),
    ("Profile parameter '{}' must be numeric.", "El parámetro del perfil '{0}' debe ser numérico."),
    ("Unknown profile type {}. Available: {}.", "Tipo de perfil desconocido {0}. Disponibles: {1}."),
    ("Profile '{}' requires rise_time > 0.", "El perfil '{0}' requiere un tiempo de subida (rise_time) > 0."),
    ("recovery_time ({}) must be >= start ({}).",
     "El instante de recuperación ({0}) debe ser mayor o igual que el de inicio ({1})."),
    ("Profile 'recovery' requires recovery_time or duration.",
     "El perfil de recuperación requiere el instante de recuperación o la duración."),
    ("recovery_shape must be 'linear' or 'exponential'.", "La forma de recuperación debe ser 'linear' o 'exponential'."),
    ("Exponential recovery requires recovery_duration > 0.",
     "La recuperación exponencial requiere una duración de recuperación > 0."),
    ("The linear Kalman filter requires a linear model; '{}' is nonlinear (use EKF/UKF).",
     "El filtro de Kalman lineal requiere un modelo lineal; '{0}' es no lineal (use EKF o UKF)."),
    ("Unknown filter {}; available: {}.", "Filtro desconocido {0}; disponibles: {1}."),
    ("The linear Kalman filter requires linear sensors; '{}' is nonlinear (use EKF/UKF).",
     "El filtro de Kalman lineal requiere sensores lineales; '{0}' es no lineal (use EKF o UKF)."),
    ("Unknown benchmark {}. Available: {}.", "Benchmark desconocido {0}. Disponibles: {1}."),
    ("The experiment configuration must be a mapping.", "La configuración del experimento debe ser un diccionario (clave: valor)."),
    ("Unknown configuration section(s): {}.", "Sección(es) de configuración desconocida(s): {0}."),
    ("simulation.duration must be > 0 s.", "La duración de la simulación debe ser > 0 s."),
    ("simulation.fs must be > 0 Hz.", "La frecuencia de muestreo debe ser > 0 Hz."),
    ("The horizon must contain at least 2 samples (duration * fs >= 2).",
     "El horizonte debe contener al menos 2 muestras (duración × frecuencia >= 2)."),
    ("Horizon too long (more than 5e6 samples).", "Horizonte demasiado largo (más de 5 millones de muestras)."),
    ("At least one sensor must be configured.", "Debe configurarse al menos un sensor."),
    ("Sensor names must be unique, got {}.", "Los nombres de los sensores deben ser únicos; se recibió {0}."),
    ("'methods' must be a non-empty list.", "Debe seleccionarse al menos un método de fusión."),
    ("Method labels must be unique (use 'label' to distinguish variants).",
     "Las etiquetas de los métodos deben ser únicas (use 'label' para distinguir variantes)."),
    ("metrics.warmup must be in [0, duration).",
     "El tiempo de calentamiento de las métricas debe estar en [0, duración)."),
    ("Configuration file not found: {}", "No se encontró el archivo de configuración: {0}"),
    ("Configuration file {} does not contain a mapping.", "El archivo de configuración {0} no contiene un diccionario válido."),
    ("simulation.duration and simulation.fs must be numeric.",
     "La duración y la frecuencia de muestreo de la simulación deben ser numéricas."),
    ("Sensor #{} must be a mapping.", "El sensor n.º {0} debe definirse como un diccionario."),
    ("Cannot parse configuration file {}: {}", "No se puede interpretar el archivo de configuración {0}: {1}"),
    ("Method #{} must be a name or a mapping with 'name'.",
     "El método n.º {0} debe ser un nombre o un diccionario con 'name'."),
    ("Sensor '{}': degradation {}={} s is outside the horizon [0, {}] s.",
     "Sensor '{0}': el instante de la degradación ({1} = {2} s) está fuera del horizonte de simulación [0, {3}] s."),
    ("Unknown sampler {}; use one of {}.", "Diseño de muestreo desconocido {0}; use uno de {1}."),
    ("Each varied parameter requires a 'path'.", "Cada parámetro variable requiere una ruta ('path')."),
    ("Unknown distribution {}; use one of {}.", "Distribución desconocida {0}; use una de {1}."),
    ("Parameter {}: '{}' requires std > 0.", "Parámetro {0}: la distribución '{1}' requiere desviación estándar > 0."),
    ("Parameter {}: 'choice' requires a non-empty 'values' list.",
     "Parámetro {0}: la distribución 'choice' requiere una lista de valores no vacía."),
    ("Grid 'levels' must be >= 2.", "El número de niveles de la rejilla debe ser >= 2."),
    ("Parameter {}: '{}' requires low <= high.", "Parámetro {0}: la distribución '{1}' requiere mínimo <= máximo."),
    ("Parameter {}: loguniform requires low > 0.", "Parámetro {0}: la distribución log-uniforme requiere mínimo > 0."),
    ("Sensor '{}' of the configuration has no column in the dataset.",
     "El sensor '{0}' de la configuración no tiene columna en el conjunto de datos."),
    ("Sensor '{}': dataset has {} columns, model expects {}.",
     "Sensor '{0}': el conjunto de datos tiene {1} columnas y el modelo espera {2}."),
    ("Ground truth has {} columns; model has {} states.",
     "El valor verdadero tiene {0} columnas y el modelo tiene {1} estados."),
    ("No fusion method could be executed: {}", "No se pudo ejecutar ningún método de fusión: {0}"),
    ("oracle methods require ground truth", "los métodos oráculo requieren el valor verdadero"),
    ("Monte Carlo requires at least 1 run.", "Monte Carlo requiere al menos 1 corrida."),
    ("analysis.criterion must be 'mean' or 'success_rate'.",
     "El criterio del análisis debe ser 'mean' (media) o 'success_rate' (tasa de éxito)."),
    ("Robustness analysis requires analysis: {type: robustness, parameter: {...}, spec: value}.",
     "El análisis de robustez requiere el tipo, el parámetro variado y la especificación."),
    ("analysis.method must be one of {}.", "El método de sensibilidad debe ser uno de {0}."),
    ("Sensitivity analysis requires analysis.parameters with path, low, high.",
     "El análisis de sensibilidad requiere al menos un parámetro con ruta, mínimo y máximo."),
    ("analysis.output.transform must be 'none' or 'log10'.", "La transformación de la salida debe ser 'none' o 'log10'."),
    ("Sensitivity parameter {}: requires low < high.", "Parámetro de sensibilidad {0}: se requiere mínimo < máximo."),
    ("analysis.parameters must list at least one parameter to sweep.",
     "El barrido requiere al menos un parámetro."),
    ("Unknown r_strategy {}; use one of {}.", "Estrategia de covarianza desconocida {0}; use una de {1}."),
    ("innovation_gate must be a significance level in (0, 1) or null.",
     "La compuerta de innovación debe ser un nivel de significancia en (0, 1) o estar desactivada."),
    ("adaptive.forgetting_factor must be in (0, 1).", "El factor de olvido del filtro adaptativo debe estar en (0, 1)."),
    ("The oracle-R filter requires ground-truth noise variances.",
     "El filtro con R verdadera (oráculo) requiere las varianzas reales del ruido."),
    ("A detector requires a trust estimator (set 'trust').",
     "Un detector requiere un estimador de confianza (configure 'trust')."),
    ("use_sensors: unknown sensor(s) {}; available {}.", "Sensores a usar: sensor(es) desconocido(s) {0}; disponibles {1}."),
    ("estimate_level: weighting must be 'equal' or 'trust'.",
     "Fusión de estimaciones: la ponderación debe ser 'equal' (igual) o 'trust' (confianza)."),
    ("A detector requires a trust estimator.", "Un detector requiere un estimador de confianza."),
    ("Method specification requires 'name'.", "La especificación del método requiere un nombre ('name')."),
    ("Unknown fusion method {}. Available: {}.", "Método de fusión desconocido {0}. Disponibles: {1}."),
    ("Invalid options for method {}: {}", "Opciones no válidas para el método {0}: {1}"),
    ("Measurement-level averaging requires all sensors to measure the same state component directly (sensors measure: {}).",
     "El promedio de mediciones requiere que todos los sensores midan directamente la misma variable "
     "(los sensores miden: {0})."),
    ("fixed_weights requires 'weights' (one per sensor).", "El promedio con pesos fijos requiere un peso por sensor."),
    ("fixed_weights: need {} non-negative weights with positive sum.",
     "Pesos fijos: se requieren {0} pesos no negativos con suma positiva."),
    ("best_sensor_oracle requires ground-truth (oracle) information.",
     "El mejor sensor (oráculo) requiere información del valor verdadero."),
    ("Dataset file not found: {}", "No se encontró el archivo de datos: {0}"),
    ("Dataset {} is empty.", "El conjunto de datos {0} está vacío."),
    ("Time column '{}' not found. Columns: {}.", "No se encontró la columna de tiempo '{0}'. Columnas: {1}."),
    ("The time column contains non-numeric or missing values.", "La columna de tiempo contiene valores no numéricos o faltantes."),
    ("The time column must be strictly increasing with at least 2 samples.",
     "La columna de tiempo debe ser estrictamente creciente y tener al menos 2 muestras."),
    ("The time column is not uniformly sampled; resample the data before importing.",
     "La columna de tiempo no está muestreada uniformemente; remuestree los datos antes de importarlos."),
    ("No sensor columns were selected.", "No se seleccionaron columnas de sensores."),
    ("Cannot read dataset {}: {}", "No se puede leer el conjunto de datos {0}: {1}"),
    ("Sensor '{}': column(s) not found: {}.", "Sensor '{0}': columna(s) no encontrada(s): {1}."),
    ("Ground-truth column(s) not found: {}.", "Columna(s) de valor verdadero no encontrada(s): {0}."),
    ("Ground-truth columns contain missing or non-numeric values.",
     "Las columnas de valor verdadero contienen valores faltantes o no numéricos."),
    ("Fault column '{}' not found.", "No se encontró la columna de fallo '{0}'."),
    ("Unsupported dataset format '{}' (use CSV, TXT or JSON).", "Formato de datos no soportado '{0}' (use CSV, TXT o JSON)."),
    ("Could not reserve a new experiment identifier.", "No se pudo reservar un nuevo identificador de experimento."),
    ("Unknown input signal type {}.", "Tipo de señal de entrada desconocido {0}."),
    ("order must be 1, 2 or 3", "el orden debe ser 1, 2 o 3"),
    ("Model parameter '{}' must be finite.", "El parámetro del modelo '{0}' debe ser un número finito."),
    ("Model parameter '{}' must be > 0, got {}.", "El parámetro del modelo '{0}' debe ser > 0 (valor recibido: {1})."),
    ("Model parameter '{}' must be >= 0, got {}.", "El parámetro del modelo '{0}' debe ser >= 0 (valor recibido: {1})."),
    ("Model parameter '{}' is required.", "El parámetro del modelo '{0}' es obligatorio."),
    ("Model parameter '{}' must have shape {}, got {}.",
     "El parámetro del modelo '{0}' debe tener dimensiones {1} (se recibió {2})."),
    ("input 'steps': 'times' and 'values' must have equal length.",
     "Entrada por escalones: los instantes y los valores deben tener la misma longitud."),
    ("Sampling period Ts must be positive, got {}.", "El periodo de muestreo debe ser positivo (valor recibido: {0})."),
    ("State index {} out of range for model '{}'.", "El índice de estado {0} está fuera de rango para el modelo '{1}'."),
    ("state_units must match the number of states.", "Las unidades de los estados deben coincidir con el número de estados."),
    ("Process noise Q must be {}x{}, got {}.", "La covarianza del ruido de proceso Q debe ser {0}x{1} (se recibió {2})."),
    ("default initial state has wrong dimension.", "El estado inicial por defecto tiene una dimensión incorrecta."),
    ("A must be {}x{}, got {}.", "La matriz A debe ser {0}x{1} (se recibió {2})."),
    ("B must be {}x{}, got {}.", "La matriz B debe ser {0}x{1} (se recibió {2})."),
    ("Model parameter '{}' must be numeric.", "El parámetro del modelo '{0}' debe ser numérico."),
    ("Sensor measures state index {}, but model '{}' has {} states ({}).",
     "El sensor mide el estado de índice {0}, pero el modelo '{1}' tiene {2} estados ({3})."),
    ("Unknown state {}; available: {}.", "Estado desconocido {0}; disponibles: {1}."),
    ("custom_linear: {} inputs in B but {} input specs.",
     "Modelo personalizado: B tiene {0} entradas pero se definieron {1} señales de entrada."),
    ("Unknown model {}. Available: {}.", "Modelo desconocido {0}. Disponibles: {1}."),
    ("Noise parameter '{}' must be {} 0, got {}.", "El parámetro del ruido '{0}' debe ser {1} 0 (valor recibido: {2})."),
    ("Noise 'std' must have 1 or {} entries, got {}.",
     "La desviación del ruido debe tener 1 o {0} valores (se recibieron {1})."),
    ("Noise 'std' must be finite and non-negative.", "La desviación del ruido debe ser finita y no negativa."),
    ("Noise parameter '{}' must be numeric.", "El parámetro del ruido '{0}' debe ser numérico."),
    ("time_varying noise: 'amplitude' must be < 1 (relative).",
     "Ruido variante en el tiempo: la amplitud relativa debe ser < 1."),
    ("correlated noise: 'phi' must be in [0, 1).", "Ruido correlacionado: el coeficiente phi debe estar en [0, 1)."),
    ("impulsive noise: 'probability' must be <= 1.", "Ruido impulsivo: la probabilidad debe ser <= 1."),
    ("Unknown noise type {}. Available: {}.", "Tipo de ruido desconocido {0}. Disponibles: {1}."),
    ("Sensor #{} configuration must be a mapping.", "La configuración del sensor n.º {0} debe ser un diccionario."),
    ("Sensor '{}': rate must be > 0 Hz.", "Sensor '{0}': la frecuencia debe ser > 0 Hz."),
    ("Sensor '{}': rate {} Hz exceeds the simulation sampling frequency {} Hz.",
     "Sensor '{0}': la frecuencia {1} Hz supera la frecuencia de muestreo de la simulación ({2} Hz)."),
    ("Sensor '{}': resolution must be > 0.", "Sensor '{0}': la resolución debe ser > 0."),
    ("Sensor '{}': delay must be >= 0.", "Sensor '{0}': el retardo debe ser >= 0."),
    ("Sensor '{}': dropout probability must be in [0, 1].",
     "Sensor '{0}': la probabilidad de pérdida de muestras debe estar en [0, 1]."),
    ("Sensor '{}': nominal_reliability must be in (0, 1].", "Sensor '{0}': la confiabilidad nominal debe estar en (0, 1]."),
    ("Sensor '{}': '{}' must have 1 or {} entries.", "Sensor '{0}': '{1}' debe tener 1 o {2} valores."),
    ("Sensor '{}': range must be [low, high] with low < high.",
     "Sensor '{0}': el rango debe ser [mínimo, máximo] con mínimo < máximo."),
    ("Sensor '{}': model '{}' has no measurement function '{}'. Available: {}.",
     "Sensor '{0}': el modelo '{1}' no tiene la función de medición '{2}'. Disponibles: {3}."),
    ("Sensor '{}': H has {} columns but the model has {} states.",
     "Sensor '{0}': la matriz H tiene {1} columnas pero el modelo tiene {2} estados."),
    ("Model sampling period differs from 1/fs.", "El periodo de muestreo del modelo difiere de 1/fs."),
    ("The ground-truth trajectory diverged (non-finite values). Check the model parameters.",
     "La trayectoria del valor verdadero divergió (valores no finitos). Revise los parámetros del modelo."),
    ("model.x0 must have {} entries ({}).", "El estado inicial verdadero debe tener {0} valores ({1})."),
    ("estimation.x0 must have {} entries.", "La media a priori del estimador debe tener {0} valores."),
    ("Reference method {} not present.", "El método de referencia {0} no está entre los resultados."),
    ("Detector alarm_on_samples and alarm_off_samples must be >= 1.",
     "Detector: las muestras para activar y desactivar la alarma deben ser >= 1."),
    ("moment_test: significance must be in (0, 1).", "Prueba de momentos: el nivel de significancia debe estar en (0, 1)."),
    ("The 'moment_test' detector requires a moment-based trust estimator (first_two_moments / first_moment_only / second_moment_only).",
     "El detector por prueba de momentos requiere un estimador de confianza basado en momentos "
     "(dos primeros momentos, solo primer momento o solo segundo momento)."),
    ("trust_threshold: require 0 < alarm_trust_below <= clear_trust_above < 1.",
     "Umbral de confianza: se requiere 0 < umbral de alarma <= umbral de liberación < 1."),
    ("Unknown detector {}; available: {}.", "Detector desconocido {0}; disponibles: {1}."),
    ("Invalid detector parameters: {}", "Parámetros del detector no válidos: {0}"),
    ("Trust parameter trust_floor must be in (0, 1).", "La confianza mínima debe estar en (0, 1)."),
    ("nis_gate: gate_significance must be in (0, 1).", "Compuerta NIS: el nivel de significancia debe estar en (0, 1)."),
    ("first_two_moments: dispersion must be 'variance' or 'second_moment'.",
     "Dos primeros momentos: el estadístico de dispersión debe ser 'variance' o 'second_moment'."),
    ("first_two_moments: at least one moment must be used.", "Dos primeros momentos: debe usarse al menos un momento."),
    ("first_two_moments: consensus_memory_samples must be >= 0.",
     "Dos primeros momentos: la memoria del consenso debe ser >= 0 muestras."),
    ("first_two_moments: consensus_trust_fraction must be in (0, 1].",
     "Dos primeros momentos: la fracción de confianza del consenso debe estar en (0, 1]."),
    ("first_two_moments: consensus_min_sensors must be >= 3 (with two sensors a disagreement cannot be attributed).",
     "Dos primeros momentos: el consenso requiere al menos 3 sensores (con dos sensores no se puede atribuir un desacuerdo)."),
    ("Unknown trust method {}. Available: {}.", "Método de confianza desconocido {0}. Disponibles: {1}."),
    ("first_two_moments: {} must be >= 0.", "Dos primeros momentos: {0} debe ser >= 0."),
    ("first_two_moments: {} must be > 0.", "Dos primeros momentos: {0} debe ser > 0."),
    ("first_two_moments: {} must be in (0, 1].", "Dos primeros momentos: {0} debe estar en (0, 1]."),
    ("Unknown moment estimator {}; use one of {}.", "Estimador de momentos desconocido {0}; use uno de {1}."),
    ("EWMA factor 'lam' must be in (0, 1).", "El factor de olvido EWMA debe estar en (0, 1)."),
    ("Moment window must contain at least 2 samples.", "La ventana de momentos debe contener al menos 2 muestras."),
    ("{}: cannot interpret array with {} dimensions.", "{0}: no se puede interpretar un arreglo de {1} dimensiones."),
    ("{} must be square, got shape {}.", "{0} debe ser una matriz cuadrada (dimensiones recibidas: {1})."),
    ("{} contains non-finite values.", "{0} contiene valores no finitos."),
    ("{} is not symmetric.", "{0} no es simétrica."),
    ("{}: expected {} diagonal entries, got {}.", "{0}: se esperaban {1} valores de la diagonal (se recibieron {2})."),
    ("{}: expected shape ({}, {}), got {}.", "{0}: se esperaban dimensiones ({1}, {2}) (se recibió {3})."),
    ("{} is not positive semidefinite (min eigenvalue {}).",
     "{0} no es semidefinida positiva (valor propio mínimo {1})."),
    ("{} is not positive definite (min eigenvalue {}).", "{0} no es definida positiva (valor propio mínimo {1})."),
    ("Invalid parameter path {}.", "Ruta de parámetro no válida {0}."),
    ("Parameter path {} does not exist.", "La ruta de parámetro {0} no existe."),
    ("Parameter path {} does not exist in the configuration.", "La ruta de parámetro {0} no existe en la configuración."),
    ("Invalid seed {}: booleans are not accepted.", "Semilla no válida {0}: no se aceptan valores booleanos."),
    ("Invalid seed {}: must be an integer.", "Semilla no válida {0}: debe ser un número entero."),
    ("Invalid seed {}: must be in [0, 2**63-1].", "Semilla no válida {0}: debe estar en [0, 2**63-1]."),
    ("Invalid seed {}: must be a non-negative integer.", "Semilla no válida {0}: debe ser un entero no negativo."),
    ("Unsupported figure format {}; use {}.", "Formato de figura no soportado {0}; use {1}."),
    ("Results directory {} not found or incomplete (manifest.json and config.yaml are required).",
     "No se encontró el directorio de resultados {0} o está incompleto (se requieren manifest.json y config.yaml)."),
    ("The manifest of {} has no reproducibility fingerprints.",
     "El manifiesto de {0} no contiene huellas de reproducibilidad."),
    ("Cannot parse configuration file {}: syntax error at line {}, column {}.",
     "No se puede interpretar el archivo de configuración {0}: error de sintaxis en la línea {1}, columna {2}."),
    ("Cannot write to the results directory {}: {}", "No se puede escribir en el directorio de resultados {0}: {1}"),
    # messages raised by the interface itself
    ("Fixed weights must be numbers separated by commas.", "Los pesos fijos deben ser números separados por coma."),
    ("Fixed weights: {} values given but there are {} sensors.",
     "Pesos fijos: se indicaron {0} valores pero hay {1} sensores."),
    ("Custom model YAML: syntax error at line {}, column {}.",
     "YAML del modelo personalizado: error de sintaxis en la línea {0}, columna {1}."),
    ("Select at least one fusion method.", "Seleccione al menos un método de fusión."),
    ("Custom model YAML: {}", "YAML del modelo personalizado: {0}"),
    ("Invalid list of numbers: {}", "Lista de números no válida: {0}"),
    ("cancelled by user", "cancelado por el usuario"),
]


def template_regex(template: str) -> re.Pattern:
    """Regex of an English template: ``{}`` is a variable part, ``{{``/``}}`` are literal braces."""
    out, i = [], 0
    while i < len(template):
        if template.startswith("{}", i):
            out.append("(.+?)")
            i += 2
        elif template.startswith("{{", i) or template.startswith("}}", i):
            out.append(re.escape(template[i]))
            i += 2
        else:
            out.append(re.escape(template[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$", re.S)


# most specific templates first (more literal text), so that e.g. "...: syntax error at line {}, column {}."
# wins over the generic "...: {}"
_COMPILED = [(template_regex(en), es) for en, es in
             sorted(ERROR_TEMPLATES_ES, key=lambda t: -len(t[0].replace("{}", "")))]
_OS_ERRORS = {
    "No such file or directory": "No existe el archivo o directorio",
    "Permission denied": "Permiso denegado",
    "No space left on device": "No queda espacio en el disco",
    "Read-only file system": "Sistema de archivos de solo lectura",
    "Is a directory": "Es un directorio",
    "Not a directory": "No es un directorio",
    "File exists": "El archivo ya existe",
    "Operation not permitted": "Operación no permitida",
}
_OS_RX = re.compile(r"^(?:\[(?:Errno|WinError) -?\d+\] )?(" + "|".join(map(re.escape, _OS_ERRORS)) + r")(.*)$", re.S)
_PREFIX = re.compile(r"^(?:[A-Za-z_]*Error|MethodNotApplicableError|Exception): ", re.S)


def translate_error(message: str, lang: str | None = None) -> str:
    """Message of an engine error in the interface language (Spanish translation or the original text)."""
    from .i18n import get_language
    lang = lang or get_language()
    text = str(message)
    if lang != "es":
        return text
    text = _PREFIX.sub("", text, count=1)
    for rx, es in _COMPILED:
        m = rx.match(text)
        if m:
            parts = list(m.groups())
            if es.startswith("No se pudo ejecutar ningún método") and parts:
                items = re.split(r"; (?=[\w .()+-]+: )", parts[0])
                parts[0] = "; ".join(f"{k}: {translate_error(v, 'es')}" if v else k
                                     for k, _, v in (it.partition(": ") for it in items))
            return es.format(*[translate_error(p, "es") if isinstance(p, str) and p.startswith("[Errno") else p
                               for p in parts])
    m = _OS_RX.match(text)
    if m:
        return _OS_ERRORS[m.group(1)] + m.group(2)
    return text

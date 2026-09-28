"""Readable, bilingual names of metrics, state variables, parameters and option values.

Internal identifiers (``rmse_during_fault``, ``position``,
``sensors[1].noise.std``, ``lhs``...) are kept in configurations, result
files and the programming interface; these functions give the text shown to
the user in the language of the interface.
"""

from __future__ import annotations

import re

METRICS: dict[str, tuple[str, str]] = {
    # estimation
    "rmse": ("RMSE", "RMSE"),
    "mae": ("MAE", "MAE"),
    "mse": ("MSE", "MSE"),
    "nrmse": ("Normalized RMSE", "RMSE normalizado"),
    "max_error": ("Maximum error", "Error máximo"),
    "bias": ("Bias (mean error)", "Sesgo (error medio)"),
    "error_variance": ("Error variance", "Varianza del error"),
    "final_error": ("Final error", "Error final"),
    "rmse_pre_fault": ("RMSE before the fault", "RMSE antes del fallo"),
    "rmse_during_fault": ("RMSE during the fault", "RMSE durante el fallo"),
    "rmse_post_fault": ("RMSE after the fault", "RMSE después del fallo"),
    # consistency
    "nis_normalized": ("Normalized NIS", "NIS normalizado"),
    "nis_in_95": ("NIS inside 95 % bounds", "NIS dentro de límites 95 %"),
    "nees_normalized": ("Normalized NEES", "NEES normalizado"),
    "nees_in_95": ("NEES inside 95 % bounds", "NEES dentro de límites 95 %"),
    # detection
    "mean_detection_delay": ("Mean detection delay [s]", "Retardo medio de detección [s]"),
    "n_detected": ("Faults detected", "Fallos detectados"),
    "n_faults": ("Faults", "Fallos"),
    "mean_f1": ("Mean F1 score", "F1 medio"),
    "mean_false_alarm_rate": ("Mean false-alarm rate", "Tasa media de falsas alarmas"),
    "detected": ("Detected", "Detectado"),
    "detection_delay": ("Detection delay [s]", "Retardo de detección [s]"),
    "f1": ("F1 score", "F1"),
    "precision": ("Precision", "Precisión"),
    "recall": ("Recall", "Exhaustividad"),
    "tp": ("True positives", "Verdaderos positivos"),
    "fp": ("False positives", "Falsos positivos"),
    "tn": ("True negatives", "Verdaderos negativos"),
    "fn": ("False negatives", "Falsos negativos"),
    "false_alarm_rate": ("False-alarm rate", "Tasa de falsas alarmas"),
    "false_alarm_events": ("False-alarm events", "Eventos de falsa alarma"),
    "missed_detection_rate": ("Missed-detection rate", "Tasa de detección fallida"),
    # trust and weights
    "degraded_sensor_trust_min": ("Minimum trust of the degraded sensor", "Confianza mínima del sensor degradado"),
    "trust_response_time": ("Trust response time [s]", "Tiempo de respuesta de la confianza [s]"),
    "trust_recovery_time": ("Trust recovery time [s]", "Tiempo de recuperación de la confianza [s]"),
    "trust_nominal_mean": ("Mean trust (nominal)", "Confianza media (nominal)"),
    "trust_nominal_variance": ("Trust variance (nominal)", "Varianza de la confianza (nominal)"),
    "trust_min": ("Minimum trust", "Confianza mínima"),
    "trust_variance": ("Trust variance", "Varianza de la confianza"),
    "trust_degraded_mean": ("Mean trust during the fault", "Confianza media durante el fallo"),
    "trust_degraded_min": ("Minimum trust during the fault", "Confianza mínima durante el fallo"),
    "trust_reduction_ratio": ("Trust reduction ratio", "Razón de reducción de la confianza"),
    "weight_share_during_fault_pct": ("Weight share during the fault [%]", "Participación del peso durante el fallo [%]"),
    "weight_mean": ("Mean weight", "Peso medio"),
    "weight_min": ("Minimum weight", "Peso mínimo"),
    "weight_nominal_mean": ("Mean weight (nominal)", "Peso medio (nominal)"),
    "weight_reduction_time": ("Weight reduction time [s]", "Tiempo de reducción del peso [s]"),
    "weight_recovery_time": ("Weight recovery time [s]", "Tiempo de recuperación del peso [s]"),
    # robustness and overall
    "performance_loss": ("Performance loss", "Pérdida de desempeño"),
    "recovery_quality": ("Recovery quality", "Calidad de recuperación"),
    "success": ("Success", "Éxito"),
    "success_rate": ("Success rate", "Tasa de éxito"),
    "diverged": ("Diverged", "Divergió"),
    "divergence_rate": ("Divergence rate", "Tasa de divergencia"),
    "runtime_s": ("Run time [s]", "Tiempo de ejecución [s]"),
    "tolerable_severity": ("Tolerable severity", "Severidad tolerable"),
    "limit_reached": ("Limit reached", "Límite alcanzado"),
    # generic table columns
    "method": ("Method", "Método"), "category": ("Category", "Categoría"), "oracle": ("Oracle", "Oráculo"),
    "sensor": ("Sensor", "Sensor"), "degraded": ("Degraded", "Degradado"), "metric": ("Metric", "Métrica"),
    "reference": ("Reference", "Referencia"), "parameter": ("Parameter", "Parámetro"), "n": ("N", "N"),
    "mean": ("Mean", "Media"), "median": ("Median", "Mediana"), "std": ("Std. dev.", "Desv. estándar"),
    "min": ("Min.", "Mín."), "max": ("Max.", "Máx."), "p05": ("5th percentile", "Percentil 5"),
    "p95": ("95th percentile", "Percentil 95"), "ci_low": ("95 % CI low", "IC 95 % inferior"),
    "ci_high": ("95 % CI high", "IC 95 % superior"), "mean_diff": ("Mean difference", "Diferencia media"),
    "median_diff": ("Median difference", "Diferencia mediana"),
    "win_rate_a": ("Fraction won by reference", "Fracción ganada por la referencia"),
    "p_value": ("p-value", "Valor p"), "p_value_holm": ("p-value (Holm)", "Valor p (Holm)"),
    "mu_star": ("mu* (Morris)", "mu* (Morris)"), "mu_star_conf": ("mu* 95 % CI", "IC 95 % de mu*"),
    "sigma": ("sigma (Morris)", "sigma (Morris)"), "S1": ("First-order index S1", "Índice de primer orden S1"),
    "S1_conf": ("S1 95 % CI", "IC 95 % de S1"), "ST": ("Total index ST", "Índice total ST"),
    "ST_conf": ("ST 95 % CI", "IC 95 % de ST"),
    "output_min": ("Output minimum", "Salida mínima"), "output_max": ("Output maximum", "Salida máxima"),
    "output_range": ("Output range", "Rango de la salida"), "severity": ("Severity", "Severidad"), "value": ("Value", "Valor"), "effect": ("Effect", "Efecto"),
}

STATES: dict[str, str] = {
    "signal": "señal", "position": "posición", "velocity": "velocidad", "acceleration": "aceleración",
    "heater_temperature": "temperatura del calefactor", "object_temperature": "temperatura del objeto",
    "angle": "ángulo", "angular_rate": "velocidad angular", "horizontal_position": "posición horizontal",
    "vertical_position": "posición vertical",
}

MODEL_DESCRIPTIONS_ES: dict[str, str] = {
    "scalar_signal": "Señal escalar x(t) = x0 + A sen(2 pi f t) perturbada por una caminata aleatoria.",
    "constant_velocity": "Modelo cinemático x = [p, v]; la aceleración es ruido blanco continuo de DEP q.",
    "constant_acceleration": "Modelo cinemático x = [p, v, a]; la sobreaceleración es ruido blanco continuo de DEP q.",
    "mass_spring_damper": "Oscilador amortiguado forzado m p'' + c p' + k p = F(t) + w(t).",
    "thermal": "Modelo térmico concentrado de dos nodos (calefactor y objeto) con potencia en onda cuadrada.",
    "pendulum": "Péndulo amortiguado con dinámica no lineal y sensores de posición no lineales.",
    "custom_linear": "Modelo lineal invariante en el tiempo definido por el usuario.",
}

VALUES: dict[str, tuple[str, str]] = {
    # Monte Carlo sampling designs and distributions
    "random": ("Random", "Aleatorio"), "lhs": ("Latin hypercube", "Hipercubo latino"),
    "sobol": ("Sobol sequence", "Secuencia de Sobol"), "grid": ("Grid", "Rejilla"),
    "uniform": ("Uniform", "Uniforme"), "loguniform": ("Log-uniform", "Log-uniforme"),
    "normal": ("Normal", "Normal"), "integer": ("Integer", "Entero"),
    # analyses
    "mean": ("Mean over repetitions", "Media de las repeticiones"),
    "success_rate": ("Success rate", "Tasa de éxito"),
    "common": ("Common random numbers", "Números aleatorios comunes"),
    "oat": ("One-at-a-time (OAT)", "Uno a la vez (OAT)"), "morris": ("Morris screening", "Cribado de Morris"),
    "sobol_indices": ("Sobol indices", "Índices de Sobol"),
}

DEGRADATIONS: dict[str, tuple[str, str]] = {
    "bias": ("bias", "sesgo"), "progressive_bias": ("progressive bias", "sesgo progresivo"),
    "drift": ("drift", "deriva"), "random_walk_drift": ("random-walk drift", "deriva de caminata aleatoria"),
    "periodic": ("periodic", "periódica"), "intermittent": ("intermittent", "intermitente"),
    "variance_increase": ("variance increase", "aumento de varianza"), "burst_fault": ("burst fault", "fallo en ráfagas"),
    "outliers": ("outliers", "valores atípicos"), "sensitivity_loss": ("sensitivity loss", "pérdida de sensibilidad"),
    "saturation": ("saturation", "saturación"), "quantization": ("quantization", "cuantización"),
    "stuck": ("stuck", "congelado"), "dropout": ("dropout", "pérdida de muestras"),
    "packet_loss": ("packet loss", "pérdida de paquetes"), "combined": ("combined", "combinada"),
    "external_label": ("labelled fault", "fallo etiquetado"),
}


def degradation_label(key: str, lang: str | None = None) -> str:
    en_es = DEGRADATIONS.get(key, (key, key))
    return en_es[1] if _lang(lang) == "es" else en_es[0]


_PATH_WORDS = {
    "noise std": ("noise std", "desviación del ruido"), "bias": ("nominal bias", "sesgo nominal"),
    "severity": ("degradation severity", "severidad de la degradación"),
    "profile start": ("degradation onset", "inicio de la degradación"),
    "profile rise time": ("degradation rise time", "tiempo de subida de la degradación"),
    "rate": ("rate", "frecuencia"), "delay": ("delay", "retardo"),
}
_TRUST_WORDS = {
    "ewma_lambda": ("EWMA forgetting factor", "factor de olvido EWMA"),
    "mean_deadzone": ("first-moment dead zone", "zona muerta del primer momento"),
    "dispersion_deadzone": ("dispersion dead zone", "zona muerta de dispersión"),
    "mean_scale": ("first-moment scale", "escala del primer momento"),
    "dispersion_scale": ("dispersion scale", "escala de dispersión"),
    "trust_decrease_rate": ("trust decrease rate", "tasa de disminución de la confianza"),
    "trust_recovery_rate": ("trust recovery rate", "tasa de recuperación de la confianza"),
}


def _lang(lang):
    if lang:
        return lang
    from .i18n import get_language
    return get_language()


def metric_label(key: str, lang: str | None = None) -> str:
    en_es = METRICS.get(key)
    if en_es is None:
        return key
    return en_es[1] if _lang(lang) == "es" else en_es[0]


def state_label(name: str, lang: str | None = None) -> str:
    if _lang(lang) == "es":
        return STATES.get(name, name.replace("_", " "))
    return name.replace("_", " ")


def value_label(value, lang: str | None = None) -> str:
    en_es = VALUES.get(str(value))
    if en_es is None:
        return str(value)
    return en_es[1] if _lang(lang) == "es" else en_es[0]


def model_description(key: str, english: str, lang: str | None = None) -> str:
    return MODEL_DESCRIPTIONS_ES.get(key, english) if _lang(lang) == "es" else english


def path_label(path: str, cfg: dict | None = None, lang: str | None = None) -> str:
    """Readable label of a parameter path, e.g. ``sensors[1].noise.std`` -> ``S2: noise std``."""
    es = _lang(lang) == "es"
    cfg = cfg or {}
    m = re.match(r"sensors\[(\d+)\]\.(.*)", path)
    if m:
        i = int(m.group(1))
        sensors = cfg.get("sensors", [])
        name = sensors[i]["name"] if i < len(sensors) else f"S{i + 1}"
        dm = re.match(r"degradations\[(\d+)\]\.(.*)", m.group(2))
        rest = (dm.group(2) if dm else m.group(2)).replace(".", " ").replace("_", " ")
        word = _PATH_WORDS.get(rest, (rest, rest))[1 if es else 0]
        if dm and len((sensors[i] if i < len(sensors) else {}).get("degradations", [])) > 1:
            word += f" {int(dm.group(1)) + 1}"
        return f"{name}: {word}"
    m = re.match(r"methods\[(\d+)\]\.trust\.(.*)", path)
    if m:
        k = int(m.group(1))
        methods = cfg.get("methods", [])
        mname = (methods[k].get("label") or methods[k].get("name")) if k < len(methods) else f"#{k + 1}"
        if es:
            from .i18n import tr
            mname = tr(mname) if mname else mname
        word = _TRUST_WORDS.get(m.group(2), (m.group(2).replace("_", " "),) * 2)[1 if es else 0]
        return f"{mname}: {word}"
    fixed = {"simulation.duration": ("simulation duration", "duración de la simulación"),
             "simulation.fs": ("sampling frequency", "frecuencia de muestreo"),
             "model.params.q": ("process noise intensity q", "intensidad del ruido de proceso q")}
    if path in fixed:
        return fixed[path][1 if es else 0]
    return path.split(".")[-1].replace("_", " ")

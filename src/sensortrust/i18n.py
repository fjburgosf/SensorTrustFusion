"""Bilingual (English / Spanish) user-facing texts.

Texts are keyed by their English version.  ``tr(text)`` returns the text in
the active language (``set_language("es")`` or ``"en"``).  Missing Spanish
entries fall back to English, so no text is ever lost.  Figure labels and the
graphical interface share this table; the scientific engine itself (logs,
configuration keys, CSV/JSON column names) always uses English identifiers so
that result files are language-independent.
"""

from __future__ import annotations

_LANG = "en"
LANGUAGES = {"en": "English", "es": "Español"}

ES: dict[str, str] = {
    # ---- generic ------------------------------------------------------
    "Time [s]": "Tiempo [s]",
    "Time": "Tiempo",
    "Ground truth": "Valor verdadero",
    "Fault period": "Periodo de fallo",
    "Measurements": "Mediciones",
    "Estimate": "Estimación",
    "Method": "Método",
    "Sensor": "Sensor",
    "Value": "Valor",
    "Metric": "Métrica",
    "oracle": "oráculo",
    "Trust": "Confianza",
    "Weight": "Peso",
    "Alarm": "Alarma",
    "Severity": "Severidad",
    "Success rate": "Tasa de éxito",
    "Probability": "Probabilidad",
    "Count": "Frecuencia",
    "Specification": "Especificación",
    "Parameter": "Parámetro",
    # ---- figure titles -------------------------------------------------
    "Ground truth vs measurements": "Valor verdadero vs mediciones",
    "Ground truth vs fused estimates": "Valor verdadero vs estimaciones fusionadas",
    "Estimation error": "Error de estimación",
    "Innovation": "Innovación",
    "Innovation vs time": "Innovación vs tiempo",
    "Innovation mean (first moment)": "Media de la innovación (primer momento)",
    "Innovation second moment and variance": "Segundo momento y varianza de la innovación",
    "Sensor trust": "Confianza de los sensores",
    "Sensor weights": "Pesos de los sensores",
    "Noise variance vs time": "Varianza del ruido vs tiempo",
    "Fault timeline": "Línea de tiempo de fallos",
    "RMSE comparison": "Comparación de RMSE",
    "Detection delay": "Retardo de detección",
    "Monte Carlo distribution": "Distribución Monte Carlo",
    "Empirical CDF": "Función de distribución empírica",
    "Histogram": "Histograma",
    "Parametric sweep": "Barrido paramétrico",
    "Robustness envelope": "Envolvente de robustez",
    "Robustness map": "Mapa de robustez",
    "One-at-a-time sensitivity": "Sensibilidad uno a la vez (OAT)",
    "Morris screening": "Cribado de Morris",
    "Sobol indices": "Índices de Sobol",
    # ---- axis labels ---------------------------------------------------
    "Standardised innovation mean [-]": "Media de la innovación estandarizada [-]",
    "Second moment / variance [-]": "Segundo momento / varianza [-]",
    "Trust T_i [-]": "Confianza T_i [-]",
    "Weight w_i [-]": "Peso w_i [-]",
    "Variance [unit^2]": "Varianza [unidad^2]",
    "Detection delay [s]": "Retardo de detección [s]",
    "RMSE": "RMSE",
    "RMSE (whole run)": "RMSE (toda la corrida)",
    "RMSE during fault": "RMSE durante el fallo",
    "Mean elementary effect |mu*|": "Efecto elemental medio |mu*|",
    "Std of elementary effects sigma": "Desv. estándar de efectos elementales sigma",
    "First-order S1": "Primer orden S1",
    "Total-order ST": "Orden total ST",
    "Output": "Salida",
    # ---- legend fragments ---------------------------------------------
    "second moment": "segundo momento",
    "variance": "varianza",
    "nominal (=1)": "nominal (=1)",
    "dead zone": "zona muerta",
    "true R": "R verdadera",
    "estimated R (adaptive KF)": "R estimada (KF adaptativo)",
    "effective R (SensorTrust)": "R efectiva (SensorTrust)",
    "effective R": "R efectiva",
    "nominal R": "R nominal",
    "fault (ground truth)": "fallo (valor verdadero)",
    "alarm D_i": "alarma D_i",
    "not detected": "no detectado",
    "±2 std (innovation)": "±2 desv. (innovación)",
    "tolerable limit": "límite tolerable",
    # ---- method labels (display only; result files keep the English identifiers) ----
    "Simple average": "Promedio simple",
    "Fixed weighted average": "Promedio ponderado fijo",
    "Inverse-variance weighting": "Ponderación por inverso de varianza",
    "Best-sensor oracle": "Mejor sensor (oráculo)",
    "Kalman filter (nominal R)": "Filtro de Kalman (R nominal)",
    "KF with true R (oracle)": "KF con R verdadera (oráculo)",
    "Adaptive KF (Sage-Husa R)": "KF adaptativo (R Sage-Husa)",
    "SensorTrust KF (first two moments)": "SensorTrust KF (dos primeros momentos)",
    "SensorTrust KF (first moment only)": "SensorTrust KF (solo primer momento)",
    "SensorTrust KF (second moment only)": "SensorTrust KF (solo segundo momento)",
    "SensorTrust KF (no consensus, ablation)": "SensorTrust KF (sin consenso, ablación)",
    "KF with NIS gate": "KF con compuerta NIS",
    "KF single sensor": "KF de un solo sensor",
    "EKF (nominal R)": "EKF (R nominal)",
    "Adaptive EKF (Sage-Husa R)": "EKF adaptativo (R Sage-Husa)",
    "UKF (nominal R)": "UKF (R nominal)",
    "Estimate-level CI (equal weights)": "Fusión de estimaciones CI (pesos iguales)",
    "SensorTrust estimate-level CI": "SensorTrust fusión de estimaciones CI",
    "KF (S1 only)": "KF (solo S1)", "KF (S2 only)": "KF (solo S2)", "KF (S3 only)": "KF (solo S3)",
    # ---- categories ----
    "baseline": "referencia", "filter": "filtro", "adaptive": "adaptativo", "trust": "confianza",
    "oracle": "oráculo", "estimate_level": "nivel de estimaciones", "Key": "Clave",
}


def set_language(lang: str) -> None:
    global _LANG
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported language {lang!r}; use one of {list(LANGUAGES)}.")
    _LANG = lang


def get_language() -> str:
    return _LANG


def tr(text: str, lang: str | None = None) -> str:
    lang = lang or _LANG
    if lang == "es":
        return ES.get(text, text)
    return text

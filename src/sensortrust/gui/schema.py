"""Parameter schemas of the graphical interface.

Every editable parameter has a configuration key, an English and a Spanish
label with its symbol and unit, a default value, a widget kind and optional
limits / choices / help text.  Forms are generated from these schemas, so
the interface never shows anonymous "parameter 1, parameter 2" fields.

``unit`` may contain ``{u}``, replaced by the unit of the measured quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class P:
    key: str
    en: str
    es: str
    default: object = 0.0
    kind: str = "float"          # float | int | bool | choice | text
    unit: str = ""
    lo: float = -1e12
    hi: float = 1e12
    decimals: int = 4
    choices: tuple = ()
    help_en: str = ""
    help_es: str = ""
    step: float | None = None
    extra: dict = field(default_factory=dict)
    unit_es: str | None = None

    def label(self, lang: str, unit_sub: str = "") -> str:
        txt = self.es if lang == "es" else self.en
        unit = self.unit_es if (lang == "es" and self.unit_es is not None) else self.unit
        u = unit.replace("{u}", unit_sub or "u")
        return f"{txt} [{u}]" if u else txt

    def help(self, lang: str) -> str:
        return self.help_es if lang == "es" else self.help_en


# ---------------------------------------------------------------------------
# dynamic models
# ---------------------------------------------------------------------------
MODEL_PARAMS: dict[str, list[P]] = {
    "scalar_signal": [
        P("amplitude", "Signal amplitude A", "Amplitud de la señal A", 1.0, unit="a.u.", lo=0),
        P("frequency", "Signal frequency f", "Frecuencia de la señal f", 0.1, unit="Hz", lo=0),
        P("q", "Process noise intensity q", "Intensidad del ruido de proceso q", 1e-3, unit="a.u.^2/s", lo=0,
          decimals=6),
    ],
    "constant_velocity": [
        P("q", "Acceleration noise PSD q", "DEP del ruido de aceleración q", 0.05, unit="m^2/s^3", lo=0, decimals=5,
          help_en="Power spectral density of the white acceleration driving the target.",
          help_es="Densidad espectral de potencia de la aceleración blanca que excita el blanco."),
    ],
    "constant_acceleration": [
        P("q", "Jerk noise PSD q", "DEP del ruido de sobreaceleración q", 0.01, unit="m^2/s^5", lo=0, decimals=5),
    ],
    "mass_spring_damper": [
        P("mass", "Mass m", "Masa m", 1.0, unit="kg", lo=1e-6),
        P("stiffness", "Spring stiffness k", "Rigidez del resorte k", 4.0, unit="N/m", lo=0),
        P("damping", "Damping coefficient c", "Coeficiente de amortiguamiento c", 0.4, unit="N s/m", lo=0),
        P("q", "Random force PSD q", "DEP de la fuerza aleatoria q", 0.01, unit="N^2 s", lo=0, decimals=5),
        P("force.amplitude", "Forcing amplitude F0", "Amplitud de la fuerza F0", 1.0, unit="N"),
        P("force.frequency", "Forcing frequency", "Frecuencia de la fuerza", 0.2, unit="Hz", lo=0),
    ],
    "thermal": [
        P("C_heater", "Heater heat capacity C_h", "Capacidad térmica del calefactor C_h", 5.0, unit="J/K", lo=1e-6),
        P("C_object", "Object heat capacity C_o", "Capacidad térmica del objeto C_o", 20.0, unit="J/K", lo=1e-6),
        P("R_heater_object", "Heater-object thermal resistance R_ho", "Resistencia térmica calefactor-objeto R_ho",
          0.5, unit="K/W", lo=1e-6),
        P("R_object_ambient", "Object-ambient thermal resistance R_oa", "Resistencia térmica objeto-ambiente R_oa",
          1.0, unit="K/W", lo=1e-6),
        P("T_ambient", "Ambient temperature T_amb", "Temperatura ambiente T_amb", 25.0, unit="degC"),
        P("q", "Heat-flow noise intensity q", "Intensidad del ruido de flujo de calor q", 0.01, unit="K^2/s", lo=0,
          decimals=5),
        P("power.amplitude", "Heating power (square wave)", "Potencia de calefacción (onda cuadrada)", 20.0, unit="W"),
        P("power.frequency", "Heating cycle frequency", "Frecuencia del ciclo de calefacción", 1 / 60, unit="Hz",
          lo=0, decimals=5),
    ],
    "pendulum": [
        P("length", "Pendulum length L", "Longitud del péndulo L", 1.0, unit="m", lo=1e-6),
        P("gravity", "Gravity g", "Gravedad g", 9.81, unit="m/s^2", lo=1e-6),
        P("damping", "Viscous damping b", "Amortiguamiento viscoso b", 0.1, unit="1/s", lo=0),
        P("q", "Angular acceleration noise PSD q", "DEP del ruido de aceleración angular q", 0.01, unit="rad^2/s^3",
          lo=0, decimals=5),
        P("torque.amplitude", "Driving torque amplitude", "Amplitud del torque de excitación", 0.5, unit="rad/s^2"),
        P("torque.frequency", "Driving torque frequency", "Frecuencia del torque de excitación", 0.3, unit="Hz", lo=0),
    ],
    "custom_linear": [],
}

MODEL_LABELS = {
    "scalar_signal": ("Model 1 - Scalar variable signal", "Modelo 1 - Señal escalar variable"),
    "constant_velocity": ("Model 2 - Constant velocity", "Modelo 2 - Velocidad constante"),
    "constant_acceleration": ("Model 3 - Constant acceleration", "Modelo 3 - Aceleración constante"),
    "mass_spring_damper": ("Model 4 - Mass-spring-damper oscillator", "Modelo 4 - Oscilador masa-resorte-amortiguador"),
    "thermal": ("Model 5 - Simplified thermal system", "Modelo 5 - Sistema térmico simplificado"),
    "pendulum": ("Model 6 - Nonlinear pendulum", "Modelo 6 - Péndulo no lineal"),
    "custom_linear": ("Custom linear model (YAML matrices)", "Modelo lineal personalizado (matrices YAML)"),
}

# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
STD = P("std", "Noise standard deviation sigma", "Desviación estándar del ruido sigma", 0.1, unit="{u}", lo=0,
        decimals=5)
NOISE_PARAMS: dict[str, list[P]] = {
    "gaussian": [STD],
    "uniform": [STD],
    "time_varying": [STD, P("amplitude", "Relative std modulation A", "Modulación relativa de la desviación A", 0.5,
                            lo=0, hi=0.99),
                     P("period", "Modulation period", "Periodo de modulación", 20.0, unit="s", lo=1e-6)],
    "time_dependent": [STD, P("slope", "Std growth rate", "Tasa de crecimiento de la desviación", 0.0, unit="{u}/s",
                              decimals=5)],
    "heteroscedastic": [STD, P("gain", "Signal-dependent std gain g", "Ganancia de desviación dependiente de la señal g",
                               0.05, lo=0, decimals=5)],
    "correlated": [STD, P("correlation_time", "Correlation time tau_c", "Tiempo de correlación tau_c", 0.1, unit="s",
                          lo=1e-6)],
    "burst": [STD, P("factor", "Std multiplier during bursts", "Multiplicador de desviación en ráfagas", 5.0, lo=1),
              P("burst_rate", "Burst rate", "Tasa de ráfagas", 0.05, unit="1/s", lo=0),
              P("burst_duration", "Mean burst duration", "Duración media de ráfaga", 0.5, unit="s", lo=1e-6)],
    "impulsive": [STD, P("probability", "Impulse probability per sample", "Probabilidad de impulso por muestra", 0.01,
                         lo=0, hi=1, decimals=5),
                  P("impulse_std", "Impulse standard deviation", "Desviación estándar del impulso", 1.0, unit="{u}",
                    lo=0)],
}
NOISE_LABELS = {
    "gaussian": ("Gaussian", "Gaussiano"), "uniform": ("Uniform", "Uniforme"),
    "time_varying": ("Time-varying Gaussian", "Gaussiano variante en el tiempo"),
    "time_dependent": ("Time-dependent (linear std)", "Dependiente del tiempo (desv. lineal)"),
    "heteroscedastic": ("Heteroscedastic", "Heterocedástico"), "correlated": ("Correlated AR(1)", "Correlacionado AR(1)"),
    "burst": ("Burst noise", "Ruido en ráfagas"), "impulsive": ("Impulsive", "Impulsivo"),
}

# ---------------------------------------------------------------------------
# degradations
# ---------------------------------------------------------------------------
DEG_SEVERITY: dict[str, P] = {
    "bias": P("severity", "Bias magnitude b", "Magnitud del sesgo b", 1.0, unit="{u}"),
    "progressive_bias": P("severity", "Bias growth rate alpha", "Tasa de crecimiento del sesgo alpha", 0.05,
                          unit="{u}/s", decimals=5),
    "drift": P("severity", "Drift rate", "Tasa de deriva", 0.02, unit="{u}/s", decimals=5),
    "random_walk_drift": P("severity", "Random-walk drift intensity", "Intensidad de deriva de caminata aleatoria",
                           0.05, unit="{u}/sqrt(s)", decimals=5),
    "periodic": P("severity", "Periodic bias amplitude", "Amplitud del sesgo periódico", 0.8, unit="{u}"),
    "intermittent": P("severity", "Intermittent bias magnitude", "Magnitud del sesgo intermitente", 1.0, unit="{u}"),
    "variance_increase": P("severity", "Final variance factor R/R0", "Factor final de varianza R/R0", 20.0, lo=1),
    "burst_fault": P("severity", "Burst noise standard deviation", "Desviación estándar del ruido de ráfaga", 1.0,
                     unit="{u}", lo=0),
    "outliers": P("severity", "Outlier amplitude", "Amplitud de los valores atípicos", 3.0, unit="{u}", lo=0),
    "sensitivity_loss": P("severity", "Gain loss fraction", "Fracción de pérdida de ganancia", 0.3, lo=0, hi=1),
    "saturation": P("severity", "Symmetric degraded range limit", "Límite simétrico del rango degradado", 1.0,
                    unit="{u}", lo=0),
    "quantization": P("severity", "Degraded quantization step", "Paso de cuantización degradado", 0.5, unit="{u}",
                      lo=1e-9),
    "stuck": P("severity", "Stuck (no magnitude; keep 1)", "Congelado (sin magnitud; dejar 1)", 1.0),
    "dropout": P("severity", "Sample-loss probability", "Probabilidad de pérdida de muestras", 0.5, lo=0, hi=1),
    "packet_loss": P("severity", "Probability good-to-bad state", "Probabilidad de paso a estado de pérdida", 0.02,
                     lo=0, hi=1, decimals=5),
}
DEG_EXTRA: dict[str, list[P]] = {
    "progressive_bias": [P("b0", "Initial bias b0", "Sesgo inicial b0", 0.0, unit="{u}")],
    "periodic": [P("period", "Period of the periodic bias", "Periodo del sesgo periódico", 5.0, unit="s", lo=1e-6)],
    "intermittent": [P("mean_on", "Mean duration of fault episodes", "Duración media de los episodios de fallo", 1.0,
                       unit="s", lo=1e-6),
                     P("mean_off", "Mean duration between episodes", "Duración media entre episodios", 2.0, unit="s",
                       lo=1e-6)],
    "burst_fault": [P("burst_rate", "Burst rate", "Tasa de ráfagas", 0.2, unit="1/s", lo=0),
                    P("burst_duration", "Mean burst duration", "Duración media de ráfaga", 0.5, unit="s", lo=1e-6)],
    "outliers": [P("probability", "Outlier probability per sample", "Probabilidad de atípico por muestra", 0.05, lo=0,
                   hi=1)],
    "saturation": [P("low", "Degraded range lower limit", "Límite inferior del rango degradado", -1.0, unit="{u}"),
                   P("high", "Degraded range upper limit", "Límite superior del rango degradado", 1.0, unit="{u}")],
    "packet_loss": [P("burst_length", "Mean loss burst length", "Longitud media de ráfaga de pérdida", 10.0,
                      unit="samples", unit_es="muestras", lo=1)],
}
DEG_LABELS = {
    "bias": ("Constant bias", "Sesgo constante"), "progressive_bias": ("Progressive bias", "Sesgo progresivo"),
    "drift": ("Drift (cumulative)", "Deriva (acumulativa)"), "random_walk_drift": ("Random-walk drift",
                                                                                  "Deriva de caminata aleatoria"),
    "periodic": ("Periodic degradation", "Degradación periódica"), "intermittent": ("Intermittent fault",
                                                                                    "Fallo intermitente"),
    "variance_increase": ("Increasing variance", "Varianza creciente"), "burst_fault": ("Burst fault",
                                                                                          "Fallo en ráfagas"),
    "outliers": ("Outliers", "Valores atípicos"), "sensitivity_loss": ("Sensitivity loss", "Pérdida de sensibilidad"),
    "saturation": ("Saturation", "Saturación"), "quantization": ("Quantization degradation",
                                                                  "Degradación de cuantización"),
    "stuck": ("Stuck sensor", "Sensor congelado"), "dropout": ("Dropout", "Pérdida de muestras (dropout)"),
    "packet_loss": ("Packet loss (bursty)", "Pérdida de paquetes (en ráfagas)"),
}

PROFILE_LABELS = {
    "step": ("Step (abrupt)", "Escalón (abrupto)"), "ramp": ("Ramp (linear)", "Rampa (lineal)"),
    "exponential": ("Exponential", "Exponencial"), "sigmoid": ("Sigmoid (smooth)", "Sigmoide (suave)"),
    "sinusoidal": ("Sinusoidal", "Sinusoidal"), "random_walk": ("Random walk", "Caminata aleatoria"),
    "intermittent": ("Intermittent", "Intermitente"), "recovery": ("Recovery", "Recuperación"),
}
PROFILE_PARAMS = [
    P("start", "Onset time t_f", "Instante de inicio t_f", 20.0, unit="s", lo=0),
    P("rise_time", "Transition / rise time", "Tiempo de transición / subida", 5.0, unit="s", lo=0),
    P("period", "Profile period", "Periodo del perfil", 10.0, unit="s", lo=1e-6),
    P("step_std", "Random-walk step intensity", "Intensidad del paso de caminata", 0.3, unit="1/sqrt(s)", lo=0),
    P("mean_on", "Mean active duration", "Duración media activa", 2.0, unit="s", lo=1e-6),
    P("mean_off", "Mean inactive duration", "Duración media inactiva", 3.0, unit="s", lo=1e-6),
    P("recovery", "Sensor recovers", "El sensor se recupera", False, kind="bool"),
    P("recovery_time", "Recovery start time", "Instante de inicio de la recuperación", 40.0, unit="s", lo=0),
    P("recovery_duration", "Recovery duration", "Duración de la recuperación", 5.0, unit="s", lo=0),
    P("recovery_shape", "Recovery shape", "Forma de la recuperación", "linear", kind="choice",
      choices=("linear", "exponential")),
]
PROFILE_USES = {"step": {"start"}, "ramp": {"start", "rise_time"}, "exponential": {"start", "rise_time"},
                "sigmoid": {"start", "rise_time"}, "sinusoidal": {"start", "period"},
                "random_walk": {"start", "step_std"}, "intermittent": {"start", "mean_on", "mean_off"},
                "recovery": {"start", "rise_time"}}

# ---------------------------------------------------------------------------
# trust / detector / adaptive
# ---------------------------------------------------------------------------
TRUST_PARAMS = [
    P("estimator", "Moment estimator", "Estimador de momentos", "ewma", kind="choice",
      choices=("ewma", "window", "recursive"),
      help_en="How the innovation moments are estimated online: exponential forgetting (EWMA), a sliding window "
              "of N samples, or the cumulative (recursive) mean.",
      help_es="Cómo se estiman en línea los momentos de la innovación: olvido exponencial (EWMA), ventana "
              "deslizante de N muestras o media acumulada (recursiva)."),
    P("ewma_lambda", "EWMA forgetting factor lambda", "Factor de olvido EWMA lambda", 0.98, lo=0.5, hi=0.9999,
      decimals=4, help_en="Effective sample size N_eff = (1+lambda)/(1-lambda).",
      help_es="Tamaño efectivo de muestra N_eff = (1+lambda)/(1-lambda)."),
    P("window_length", "Sliding window length N", "Longitud de ventana deslizante N", 100, kind="int", unit="samples",
      unit_es="muestras",
      lo=2, hi=100000),
    P("mean_deadzone", "First-moment dead zone gamma1", "Zona muerta del primer momento gamma1", 3.0, unit="std",
      unit_es="desv.", lo=0, help_en="Number of standard deviations of the mean estimate tolerated before trust decreases.",
      help_es="Número de desviaciones estándar de la estimación de la media toleradas antes de reducir la confianza."),
    P("dispersion_deadzone", "Dispersion dead zone gamma2", "Zona muerta de dispersión gamma2", 3.0, unit="std",
      unit_es="desv.", lo=0),
    P("mean_scale", "First-moment trust scale beta1", "Escala de confianza del primer momento beta1", 0.5,
      unit="innovation std", unit_es="desv. de la innovación", lo=1e-6,
      help_en="Mean excess (in innovation standard deviations) that divides trust by e.",
      help_es="Exceso de media (en desviaciones de la innovación) que divide la confianza por e."),
    P("dispersion_scale", "Dispersion trust scale beta2", "Escala de confianza de dispersión beta2", 1.0,
      unit="nominal variance", unit_es="varianza nominal", lo=1e-6),
    P("trust_decrease_rate", "Trust decrease rate eta_down", "Tasa de disminución de confianza eta_down", 0.5,
      lo=1e-4, hi=1),
    P("trust_recovery_rate", "Trust recovery rate eta_up", "Tasa de recuperación de confianza eta_up", 0.02, lo=1e-5,
      hi=1, decimals=5),
    P("trust_floor", "Minimum trust T_min", "Confianza mínima T_min", 0.01, lo=1e-4, hi=0.99),
    P("use_first_moment", "Use first moment (mean)", "Usar primer momento (media)", True, kind="bool"),
    P("use_second_moment", "Use second moment (dispersion)", "Usar segundo momento (dispersión)", True, kind="bool"),
    P("dispersion", "Dispersion statistic", "Estadístico de dispersión", "variance", kind="choice",
      choices=("variance", "second_moment"),
      help_en="Variance = second moment minus squared mean (separates a bias from a noise increase); the raw second "
              "moment reacts to both.",
      help_es="Varianza = segundo momento menos la media al cuadrado (separa un sesgo de un aumento de ruido); el "
              "segundo momento sin centrar reacciona a ambos."),
    P("consensus", "Consensus reference of the first moment", "Referencia de consenso del primer momento", True,
      kind="bool", help_en="Reference innovation means to the median of trusted redundant sensors.",
      help_es="Referencia las medias de innovación a la mediana de los sensores redundantes confiables."),
    P("consensus_min_sensors", "Minimum redundant sensors for consensus", "Mínimo de sensores redundantes para consenso",
      3, kind="int", lo=3, hi=100),
    P("consensus_trust_fraction", "Trusted-member fraction f_c", "Fracción de miembro confiable f_c", 0.5, lo=0.01,
      hi=1),
]
DETECTOR_PARAMS = [
    P("method", "Detector", "Detector", "moment_test", kind="choice", choices=("moment_test", "trust_threshold", "none")),
    P("significance", "Significance level alpha", "Nivel de significancia alpha", 1e-3, lo=1e-9, hi=0.5, decimals=6),
    P("alarm_on_samples", "Consecutive samples to raise alarm", "Muestras consecutivas para activar alarma", 5,
      kind="int", lo=1, hi=100000),
    P("alarm_off_samples", "Consecutive samples to clear alarm", "Muestras consecutivas para desactivar alarma", 50,
      kind="int", lo=1, hi=100000),
    P("alarm_trust_below", "Alarm when trust below", "Alarma si la confianza es menor que", 0.3, lo=0.001, hi=0.999),
    P("clear_trust_above", "Clear when trust above", "Desactivar si la confianza es mayor que", 0.7, lo=0.001, hi=0.999),
]
GATE_PARAM = P("innovation_gate", "Innovation gate significance", "Significancia de la compuerta de innovación", 1e-6,
               lo=0, hi=0.5, decimals=9,
               help_en="Samples beyond the chi-square gate at this significance are excluded from the state update "
                       "(0 disables the gate).",
               help_es="Las muestras que superan la compuerta chi-cuadrado a esta significancia se excluyen de la "
                       "actualización del estado (0 la desactiva).")
ADAPTIVE_PARAMS = [
    P("forgetting_factor", "Sage-Husa forgetting factor b", "Factor de olvido Sage-Husa b", 0.99, lo=0.5, hi=0.99999,
      decimals=5),
    P("min_variance_ratio", "Minimum estimated/nominal R ratio", "Razón mínima R estimada / R nominal", 0.1, lo=1e-6,
      hi=1),
]

METRIC_PARAMS = [
    P("warmup", "Warm-up excluded from metrics", "Calentamiento excluido de métricas", 1.0, unit="s", lo=0),
    P("rmse_max", "RMSE specification (success criterion)", "Especificación de RMSE (criterio de éxito)", 0.05,
      unit="{u}", lo=0, decimals=5),
    P("trust_threshold", "Trust response threshold", "Umbral de respuesta de confianza", 0.5, lo=0.01, hi=0.99),
    P("recovery_fraction", "Recovery fraction of nominal trust", "Fracción de recuperación de la confianza nominal",
      0.9, lo=0.01, hi=1),
]

# ---------------------------------------------------------------------------
# display labels of choice values and explanations (tooltips) of every parameter
# ---------------------------------------------------------------------------
CHOICE_LABELS = {
    "ewma": ("EWMA (exponential forgetting)", "EWMA (olvido exponencial)"),
    "window": ("Sliding window", "Ventana deslizante"),
    "recursive": ("Recursive (cumulative) mean", "Media recursiva (acumulada)"),
    "variance": ("Variance (central second moment)", "Varianza (segundo momento central)"),
    "second_moment": ("Raw second moment", "Segundo momento sin centrar"),
    "moment_test": ("Chi-square tests on the moments", "Pruebas chi-cuadrado sobre los momentos"),
    "trust_threshold": ("Trust thresholds", "Umbrales de confianza"),
    "none": ("None", "Ninguno"),
    "linear": ("Linear", "Lineal"),
    "exponential": ("Exponential", "Exponencial"),
}


def choice_label(value, lang: str) -> str:
    en_es = CHOICE_LABELS.get(value)
    return (en_es[1] if lang == "es" else en_es[0]) if en_es else str(value)


_H = {  # (group, key) or key -> (English help, Spanish help)
    ("scalar_signal", "amplitude"): ("Amplitude of the sinusoidal component of the true signal.",
                                     "Amplitud de la componente sinusoidal de la señal verdadera."),
    ("scalar_signal", "frequency"): ("Frequency of the sinusoidal component of the true signal.",
                                     "Frecuencia de la componente sinusoidal de la señal verdadera."),
    ("scalar_signal", "q"): ("Intensity of the random process noise added to the signal (random-walk component).",
                             "Intensidad del ruido de proceso aleatorio añadido a la señal (componente de caminata aleatoria)."),
    ("constant_acceleration", "q"): ("Power spectral density of the white jerk (derivative of acceleration).",
                                     "Densidad espectral de potencia de la sobreaceleración blanca (derivada de la aceleración)."),
    ("mass_spring_damper", "mass"): ("Mass of the oscillator.", "Masa del oscilador."),
    ("mass_spring_damper", "stiffness"): ("Spring constant; natural frequency = sqrt(k/m)/(2 pi).",
                                          "Constante del resorte; frecuencia natural = sqrt(k/m)/(2 pi)."),
    ("mass_spring_damper", "damping"): ("Viscous damping coefficient of the oscillator.",
                                        "Coeficiente de amortiguamiento viscoso del oscilador."),
    ("mass_spring_damper", "q"): ("Power spectral density of the random force acting on the mass.",
                                  "Densidad espectral de potencia de la fuerza aleatoria que actúa sobre la masa."),
    ("mass_spring_damper", "force.amplitude"): ("Amplitude of the deterministic sinusoidal forcing.",
                                                "Amplitud de la fuerza sinusoidal determinista."),
    ("mass_spring_damper", "force.frequency"): ("Frequency of the deterministic sinusoidal forcing.",
                                                "Frecuencia de la fuerza sinusoidal determinista."),
    ("thermal", "C_heater"): ("Heat capacity of the heater node.", "Capacidad térmica del nodo calefactor."),
    ("thermal", "C_object"): ("Heat capacity of the heated object.", "Capacidad térmica del objeto calentado."),
    ("thermal", "R_heater_object"): ("Thermal resistance between heater and object.",
                                     "Resistencia térmica entre el calefactor y el objeto."),
    ("thermal", "R_object_ambient"): ("Thermal resistance between object and ambient.",
                                      "Resistencia térmica entre el objeto y el ambiente."),
    ("thermal", "T_ambient"): ("Constant ambient temperature.", "Temperatura ambiente constante."),
    ("thermal", "q"): ("Intensity of the random heat-flow disturbance.", "Intensidad de la perturbación aleatoria del flujo de calor."),
    ("thermal", "power.amplitude"): ("Heating power of the square-wave heater command.",
                                     "Potencia de calefacción de la orden en onda cuadrada."),
    ("thermal", "power.frequency"): ("Frequency of the on/off heating cycle.", "Frecuencia del ciclo de encendido/apagado."),
    ("pendulum", "length"): ("Length of the pendulum.", "Longitud del péndulo."),
    ("pendulum", "gravity"): ("Gravitational acceleration.", "Aceleración de la gravedad."),
    ("pendulum", "damping"): ("Viscous damping of the angular motion.", "Amortiguamiento viscoso del movimiento angular."),
    ("pendulum", "q"): ("Power spectral density of the random angular acceleration.",
                        "Densidad espectral de potencia de la aceleración angular aleatoria."),
    ("pendulum", "torque.amplitude"): ("Amplitude of the sinusoidal driving torque (per unit inertia).",
                                       "Amplitud del torque sinusoidal de excitación (por unidad de inercia)."),
    ("pendulum", "torque.frequency"): ("Frequency of the driving torque.", "Frecuencia del torque de excitación."),
    # noise
    ("noise", "std"): ("Nominal standard deviation of the measurement noise (the datasheet value).",
                       "Desviación estándar nominal del ruido de medición (valor de la hoja de datos)."),
    ("noise", "amplitude"): ("Relative sinusoidal modulation of the noise standard deviation (0 = constant).",
                             "Modulación sinusoidal relativa de la desviación del ruido (0 = constante)."),
    ("noise", "period"): ("Period of the modulation of the noise standard deviation.",
                          "Periodo de la modulación de la desviación del ruido."),
    ("noise", "slope"): ("Linear growth of the noise standard deviation with time.",
                         "Crecimiento lineal de la desviación del ruido con el tiempo."),
    ("noise", "gain"): ("Additional standard deviation proportional to the magnitude of the measured value.",
                        "Desviación adicional proporcional a la magnitud del valor medido."),
    ("noise", "correlation_time"): ("Time constant of the first-order (AR(1)) correlation of the noise.",
                                    "Constante de tiempo de la correlación de primer orden (AR(1)) del ruido."),
    ("noise", "factor"): ("Factor multiplying the noise standard deviation during a burst.",
                          "Factor que multiplica la desviación del ruido durante una ráfaga."),
    ("noise", "burst_rate"): ("Average number of bursts per second.", "Número medio de ráfagas por segundo."),
    ("noise", "burst_duration"): ("Average duration of a burst.", "Duración media de una ráfaga."),
    ("noise", "probability"): ("Probability that a sample contains an impulse.", "Probabilidad de que una muestra contenga un impulso."),
    ("noise", "impulse_std"): ("Standard deviation of the impulses.", "Desviación estándar de los impulsos."),
    # degradation severity (per type) and extra parameters
    ("deg", "bias"): ("Constant offset added to the measurement once the fault is active.",
                      "Desplazamiento constante sumado a la medición cuando el fallo está activo."),
    ("deg", "progressive_bias"): ("Rate at which the bias grows after the onset.", "Tasa de crecimiento del sesgo tras el inicio."),
    ("deg", "drift"): ("Rate of the cumulative drift (integrated by the activation profile).",
                       "Tasa de la deriva acumulativa (integrada según el perfil de activación)."),
    ("deg", "random_walk_drift"): ("Intensity of the random-walk drift.", "Intensidad de la deriva de caminata aleatoria."),
    ("deg", "periodic"): ("Amplitude of the periodic bias.", "Amplitud del sesgo periódico."),
    ("deg", "intermittent"): ("Bias during the fault episodes.", "Sesgo durante los episodios de fallo."),
    ("deg", "variance_increase"): ("Final ratio between the degraded and the nominal noise variance (>= 1).",
                                   "Razón final entre la varianza degradada y la nominal del ruido (>= 1)."),
    ("deg", "burst_fault"): ("Standard deviation of the additional noise during bursts.",
                             "Desviación estándar del ruido adicional durante las ráfagas."),
    ("deg", "outliers"): ("Standard deviation (amplitude) of the outliers.", "Desviación estándar (amplitud) de los valores atípicos."),
    ("deg", "sensitivity_loss"): ("Fraction of the sensor gain that is lost (0 = none, 1 = total).",
                                  "Fracción de la ganancia del sensor que se pierde (0 = ninguna, 1 = total)."),
    ("deg", "saturation"): ("Symmetric limit of the degraded measurement range.", "Límite simétrico del rango de medición degradado."),
    ("deg", "quantization"): ("Quantization step applied to the measurement once degraded.",
                              "Paso de cuantización aplicado a la medición degradada."),
    ("deg", "stuck"): ("The sensor repeats its last value; the magnitude is not used.",
                       "El sensor repite su último valor; la magnitud no se usa."),
    ("deg", "dropout"): ("Probability that a sample is lost while the fault is active.",
                         "Probabilidad de que una muestra se pierda mientras el fallo está activo."),
    ("deg", "packet_loss"): ("Probability of entering the loss state (Gilbert-Elliott model).",
                             "Probabilidad de entrar en el estado de pérdida (modelo de Gilbert-Elliott)."),
    ("deg", "b0"): ("Bias already present at the onset.", "Sesgo ya presente en el instante de inicio."),
    ("deg", "period"): ("Period of the periodic bias.", "Periodo del sesgo periódico."),
    ("deg", "mean_on"): ("Average duration of each fault episode.", "Duración media de cada episodio de fallo."),
    ("deg", "mean_off"): ("Average time between fault episodes.", "Tiempo medio entre episodios de fallo."),
    ("deg", "burst_rate"): ("Average number of bursts per second.", "Número medio de ráfagas por segundo."),
    ("deg", "burst_duration"): ("Average duration of a burst.", "Duración media de una ráfaga."),
    ("deg", "probability"): ("Probability that a sample is an outlier.", "Probabilidad de que una muestra sea atípica."),
    ("deg", "low"): ("Lower limit of the degraded range.", "Límite inferior del rango degradado."),
    ("deg", "high"): ("Upper limit of the degraded range.", "Límite superior del rango degradado."),
    ("deg", "burst_length"): ("Average number of consecutive lost samples.", "Número medio de muestras perdidas consecutivas."),
    # temporal profile
    ("profile", "start"): ("Time at which the degradation starts (fault onset).", "Instante en que inicia la degradación (inicio del fallo)."),
    ("profile", "rise_time"): ("Time the degradation takes to reach its full magnitude.",
                               "Tiempo que tarda la degradación en alcanzar su magnitud completa."),
    ("profile", "period"): ("Period of the sinusoidal activation.", "Periodo de la activación sinusoidal."),
    ("profile", "step_std"): ("Intensity of the random-walk activation.", "Intensidad de la activación de caminata aleatoria."),
    ("profile", "mean_on"): ("Average duration of the active intervals.", "Duración media de los intervalos activos."),
    ("profile", "mean_off"): ("Average duration of the inactive intervals.", "Duración media de los intervalos inactivos."),
    ("profile", "recovery"): ("If checked, the degradation disappears again (sensor recovery).",
                              "Si se marca, la degradación desaparece de nuevo (recuperación del sensor)."),
    ("profile", "recovery_time"): ("Time at which the recovery starts.", "Instante en que inicia la recuperación."),
    ("profile", "recovery_duration"): ("Duration of the recovery (0 = instantaneous).", "Duración de la recuperación (0 = instantánea)."),
    ("profile", "recovery_shape"): ("Shape of the recovery: linear or exponential.", "Forma de la recuperación: lineal o exponencial."),
    # trust
    ("trust", "window_length"): ("Number of samples of the sliding window (only for the window estimator).",
                                 "Número de muestras de la ventana deslizante (solo para el estimador de ventana)."),
    ("trust", "dispersion_deadzone"): ("Number of standard deviations of the dispersion estimate tolerated before trust decreases.",
                                       "Número de desviaciones estándar de la dispersión toleradas antes de reducir la confianza."),
    ("trust", "dispersion_scale"): ("Excess of the normalized variance that divides trust by e.",
                                    "Exceso de la varianza normalizada que divide la confianza por e."),
    ("trust", "trust_decrease_rate"): ("Fraction of the gap closed per sample when trust decreases (fast reaction).",
                                       "Fracción de la diferencia cerrada por muestra cuando la confianza disminuye (reacción rápida)."),
    ("trust", "trust_recovery_rate"): ("Fraction of the gap closed per sample when trust recovers (slow, cautious recovery).",
                                       "Fracción de la diferencia cerrada por muestra cuando la confianza se recupera (recuperación lenta)."),
    ("trust", "trust_floor"): ("Lowest trust a sensor can reach; its covariance is inflated at most by 1/T_min.",
                               "Confianza mínima de un sensor; su covarianza se infla como máximo en 1/T_min."),
    ("trust", "use_first_moment"): ("Use the mean of the innovation (detects bias, drift, stuck sensors).",
                                    "Usar la media de la innovación (detecta sesgo, deriva, sensor congelado)."),
    ("trust", "use_second_moment"): ("Use the dispersion of the innovation (detects noise growth, outliers).",
                                     "Usar la dispersión de la innovación (detecta aumento de ruido y atípicos)."),
    ("trust", "consensus_min_sensors"): ("Minimum number of redundant sensors needed to use the consensus reference.",
                                         "Número mínimo de sensores redundantes para usar la referencia de consenso."),
    ("trust", "consensus_trust_fraction"): ("A sensor enters the consensus if its trust is at least f_c times the highest trust.",
                                            "Un sensor entra en el consenso si su confianza es al menos f_c veces la mayor confianza."),
    # detector
    ("det", "method"): ("Degradation detector: chi-square tests on the innovation moments, thresholds on trust, or none.",
                        "Detector de degradación: pruebas chi-cuadrado sobre los momentos, umbrales de confianza o ninguno."),
    ("det", "significance"): ("False-alarm probability of each chi-square test.", "Probabilidad de falsa alarma de cada prueba chi-cuadrado."),
    ("det", "alarm_on_samples"): ("Consecutive raw alarms needed to declare a degradation (hysteresis).",
                                  "Alarmas crudas consecutivas necesarias para declarar una degradación (histéresis)."),
    ("det", "alarm_off_samples"): ("Consecutive samples without alarm needed to clear it.",
                                   "Muestras consecutivas sin alarma necesarias para desactivarla."),
    ("det", "alarm_trust_below"): ("Trust threshold that raises the alarm (trust-threshold detector).",
                                   "Umbral de confianza que activa la alarma (detector por umbrales)."),
    ("det", "clear_trust_above"): ("Trust threshold that clears the alarm (trust-threshold detector).",
                                   "Umbral de confianza que desactiva la alarma (detector por umbrales)."),
    # adaptive KF
    ("adaptive", "forgetting_factor"): ("Sage-Husa forgetting factor; closer to 1 means longer memory of the noise estimate.",
                                        "Factor de olvido de Sage-Husa; más cercano a 1 implica mayor memoria de la estimación del ruido."),
    ("adaptive", "min_variance_ratio"): ("Lower bound of the estimated R relative to the nominal R (avoids overconfidence).",
                                         "Límite inferior de la R estimada relativa a la nominal (evita exceso de confianza)."),
    # metrics
    ("metric", "warmup"): ("Initial interval excluded from the metrics (filter convergence).",
                           "Intervalo inicial excluido de las métricas (convergencia del filtro)."),
    ("metric", "rmse_max"): ("A run is successful if its RMSE is below this value.", "Una corrida es exitosa si su RMSE es menor que este valor."),
    ("metric", "trust_threshold"): ("Trust level used to measure the trust response time after a fault.",
                                    "Nivel de confianza usado para medir el tiempo de respuesta de la confianza tras un fallo."),
    ("metric", "recovery_fraction"): ("Fraction of the nominal trust that defines recovery after a fault ends.",
                                      "Fracción de la confianza nominal que define la recuperación cuando termina el fallo."),
}


def _fill_help(params, group):
    for p in params:
        if not (p.help_en and p.help_es):
            en, es = _H.get((group, p.key), ("", ""))
            p.help_en = p.help_en or en
            p.help_es = p.help_es or es


for _g, _ps in MODEL_PARAMS.items():
    _fill_help(_ps, _g)
for _ps in NOISE_PARAMS.values():
    _fill_help(_ps, "noise")
for _t, _p in DEG_SEVERITY.items():
    if not (_p.help_en and _p.help_es):
        _p.help_en, _p.help_es = _H.get(("deg", _t), ("", ""))
for _ps in DEG_EXTRA.values():
    _fill_help(_ps, "deg")
_fill_help(PROFILE_PARAMS, "profile")
_fill_help(TRUST_PARAMS, "trust")
_fill_help(DETECTOR_PARAMS, "det")
_fill_help(ADAPTIVE_PARAMS, "adaptive")
_fill_help(METRIC_PARAMS, "metric")

# SensorTrust Fusion v1.0.0

**A Scientific Platform for Adaptive Multisensor Fusion Under Sensor Degradation and Uncertainty**
*Plataforma científica para fusión multisensor adaptativa bajo degradación e incertidumbre de sensores*

![Windows executable](https://github.com/fjburgosf/SensorTrustFusion/actions/workflows/build-windows-exe.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Hardware](https://img.shields.io/badge/hardware-not%20required-lightgrey)

SensorTrust Fusion es un software científico para formular, simular, ejecutar, analizar y reproducir experimentos de **fusión multisensor adaptativa bajo degradación no estacionaria, incertidumbre y fallos de sensores**. Genera el valor verdadero de un sistema dinámico, simula cualquier número de sensores virtuales con ruido y degradaciones configurables, ejecuta varios algoritmos de fusión sobre **exactamente las mismas mediciones**, estima dinámicamente la **confianza** $T_i(k)\in[0,1]$ de cada sensor a partir de los **dos primeros momentos de la innovación** y cuantifica el desempeño con métricas, Monte Carlo, envolventes de robustez y análisis de sensibilidad.

> **No requiere hardware.** Todos los experimentos y benchmarks se obtienen exclusivamente con simulación computacional y sensores virtuales. La importación de datos externos (CSV/TXT/JSON) es opcional.

## Objetivo científico

Proporcionar una plataforma computacional reproducible para modelar, simular, analizar y comparar algoritmos de fusión multisensor bajo condiciones nominales, incertidumbre, fallos y degradación no estacionaria, permitiendo estimar dinámicamente la confiabilidad de las mediciones y cuantificar su impacto sobre el desempeño de estimación.

Preguntas que permite estudiar: ¿cómo responde un algoritmo ante sesgo, aumento de varianza o deriva? ¿Basta la media de la innovación o qué aporta el segundo momento? ¿Cuánto tarda en reaccionar y en recuperar la confianza? ¿Qué pasa con dos sensores degradados? ¿Qué severidad es tolerable antes de incumplir una especificación? ¿Qué parámetros dominan el desempeño?

## Funcionalidades

| Componente | Contenido |
|---|---|
| Modelos dinámicos | señal escalar, velocidad constante, aceleración constante, oscilador masa–resorte–amortiguador, sistema térmico, péndulo no lineal, modelos lineales del usuario |
| Sensores virtuales | número ilimitado; variable medida o $h(x)$ no lineal, frecuencia, resolución, rango, sesgo, retardo, pérdida, confiabilidad nominal |
| Ruido | gaussiano, uniforme, variante en el tiempo, dependiente del tiempo, heterocedástico, correlacionado AR(1), en ráfagas, impulsivo |
| Degradación | sesgo constante/progresivo, deriva, deriva de caminata aleatoria, periódica, intermitente, varianza creciente, ráfagas, atípicos, pérdida de sensibilidad, saturación, cuantización, congelamiento, dropout, pérdida de paquetes; perfiles escalón, rampa, exponencial, sigmoide, sinusoidal, caminata aleatoria, intermitente y recuperación |
| Estimación y fusión | promedio simple, ponderado fijo, inverso de varianza, mejor sensor (oráculo), KF, EKF, UKF, KF adaptativo Sage–Husa, KF con R verdadera (oráculo), compuerta NIS, SensorTrust KF/EKF/UKF, fusión a nivel de estimaciones (intersección de covarianzas) |
| SensorTrust | innovación estandarizada, primer y segundo momento (EWMA / ventana / recursivo), referencia de consenso entre sensores redundantes, confianza asimétrica, compuerta de innovación, detector de degradación independiente |
| Métricas | RMSE, MAE, MSE, NRMSE, error máximo, sesgo, RMSE antes/durante/después del fallo, NIS, NEES, retardo de detección, precisión, exhaustividad, F1, falsas alarmas, métricas de confianza y de pesos, éxito, divergencia, pérdida de desempeño |
| Estudios | Monte Carlo (aleatorio, LHS, Sobol), barridos paramétricos, envolvente y mapa de robustez, sensibilidad OAT/Morris/Sobol, IC y pruebas pareadas de Wilcoxon con Holm |
| Interfaces | GUI bilingüe (español/inglés), línea de comandos, API de Python |
| Reproducibilidad | semillas por proceso, comparación sobre un único conjunto de datos, identificadores ordenados, manifiestos con SHA-256, `sensortrust verify` |

## Instalación

**Windows, sin Python:** descargue `SensorTrustFusion-1.0.0-win64.exe` desde [Releases](https://github.com/fjburgosf/SensorTrustFusion/releases) y ejecútelo (el ejecutable se compila en GitHub Actions a partir de este código fuente).

**Desde el código fuente (Windows, Linux, macOS; Python ≥ 3.10):**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
pip install -e .
```

## Ejecución

```bash
python -m sensortrust                                   # interfaz gráfica
python -m sensortrust gui --lang es                     # interfaz en español
python -m sensortrust run configs/abrupt_bias.yaml      # un experimento
python -m sensortrust benchmark all                     # ST-BENCH-01 ... ST-BENCH-16
python -m sensortrust montecarlo configs/montecarlo_abrupt_bias.yaml
python -m sensortrust robustness configs/robustness_abrupt_bias.yaml
python -m sensortrust sensitivity configs/sensitivity_morris.yaml
python -m sensortrust verify results/ST_EXP_2026_000001  # reproducibilidad
python -m sensortrust list                              # catálogo
```

```python
from sensortrust.experiments import ExperimentManager, get_benchmark
res = ExperimentManager("results").run(get_benchmark("ST-BENCH-02"))
print(res.table[["method", "rmse", "mean_detection_delay"]])
```

## Metodología (resumen)

Para cada sensor, la innovación en el estado a priori se estandariza con la covarianza **nominal**: $\varepsilon_{i,k}=L_{i,k}^{-1}(z_{i,k}-\hat z_{i,k|k-1})$, $S_{i,k}=H_iP_{k|k-1}H_i^T+R_i=L_{i,k}L_{i,k}^T$. Se estiman en línea el primer momento $\hat\mu_i$, el segundo momento $\hat m_{2,i}$ y la varianza $\hat s^2_i=\hat m_{2,i}-\lVert\hat\mu_i\rVert^2/m$. La confianza es

$$
T_i \leftarrow T_i+\eta\,[\rho_i^0 e^{-d_1/\beta_1-d_2/\beta_2}-T_i],\quad
d_1=\max(0,\lVert\hat\mu^c_i\rVert/\sqrt m-\gamma_1/\sqrt{N_{eff}}),\quad
d_2=\max(0,\hat s^2_i-1-\gamma_2\sqrt{2/N_{eff}}),
$$

donde $\hat\mu^c_i$ es la media referida a la mediana de los sensores redundantes confiables (consenso), y la fusión usa $R^{eff}_i=R_i/\max(T_{min},T_i/\max_jT_j)$. La formulación completa, sus decisiones de diseño y todas las demás ecuaciones se documentan en el [Manual Técnico](Manuales/Manual_Tecnico_SensorTrust_Fusion.docx). La función de confianza está aislada en `trust/estimators.py` y puede sustituirse por formulaciones alternativas para compararlas sobre los mismos escenarios.

## Benchmarks y resultados

RMSE de posición [m] (60 s a 100 Hz, tres sensores de 0,10/0,15/0,20 m, semilla 42; especificación 0,05 m), obtenidos con `python -m sensortrust benchmark all` a partir de `benchmarks/ST-BENCH-*.yaml`:

| Benchmark | Promedio simple | KF (R nominal) | KF adaptativo | KF + compuerta NIS | SensorTrust KF | KF R verdadera (oráculo) |
|---|---|---|---|---|---|---|
| ST-BENCH-01 Nominal | 0.0898 | 0.0206 | 0.0207 | 0.0219 | 0.0207 | 0.0206 |
| ST-BENCH-02 Sesgo abrupto | 0.2889 | 0.2174 | 0.0241 | 0.0239 | 0.0224 | 0.2174 |
| ST-BENCH-03 Sesgo progresivo | 0.3297 | 0.2510 | 0.0256 | 0.0315 | 0.0227 | 0.2510 |
| ST-BENCH-04 Varianza creciente | 0.2653 | 0.0536 | 0.0222 | 0.0253 | 0.0223 | 0.0222 |
| ST-BENCH-05 Deriva | 0.1575 | 0.1045 | 0.0273 | 0.0424 | 0.0225 | 0.1045 |
| ST-BENCH-06 Valores atípicos | 0.1786 | 0.0341 | 0.0220 | 0.0219 | 0.0221 | 0.0219 |
| ST-BENCH-07 Sensor congelado | 0.8275 | 0.6473 | 0.0256 | 0.0327 | 0.0233 | 0.6473 |
| ST-BENCH-08 Pérdida de muestras | 0.0941 | 0.0221 | 0.0222 | 0.0234 | 0.0221 | 0.0221 |
| ST-BENCH-09 Recuperación | 0.2806 | 0.2105 | 0.0228 | 0.0270 | 0.0222 | 0.2105 |
| ST-BENCH-10 Dos fallos simultáneos | 0.4090 | 0.2177 | 0.0262 | 0.0256 | 0.0238 | 0.2533 |
| ST-BENCH-11 Sesgo + ruido | 0.2866 | 0.1681 | 0.0232 | 0.0258 | 0.0224 | 0.0252 |
| ST-BENCH-12 Múltiples perfiles de degradación | 0.3492 | 0.2954 | 0.0334 | 0.0336 | 0.0215 | 0.2954 |

Sensor dominante degradado (ST-BENCH-16): SensorTrust KF 0.0215 m frente a 0.4531 m sin la referencia de consenso y 0.3463 m del KF nominal.

## Reproducibilidad

Cada experimento crea `results/ST_EXP_<año>_<n>/` con `config.yaml`, `manifest.json` (versión, semilla, SHA-256 de configuración y datos, resumen de métricas), `dataset.csv`, `estimates.csv`, `trust.csv`, `innovations.csv`, `metrics.json/csv`, `figures/` y `run.log`. Misma configuración + misma semilla ⇒ mismas mediciones y resultados (probado); `sensortrust verify` lo comprueba.

## Estructura

```text
src/sensortrust/   models, simulation, sensors, degradation, estimators, trust, fusion,
                   metrics, statistics, experiments, visualization, io, gui, utils
configs/           configuraciones con nombre (benchmarks y estudios)
examples/          12 ejemplos científicos de la interfaz
benchmarks/        ST-BENCH-01..16 (YAML)
datasets/          conjuntos de datos sintéticos de ejemplo (importación)
Manuales/          Descripción del Software, Manual Técnico y Manual de Usuario (DOCX)
packaging/         PyInstaller (ejecutable de Windows)
```

## Documentación

* [Descripción del Software](Manuales/Descripcion_del_Software_SensorTrust_Fusion.docx)
* [Manual Técnico](Manuales/Manual_Tecnico_SensorTrust_Fusion.docx)
* [Manual de Usuario](Manuales/Manual_de_Usuario_SensorTrust_Fusion.docx)

## Autor

Francisco Javier Burgos Flórez (fjburgosf@gmail.com), autor y titular de los derechos.

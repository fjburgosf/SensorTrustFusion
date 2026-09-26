r"""Degradation detectors ``D_i(k) in {0, 1}``.

Detection (declaring an alarm) is conceptually separated from adaptive
weighting (reducing trust): a method may reduce the weight of a sensor
without ever declaring it faulty, and a detector may run without changing
any weight.  Detectors only consume statistics already produced by the trust
layer.

``moment_test`` (:class:`MomentTestDetector`)
    Uses the innovation-moment estimates of the sensor.  Under ``H0``
    (``eps ~ N(0, I_m)``, white):

    * mean test:     ``N_eff ||mu_hat^c||^2 ~ chi2_m``  -> raw alarm if
      ``> chi2_{m, 1-alpha}`` (``mu_hat^c``: consensus-referenced mean when the
      trust estimator uses the consensus reference, otherwise ``mu_hat``);
    * dispersion test: the EWMA / window average of ``||eps||^2/m`` is
      approximated by a scaled chi-square, ``nu s2_hat ~ chi2_nu`` with
      ``nu = m N_eff`` -> raw alarm if ``nu s2_hat > chi2_{nu, 1-alpha}``
      (more accurate in the right tail than the Gaussian approximation).

    ``D_i`` switches to 1 after ``alarm_on_samples`` consecutive raw alarms and back to 0
    after ``alarm_off_samples`` consecutive samples without raw alarm (hysteresis).
    Requires a moment-based trust estimator.

``trust_threshold`` (:class:`TrustThresholdDetector`)
    ``D_i`` switches to 1 after ``alarm_on_samples`` consecutive samples with
    ``T_i < tau_on`` and back to 0 after ``alarm_off_samples`` samples with ``T_i > tau_off``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..utils.errors import ConfigurationError
from .estimators import TrustEstimator


class Detector:
    key = "abstract"

    def __init__(self, alarm_on_samples: int = 5, alarm_off_samples: int = 50, **params):
        self.alarm_on_samples, self.alarm_off_samples = int(alarm_on_samples), int(alarm_off_samples)
        if self.alarm_on_samples < 1 or self.alarm_off_samples < 1:
            raise ConfigurationError("Detector alarm_on_samples and alarm_off_samples must be >= 1.")
        self.params = {"alarm_on_samples": self.alarm_on_samples, "alarm_off_samples": self.alarm_off_samples, **params}

    def reset(self, N: int, K: int) -> None:
        self.D = np.zeros(N, dtype=bool)
        self.c_on = np.zeros(N, dtype=int)
        self.c_off = np.zeros(N, dtype=int)
        self.hist = np.zeros((K, N), dtype=bool)

    def _hyst(self, i: int, raw: bool) -> None:
        if raw:
            self.c_on[i] += 1
            self.c_off[i] = 0
            if not self.D[i] and self.c_on[i] >= self.alarm_on_samples:
                self.D[i] = True
        else:
            self.c_off[i] += 1
            self.c_on[i] = 0
            if self.D[i] and self.c_off[i] >= self.alarm_off_samples:
                self.D[i] = False

    def update(self, k: int, trust: TrustEstimator, available: list[bool]) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"method": self.key, **self.params}


class MomentTestDetector(Detector):
    key = "moment_test"

    def __init__(self, significance: float = 1e-3, alarm_on_samples: int = 5, alarm_off_samples: int = 50, **kw):
        super().__init__(alarm_on_samples, alarm_off_samples, significance=significance)
        self.alpha = float(significance)
        if not 0.0 < self.alpha < 1.0:
            raise ConfigurationError("moment_test: significance must be in (0, 1).")

    def reset(self, N, K):
        super().reset(N, K)
        self.stat1 = np.full((K, N), np.nan)
        self.stat2 = np.full((K, N), np.nan)
        self._chi = {}

    def update(self, k, trust, available):
        if not getattr(trust, "uses_moments", False):
            raise ConfigurationError("The 'moment_test' detector requires a moment-based trust estimator "
                                     "(first_two_moments / first_moment_only / second_moment_only).")
        for i, tr in enumerate(trust.trackers):  # type: ignore[attr-defined]
            if not available[i] or not tr.ready:
                self.hist[k, i] = self.D[i]
                continue
            ne = tr.n_eff
            thr = self._chi.get(tr.m)
            if thr is None:
                thr = self._chi[tr.m] = float(chi2.ppf(1 - self.alpha, tr.m))
            mu = trust.mu_eff[i] if hasattr(trust, "mu_eff") else tr.mu
            s1 = ne * float(mu @ mu)
            nu = tr.m * ne
            key2 = (tr.m, round(ne, 6))
            thr2 = self._chi.get(key2)
            if thr2 is None:
                thr2 = self._chi[key2] = float(chi2.ppf(1 - self.alpha, nu))
            s2 = nu * tr.variance
            self.stat1[k, i], self.stat2[k, i] = s1 / thr, s2 / thr2
            raw = False
            if getattr(trust, "use_first_moment", True) and s1 > thr:
                raw = True
            if getattr(trust, "use_second_moment", True) and s2 > thr2:
                raw = True
            self._hyst(i, raw)
            self.hist[k, i] = self.D[i]
        return self.D


class TrustThresholdDetector(Detector):
    key = "trust_threshold"

    def __init__(self, alarm_trust_below: float = 0.3, clear_trust_above: float = 0.7, alarm_on_samples: int = 5, alarm_off_samples: int = 50, **kw):
        super().__init__(alarm_on_samples, alarm_off_samples, alarm_trust_below=alarm_trust_below,
                         clear_trust_above=clear_trust_above)
        self.tau_on, self.tau_off = float(alarm_trust_below), float(clear_trust_above)
        if not 0 < self.tau_on <= self.tau_off < 1:
            raise ConfigurationError("trust_threshold: require 0 < alarm_trust_below <= clear_trust_above < 1.")

    def update(self, k, trust, available):
        T = trust.T
        for i in range(T.size):
            if available[i]:
                if self.D[i]:
                    self._hyst(i, not (T[i] > self.tau_off))
                else:
                    self._hyst(i, T[i] < self.tau_on)
            self.hist[k, i] = self.D[i]
        return self.D


DETECTORS = {"moment_test": MomentTestDetector, "trust_threshold": TrustThresholdDetector}


def create_detector(spec: dict | str | None) -> Detector | None:
    if spec in (None, False, "none"):
        return None
    if isinstance(spec, str):
        spec = {"method": spec}
    spec = dict(spec)
    method = spec.pop("method", "moment_test")
    try:
        return DETECTORS[method](**spec)
    except KeyError:
        raise ConfigurationError(f"Unknown detector {method!r}; available: {', '.join(DETECTORS)}.") from None
    except TypeError as exc:
        raise ConfigurationError(f"Invalid detector parameters: {exc}") from None

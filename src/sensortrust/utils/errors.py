"""Exception hierarchy.

All errors raised on purpose by the scientific engine derive from
:class:`SensorTrustError`, so that the GUI and the CLI can show a clear,
user-level message instead of a Python traceback.
"""


class SensorTrustError(Exception):
    """Base class of all SensorTrust Fusion errors."""


class ConfigurationError(SensorTrustError, ValueError):
    """Invalid or inconsistent experiment configuration."""


class DimensionError(ConfigurationError):
    """Incompatible matrix / vector dimensions."""


class CovarianceError(ConfigurationError):
    """A covariance matrix is not symmetric positive (semi)definite."""


class DatasetError(SensorTrustError, ValueError):
    """Corrupt, incomplete or incompatible dataset file."""


class MethodNotApplicableError(SensorTrustError):
    """A fusion method cannot be applied to the given scenario.

    Example: a measurement-level average when the sensors do not measure the
    same physical quantity. The experiment manager records the reason and
    continues with the remaining methods.
    """

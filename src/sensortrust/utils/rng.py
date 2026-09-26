"""Reproducible random number streams.

Every stochastic process of a scenario (process noise, initial state, noise of
each sensor, each degradation, each Monte Carlo draw) receives its own
independent :class:`numpy.random.Generator`, derived from a single master seed
with :class:`numpy.random.SeedSequence` and a *named* key.  Two consequences:

* the same seed always reproduces exactly the same dataset;
* removing a degradation does not change the noise realisation of the
  sensors (the streams are independent), which enables *paired* nominal vs
  degraded comparisons with common random numbers.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable

import numpy as np

from .errors import ConfigurationError

MAX_SEED = 2**63 - 1


def validate_seed(seed) -> int:
    """Return ``seed`` as a non-negative int or raise :class:`ConfigurationError`."""
    if isinstance(seed, bool):
        raise ConfigurationError(f"Invalid seed {seed!r}: booleans are not accepted.")
    try:
        s = int(seed)
    except (TypeError, ValueError):
        raise ConfigurationError(f"Invalid seed {seed!r}: must be a non-negative integer.") from None
    if isinstance(seed, float) and not float(seed).is_integer():
        raise ConfigurationError(f"Invalid seed {seed!r}: must be an integer.")
    if s < 0 or s > MAX_SEED:
        raise ConfigurationError(f"Invalid seed {seed!r}: must be in [0, 2**63-1].")
    return s


def _key_to_ints(key: str) -> list[int]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]


def stream(seed: int, *keys: str | int) -> np.random.Generator:
    """Independent generator identified by ``seed`` and a tuple of keys.

    ``stream(42, "sensor", 1, "noise")`` is always the same stream, and is
    statistically independent from ``stream(42, "sensor", 2, "noise")``.
    """
    seed = validate_seed(seed)
    entropy = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF]
    for k in keys:
        entropy.extend(_key_to_ints(str(k)))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def derive_seeds(master_seed: int, n: int, key: str = "runs") -> list[int]:
    """Derive ``n`` reproducible 63-bit run seeds from a master seed."""
    master_seed = validate_seed(master_seed)
    ss = np.random.SeedSequence([master_seed & 0xFFFFFFFF, (master_seed >> 32) & 0xFFFFFFFF,
                                 *_key_to_ints(key)])
    states = ss.generate_state(n * 2, dtype=np.uint32).reshape(n, 2).astype(np.uint64)
    seeds = (states[:, 0] | (states[:, 1] << np.uint64(32))) & np.uint64(MAX_SEED)
    return [int(s) for s in seeds]


def seed_global(seed: int) -> None:
    """Seed the global Python and NumPy legacy generators.

    The engine itself never uses global generators (it only uses
    :func:`stream`), but third-party code called by user extensions might.
    """
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))


def array_hash(arrays: Iterable[np.ndarray]) -> str:
    """SHA-256 fingerprint of a sequence of arrays (NaN-safe, shape-aware)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()

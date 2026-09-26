"""Access to nested configuration values through dotted paths.

Paths such as ``sensors[1].degradations[0].severity`` or
``methods[2].trust.lam`` are used by the Monte Carlo engine, parametric sweeps,
robustness envelopes and sensitivity analyses to modify one parameter of an
experiment configuration without special-case code.
"""

from __future__ import annotations

import re
from typing import Any

from .errors import ConfigurationError

_TOKEN = re.compile(r"([^.\[\]]+)|\[(-?\d+)\]")


def _tokens(path: str) -> list[str | int]:
    if not path or not isinstance(path, str):
        raise ConfigurationError(f"Invalid parameter path {path!r}.")
    out: list[str | int] = []
    pos = 0
    for m in _TOKEN.finditer(path):
        if m.start() != pos and path[pos:m.start()] != ".":
            raise ConfigurationError(f"Invalid parameter path {path!r}.")
        out.append(m.group(1) if m.group(1) is not None else int(m.group(2)))
        pos = m.end()
    if pos != len(path):
        raise ConfigurationError(f"Invalid parameter path {path!r}.")
    return out


def get_path(cfg: Any, path: str) -> Any:
    cur = cfg
    for t in _tokens(path):
        try:
            cur = cur[t]
        except (KeyError, IndexError, TypeError):
            raise ConfigurationError(f"Parameter path {path!r} does not exist in the configuration.") from None
    return cur


def set_path(cfg: Any, path: str, value: Any, create: bool = True) -> None:
    toks = _tokens(path)
    cur = cfg
    for t, nxt in zip(toks[:-1], toks[1:]):
        try:
            cur = cur[t]
        except KeyError:
            if not create or not isinstance(t, str):
                raise ConfigurationError(f"Parameter path {path!r} does not exist.") from None
            cur[t] = [] if isinstance(nxt, int) else {}
            cur = cur[t]
        except (IndexError, TypeError):
            raise ConfigurationError(f"Parameter path {path!r} does not exist.") from None
    last = toks[-1]
    try:
        cur[last] = value
    except (IndexError, TypeError):
        raise ConfigurationError(f"Parameter path {path!r} does not exist.") from None


def _writable(d) -> bool:
    import os
    import tempfile
    from pathlib import Path
    try:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=d, prefix=".st_write_check_"):
            pass
        return os.access(d, os.W_OK)
    except OSError:
        return False


def user_data_dir():
    """Per-user folder of SensorTrust Fusion (``Documents/SensorTrustFusion`` or ``~/SensorTrustFusion``)."""
    from pathlib import Path
    home = Path.home()
    docs = home / "Documents"
    return (docs if docs.is_dir() else home) / "SensorTrustFusion"


def default_results_dir() -> str:
    """Default results directory of the interface.

    ``results`` (relative to the working directory) when running from the source
    code in a writable folder; for the frozen executable, or when the working
    directory is read-only, ``<user data dir>/results``.
    """
    import sys
    from pathlib import Path
    if not getattr(sys, "frozen", False) and _writable(Path.cwd()):
        return "results"
    return str(user_data_dir() / "results")

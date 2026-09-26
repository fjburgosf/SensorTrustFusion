"""Logging configuration.

The engine logs through the standard :mod:`logging` module under the
``sensortrust`` logger hierarchy.  Each experiment additionally writes a
``run.log`` file inside its result directory.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "sensortrust"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int | str = logging.INFO) -> None:
    """Console logging for CLI / GUI sessions (idempotent)."""
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level)
    if not any(getattr(h, "_st_console", False) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        h._st_console = True  # type: ignore[attr-defined]
        root.addHandler(h)


@contextmanager
def file_log(path: Path, level: int = logging.INFO):
    """Temporarily attach a file handler writing to ``path``."""
    root = logging.getLogger(LOGGER_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    old_level = root.level
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(old_level)

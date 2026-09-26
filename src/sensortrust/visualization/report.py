"""Batch export of the figures of an experiment."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from ..utils.errors import ConfigurationError
from ..utils.logging import get_logger
from .plots import SINGLE_FIGURES

log = get_logger("visualization")
FORMATS = ("png", "svg", "pdf")


def save_figure(fig: Figure, stem: Path, formats=("png",), dpi: int = 150) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        f = str(f).lower()
        if f not in FORMATS:
            raise ConfigurationError(f"Unsupported figure format {f!r}; use {FORMATS}.")
        p = stem.with_suffix("." + f)
        fig.savefig(p, dpi=dpi)
        out.append(p)
    return out


def save_experiment_figures(result, directory: Path, formats=("png",), dpi: int = 150) -> list[Path]:
    files: list[Path] = []
    for i, (name, fn) in enumerate(SINGLE_FIGURES.items(), start=1):
        try:
            fig = fn(result)
        except Exception as exc:  # a figure that cannot be drawn must not stop the export
            log.warning("Figure %s skipped: %s", name, exc)
            continue
        files += save_figure(fig, Path(directory) / f"{i:02d}_{name}", formats, dpi)
    return files

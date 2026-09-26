"""Scientific figures (matplotlib, no global pyplot state)."""

from .plots import FIGURE_TITLES, SINGLE_FIGURES
from .report import save_experiment_figures, save_figure

__all__ = ["FIGURE_TITLES", "SINGLE_FIGURES", "save_experiment_figures", "save_figure"]

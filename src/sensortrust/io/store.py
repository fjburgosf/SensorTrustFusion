"""Result directories and ordered experiment identifiers.

Identifiers have the form ``<PREFIX>_<YEAR>_<NNNNNN>`` (e.g.
``ST_EXP_2026_000001``), are strictly increasing within a results root and
are reserved atomically (directory creation), so concurrent processes never
obtain the same identifier.

Layout of one experiment::

    results/ST_EXP_2026_000001/
        config.yaml        normalized configuration (re-runnable)
        manifest.json      reproducibility manifest
        dataset.csv        scenario dataset (+ dataset_meta.json)
        estimates.csv      estimates (+ std) of every method
        trust.csv          trust, weights, alarms, innovation moments
        innovations.csv    innovations, innovation std and per-sensor NIS
        metrics.json       full metrics
        metrics.csv        one-row-per-method summary table
        figures/           PNG / SVG / PDF figures
        run.log            execution log
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

from ..utils.errors import SensorTrustError

_ID = re.compile(r"^(?P<prefix>[A-Z_]+)_(?P<year>\d{4})_(?P<num>\d{6})$")


def slug(text: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")
    return s or "item"


class ResultStore:
    def __init__(self, root: str | Path = "results"):
        self.root = Path(root)

    def _existing(self, prefix: str, year: int) -> list[int]:
        if not self.root.exists():
            return []
        out = []
        for p in self.root.iterdir():
            m = _ID.match(p.name)
            if m and m.group("prefix") == prefix and int(m.group("year")) == year:
                out.append(int(m.group("num")))
        return out

    def new_directory(self, prefix: str = "ST_EXP", year: int | None = None) -> tuple[str, Path]:
        year = year or _dt.date.today().year
        self.root.mkdir(parents=True, exist_ok=True)
        nxt = max(self._existing(prefix, year), default=0) + 1
        for _ in range(10000):
            eid = f"{prefix}_{year}_{nxt:06d}"
            path = self.root / eid
            try:
                path.mkdir(parents=False, exist_ok=False)
                return eid, path
            except FileExistsError:
                nxt += 1
        raise SensorTrustError("Could not reserve a new experiment identifier.")

    def list(self, prefix: str | None = None) -> list[Path]:
        if not self.root.exists():
            return []
        items = [p for p in self.root.iterdir() if p.is_dir() and _ID.match(p.name)]
        if prefix:
            items = [p for p in items if p.name.startswith(prefix + "_")]
        return sorted(items, key=lambda p: p.name)

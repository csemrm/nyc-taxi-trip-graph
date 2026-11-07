"""Single configuration object; every setting is recorded as run metadata (NFR-04/06)."""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass(frozen=True)
class Config:
    csv_path: str = ""
    max_rows: Optional[int] = None          # deterministic subset: first N data rows
    # NYC bounding box (lat_min, lat_max, lon_min, lon_max)
    lat_min: float = 40.45
    lat_max: float = 41.00
    lon_min: float = -74.30
    lon_max: float = -73.65
    cell_deg: float = 0.005                 # grid cell size in degrees (~500 m)
    min_duration: float = 1.0               # seconds; > 0 required
    max_duration: float = 6 * 3600.0        # seconds; documented, configurable cutoff
    stat: str = "mean"                      # "mean" or "median"
    min_trips: int = 1                      # drop edges seen fewer times
    hour_range: Optional[Tuple[int, int]] = None  # [start, end) pickup hour filter (FR-15)

    def as_dict(self):
        return asdict(self)

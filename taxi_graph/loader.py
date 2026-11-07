"""DataLoader + DataCleaner: stream CSV rows, validate, count exclusions by reason.

One pass, O(T) time, O(1) extra memory (generator).
"""
import csv
import math
from collections import Counter

COLS = {
    "plon": "pickup_longitude", "plat": "pickup_latitude",
    "dlon": "dropoff_longitude", "dlat": "dropoff_latitude",
    "dur": "trip_duration", "time": "pickup_datetime",
}


class CleaningSummary:
    def __init__(self):
        self.read = 0
        self.kept = 0
        self.removed = Counter()

    def as_dict(self):
        return {"read": self.read, "kept": self.kept, "removed": dict(self.removed)}


def iter_clean_trips(cfg, summary):
    """Yield (plat, plon, dlat, dlon, duration_s) for valid rows.

    Raises FileNotFoundError / ValueError with a clear message (US-DP-01).
    """
    try:
        f = open(cfg.csv_path, newline="")
    except OSError as e:
        raise FileNotFoundError(f"cannot read dataset '{cfg.csv_path}': {e.strerror}") from e
    with f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"dataset '{cfg.csv_path}' is empty")
        try:
            idx = {k: header.index(v) for k, v in COLS.items() if k != "time"}
        except ValueError as e:
            raise ValueError(f"dataset missing required column: {e}") from e
        t_idx = header.index(COLS["time"]) if COLS["time"] in header else None
        if cfg.hour_range is not None and t_idx is None:
            raise ValueError("hour_range set but dataset has no pickup_datetime column")

        i_plon, i_plat, i_dlon, i_dlat, i_dur = (
            idx["plon"], idx["plat"], idx["dlon"], idx["dlat"], idx["dur"])
        lat0, lat1, lon0, lon1 = cfg.lat_min, cfg.lat_max, cfg.lon_min, cfg.lon_max
        dmin, dmax = cfg.min_duration, cfg.max_duration
        hr = cfg.hour_range
        limit = cfg.max_rows
        removed = summary.removed
        isfinite = math.isfinite

        for row in reader:
            if limit is not None and summary.read >= limit:
                break
            summary.read += 1
            try:
                plat = float(row[i_plat]); plon = float(row[i_plon])
                dlat = float(row[i_dlat]); dlon = float(row[i_dlon])
                dur = float(row[i_dur])
            except (ValueError, IndexError):
                removed["missing_or_unparseable"] += 1
                continue
            if not (isfinite(plat) and isfinite(plon) and isfinite(dlat)
                    and isfinite(dlon) and isfinite(dur)):
                removed["non_finite"] += 1
                continue
            if not (lat0 <= plat <= lat1 and lon0 <= plon <= lon1
                    and lat0 <= dlat <= lat1 and lon0 <= dlon <= lon1):
                removed["outside_nyc_bounds"] += 1
                continue
            if dur < dmin:
                removed["duration_too_small"] += 1
                continue
            if dur > dmax:
                removed["duration_too_large"] += 1
                continue
            if hr is not None:
                try:
                    h = int(row[t_idx][11:13])
                except (ValueError, IndexError):
                    removed["bad_pickup_time"] += 1
                    continue
                if not (hr[0] <= h < hr[1]):
                    removed["outside_hour_range"] += 1
                    continue
            summary.kept += 1
            yield plat, plon, dlat, dlon, dur

"""LocationNormalizer + LocationIndex.

Normalizer: coordinate -> single int grid-cell key, O(1), pure arithmetic.
Index: hash map key -> dense vertex id 0..V-1 (first-seen order => deterministic).
"""
import math


class LocationNormalizer:
    def __init__(self, cfg):
        self.lat0 = cfg.lat_min
        self.lon0 = cfg.lon_min
        self.lat1 = cfg.lat_max
        self.lon1 = cfg.lon_max
        self.cell = cfg.cell_deg
        if self.cell <= 0:
            raise ValueError("cell_deg must be > 0")
        self.ncols = int(math.floor((cfg.lon_max - cfg.lon_min) / self.cell)) + 1

    def key(self, lat, lon):
        """Return grid-cell key, or None if the coordinate is invalid / out of bounds."""
        if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
            return None
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        if not (self.lat0 <= lat <= self.lat1 and self.lon0 <= lon <= self.lon1):
            return None
        r = int((lat - self.lat0) / self.cell)
        c = int((lon - self.lon0) / self.cell)
        return r * self.ncols + c

    def center(self, key):
        r, c = divmod(key, self.ncols)
        return (self.lat0 + (r + 0.5) * self.cell, self.lon0 + (c + 0.5) * self.cell)


class LocationIndex:
    def __init__(self):
        self._ids = {}
        self.keys = []          # vertex id -> location key

    def get_or_add(self, key):
        v = self._ids.get(key)
        if v is None:
            v = len(self.keys)
            self._ids[key] = v
            self.keys.append(key)
        return v

    def lookup(self, key):
        """Vertex id or None ("not found")."""
        return self._ids.get(key)

    def __len__(self):
        return len(self.keys)

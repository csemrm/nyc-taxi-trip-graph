"""Deterministic synthetic trip generator (same seed => identical CSV).

Used for tests/benchmarks when the real NYC dataset is not available. Uses the
Kaggle "NYC Taxi Trip Duration" column names. A few bad rows are injected so the
cleaner is exercised.
"""
import csv
import math
import random

HEADER = ["id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
          "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
          "store_and_fwd_flag", "trip_duration"]


def generate(path, n_trips, seed=42, bad_fraction=0.01):
    rng = random.Random(seed)
    # Gaussian hotspot around Manhattan; trips cluster so edges repeat.
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i in range(n_trips):
            plat = rng.gauss(40.75, 0.04)
            plon = rng.gauss(-73.98, 0.04)
            dlat = plat + rng.gauss(0, 0.03)
            dlon = plon + rng.gauss(0, 0.03)
            km = math.hypot((dlat - plat) * 111.0, (dlon - plon) * 84.0)
            dur = max(60.0, km / 25.0 * 3600.0 * rng.uniform(0.7, 1.5))
            hour = rng.randrange(24)
            r = rng.random()
            if r < bad_fraction / 3:
                dur = -5
            elif r < 2 * bad_fraction / 3:
                plat = 0.0
            elif r < bad_fraction:
                dur = ""
            w.writerow([f"id{i}", 1, f"2016-03-14 {hour:02d}:00:00", "", 1,
                        f"{plon:.6f}", f"{plat:.6f}", f"{dlon:.6f}", f"{dlat:.6f}", "N",
                        dur if dur == "" else f"{dur:.0f}"])

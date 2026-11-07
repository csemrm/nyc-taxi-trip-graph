import os
import tempfile
import unittest

from taxi_graph.config import Config
from taxi_graph.pipeline import Engine, build
from taxi_graph.shortest_path import INVALID_COORDINATES, SUCCESS
from taxi_graph.synthetic import generate

HDR = ("id,vendor_id,pickup_datetime,dropoff_datetime,passenger_count,pickup_longitude,"
       "pickup_latitude,dropoff_longitude,dropoff_latitude,store_and_fwd_flag,trip_duration\n")


def row(plon, plat, dlon, dlat, dur, t="2016-03-14 08:00:00"):
    return f"x,1,{t},,1,{plon},{plat},{dlon},{dlat},N,{dur}\n"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "t.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, *rows):
        with open(self.path, "w") as f:
            f.write(HDR + "".join(rows))

    def test_cleaning_reasons_and_mean_weight(self):
        self.write(
            row(-73.98, 40.75, -73.95, 40.78, 100),
            row(-73.98, 40.75, -73.95, 40.78, 300),          # same edge, mean 200
            row(-73.98, 40.75, -73.95, 40.78, 0),            # too small
            row(-73.98, 40.75, -73.95, 40.78, 999999),       # too large
            row(0, 0, -73.95, 40.78, 100),                   # outside NYC
            row(-73.98, 40.75, -73.95, 40.78, ""),           # missing
            row(-73.98, 40.75, -73.95, 40.78, "nan"),        # non-finite
            row(-73.98, 40.75, -73.98, 40.75, 50),           # self loop
        )
        cfg = Config(csv_path=self.path, cell_deg=0.01)
        b = build(cfg)
        self.assertEqual(b.cleaning.read, 8)
        self.assertEqual(b.cleaning.kept, 3)
        rem = b.cleaning.removed
        self.assertEqual(rem["duration_too_small"], 1)
        self.assertEqual(rem["duration_too_large"], 1)
        self.assertEqual(rem["outside_nyc_bounds"], 1)
        self.assertEqual(rem["missing_or_unparseable"], 1)
        self.assertEqual(rem["non_finite"], 1)
        self.assertEqual(b.self_loops, 1)
        self.assertEqual((b.graph.n, b.graph.m), (2, 1))
        self.assertAlmostEqual(b.graph.weights[0], 200.0)

    def test_median_and_min_trips(self):
        self.write(*[row(-73.98, 40.75, -73.95, 40.78, d) for d in (10, 20, 1000)])
        b = build(Config(csv_path=self.path, stat="median"))
        self.assertAlmostEqual(b.graph.weights[0], 20.0)
        b = build(Config(csv_path=self.path, min_trips=4))
        self.assertEqual(b.graph.m, 0)

    def test_hour_filter(self):
        self.write(row(-73.98, 40.75, -73.95, 40.78, 100, "2016-03-14 08:10:00"),
                   row(-73.98, 40.75, -73.95, 40.78, 900, "2016-03-14 20:10:00"))
        b = build(Config(csv_path=self.path, hour_range=(6, 10)))
        self.assertEqual(b.graph.weights[0], 100.0)

    def test_missing_file_clear_error(self):
        with self.assertRaises(FileNotFoundError) as cm:
            build(Config(csv_path=os.path.join(self.tmp.name, "nope.csv")))
        self.assertIn("cannot read dataset", str(cm.exception))

    def test_engine_statuses(self):
        self.write(row(-73.98, 40.75, -73.95, 40.78, 100),
                   row(-73.95, 40.78, -73.90, 40.80, 200))
        eng = Engine(build(Config(csv_path=self.path)))
        r = eng.route(40.75, -73.98, 40.80, -73.90)
        self.assertEqual(r.status, SUCCESS)
        self.assertAlmostEqual(r.total_duration, 300.0)
        self.assertEqual(eng.route(0, 0, 40.80, -73.90).status, INVALID_COORDINATES)
        self.assertEqual(eng.route(40.60, -74.10, 40.80, -73.90).status,
                         "LOCATION_NOT_IN_GRAPH")
        self.assertEqual(eng.route(40.80, -73.90, 40.75, -73.98).status, "NO_PATH")

    def test_estimate_prefers_direct_edge(self):
        # A->B 100 s, B->C 200 s, A->C direct 1000 s: shortest path 300, direct 1000.
        self.write(row(-73.98, 40.75, -73.95, 40.78, 100),
                   row(-73.95, 40.78, -73.90, 40.80, 200),
                   row(-73.98, 40.75, -73.90, 40.80, 1000))
        eng = Engine(build(Config(csv_path=self.path)))
        r, est, basis = eng.estimate(40.75, -73.98, 40.80, -73.90)
        self.assertAlmostEqual(r.total_duration, 300.0)
        self.assertEqual((basis, est), ("direct", 1000.0))
        r, est, basis = eng.estimate(40.75, -73.98, 40.78, -73.95)   # A->B direct
        self.assertEqual((basis, est), ("direct", 100.0))
        self.assertEqual(eng.estimate(0, 0, 40.8, -73.9)[1:], (None, None))  # invalid coords

    def test_estimate_falls_back_to_path(self):
        self.write(row(-73.98, 40.75, -73.95, 40.78, 100),
                   row(-73.95, 40.78, -73.90, 40.80, 200))
        eng = Engine(build(Config(csv_path=self.path)))
        r, est, basis = eng.estimate(40.75, -73.98, 40.80, -73.90)
        self.assertEqual((basis, est), ("path", 300.0))

    def test_period_graph_used_only_inside_its_hours(self):
        night, noon = "2016-03-14 02:00:00", "2016-03-14 12:00:00"
        # Same A->B edge: night trips take 100 s, noon trips take 500 s (all-day mean 300 s).
        self.write(row(-73.98, 40.75, -73.95, 40.78, 100, night),
                   row(-73.98, 40.75, -73.95, 40.78, 500, noon))
        from taxi_graph.pipeline import selected_periods
        eng = Engine(build(Config(csv_path=self.path), periods=selected_periods()))
        args = (40.75, -73.98, 40.78, -73.95)
        self.assertEqual(eng.estimate(*args, hour=2)[1:], (100.0, "period-direct"))
        self.assertEqual(eng.estimate(*args, hour=12)[1:], (300.0, "direct"))   # no midday graph
        self.assertEqual(eng.estimate(*args)[1:], (300.0, "direct"))            # no hour given
        # all-day graph is unchanged by building period graphs
        self.assertAlmostEqual(eng.route(*args).total_duration, 300.0)

    def test_period_falls_back_when_period_graph_lacks_pair(self):
        self.write(row(-73.98, 40.75, -73.95, 40.78, 100, "2016-03-14 12:00:00"))
        from taxi_graph.pipeline import selected_periods
        eng = Engine(build(Config(csv_path=self.path), periods=selected_periods()))
        self.assertEqual(eng.estimate(40.75, -73.98, 40.78, -73.95, hour=2)[1:], (100.0, "direct"))

    def test_determinism(self):
        generate(self.path, 3000, seed=3)
        a = build(Config(csv_path=self.path))
        b = build(Config(csv_path=self.path))
        self.assertEqual((a.graph.offsets, a.graph.targets, a.graph.weights),
                         (b.graph.offsets, b.graph.targets, b.graph.weights))


if __name__ == "__main__":
    unittest.main()

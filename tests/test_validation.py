import os
import random
import tempfile
import unittest

from taxi_graph.config import Config
from taxi_graph.graph import TravelGraph
from taxi_graph.loader import CleaningSummary, iter_clean_trips
from taxi_graph.shortest_path import Dijkstra, INF
from taxi_graph.synthetic import generate
from taxi_graph.validation import PERIODS, metrics, run_validation


class DistancesTests(unittest.TestCase):
    def test_matches_single_queries_on_random_graphs(self):
        rng = random.Random(11)
        for _ in range(30):
            n = rng.randint(2, 25)
            edges = [(rng.randrange(n), rng.randrange(n), rng.uniform(0.1, 40), 1)
                     for _ in range(rng.randint(0, 70))]
            edges = [e for e in edges if e[0] != e[1]]
            d = Dijkstra(TravelGraph.from_edges(n, edges))
            for s in range(n):
                targets = {rng.randrange(n) for _ in range(5)}
                got, _ = d.distances(s, targets)
                for t in targets - {s}:
                    one = d.route(s, t)
                    if one.status == "SUCCESS":
                        self.assertAlmostEqual(got[t], one.total_duration, places=9)
                    else:
                        self.assertNotIn(t, got)
        self.assertTrue(all(x == INF for x in d._dist))

    def test_early_stop_visits_less(self):
        # chain 0->1->...->9 ; target 2 settles after 3 pops
        g = TravelGraph.from_edges(10, [(i, i + 1, 1.0, 1) for i in range(9)])
        found, visited = Dijkstra(g).distances(0, {2})
        self.assertEqual(found, {2: 2.0})
        self.assertEqual(visited, 3)


class MetricsTests(unittest.TestCase):
    def test_known_values(self):
        m = metrics([(110, 100), (90, 100), (150, 100)])
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["mae_s"], (10 + 10 + 50) / 3)
        self.assertAlmostEqual(m["bias_s"], (10 - 10 + 50) / 3)
        self.assertAlmostEqual(m["median_ae_s"], 10)
        self.assertAlmostEqual(m["mape"], (0.1 + 0.1 + 0.5) / 3)
        self.assertAlmostEqual(m["within_20pct"], 2 / 3)
        self.assertIsNone(metrics([]))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "t.csv")
        generate(self.path, 20000, seed=5)
        self.cfg = Config(csv_path=self.path, cell_deg=0.01)

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_and_accounting(self):
        r = run_validation(self.cfg, fold=5)
        s = r["split"]
        self.assertEqual(s["train_trips"] + s["test_trips"], r["cleaning"]["kept"])
        self.assertEqual(s["test_trips"], r["cleaning"]["kept"] // 5)
        a = r["test_trip_accounting"]
        self.assertEqual(a["same_cell_excluded"] + a["origin_or_dest_unseen_in_train"]
                         + a["no_path"] + a["covered"], s["test_trips"])
        self.assertEqual(r["overall"]["shortest_path"]["n"], a["covered"])

    def test_no_test_leakage_in_baseline(self):
        r = run_validation(self.cfg, fold=5)
        summary = CleaningSummary()
        trips = list(iter_clean_trips(self.cfg, summary, with_hour=True))
        train = [t[4] for i, t in enumerate(trips, 1) if i % 5 != 0]
        self.assertAlmostEqual(r["split"]["global_mean_s"], sum(train) / len(train))

    def test_periods_and_determinism(self):
        r1 = run_validation(self.cfg, fold=4)
        r2 = run_validation(self.cfg, fold=4)
        self.assertEqual(r1, r2)
        per = r1["time_of_day"]["per_period"]
        self.assertEqual([p["period"] for p in per], [p[0] for p in PERIODS])
        self.assertEqual(sum(p["test_trips"] for p in per),
                         r1["test_trip_accounting"]["covered"]
                         + r1["test_trip_accounting"]["no_path"])
        for p in per:   # period graph is sparser: never covers more than all-day
            if p["test_trips"]:
                self.assertLessEqual(p["coverage_period_graph"], p["coverage_all_day"])

    def test_direct_first_consistent(self):
        r = run_validation(self.cfg, fold=5)
        ov = r["overall"]
        n_direct = ov["where_direct_edge_exists"]["trips"]
        n_none = ov["where_no_direct_edge"]["trips"]
        self.assertEqual(n_direct + n_none, ov["shortest_path"]["n"])
        self.assertEqual(ov["direct_first"]["n"], ov["shortest_path"]["n"])
        # direct-first == direct edge on direct subset, shortest path elsewhere
        total = (ov["where_direct_edge_exists"]["direct_edge"]["mae_s"] * n_direct
                 + ov["where_no_direct_edge"]["shortest_path"]["mae_s"] * n_none)
        self.assertAlmostEqual(ov["direct_first"]["mae_s"], total / (n_direct + n_none), places=6)
        tod = r["time_of_day"]["all_periods"]
        self.assertEqual(tod["period_direct_first_on_covered"]["n"],
                         tod["all_day_direct_first_on_covered"]["n"])

    def test_bad_fold(self):
        with self.assertRaises(ValueError):
            run_validation(self.cfg, fold=1)

    def test_hour_requires_datetime_column(self):
        bad = os.path.join(self.tmp.name, "nohour.csv")
        with open(bad, "w") as f:
            f.write("pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,trip_duration\n")
        with self.assertRaises(ValueError):
            run_validation(Config(csv_path=bad))


if __name__ == "__main__":
    unittest.main()

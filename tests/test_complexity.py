import math
import os
import tempfile
import unittest

from taxi_graph import traversal
from taxi_graph.complexity import (CountingHeap, full_dijkstra_counts, lazy_dijkstra,
                                   loglog_slope, random_graph, run_all, svg_loglog)
from taxi_graph.shortest_path import Dijkstra, INF
from taxi_graph.synthetic import generate


class MeasurementInvariants(unittest.TestCase):
    def setUp(self):
        self.g = random_graph(500, 6, seed=2)

    def test_random_graph_is_strongly_connected_and_sized(self):
        g = self.g
        self.assertEqual((g.n, g.m), (500, 3000))
        self.assertEqual(traversal.bfs(g, 0).reached, g.n)
        self.assertEqual(traversal.weak_components(g).count, 1)

    def test_full_dijkstra_touches_every_edge_once(self):
        c = full_dijkstra_counts(self.g)
        self.assertEqual(c["visited"], self.g.n)          # every vertex settled
        self.assertEqual(c["relaxed"], self.g.m)          # each edge examined exactly once
        self.assertEqual(c["inserts"], self.g.n)          # each vertex enters the heap once
        self.assertEqual(c["pops"], self.g.n)
        self.assertLessEqual(c["decrease_keys"], self.g.m - (self.g.n - 1))

    def test_bfs_examines_every_edge_once(self):
        r = traversal.bfs(self.g, 0)
        self.assertEqual((r.edges_examined, r.reached), (self.g.m, self.g.n))

    def test_lazy_reference_agrees_with_project_dijkstra(self):
        dist, pushes, pops = lazy_dijkstra(self.g, 0)
        dj = Dijkstra(self.g)
        for t in (1, 50, 499):
            self.assertAlmostEqual(dj.route(0, t).total_duration, dist[t], places=9)
        self.assertGreaterEqual(pushes, self.g.n)         # lazy heap holds duplicates
        self.assertEqual(pops, pushes)

    def test_counting_heap_counts(self):
        keys = [3.0, 1.0, 2.0]
        h = CountingHeap(keys)
        for v in range(3):
            h.push_or_update(v)
        keys[0] = 0.5
        h.push_or_update(0)
        while len(h):
            h.pop_min()
        self.assertEqual((h.inserts, h.updates, h.pops), (3, 1, 3))


class HelperTests(unittest.TestCase):
    def test_slope_recovers_exponents(self):
        xs = [10, 100, 1000, 10000]
        self.assertAlmostEqual(loglog_slope(xs, [x for x in xs]), 1.0, places=9)
        self.assertAlmostEqual(loglog_slope(xs, [x * x for x in xs]), 2.0, places=9)
        n_log_n = loglog_slope(xs, [x * math.log2(x) for x in xs])
        self.assertTrue(1.0 < n_log_n < 1.3)

    def test_svg_written(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.svg")
            svg_loglog(p, "t", "x", "y", [("a", [1, 10, 100], [0.1, 1, 10], "#000")])
            s = open(p).read()
            self.assertTrue(s.startswith("<svg") and s.rstrip().endswith("</svg>"))


class SmokeTest(unittest.TestCase):
    def test_quick_run_all_produces_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            csv = os.path.join(d, "t.csv")
            generate(csv, 30000, seed=1)
            res = run_all(csv, os.path.join(d, "out"), quick=True)
            for name in ("results.json", "build_vs_T.svg", "dijkstra_vs_size.svg",
                         "traversal_vs_size.svg", "heap_vs_n.svg"):
                self.assertTrue(os.path.exists(os.path.join(d, "out", name)), name)
            for row in res["dijkstra_size"]:                  # counts hold in the experiment too
                self.assertEqual(row["relaxed"], row["E"])
                self.assertEqual(row["visited"], row["V"])


if __name__ == "__main__":
    unittest.main()

import random
import unittest

from taxi_graph.graph import TravelGraph
from taxi_graph.heap import IndexedMinHeap
from taxi_graph.mst import kruskal
from taxi_graph.shortest_path import (Dijkstra, INF, LOCATION_NOT_IN_GRAPH, NO_PATH,
                                      SUCCESS)
from taxi_graph import traversal


def G(n, edges):
    return TravelGraph.from_edges(n, [(u, v, w, 1) for u, v, w in edges])


def bellman_ford(n, edges, s):
    d = [INF] * n
    d[s] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if d[u] + w < d[v]:
                d[v] = d[u] + w
    return d


class HeapTests(unittest.TestCase):
    def test_pop_order_and_decrease_key(self):
        keys = [5.0, 3.0, 8.0, 1.0, 9.0]
        h = IndexedMinHeap(keys)
        for v in range(5):
            h.push_or_update(v)
        keys[4] = 0.5
        h.push_or_update(4)
        order = [h.pop_min() for _ in range(5)]
        self.assertEqual(order, [4, 3, 1, 0, 2])
        self.assertEqual(len(h), 0)

    def test_clear_allows_reuse(self):
        keys = [2.0, 1.0]
        h = IndexedMinHeap(keys)
        h.push_or_update(0)
        h.push_or_update(1)
        h.clear()
        self.assertNotIn(0, h)
        h.push_or_update(0)
        self.assertEqual(h.pop_min(), 0)


class GraphTests(unittest.TestCase):
    def test_direction_preserved(self):
        g = G(2, [(0, 1, 5.0)])
        self.assertEqual(g.edge_weight(0, 1), 5.0)
        self.assertIsNone(g.edge_weight(1, 0))
        self.assertEqual((g.n, g.m), (2, 1))

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            G(2, [(0, 1, -1.0)])


class DijkstraTests(unittest.TestCase):
    def test_competing_paths(self):
        g = G(4, [(0, 1, 10), (0, 2, 3), (2, 1, 3), (1, 3, 1), (2, 3, 20)])
        r = Dijkstra(g).route(0, 3)
        self.assertEqual(r.status, SUCCESS)
        self.assertEqual(r.path, [0, 2, 1, 3])
        self.assertAlmostEqual(r.total_duration, 7)
        self.assertAlmostEqual(sum(r.edge_durations), r.total_duration)

    def test_same_vertex_zero(self):
        r = Dijkstra(G(2, [])).route(1, 1)
        self.assertEqual((r.status, r.path, r.total_duration), (SUCCESS, [1], 0.0))

    def test_unreachable_and_unknown(self):
        g = G(3, [(0, 1, 1)])
        d = Dijkstra(g)
        self.assertEqual(d.route(1, 0).status, NO_PATH)
        self.assertEqual(d.route(0, 2).status, NO_PATH)
        self.assertEqual(d.route(None, 1).status, LOCATION_NOT_IN_GRAPH)

    def test_workspace_reuse_is_clean(self):
        g = G(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        d = Dijkstra(g)
        for _ in range(3):
            self.assertAlmostEqual(d.route(0, 3).total_duration, 3)
            self.assertEqual(d.route(3, 0).status, NO_PATH)
        self.assertTrue(all(x == INF for x in d._dist))
        self.assertTrue(all(p == -1 for p in d._parent))

    def test_matches_bellman_ford_on_random_graphs(self):
        rng = random.Random(7)
        for _ in range(40):
            n = rng.randint(2, 25)
            edges = [(rng.randrange(n), rng.randrange(n), rng.uniform(0.1, 50))
                     for _ in range(rng.randint(0, 80))]
            edges = [e for e in edges if e[0] != e[1]]
            g = G(n, edges)
            d = Dijkstra(g)
            for s in range(n):
                ref = bellman_ford(n, edges, s)
                for t in range(n):
                    r = d.route(s, t)
                    if ref[t] == INF:
                        self.assertEqual(r.status, NO_PATH)
                    else:
                        self.assertAlmostEqual(r.total_duration, ref[t], places=9)


class TraversalTests(unittest.TestCase):
    def setUp(self):
        # 0->1->2, 3->2, 4 isolated
        self.g = G(5, [(0, 1, 1), (1, 2, 1), (3, 2, 1)])

    def test_bfs_levels(self):
        r = traversal.bfs(self.g, 0)
        self.assertEqual(r.levels, [0, 1, 2, -1, -1])
        self.assertEqual(r.reached, 3)

    def test_dfs_directed_reachability(self):
        self.assertEqual(sorted(traversal.dfs_reachable(self.g, 3)), [2, 3])

    def test_weak_components_incl_isolated(self):
        c = traversal.weak_components(self.g)
        self.assertEqual(c.count, 2)
        self.assertEqual(sorted(c.sizes), [1, 4])
        self.assertEqual(c.largest, 4)


class MSTTests(unittest.TestCase):
    def test_known_mst(self):
        g = G(4, [(0, 1, 1), (1, 0, 9), (1, 2, 2), (2, 3, 3), (0, 3, 10), (0, 2, 4)])
        edges, total = kruskal(g)
        self.assertEqual(len(edges), 3)
        self.assertAlmostEqual(total, 6)


if __name__ == "__main__":
    unittest.main()

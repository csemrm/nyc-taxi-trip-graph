"""Dijkstra with indexed binary min-heap: O((V + E) log V).

Workspace (dist/parent/heap) is allocated once. Each query resets only the
vertices it touched, so a query with early stop costs O(touched log touched),
not O(V) -- important for repeated queries on a large graph.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .heap import IndexedMinHeap

INF = float("inf")
SUCCESS = "SUCCESS"
NO_PATH = "NO_PATH"
LOCATION_NOT_IN_GRAPH = "LOCATION_NOT_IN_GRAPH"
INVALID_COORDINATES = "INVALID_COORDINATES"


@dataclass
class RouteResult:
    status: str
    source: Optional[int] = None
    destination: Optional[int] = None
    path: List[int] = field(default_factory=list)
    edge_durations: List[float] = field(default_factory=list)
    total_duration: Optional[float] = None
    visited_vertices: int = 0
    relaxed_edges: int = 0
    elapsed_s: float = 0.0
    graph_v: int = 0
    graph_e: int = 0

    def message(self):
        if self.status == NO_PATH:
            return "No path found."
        if self.status == LOCATION_NOT_IN_GRAPH:
            return "Location not in graph."
        if self.status == INVALID_COORDINATES:
            return "Invalid coordinates."
        return fmt_duration(self.total_duration)


def fmt_duration(sec):
    s = int(round(sec))
    return f"{s // 60} min {s % 60:02d} s"


class Dijkstra:
    def __init__(self, graph):
        self.g = graph
        self._dist = [INF] * graph.n
        self._parent = [-1] * graph.n
        self._heap = IndexedMinHeap(self._dist)

    def run(self, src, dst):
        """Return (found, total, path, visited, relaxed). Early stop at dst."""
        g = self.g
        off, tg, wt = g.offsets, g.targets, g.weights
        dist, parent, heap = self._dist, self._parent, self._heap
        touched = [src]
        dist[src] = 0.0
        heap.push_or_update(src)
        visited = relaxed = 0
        found = False
        while len(heap):
            u = heap.pop_min()
            visited += 1
            if u == dst:
                found = True
                break
            du = dist[u]
            for i in range(off[u], off[u + 1]):
                v = tg[i]
                nd = du + wt[i]
                relaxed += 1
                if nd < dist[v]:
                    if dist[v] == INF:
                        touched.append(v)
                    dist[v] = nd
                    parent[v] = u
                    heap.push_or_update(v)
        total = dist[dst] if found else None
        path = []
        if found:
            v = dst
            while v != -1:
                path.append(v)
                v = parent[v]
            path.reverse()
        heap.clear()
        for v in touched:
            dist[v] = INF
            parent[v] = -1
        return found, total, path, visited, relaxed

    def distances(self, src, targets):
        """Single-source Dijkstra that stops once every vertex in `targets` is settled.

        Returns (dict target -> shortest distance for reachable targets, visited).
        Lets many queries sharing a source cost one search: O((V+E) log V) per
        distinct source instead of per query.
        """
        g = self.g
        off, tg, wt = g.offsets, g.targets, g.weights
        dist, heap = self._dist, self._heap
        want = set(targets)
        want.discard(src)
        found = {}
        touched = [src]
        dist[src] = 0.0
        heap.push_or_update(src)
        visited = 0
        remaining = len(want)
        while remaining and len(heap):
            u = heap.pop_min()
            visited += 1
            du = dist[u]
            if u in want:
                found[u] = du
                remaining -= 1
                if not remaining:
                    break
            for i in range(off[u], off[u + 1]):
                v = tg[i]
                nd = du + wt[i]
                if nd < dist[v]:
                    if dist[v] == INF:
                        touched.append(v)
                    dist[v] = nd
                    heap.push_or_update(v)
        heap.clear()
        for v in touched:
            dist[v] = INF
        return found, visited

    def route(self, src, dst):
        """Full RouteResult for vertex ids; None ids => LOCATION_NOT_IN_GRAPH."""
        import time
        g = self.g
        res = RouteResult(status=NO_PATH, source=src, destination=dst,
                          graph_v=g.n, graph_e=g.m)
        if src is None or dst is None or not (0 <= src < g.n and 0 <= dst < g.n):
            res.status = LOCATION_NOT_IN_GRAPH
            return res
        t0 = time.perf_counter()
        if src == dst:
            res.status, res.path, res.total_duration = SUCCESS, [src], 0.0
            res.visited_vertices = 1
        else:
            found, total, path, vis, rel = self.run(src, dst)
            res.visited_vertices, res.relaxed_edges = vis, rel
            if found:
                res.status, res.path, res.total_duration = SUCCESS, path, total
                res.edge_durations = [g.edge_weight(a, b) for a, b in zip(path, path[1:])]
        res.elapsed_s = time.perf_counter() - t0
        return res

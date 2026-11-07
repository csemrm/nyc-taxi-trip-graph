"""EdgeAggregator + TravelGraph.

Aggregation: single pass over T trips, O(T) with hash map keyed by one packed int.
Graph: CSR adjacency lists (offsets/targets/weights), built by counting sort in
O(V + E). Total storage O(V + E). Plain lists are used for the hot-loop arrays
because list indexing is faster than array('d') in CPython.
"""
from statistics import median

_SHIFT = 32
_MASK = (1 << _SHIFT) - 1


class EdgeAggregator:
    def __init__(self, stat="mean"):
        if stat not in ("mean", "median"):
            raise ValueError("stat must be 'mean' or 'median'")
        self.stat = stat
        self._acc = {}          # packed (u,v) -> [sum, count] or list of durations
        self.self_loops = 0     # trips that start and end in the same cell

    def add(self, u, v, dur):
        if u == v:
            self.self_loops += 1
            return
        k = (u << _SHIFT) | v
        a = self._acc.get(k)
        if self.stat == "mean":
            if a is None:
                self._acc[k] = [dur, 1]
            else:
                a[0] += dur
                a[1] += 1
        else:
            if a is None:
                self._acc[k] = [dur]
            else:
                a.append(dur)

    def edges(self, min_trips=1):
        """Yield (u, v, weight, count) in insertion order (deterministic)."""
        mean = self.stat == "mean"
        for k, a in self._acc.items():
            if mean:
                cnt = a[1]
                if cnt < min_trips:
                    continue
                w = a[0] / cnt
            else:
                cnt = len(a)
                if cnt < min_trips:
                    continue
                w = median(a)
            yield k >> _SHIFT, k & _MASK, w, cnt


class TravelGraph:
    """Directed weighted graph in CSR form. Edge u->v does not imply v->u."""

    def __init__(self, n, offsets, targets, weights, counts):
        self.n = n
        self.offsets = offsets
        self.targets = targets
        self.weights = weights
        self.counts = counts
        self._rev = None

    @property
    def m(self):
        return len(self.targets)

    @classmethod
    def from_edges(cls, n, edges):
        """Counting-sort edges by source: O(V + E). Non-negative weights enforced."""
        edges = list(edges)
        deg = [0] * (n + 1)
        for u, v, w, c in edges:
            if w < 0:
                raise ValueError("negative edge weight not allowed")
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError("edge endpoint out of range")
            deg[u + 1] += 1
        for i in range(n):
            deg[i + 1] += deg[i]
        offsets = deg
        pos = offsets[:-1]
        m = len(edges)
        targets = [0] * m
        weights = [0.0] * m
        counts = [0] * m
        for u, v, w, c in edges:
            p = pos[u]
            targets[p] = v
            weights[p] = w
            counts[p] = c
            pos[u] = p + 1
        return cls(n, offsets, targets, weights, counts)

    def neighbors(self, u):
        o = self.offsets
        return [(self.targets[i], self.weights[i]) for i in range(o[u], o[u + 1])]

    def edge_weight(self, u, v):
        """Direct-edge weight or None. O(out-degree(u))."""
        o, t = self.offsets, self.targets
        for i in range(o[u], o[u + 1]):
            if t[i] == v:
                return self.weights[i]
        return None

    def reverse_adjacency(self):
        """Reverse CSR (offsets, sources); cached. O(V + E)."""
        if self._rev is None:
            n, o, t = self.n, self.offsets, self.targets
            roff = [0] * (n + 1)
            for v in t:
                roff[v + 1] += 1
            for i in range(n):
                roff[i + 1] += roff[i]
            pos = roff[:-1]
            src = [0] * len(t)
            for u in range(n):
                for i in range(o[u], o[u + 1]):
                    v = t[i]
                    src[pos[v]] = u
                    pos[v] += 1
            self._rev = (roff, src)
        return self._rev

    def undirected_edges(self):
        """Yield (u, v, w) with u<v, w = min of the two directions (documented MST rule)."""
        best = {}
        o, t, w = self.offsets, self.targets, self.weights
        for u in range(self.n):
            for i in range(o[u], o[u + 1]):
                v = t[i]
                k = (u, v) if u < v else (v, u)
                cur = best.get(k)
                if cur is None or w[i] < cur:
                    best[k] = w[i]
        for (u, v), wt in best.items():
            yield u, v, wt

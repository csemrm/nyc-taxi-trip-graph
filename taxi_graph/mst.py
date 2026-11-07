"""Optional (FR-16): Kruskal MST on the undirected cost graph. O(E log E).

Undirected rule: weight = min of the two directed weights. This is connectivity
analysis, NOT a trip-duration estimate.
"""


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]       # path halving
            x = p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1
        return True


def kruskal(g):
    """Return (edges, total_cost): a minimum spanning forest (V - components edges)."""
    edges = sorted(g.undirected_edges(), key=lambda e: (e[2], e[0], e[1]))
    uf = UnionFind(g.n)
    chosen, total = [], 0.0
    for u, v, w in edges:
        if uf.union(u, v):
            chosen.append((u, v, w))
            total += w
    return chosen, total

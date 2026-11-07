"""BFS and DFS, all O(V + E), iterative (no recursion limit issues).

DFS convention (stated explicitly per SP3 6.2):
  * dfs_reachable      -> DIRECTED reachability from a source.
  * weak_components    -> WEAKLY connected components (edge direction ignored),
                          using forward + reverse adjacency.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class BFSResult:
    levels: List[int]          # hop level per vertex, -1 if unreachable
    reached: int
    edges_examined: int


def bfs(g, src):
    off, tg = g.offsets, g.targets
    level = [-1] * g.n
    level[src] = 0
    q = [src]
    head = 0
    examined = 0
    while head < len(q):
        u = q[head]
        head += 1
        lu = level[u] + 1
        for i in range(off[u], off[u + 1]):
            examined += 1
            v = tg[i]
            if level[v] == -1:
                level[v] = lu
                q.append(v)
    return BFSResult(level, len(q), examined)


def dfs_reachable(g, src):
    """Directed DFS from src. Returns visited vertices in discovery order."""
    off, tg = g.offsets, g.targets
    seen = [False] * g.n
    seen[src] = True
    stack = [src]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for i in range(off[u], off[u + 1]):
            v = tg[i]
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return order


@dataclass
class Components:
    comp: List[int]            # component id per vertex
    sizes: List[int]

    @property
    def count(self):
        return len(self.sizes)

    @property
    def largest(self):
        return max(self.sizes) if self.sizes else 0


def weak_components(g):
    """DFS over every unvisited vertex; isolated vertices form their own component."""
    off, tg = g.offsets, g.targets
    roff, rsrc = g.reverse_adjacency()
    comp = [-1] * g.n
    sizes = []
    for s in range(g.n):
        if comp[s] != -1:
            continue
        cid = len(sizes)
        comp[s] = cid
        stack = [s]
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for i in range(off[u], off[u + 1]):
                v = tg[i]
                if comp[v] == -1:
                    comp[v] = cid
                    stack.append(v)
            for i in range(roff[u], roff[u + 1]):
                v = rsrc[i]
                if comp[v] == -1:
                    comp[v] = cid
                    stack.append(v)
        sizes.append(size)
    return Components(comp, sizes)

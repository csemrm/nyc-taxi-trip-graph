"""SP4 complexity experiments: measure running time and operation counts against Big-O.

Controlled experiments use deterministic random graphs (sizes and density chosen
independently), so V and E can be varied separately; real-data experiments use the
NYC graph. Every experiment records (size, time) pairs and a fitted log-log slope
(slope ~1 => linear; n log n gives a slightly larger slope over these ranges).
Standard library only; SVG plots are written by hand.
"""
import heapq
import json
import math
import os
import random
import time
import tracemalloc

from . import traversal
from .config import Config
from .graph import EdgeAggregator, TravelGraph
from .heap import IndexedMinHeap
from .loader import CleaningSummary, iter_clean_trips
from .location import LocationIndex, LocationNormalizer
from .mst import kruskal
from .pipeline import build
from .shortest_path import Dijkstra, INF


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def best_of(fn, repeats=3):
    """Return (result_of_last_call, best elapsed seconds)."""
    best = INF
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def loglog_slope(xs, ys):
    """Least-squares slope of log(y) vs log(x)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(xs)
    mx, my = sum(lx) / n, sum(ly) / n
    den = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den else float("nan")


def random_graph(n, avg_deg, seed=1):
    """Strongly connected sparse digraph: a ring plus random edges, positive weights."""
    rng = random.Random(seed)
    edges = {}
    for i in range(n):
        edges[(i, (i + 1) % n)] = rng.uniform(1.0, 100.0)
    target = n * avg_deg
    while len(edges) < target:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and (u, v) not in edges:
            edges[(u, v)] = rng.uniform(1.0, 100.0)
    return TravelGraph.from_edges(n, ((u, v, w, 1) for (u, v), w in edges.items()))


class CountingHeap(IndexedMinHeap):
    """Indexed heap that counts operations (used only for measurement)."""

    def __init__(self, keys):
        super().__init__(keys)
        self.inserts = self.updates = self.pops = 0

    def push_or_update(self, v):
        if v in self:
            self.updates += 1
        else:
            self.inserts += 1
        super().push_or_update(v)

    def pop_min(self):
        self.pops += 1
        return super().pop_min()


def lazy_dijkstra(g, src):
    """Reference: heapq (C implementation) with lazy deletion. Returns (dist, pushes, pops)."""
    off, tg, wt = g.offsets, g.targets, g.weights
    dist = [INF] * g.n
    dist[src] = 0.0
    pq = [(0.0, src)]
    pushes, pops = 1, 0
    while pq:
        d, u = heapq.heappop(pq)
        pops += 1
        if d > dist[u]:
            continue
        for i in range(off[u], off[u + 1]):
            v = tg[i]
            nd = d + wt[i]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
                pushes += 1
    return dist, pushes, pops


def lazy_dijkstra_to(g, src, dst):
    """Reference lazy heapq Dijkstra with early stop at dst. Returns (distance or None, pops)."""
    off, tg, wt = g.offsets, g.targets, g.weights
    dist = {src: 0.0}
    pq = [(0.0, src)]
    pops = 0
    while pq:
        d, u = heapq.heappop(pq)
        pops += 1
        if d > dist.get(u, INF):
            continue
        if u == dst:
            return d, pops
        for i in range(off[u], off[u + 1]):
            v = tg[i]
            nd = d + wt[i]
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return None, pops


def full_dijkstra_counts(g, src=0):
    """Full single-source run on the project's Dijkstra with a counting heap."""
    dj = Dijkstra(g)
    dj._heap = CountingHeap(dj._dist)
    found, total, path, visited, relaxed = dj.run(src, -1)   # dst=-1: never found -> full run
    h = dj._heap
    return {"visited": visited, "relaxed": relaxed, "inserts": h.inserts,
            "decrease_keys": h.updates, "pops": h.pops}


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def exp_build_scaling(csv_path, sizes):
    """Build pipeline vs number of trips T (real data), split into phases."""
    rows = []
    for t in sizes:
        cfg = Config(csv_path=csv_path, max_rows=t)

        def load_only():
            s = CleaningSummary()
            n = 0
            for _ in iter_clean_trips(cfg, s):
                n += 1
            return n, s.read

        (kept, read), t_load = best_of(load_only, 2)
        b, t_total = best_of(lambda: build(cfg), 2)
        rows.append({"T_read": read, "T_kept": kept, "V": b.graph.n, "E": b.graph.m,
                     "load_clean_s": t_load, "build_total_s": t_total,
                     "aggregate_and_graph_s": max(t_total - t_load, 0.0),
                     "us_per_trip": t_total / read * 1e6})
    return rows


def exp_hash(sizes, lookups=200000):
    rows = []
    rng = random.Random(3)
    for v in sizes:
        d = {k * 7919 + 13: k for k in range(v)}
        keys = [rng.randrange(v) * 7919 + 13 for _ in range(lookups)]

        def run():
            g = d.get
            s = 0
            for k in keys:
                s += g(k)
            return s
        _, t = best_of(run, 3)
        rows.append({"V": v, "ns_per_lookup": t / lookups * 1e9})
    return rows


def exp_heap(sizes):
    rows = []
    for n in sizes:
        rng = random.Random(5)
        keys = [rng.random() for _ in range(n)]

        def run():
            h = IndexedMinHeap(keys)
            for v in range(n):
                h.push_or_update(v)
            while len(h):
                h.pop_min()
        _, t = best_of(run, 2)
        rows.append({"n": n, "total_s": t, "us_per_op": t / (2 * n) * 1e6,
                     "us_per_op_per_log2n": t / (2 * n) / math.log2(n) * 1e6})
    return rows


def exp_dijkstra_vs_size(vs, deg=8):
    """Full Dijkstra vs graph size, E = deg*V; also lazy heapq reference."""
    rows = []
    for v in vs:
        g = random_graph(v, deg)
        dj = Dijkstra(g)
        _, t_idx = best_of(lambda: dj.run(0, -1), 2)
        _, t_lazy = best_of(lambda: lazy_dijkstra(g, 0), 2)
        c = full_dijkstra_counts(g)
        _, lz_push, lz_pops = lazy_dijkstra(g, 0)
        bound = (g.n + g.m) * math.log2(g.n)
        rows.append({"V": g.n, "E": g.m, "indexed_s": t_idx, "lazy_heapq_s": t_lazy,
                     "bound_(V+E)log2V": bound, "indexed_s_per_bound_us": t_idx / bound * 1e6,
                     **c, "lazy_pushes": lz_push, "lazy_pops": lz_pops})
    return rows


def exp_dijkstra_vs_density(v, degs):
    rows = []
    for deg in degs:
        g = random_graph(v, deg)
        dj = Dijkstra(g)
        _, t = best_of(lambda: dj.run(0, -1), 2)
        c = full_dijkstra_counts(g)
        rows.append({"V": g.n, "E": g.m, "avg_deg": deg, "indexed_s": t, **c})
    return rows


def exp_traversal(vs, deg=8):
    rows = []
    for v in vs:
        g = random_graph(v, deg)
        r, t_bfs = best_of(lambda: traversal.bfs(g, 0), 3)
        _, t_dfs = best_of(lambda: traversal.dfs_reachable(g, 0), 3)
        _, t_cc = best_of(lambda: traversal.weak_components(g), 2)
        rows.append({"V": g.n, "E": g.m, "V_plus_E": g.n + g.m, "bfs_s": t_bfs, "dfs_s": t_dfs,
                     "weak_cc_s": t_cc, "bfs_edges_examined": r.edges_examined,
                     "bfs_reached": r.reached})
    return rows


def exp_kruskal(es, v=20000):
    rows = []
    for deg in es:
        g = random_graph(v, deg)
        und = list(g.undirected_edges())
        _, t_sort = best_of(lambda: sorted(und, key=lambda e: (e[2], e[0], e[1])), 3)
        _, t_total = best_of(lambda: kruskal(g), 2)
        rows.append({"V": g.n, "E_directed": g.m, "E_undirected": len(und), "sort_s": t_sort,
                     "kruskal_total_s": t_total})
    return rows


def exp_space(vs, deg=8):
    rows = []
    for v in vs:
        g0 = random_graph(v, deg)
        edges = [(u, g0.targets[i], g0.weights[i], 1)
                 for u in range(g0.n) for i in range(g0.offsets[u], g0.offsets[u + 1])]
        tracemalloc.start()
        g = TravelGraph.from_edges(g0.n, edges)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"V": g.n, "E": g.m, "csr_bytes": cur, "peak_bytes": peak,
                     "bytes_per_edge": cur / g.m, "bytes_per_(V+E)": cur / (g.n + g.m)})
    return rows


def exp_real_queries(csv_path, n_queries=400, seed=11):
    """Early stopping on the real graph, compared on the SAME (source, destination) pairs.

    Only pairs with a path are used (a failed search explores the whole reachable set,
    which is a different case, reported separately as the unreachable share). For each
    pair we run, from the same source: indexed-heap Dijkstra with early stop, the same
    without early stop (full), and the lazy heapq reference with early stop.
    """
    b = build(Config(csv_path=csv_path))
    g = b.graph
    dj = Dijkstra(g)
    rng = random.Random(seed)
    full_cache = {}                       # source -> (visited, relaxed, seconds) of a full run
    rec = {"vis_early": [], "vis_full": [], "rel_early": [], "rel_full": [],
           "t_early": [], "t_full": [], "t_lazy_early": [], "pops_lazy": []}
    attempts = unreachable = 0
    while len(rec["vis_early"]) < n_queries:
        s, t = rng.randrange(g.n), rng.randrange(g.n)
        if s == t:
            continue
        attempts += 1
        t0 = time.perf_counter()
        found, total, path, vis, rel = dj.run(s, t)
        dt = time.perf_counter() - t0
        if not found:
            unreachable += 1
            continue
        if s not in full_cache:
            t0 = time.perf_counter()
            _, _, _, fvis, frel = dj.run(s, -1)
            full_cache[s] = (fvis, frel, time.perf_counter() - t0)
        fvis, frel, ft = full_cache[s]
        t0 = time.perf_counter()
        dist, pops = lazy_dijkstra_to(g, s, t)
        lt = time.perf_counter() - t0
        assert dist is not None and abs(dist - total) < 1e-6   # same answer as project Dijkstra
        rec["vis_early"].append(vis); rec["vis_full"].append(fvis)
        rec["rel_early"].append(rel); rec["rel_full"].append(frel)
        rec["t_early"].append(dt); rec["t_full"].append(ft)
        rec["t_lazy_early"].append(lt); rec["pops_lazy"].append(pops)

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(int(len(xs) * p), len(xs) - 1)]
    mean = lambda xs: sum(xs) / len(xs)
    ratio = lambda a, b_: mean([x / y for x, y in zip(a, b_)])
    return {
        "V": g.n, "E": g.m, "reachable_pairs_used": n_queries,
        "random_pairs_tried": attempts, "unreachable_share": unreachable / attempts,
        "visited_early_mean": mean(rec["vis_early"]),
        "visited_early_median": pct(rec["vis_early"], 0.5),
        "visited_early_p90": pct(rec["vis_early"], 0.9),
        "visited_full_same_source_mean": mean(rec["vis_full"]),
        "visited_ratio_early_over_full_mean": ratio(rec["vis_early"], rec["vis_full"]),
        "relaxed_ratio_early_over_full_mean": ratio(rec["rel_early"], rec["rel_full"]),
        "time_ms": {"indexed_early_mean": mean(rec["t_early"]) * 1e3,
                    "indexed_early_p90": pct(rec["t_early"], 0.9) * 1e3,
                    "indexed_full_mean": mean(rec["t_full"]) * 1e3,
                    "lazy_heapq_early_mean": mean(rec["t_lazy_early"]) * 1e3},
        "time_ratio_early_over_full_mean": ratio(rec["t_early"], rec["t_full"]),
        "time_ratio_lazy_over_indexed_early_mean": ratio(rec["t_lazy_early"], rec["t_early"]),
    }


# ---------------------------------------------------------------------------
# SVG plots (log-log)
# ---------------------------------------------------------------------------
def svg_loglog(path, title, xlabel, ylabel, series):
    """series: [(label, xs, ys, color)]. Writes a self-contained SVG."""
    W, H, L, R, T, B = 640, 400, 70, 20, 40, 55
    xs_all = [x for _, xs, _, _ in series for x in xs]
    ys_all = [y for _, _, ys, _ in series for y in ys]
    x0, x1 = math.floor(math.log10(min(xs_all))), math.ceil(math.log10(max(xs_all)))
    y0, y1 = math.floor(math.log10(min(ys_all))), math.ceil(math.log10(max(ys_all)))
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1

    def px(x):
        return L + (math.log10(x) - x0) / (x1 - x0) * (W - L - R)

    def py(y):
        return H - B - (math.log10(y) - y0) / (y1 - y0) * (H - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="sans-serif" font-size="12"><rect width="{W}" height="{H}" fill="#fff"/>',
         f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="14" font-weight="bold" '
         f'fill="#222">{title}</text>']
    for e in range(x0, x1 + 1):
        x = px(10 ** e)
        o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" stroke="#e3e6eb"/>')
        o.append(f'<text x="{x:.1f}" y="{H - B + 16}" text-anchor="middle" fill="#444">1e{e}</text>')
    for e in range(y0, y1 + 1):
        y = py(10 ** e)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="#e3e6eb"/>')
        o.append(f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#444">1e{e}</text>')
    o.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="none" stroke="#999"/>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle" fill="#222">{xlabel}</text>')
    o.append(f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" '
             f'fill="#222">{ylabel}</text>')
    for i, (label, xs, ys, color) in enumerate(series):
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            o.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>')
        ly = T + 16 + i * 17
        o.append(f'<line x1="{L + 10}" y1="{ly - 4}" x2="{L + 30}" y2="{ly - 4}" stroke="{color}" stroke-width="2"/>')
        o.append(f'<text x="{L + 36}" y="{ly}" fill="#222">{label}</text>')
    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_all(csv_path, out_dir, quick=False):
    os.makedirs(out_dir, exist_ok=True)
    if quick:
        t_sizes, hv, hp = [20000, 40000], [10 ** 3, 10 ** 4], [2 ** 10, 2 ** 12]
        dv, dens, trv, ks, sv = [2 ** 9, 2 ** 10, 2 ** 11], [2, 4, 8], [2 ** 9, 2 ** 10, 2 ** 11], [2, 4, 8], [2 ** 9, 2 ** 10]
        nq, dv_fixed = 40, 2 ** 10
    else:
        t_sizes = [50000, 100000, 200000, 400000, 800000, 1458644]
        hv, hp = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6], [2 ** k for k in range(10, 19, 2)]
        dv = [2 ** k for k in range(10, 17)]
        dens, dv_fixed = [2, 4, 8, 16, 32, 64], 2 ** 14
        trv, ks, sv = [2 ** k for k in range(10, 17)], [2, 4, 8, 16, 32], [2 ** k for k in range(10, 17)]
        nq = 400
    import platform
    import sys
    res = {"environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                           "machine": platform.machine(), "processor": platform.processor(),
                           "note": "single run on one machine, no CPU pinning; timings are best-of-N"}}
    print("build scaling ...", flush=True); res["build"] = exp_build_scaling(csv_path, t_sizes)
    print("hash ...", flush=True); res["hash"] = exp_hash(hv)
    print("heap ...", flush=True); res["heap"] = exp_heap(hp)
    print("dijkstra vs size ...", flush=True); res["dijkstra_size"] = exp_dijkstra_vs_size(dv)
    print("dijkstra vs density ...", flush=True); res["dijkstra_density"] = exp_dijkstra_vs_density(dv_fixed, dens)
    print("traversal ...", flush=True); res["traversal"] = exp_traversal(trv)
    print("kruskal ...", flush=True); res["kruskal"] = exp_kruskal(ks)
    print("space ...", flush=True); res["space"] = exp_space(sv)
    print("real queries ...", flush=True); res["real_queries"] = exp_real_queries(csv_path, nq)

    sl = {}
    b = res["build"]
    sl["build_vs_T"] = loglog_slope([r["T_read"] for r in b], [r["build_total_s"] for r in b])
    sl["load_clean_vs_T"] = loglog_slope([r["T_read"] for r in b], [r["load_clean_s"] for r in b])
    sl["hash_ns_vs_V"] = loglog_slope([r["V"] for r in res["hash"]], [r["ns_per_lookup"] for r in res["hash"]])
    h = res["heap"]
    sl["heap_total_vs_n"] = loglog_slope([r["n"] for r in h], [r["total_s"] for r in h])
    d = res["dijkstra_size"]
    sl["dijkstra_indexed_vs_V+E"] = loglog_slope([r["V"] + r["E"] for r in d], [r["indexed_s"] for r in d])
    sl["dijkstra_lazy_vs_V+E"] = loglog_slope([r["V"] + r["E"] for r in d], [r["lazy_heapq_s"] for r in d])
    dd = res["dijkstra_density"]
    sl["dijkstra_indexed_vs_E_fixedV"] = loglog_slope([r["E"] for r in dd], [r["indexed_s"] for r in dd])
    tr = res["traversal"]
    for k in ("bfs_s", "dfs_s", "weak_cc_s"):
        sl[f"{k}_vs_V+E"] = loglog_slope([r["V_plus_E"] for r in tr], [r[k] for r in tr])
    kr = res["kruskal"]
    sl["kruskal_vs_E"] = loglog_slope([r["E_undirected"] for r in kr], [r["kruskal_total_s"] for r in kr])
    sl["kruskal_sort_vs_E"] = loglog_slope([r["E_undirected"] for r in kr], [r["sort_s"] for r in kr])
    sp = res["space"]
    sl["csr_bytes_vs_V+E"] = loglog_slope([r["V"] + r["E"] for r in sp], [r["csr_bytes"] for r in sp])
    res["loglog_slopes"] = sl

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    svg_loglog(os.path.join(out_dir, "build_vs_T.svg"), "Graph build time vs trips (real data)",
               "trips read (T)", "seconds",
               [("total build", [r["T_read"] for r in b], [r["build_total_s"] for r in b], "#1a6fd6"),
                ("load + clean only", [r["T_read"] for r in b], [r["load_clean_s"] for r in b], "#d6781a")])
    svg_loglog(os.path.join(out_dir, "dijkstra_vs_size.svg"), "Full Dijkstra vs graph size (E = 8V)",
               "V + E", "seconds",
               [("indexed heap (project)", [r["V"] + r["E"] for r in d], [r["indexed_s"] for r in d], "#1a6fd6"),
                ("lazy heapq (reference)", [r["V"] + r["E"] for r in d], [r["lazy_heapq_s"] for r in d], "#2e9e4f")])
    svg_loglog(os.path.join(out_dir, "traversal_vs_size.svg"), "BFS / DFS / components vs V + E",
               "V + E", "seconds",
               [("BFS", [r["V_plus_E"] for r in tr], [r["bfs_s"] for r in tr], "#1a6fd6"),
                ("DFS", [r["V_plus_E"] for r in tr], [r["dfs_s"] for r in tr], "#d6781a"),
                ("weak components", [r["V_plus_E"] for r in tr], [r["weak_cc_s"] for r in tr], "#2e9e4f")])
    svg_loglog(os.path.join(out_dir, "heap_vs_n.svg"), "Indexed heap: n inserts + n pops",
               "n", "seconds",
               [("total time", [r["n"] for r in h], [r["total_s"] for r in h], "#1a6fd6")])
    return res

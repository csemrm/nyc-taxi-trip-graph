"""Command line: generate | stats | query | bench | mst."""
import argparse
import json
import random
import sys

from . import traversal
from .config import Config
from .mst import kruskal
from .perf import measure
from .pipeline import Engine, build
from .shortest_path import Dijkstra, SUCCESS, fmt_duration
from .synthetic import generate


def _add_data_args(p):
    p.add_argument("--csv", required=True, help="trip CSV (Kaggle NYC taxi columns)")
    p.add_argument("--max-rows", type=int, default=None, help="use first N rows")
    p.add_argument("--cell-deg", type=float, default=0.005)
    p.add_argument("--stat", choices=["mean", "median"], default="mean")
    p.add_argument("--min-trips", type=int, default=1)
    p.add_argument("--max-duration", type=float, default=6 * 3600.0)
    p.add_argument("--hours", nargs=2, type=int, metavar=("START", "END"),
                   help="only trips with pickup hour in [START, END)")


def _cfg(a):
    return Config(csv_path=a.csv, max_rows=a.max_rows, cell_deg=a.cell_deg, stat=a.stat,
                  min_trips=a.min_trips, max_duration=a.max_duration,
                  hour_range=tuple(a.hours) if a.hours else None)


def cmd_generate(a):
    generate(a.out, a.trips, seed=a.seed)
    print(f"wrote {a.trips} synthetic trips to {a.out}")


def cmd_stats(a):
    b = build(_cfg(a))
    g = b.graph
    bfs_r = traversal.bfs(g, 0) if g.n else None
    comps = traversal.weak_components(g)
    print(json.dumps({
        "config": b.config, "cleaning": b.cleaning.as_dict(),
        "self_loop_trips_skipped": b.self_loops,
        "V": g.n, "E": g.m, "build_s": round(b.build_s, 3),
        "weak_components": comps.count, "largest_component": comps.largest,
        "bfs_from_v0_reached": bfs_r.reached if bfs_r else 0,
    }, indent=2))


def cmd_query(a):
    b = build(_cfg(a))
    eng = Engine(b)
    r = eng.route(a.src[0], a.src[1], a.dst[0], a.dst[1])
    print(f"status: {r.status}")
    if r.status == SUCCESS:
        print(f"estimated duration: {fmt_duration(r.total_duration)} ({r.total_duration:.1f} s)")
        print(f"route: {' -> '.join(map(str, r.path))}")
        for (u, v), w in zip(zip(r.path, r.path[1:]), r.edge_durations):
            print(f"  {u} -> {v}: {w:.1f} s")
        direct = b.graph.edge_weight(r.source, r.destination) if r.source != r.destination else None
        if direct is not None:
            print(f"direct edge: {direct:.1f} s (shortest path is {direct - r.total_duration:.1f} s lower)")
        print(f"visited={r.visited_vertices} relaxed={r.relaxed_edges} time={r.elapsed_s * 1e3:.3f} ms "
              f"V={r.graph_v} E={r.graph_e}")
    else:
        print(r.message())


def cmd_bench(a):
    print("rows,V,E,build_s,bfs_ms,weak_cc_ms,dijkstra_avg_ms,dijkstra_avg_visited,peak_mb")
    for n in a.sizes:
        cfg = _cfg(a.__class__(**{**vars(a), "max_rows": n}))
        b, _, peak = measure(build, cfg, memory=a.memory)
        g = b.graph
        if g.n == 0:
            continue
        rng = random.Random(a.seed)
        _, t_bfs, _ = measure(traversal.bfs, g, 0)
        _, t_cc, _ = measure(traversal.weak_components, g)
        dj = Dijkstra(g)
        tot_t = tot_v = 0
        for _ in range(a.queries):
            r = dj.route(rng.randrange(g.n), rng.randrange(g.n))
            tot_t += r.elapsed_s
            tot_v += r.visited_vertices
        q = max(a.queries, 1)
        print(f"{n},{g.n},{g.m},{b.build_s:.3f},{t_bfs * 1e3:.2f},{t_cc * 1e3:.2f},"
              f"{tot_t / q * 1e3:.3f},{tot_v / q:.1f},"
              f"{'' if peak is None else f'{peak / 1e6:.1f}'}")


def cmd_mst(a):
    b = build(_cfg(a))
    edges, total = kruskal(b.graph)
    print(f"MST forest edges={len(edges)} total_cost={total:.1f} s (connectivity analysis, not a trip estimate)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="taxi_graph", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write deterministic synthetic CSV")
    g.add_argument("--out", required=True)
    g.add_argument("--trips", type=int, default=100000)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("stats", help="build graph, print cleaning + graph stats")
    _add_data_args(s)
    s.set_defaults(fn=cmd_stats)

    q = sub.add_parser("query", help="estimate duration between two coordinates")
    _add_data_args(q)
    q.add_argument("--from", dest="src", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    q.add_argument("--to", dest="dst", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    q.set_defaults(fn=cmd_query)

    bn = sub.add_parser("bench", help="scaling benchmark over row counts")
    _add_data_args(bn)
    bn.add_argument("--sizes", nargs="+", type=int, default=[10000, 50000, 100000])
    bn.add_argument("--queries", type=int, default=200)
    bn.add_argument("--seed", type=int, default=1)
    bn.add_argument("--memory", action="store_true", help="track peak memory (slower)")
    bn.set_defaults(fn=cmd_bench)

    m = sub.add_parser("mst", help="optional Kruskal MST analysis")
    _add_data_args(m)
    m.set_defaults(fn=cmd_mst)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line: generate | stats | query | bench | mst | validate."""
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
from .validation import run_validation


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
    r, est, basis = eng.estimate(a.src[0], a.src[1], a.dst[0], a.dst[1])
    print(f"status: {r.status}")
    if r.status == SUCCESS:
        print(f"best estimate: {fmt_duration(est)} ({est:.1f} s, basis: "
              f"{'direct edge' if basis == 'direct' else 'shortest path'})")
        print(f"shortest-path duration: {fmt_duration(r.total_duration)} ({r.total_duration:.1f} s)")
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


def _row(label, m):
    if m is None:
        return f"  {label:<28} (no data)"
    return (f"  {label:<28} n={m['n']:<7} MAE={m['mae_s']:7.1f}s  medAE={m['median_ae_s']:7.1f}s  "
            f"RMSE={m['rmse_s']:7.1f}s  bias={m['bias_s']:+7.1f}s  "
            f"MAPE={m['mape'] * 100:5.1f}%  medAPE={m['median_ape'] * 100:5.1f}%  "
            f"<=20%: {m['within_20pct'] * 100:4.1f}%")


def print_validation(r):
    sp, acc, ov = r["split"], r["test_trip_accounting"], r["overall"]
    print(f"split: 1 in {sp['fold']} trips held out -> train={sp['train_trips']} "
          f"test={sp['test_trips']}; graph V={r['graph']['V']} E={r['graph']['E_all_day']}")
    print("test trips: same-cell excluded={same_cell_excluded}, unseen in train="
          "{origin_or_dest_unseen_in_train}, no path={no_path}, covered={covered} "
          "({pct:.2f}% of predictable)".format(pct=acc["coverage_of_predictable"] * 100, **acc))
    print("\nOverall accuracy (covered trips):")
    print(_row("shortest path (Dijkstra)", ov["shortest_path"]))
    print(_row("direct edge, else path", ov["direct_first"]))
    print(_row("global mean baseline", ov["global_mean_baseline"]))
    nd = ov["where_no_direct_edge"]
    print(f"\nWhere no direct edge exists ({nd['trips']} trips):")
    print(_row("shortest path", nd["shortest_path"]))
    d = ov["where_direct_edge_exists"]
    print(f"\nWhere a direct edge also exists ({d['trips']} trips):")
    print(_row("shortest path", d["shortest_path"]))
    print(_row("direct edge only", d["direct_edge"]))
    tod = r["time_of_day"]
    print("\nTime-of-day graphs vs all-day graph (same trips, covered by both):")
    for p in tod["per_period"]:
        print(f"[{p['period']} {p['hours'][0]:02d}-{p['hours'][1]:02d}h] test={p['test_trips']} "
              f"mean actual={p['mean_actual_s'] or 0:.0f}s edges={p['graph_edges']} "
              f"coverage all-day={((p['coverage_all_day'] or 0) * 100):.1f}% "
              f"period={((p['coverage_period_graph'] or 0) * 100):.1f}%")
        print(_row("all-day graph", p["all_day_on_common"]))
        print(_row("period graph", p["period_on_common"]))
    a = tod["all_periods"]
    print("[all periods]")
    print(_row("all-day graph (common)", a["all_day_on_common"]))
    print(_row("period graph (common)", a["period_on_common"]))
    print(_row("all-day graph (covered)", a["all_day_on_covered"]))
    print(_row("hybrid: period else all-day", a["hybrid_on_covered"]))
    print("Direct edge first, else shortest path (covered trips):")
    print(_row("all-day graph", a["all_day_direct_first_on_covered"]))
    print(_row("period graph", a["period_direct_first_on_covered"]))


def cmd_validate(a):
    r = run_validation(_cfg(a), fold=a.fold)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
    print_validation(r)


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

    v = sub.add_parser("validate", help="held-out accuracy + time-of-day graph comparison")
    _add_data_args(v)
    v.add_argument("--fold", type=int, default=5, help="hold out every FOLD-th trip (default 5 = 20%%)")
    v.add_argument("--out", help="also write full results as JSON")
    v.set_defaults(fn=cmd_validate)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

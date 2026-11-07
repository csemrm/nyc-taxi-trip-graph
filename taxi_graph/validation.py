"""Held-out validation (accuracy of the graph estimator) and time-of-day graphs.

Not machine learning: the "model" is the aggregated graph plus Dijkstra. Trips are
split deterministically (every `fold`-th cleaned trip is test, the rest train); the
graph, location index and baseline use TRAIN trips only. Each test trip is then
predicted as the Dijkstra distance between its origin and destination cells and
compared with its actual duration.

Cost: one pass over the CSV, then one multi-target Dijkstra per distinct test
origin (not per trip).
"""
import statistics

from .graph import EdgeAggregator, TravelGraph
from .loader import CleaningSummary, iter_clean_trips
from .location import LocationIndex, LocationNormalizer
from .shortest_path import Dijkstra

PERIODS = [("night", 0, 6), ("am_rush", 6, 10), ("midday", 10, 16),
           ("pm_rush", 16, 20), ("evening", 20, 24)]


def metrics(pairs):
    """pairs: list of (predicted, actual) seconds -> error summary dict (or None)."""
    n = len(pairs)
    if not n:
        return None
    err = [p - a for p, a in pairs]
    ae = [abs(e) for e in err]
    ape = [abs(p - a) / a for p, a in pairs]
    return {
        "n": n,
        "mae_s": sum(ae) / n,
        "median_ae_s": statistics.median(ae),
        "rmse_s": (sum(e * e for e in err) / n) ** 0.5,
        "bias_s": sum(err) / n,
        "mape": sum(ape) / n,
        "median_ape": statistics.median(ape),
        "within_20pct": sum(1 for x in ape if x <= 0.2) / n,
    }


def _shortest(graph, pairs):
    """{(u, v): distance} for distinct reachable pairs; one search per source."""
    by_src = {}
    for u, v in pairs:
        by_src.setdefault(u, set()).add(v)
    dj = Dijkstra(graph)
    out = {}
    for u, vs in by_src.items():
        found, _ = dj.distances(u, vs)
        for v, d in found.items():
            out[(u, v)] = d
    return out


def run_validation(cfg, fold=5, periods=PERIODS):
    if fold < 2:
        raise ValueError("fold must be >= 2")
    norm = LocationNormalizer(cfg)
    index = LocationIndex()
    agg_all = EdgeAggregator(cfg.stat)
    agg_p = [EdgeAggregator(cfg.stat) for _ in periods]
    hour_to_p = [None] * 24
    for i, (_, a, b) in enumerate(periods):
        for h in range(a, b):
            hour_to_p[h] = i

    summary = CleaningSummary()
    tests = []                       # (origin_key, dest_key, actual_s, period_idx)
    train_n = 0
    train_sum = 0.0
    key, get_or_add = norm.key, index.get_or_add
    for plat, plon, dlat, dlon, dur, h in iter_clean_trips(cfg, summary, with_hour=True):
        ok, dk = key(plat, plon), key(dlat, dlon)
        pi = hour_to_p[h]
        if summary.kept % fold == 0:
            tests.append((ok, dk, dur, pi))
            continue
        u, v = get_or_add(ok), get_or_add(dk)
        agg_all.add(u, v, dur)
        if pi is not None:
            agg_p[pi].add(u, v, dur)
        train_n += 1
        train_sum += dur
    if not train_n or not tests:
        raise ValueError("not enough trips for a train/test split")
    global_mean = train_sum / train_n

    n = len(index)
    g_all = TravelGraph.from_edges(n, agg_all.edges(cfg.min_trips))
    g_p = [TravelGraph.from_edges(n, a.edges(cfg.min_trips)) for a in agg_p]

    # --- classify test trips -------------------------------------------------
    same_cell = unseen = 0
    resolved = []                    # (u, v, actual, period_idx)
    lookup = index.lookup
    for ok, dk, dur, pi in tests:
        if ok == dk:
            same_cell += 1
            continue
        u, v = lookup(ok), lookup(dk)
        if u is None or v is None:
            unseen += 1
            continue
        resolved.append((u, v, dur, pi))

    sp_all = _shortest(g_all, {(u, v) for u, v, _, _ in resolved})
    covered = [(u, v, a, pi, sp_all[(u, v)]) for u, v, a, pi in resolved if (u, v) in sp_all]
    no_path = len(resolved) - len(covered)

    # --- overall accuracy (FR-14 style comparison included) -------------------
    sp_pairs = [(d, a) for _, _, a, _, d in covered]
    base_pairs = [(global_mean, a) for _, _, a, _, _ in covered]
    direct_sp, direct_dir, no_direct_sp, df_pairs = [], [], [], []
    for u, v, a, _, d in covered:
        w = g_all.edge_weight(u, v)
        if w is not None:
            direct_sp.append((d, a))
            direct_dir.append((w, a))
            df_pairs.append((w, a))
        else:
            no_direct_sp.append((d, a))
            df_pairs.append((d, a))
    overall = {
        "shortest_path": metrics(sp_pairs),
        "direct_first": metrics(df_pairs),     # direct edge if it exists, else shortest path
        "global_mean_baseline": metrics(base_pairs),
        "where_no_direct_edge": {"trips": len(no_direct_sp),
                                 "shortest_path": metrics(no_direct_sp)},
        "where_direct_edge_exists": {
            "trips": len(direct_sp),
            "shortest_path": metrics(direct_sp),
            "direct_edge": metrics(direct_dir),
        },
    }

    # --- time-of-day graphs ----------------------------------------------------
    per_period = []
    agg_rows = {"all_day_common": [], "period_common": [], "all_day_cov": [], "hybrid_cov": [],
                "all_day_df_cov": [], "period_df_cov": []}
    for pi, (name, a, b) in enumerate(periods):
        trips = [r for r in resolved if r[3] == pi]
        sp_p = _shortest(g_p[pi], {(u, v) for u, v, _, _ in trips})
        cov_all = [(u, v, act) for u, v, act, _ in trips if (u, v) in sp_all]
        common = [(u, v, act) for u, v, act in cov_all if (u, v) in sp_p]
        a_common = [(sp_all[(u, v)], act) for u, v, act in common]
        p_common = [(sp_p[(u, v)], act) for u, v, act in common]
        a_cov = [(sp_all[(u, v)], act) for u, v, act in cov_all]
        h_cov = [((sp_p[(u, v)] if (u, v) in sp_p else sp_all[(u, v)]), act)
                 for u, v, act in cov_all]
        # direct-first estimators over every all-day-covered trip
        ad_df, pd_df = [], []
        for u, v, act in cov_all:
            w_all = g_all.edge_weight(u, v)
            ad_df.append((w_all if w_all is not None else sp_all[(u, v)], act))
            w_p = g_p[pi].edge_weight(u, v)
            if w_p is not None:
                est = w_p
            elif (u, v) in sp_p:
                est = sp_p[(u, v)]
            else:
                est = w_all if w_all is not None else sp_all[(u, v)]
            pd_df.append((est, act))
        agg_rows["all_day_common"] += a_common
        agg_rows["period_common"] += p_common
        agg_rows["all_day_cov"] += a_cov
        agg_rows["hybrid_cov"] += h_cov
        agg_rows["all_day_df_cov"] += ad_df
        agg_rows["period_df_cov"] += pd_df
        per_period.append({
            "period": name, "hours": [a, b],
            "test_trips": len(trips),
            "mean_actual_s": (sum(t[2] for t in trips) / len(trips)) if trips else None,
            "graph_edges": g_p[pi].m,
            "coverage_all_day": len(cov_all) / len(trips) if trips else None,
            "coverage_period_graph": len(common) / len(trips) if trips else None,
            "common_trips": len(common),
            "all_day_on_common": metrics(a_common),
            "period_on_common": metrics(p_common),
            "all_day_on_covered": metrics(a_cov),
            "hybrid_on_covered": metrics(h_cov),
            "all_day_direct_first_on_covered": metrics(ad_df),
            "period_direct_first_on_covered": metrics(pd_df),
        })
    periods_summary = {
        "all_day_on_common": metrics(agg_rows["all_day_common"]),
        "period_on_common": metrics(agg_rows["period_common"]),
        "all_day_on_covered": metrics(agg_rows["all_day_cov"]),
        "hybrid_on_covered": metrics(agg_rows["hybrid_cov"]),
        "all_day_direct_first_on_covered": metrics(agg_rows["all_day_df_cov"]),
        "period_direct_first_on_covered": metrics(agg_rows["period_df_cov"]),
    }

    return {
        "config": cfg.as_dict(),
        "cleaning": summary.as_dict(),
        "split": {"fold": fold, "train_trips": train_n, "test_trips": len(tests),
                  "global_mean_s": global_mean},
        "graph": {"V": n, "E_all_day": g_all.m},
        "test_trip_accounting": {
            "same_cell_excluded": same_cell,
            "origin_or_dest_unseen_in_train": unseen,
            "no_path": no_path,
            "covered": len(covered),
            "coverage_of_predictable": len(covered) / len(resolved) if resolved else 0.0,
        },
        "overall": overall,
        "time_of_day": {"per_period": per_period, "all_periods": periods_summary},
    }

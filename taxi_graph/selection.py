"""Train / validation / test protocol.

Choices are fitted on VALIDATION trips only and reported once on TEST (the same test
trips as `run_validation(fold=5)`): kept-trip index % 5 == 0 -> test, == 1 -> validation,
otherwise train. Graph and index use train only.

Two choices are made from data:
  * per period: use that period's graph or the all-day graph (lower validation MAE wins);
  * one scale factor for chained (no direct edge) estimates = sum(actual)/sum(predicted)
    over validation trips answered by a chained path.
Both are plain aggregation, but they are parameters estimated from data, hence the
separate validation split.
"""
from .graph import EdgeAggregator, TravelGraph
from .loader import CleaningSummary, iter_clean_trips
from .location import LocationIndex, LocationNormalizer
from .validation import PERIODS, _shortest, metrics


def run_selection(cfg, periods=PERIODS):
    norm = LocationNormalizer(cfg)
    index = LocationIndex()
    agg_all = EdgeAggregator(cfg.stat)
    agg_p = [EdgeAggregator(cfg.stat) for _ in periods]
    hour_to_p = [None] * 24
    for i, (_, a, b) in enumerate(periods):
        for h in range(a, b):
            hour_to_p[h] = i

    summary = CleaningSummary()
    val, test = [], []
    train_n = 0
    key, get_or_add = norm.key, index.get_or_add
    for plat, plon, dlat, dlon, dur, h in iter_clean_trips(cfg, summary, with_hour=True):
        ok, dk, pi = key(plat, plon), key(dlat, dlon), hour_to_p[h]
        role = summary.kept % 5
        if role == 0:
            test.append((ok, dk, dur, pi))
        elif role == 1:
            val.append((ok, dk, dur, pi))
        else:
            u, v = get_or_add(ok), get_or_add(dk)
            agg_all.add(u, v, dur)
            if pi is not None:
                agg_p[pi].add(u, v, dur)
            train_n += 1
    if not (train_n and val and test):
        raise ValueError("not enough trips for a train/validation/test split")

    n = len(index)
    g_all = TravelGraph.from_edges(n, agg_all.edges(cfg.min_trips))
    g_p = [TravelGraph.from_edges(n, a.edges(cfg.min_trips)) for a in agg_p]

    def resolve(lst):
        out = []
        for ok, dk, dur, pi in lst:
            if ok == dk:
                continue
            u, v = index.lookup(ok), index.lookup(dk)
            if u is not None and v is not None:
                out.append((u, v, dur, pi))
        return out

    rv, rt = resolve(val), resolve(test)
    both = rv + rt
    sp_all = _shortest(g_all, {(u, v) for u, v, _, _ in both})
    sp_p = [_shortest(g_p[pi], {(u, v) for u, v, _, p in both if p == pi})
            for pi in range(len(periods))]

    def est_all(u, v):
        """(seconds, chained) from the all-day graph, or None if unreachable."""
        w = g_all.edge_weight(u, v)
        if w is not None:
            return w, False
        d = sp_all.get((u, v))
        return None if d is None else (d, True)

    def est_period(u, v, pi):
        w = g_p[pi].edge_weight(u, v)
        if w is not None:
            return w, False
        d = sp_p[pi].get((u, v))
        if d is not None:
            return d, True
        return est_all(u, v)

    # ---- fit on validation only ----------------------------------------------
    use_period, detail = [], []
    for pi, (name, _, _) in enumerate(periods):
        a_err, p_err = [], []
        for u, v, act, p in rv:
            if p != pi:
                continue
            ea = est_all(u, v)
            if ea is None:
                continue
            a_err.append(abs(ea[0] - act))
            p_err.append(abs(est_period(u, v, pi)[0] - act))
        ma = sum(a_err) / len(a_err) if a_err else None
        mp = sum(p_err) / len(p_err) if p_err else None
        use_period.append(bool(a_err) and mp < ma)
        detail.append({"period": name, "val_trips": len(a_err), "val_mae_all_day_s": ma,
                       "val_mae_period_s": mp, "use_period_graph": use_period[-1]})
    chain_act = chain_pred = 0.0
    chain_n = 0
    for u, v, act, _ in rv:
        e = est_all(u, v)
        if e is not None and e[1]:
            chain_act += act
            chain_pred += e[0]
            chain_n += 1
    factor = chain_act / chain_pred if chain_pred else 1.0

    def cal(e):
        return e[0] * factor if e[1] else e[0]

    # ---- evaluate once on test -----------------------------------------------
    rows = {k: [] for k in ("all_day_df", "all_day_df_calibrated", "period_df",
                            "selected_df", "selected_df_calibrated")}
    chain_before, chain_after = [], []
    for u, v, act, pi in rt:
        ea = est_all(u, v)
        if ea is None:
            continue
        ep = est_period(u, v, pi)
        es = ep if use_period[pi] else ea
        rows["all_day_df"].append((ea[0], act))
        rows["all_day_df_calibrated"].append((cal(ea), act))
        rows["period_df"].append((ep[0], act))
        rows["selected_df"].append((es[0], act))
        rows["selected_df_calibrated"].append((cal(es), act))
        if ea[1]:
            chain_before.append((ea[0], act))
            chain_after.append((cal(ea), act))

    return {
        "config": cfg.as_dict(),
        "split": {"train_trips": train_n, "validation_trips": len(val), "test_trips": len(test),
                  "test_predictable_covered": len(rows["all_day_df"])},
        "graph": {"V": n, "E_all_day": g_all.m},
        "selected_on_validation": {"periods": detail, "chain_scale_factor": factor,
                                   "chain_fit_trips": chain_n},
        "test": {k: metrics(v) for k, v in rows.items()},
        "test_chained_only": {"before": metrics(chain_before), "after": metrics(chain_after)},
    }

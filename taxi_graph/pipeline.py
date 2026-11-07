"""Pipeline wiring: CSV -> clean -> normalize -> index -> aggregate -> TravelGraph.

Whole build is O(T + V + E): one streaming pass over T trips, then counting sort.
Optional period graphs add one aggregator per period to the same pass.
"""
import time
from dataclasses import dataclass, field

from .graph import EdgeAggregator, TravelGraph
from .loader import CleaningSummary, iter_clean_trips
from .location import LocationIndex, LocationNormalizer
from .shortest_path import (Dijkstra, RouteResult, INVALID_COORDINATES,
                            LOCATION_NOT_IN_GRAPH, SUCCESS)
from .validation import PERIODS

# Chosen on a separate validation split (see docs/results.md section 3): these periods
# beat the all-day graph; the other three periods do not.
SELECTED_PERIODS = ("night", "evening")


def selected_periods():
    return [p for p in PERIODS if p[0] in SELECTED_PERIODS]


@dataclass
class BuildResult:
    graph: TravelGraph
    index: LocationIndex
    normalizer: LocationNormalizer
    cleaning: CleaningSummary
    self_loops: int
    build_s: float
    config: dict
    # period name -> (start_hour, end_hour, TravelGraph); shares vertex ids with `graph`
    period_graphs: dict = field(default_factory=dict)


def build(cfg, periods=None):
    """Build the all-day graph; with `periods` [(name, start, end)] also one graph each."""
    t0 = time.perf_counter()
    norm = LocationNormalizer(cfg)
    index = LocationIndex()
    agg = EdgeAggregator(cfg.stat)
    summary = CleaningSummary()
    key, get_or_add, add = norm.key, index.get_or_add, agg.add
    period_aggs = {}
    if periods:
        hour_to_agg = [None] * 24
        for name, a, b in periods:
            period_aggs[name] = EdgeAggregator(cfg.stat)
            for h in range(a, b):
                hour_to_agg[h] = period_aggs[name]
        for plat, plon, dlat, dlon, dur, h in iter_clean_trips(cfg, summary, with_hour=True):
            u, v = get_or_add(key(plat, plon)), get_or_add(key(dlat, dlon))
            add(u, v, dur)
            pa = hour_to_agg[h]
            if pa is not None:
                pa.add(u, v, dur)
    else:
        for plat, plon, dlat, dlon, dur in iter_clean_trips(cfg, summary):
            add(get_or_add(key(plat, plon)), get_or_add(key(dlat, dlon)), dur)
    n = len(index)
    graph = TravelGraph.from_edges(n, agg.edges(cfg.min_trips))
    pgraphs = {name: (a, b, TravelGraph.from_edges(n, period_aggs[name].edges(cfg.min_trips)))
               for name, a, b in (periods or [])}
    return BuildResult(graph, index, norm, summary, agg.self_loops,
                       time.perf_counter() - t0, cfg.as_dict(), pgraphs)


class Engine:
    """Query layer: coordinates -> vertex ids -> Dijkstra -> RouteResult."""

    def __init__(self, built):
        self.built = built
        self.dijkstra = Dijkstra(built.graph)
        self._period = {name: (a, b, g, Dijkstra(g))
                        for name, (a, b, g) in built.period_graphs.items()}

    def vertex_for(self, lat, lon):
        k = self.built.normalizer.key(lat, lon)
        return None if k is None else self.built.index.lookup(k)

    def _period_for(self, hour):
        if hour is None:
            return None
        for name, (a, b, g, dj) in self._period.items():
            if a <= hour < b:
                return name, g, dj
        return None

    def estimate(self, plat, plon, dlat, dlon, hour=None):
        """Best duration estimate: direct edge if observed, else shortest path.

        Validation showed shortest path is biased low (about -100 s) because it takes
        the minimum over noisy chained means, while a direct edge is unbiased.

        If `hour` (0-23, pickup hour) falls in a period with its own graph, that graph
        is tried first (direct edge, then shortest path) and the all-day graph is the
        fallback. Returns (RouteResult, seconds_or_None, basis) where basis is one of
        "direct", "path", "period-direct", "period-path", or None.
        """
        r = self.route(plat, plon, dlat, dlon)
        if r.status != SUCCESS:
            return r, None, None
        if r.source == r.destination:
            return r, r.total_duration, "path"
        per = self._period_for(hour)
        if per is not None:
            _, g, dj = per
            w = g.edge_weight(r.source, r.destination)
            if w is not None:
                return r, w, "period-direct"
            pr = dj.route(r.source, r.destination)
            if pr.status == SUCCESS:
                return r, pr.total_duration, "period-path"
        w = self.built.graph.edge_weight(r.source, r.destination)
        if w is not None:
            return r, w, "direct"
        return r, r.total_duration, "path"

    def route(self, plat, plon, dlat, dlon):
        n = self.built.normalizer
        if n.key(plat, plon) is None or n.key(dlat, dlon) is None:
            return RouteResult(status=INVALID_COORDINATES)
        s, d = self.vertex_for(plat, plon), self.vertex_for(dlat, dlon)
        if s is None or d is None:
            return RouteResult(status=LOCATION_NOT_IN_GRAPH)
        return self.dijkstra.route(s, d)

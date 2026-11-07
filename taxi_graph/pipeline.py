"""Pipeline wiring: CSV -> clean -> normalize -> index -> aggregate -> TravelGraph.

Whole build is O(T + V + E): one streaming pass over T trips, then counting sort.
"""
import time
from dataclasses import dataclass

from .graph import EdgeAggregator, TravelGraph
from .loader import CleaningSummary, iter_clean_trips
from .location import LocationIndex, LocationNormalizer
from .shortest_path import (Dijkstra, RouteResult, INVALID_COORDINATES,
                            LOCATION_NOT_IN_GRAPH, SUCCESS)


@dataclass
class BuildResult:
    graph: TravelGraph
    index: LocationIndex
    normalizer: LocationNormalizer
    cleaning: CleaningSummary
    self_loops: int
    build_s: float
    config: dict


def build(cfg):
    t0 = time.perf_counter()
    norm = LocationNormalizer(cfg)
    index = LocationIndex()
    agg = EdgeAggregator(cfg.stat)
    summary = CleaningSummary()
    key, get_or_add, add = norm.key, index.get_or_add, agg.add
    for plat, plon, dlat, dlon, dur in iter_clean_trips(cfg, summary):
        add(get_or_add(key(plat, plon)), get_or_add(key(dlat, dlon)), dur)
    graph = TravelGraph.from_edges(len(index), agg.edges(cfg.min_trips))
    return BuildResult(graph, index, norm, summary, agg.self_loops,
                       time.perf_counter() - t0, cfg.as_dict())


class Engine:
    """Query layer: coordinates -> vertex ids -> Dijkstra -> RouteResult."""

    def __init__(self, built):
        self.built = built
        self.dijkstra = Dijkstra(built.graph)

    def vertex_for(self, lat, lon):
        k = self.built.normalizer.key(lat, lon)
        return None if k is None else self.built.index.lookup(k)

    def estimate(self, plat, plon, dlat, dlon):
        """Best duration estimate: direct edge if observed, else shortest path.

        Validation showed shortest path is biased low (about -100 s) because it takes
        the minimum over noisy chained means, while a direct edge is unbiased.
        Returns (RouteResult, seconds_or_None, "direct"|"path"|None).
        """
        r = self.route(plat, plon, dlat, dlon)
        if r.status != SUCCESS:
            return r, None, None
        if r.source != r.destination:
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

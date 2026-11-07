"""Small web UI: stdlib http.server, graph held in memory, JSON API + one HTML page.

Endpoints
  GET /                 map page (taxi_graph/static/index.html)
  GET /api/info         bounds, graph size, grid, time-of-day periods
  GET /api/cells        centers of all known locations (graph vertices)
  GET /api/estimate     plat, plon, dlat, dlon [, hour]  -> estimate + route

Dijkstra's workspace is shared, so queries are serialized with a lock (each takes a
few ms). Binds to 127.0.0.1 by default; there is no authentication.
"""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .pipeline import Engine, SELECTED_PERIODS
from .shortest_path import SUCCESS, fmt_duration

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://unpkg.com; "
       "style-src 'self' 'unsafe-inline' https://unpkg.com; "
       "img-src 'self' data: https://tile.openstreetmap.org; connect-src 'self'; "
       "frame-ancestors 'none'")


class BadRequest(ValueError):
    pass


def _float(q, name):
    try:
        v = float(q[name][0])
    except (KeyError, ValueError, IndexError):
        raise BadRequest(f"missing or invalid '{name}'") from None
    if not math.isfinite(v):
        raise BadRequest(f"'{name}' must be finite")
    return v


def _hour(q):
    raw = q.get("hour", [""])[0]
    if raw == "":
        return None
    try:
        h = int(raw)
    except ValueError:
        raise BadRequest("'hour' must be an integer 0-23") from None
    if not 0 <= h <= 23:
        raise BadRequest("'hour' must be an integer 0-23")
    return h


class App:
    """Shared state for the handler: engine, lock, and precomputed static payloads."""

    def __init__(self, built):
        self.built = built
        self.engine = Engine(built)
        self.lock = threading.Lock()
        norm, idx = built.normalizer, built.index
        self.centers = [norm.center(k) for k in idx.keys]
        cfg = built.config
        self.info = {
            "bounds": {"lat_min": cfg["lat_min"], "lat_max": cfg["lat_max"],
                       "lon_min": cfg["lon_min"], "lon_max": cfg["lon_max"]},
            "cell_deg": cfg["cell_deg"],
            "stat": cfg["stat"],
            "V": built.graph.n, "E": built.graph.m,
            "trips_used": built.cleaning.kept,
            "period_graphs": {name: [a, b] for name, (a, b, _) in built.period_graphs.items()},
            "selected_periods": list(SELECTED_PERIODS),
        }
        self.cells_json = json.dumps(
            [[round(la, 5), round(lo, 5)] for la, lo in self.centers],
            separators=(",", ":")).encode()

    def estimate(self, plat, plon, dlat, dlon, hour):
        with self.lock:
            r, est, basis = self.engine.estimate(plat, plon, dlat, dlon, hour=hour)
        out = {"status": r.status, "message": r.message() if r.status != SUCCESS else "OK"}
        if r.status != SUCCESS:
            return out
        g = self.built.graph
        direct = g.edge_weight(r.source, r.destination) if r.source != r.destination else None
        out.update({
            "estimate_s": est, "estimate_text": fmt_duration(est), "basis": basis,
            "hour": hour,
            "shortest_path_s": r.total_duration,
            "shortest_path_text": fmt_duration(r.total_duration),
            "direct_s": direct,
            "route": [{"vertex": v, "lat": self.centers[v][0], "lon": self.centers[v][1]}
                      for v in r.path],
            "legs_s": r.edge_durations,
            "stats": {"visited": r.visited_vertices, "relaxed": r.relaxed_edges,
                      "time_ms": round(r.elapsed_s * 1e3, 3), "V": r.graph_v, "E": r.graph_e},
        })
        return out


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "taxi-graph"

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/info":
                    self._json(200, app.info)
                elif url.path == "/api/cells":
                    self._send(200, app.cells_json, "application/json")
                elif url.path == "/api/estimate":
                    q = parse_qs(url.query)
                    res = app.estimate(_float(q, "plat"), _float(q, "plon"),
                                       _float(q, "dlat"), _float(q, "dlon"), _hour(q))
                    self._json(200, res)
                else:
                    self._json(404, {"error": "not found"})
            except BadRequest as e:
                self._json(400, {"error": str(e)})

        def do_POST(self):
            self._json(405, {"error": "method not allowed"})

        do_PUT = do_DELETE = do_PATCH = do_POST

        def log_message(self, fmt, *args):
            pass

    return Handler


def make_server(built, host="127.0.0.1", port=8000):
    """Return a ThreadingHTTPServer (port 0 picks a free port; see server.server_port)."""
    srv = ThreadingHTTPServer((host, port), make_handler(App(built)))
    srv.daemon_threads = True
    return srv

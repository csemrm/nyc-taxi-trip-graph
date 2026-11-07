import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from taxi_graph.config import Config
from taxi_graph.pipeline import build, selected_periods
from taxi_graph.web import make_server

HDR = ("id,vendor_id,pickup_datetime,dropoff_datetime,passenger_count,pickup_longitude,"
       "pickup_latitude,dropoff_longitude,dropoff_latitude,store_and_fwd_flag,trip_duration\n")


def row(plon, plat, dlon, dlat, dur, t):
    return f"x,1,{t},,1,{plon},{plat},{dlon},{dlat},N,{dur}\n"


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(cls.tmp.name, "t.csv")
        night, noon = "2016-03-14 02:00:00", "2016-03-14 12:00:00"
        with open(path, "w") as f:
            f.write(HDR
                    + row(-73.98, 40.75, -73.95, 40.78, 100, night)
                    + row(-73.98, 40.75, -73.95, 40.78, 500, noon)
                    + row(-73.95, 40.78, -73.90, 40.80, 200, noon))
        built = build(Config(csv_path=path), periods=selected_periods())
        cls.srv = make_server(built, port=0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        cls.t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=5) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def j(self, path):
        code, _, body = self.get(path)
        return code, json.loads(body)

    def test_index_and_security_headers(self):
        code, h, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn(b"NYC Taxi Trip Estimator", body)
        self.assertEqual(h["X-Content-Type-Options"], "nosniff")
        csp = h["Content-Security-Policy"]
        self.assertIn("frame-ancestors 'none'", csp)
        # The page loads tiles from this exact host; a wildcard like *.tile... would not match it.
        self.assertIn(b"https://tile.openstreetmap.org/", body)
        self.assertIn("img-src 'self' data: https://tile.openstreetmap.org", csp)

    def test_info_and_cells(self):
        code, info = self.j("/api/info")
        self.assertEqual((code, info["V"], info["E"]), (200, 3, 2))
        self.assertEqual(sorted(info["period_graphs"]), ["evening", "night"])
        code, cells = self.j("/api/cells")
        self.assertEqual((code, len(cells)), (200, 3))

    def test_estimate_success_direct_and_period(self):
        q = "plat=40.75&plon=-73.98&dlat=40.78&dlon=-73.95"
        code, d = self.j("/api/estimate?" + q)
        self.assertEqual(code, 200)
        self.assertEqual((d["status"], d["basis"], d["estimate_s"]), ("SUCCESS", "direct", 300.0))
        self.assertEqual(d["direct_s"], 300.0)
        self.assertEqual(len(d["route"]), 2)
        _, d = self.j("/api/estimate?" + q + "&hour=2")
        self.assertEqual((d["basis"], d["estimate_s"]), ("period-direct", 100.0))
        _, d = self.j("/api/estimate?" + q + "&hour=12")      # midday: no period graph
        self.assertEqual((d["basis"], d["estimate_s"]), ("direct", 300.0))

    def test_chained_route_has_legs(self):
        _, d = self.j("/api/estimate?plat=40.75&plon=-73.98&dlat=40.80&dlon=-73.90")
        self.assertEqual(d["basis"], "path")
        self.assertAlmostEqual(d["estimate_s"], 500.0)
        self.assertEqual(len(d["legs_s"]), 2)
        self.assertIsNone(d["direct_s"])

    def test_non_success_statuses(self):
        _, d = self.j("/api/estimate?plat=0&plon=0&dlat=40.78&dlon=-73.95")
        self.assertEqual(d["status"], "INVALID_COORDINATES")
        _, d = self.j("/api/estimate?plat=40.60&plon=-74.10&dlat=40.78&dlon=-73.95")
        self.assertEqual(d["status"], "LOCATION_NOT_IN_GRAPH")
        _, d = self.j("/api/estimate?plat=40.80&plon=-73.90&dlat=40.75&dlon=-73.98")
        self.assertEqual(d["status"], "NO_PATH")

    def test_bad_requests(self):
        for path in ("/api/estimate",
                     "/api/estimate?plat=abc&plon=1&dlat=2&dlon=3",
                     "/api/estimate?plat=nan&plon=1&dlat=2&dlon=3",
                     "/api/estimate?plat=40.75&plon=-73.98&dlat=40.78&dlon=-73.95&hour=24",
                     "/api/estimate?plat=40.75&plon=-73.98&dlat=40.78&dlon=-73.95&hour=x"):
            code, body = self.j(path)
            self.assertEqual(code, 400, path)
            self.assertIn("error", body)

    def test_unknown_path_and_methods(self):
        self.assertEqual(self.get("/etc/passwd")[0], 404)
        self.assertEqual(self.get("/../../etc/passwd")[0], 404)
        req = urllib.request.Request(self.base + "/api/info", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 405)

    def test_concurrent_queries_are_consistent(self):
        q = "/api/estimate?plat=40.75&plon=-73.98&dlat=40.80&dlon=-73.90"
        results = []

        def work():
            for _ in range(20):
                results.append(self.j(q)[1]["estimate_s"])
        threads = [threading.Thread(target=work) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(results), 120)
        self.assertEqual(set(results), {500.0})


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

from taxi_graph.config import Config
from taxi_graph.selection import run_selection
from taxi_graph.synthetic import generate
from taxi_graph.validation import PERIODS, run_validation


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, "t.csv")
        generate(path, 30000, seed=9)
        self.cfg = Config(csv_path=path, cell_deg=0.01)
        self.r = run_selection(self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_sizes_and_same_test_set_as_validate(self):
        s = self.r["split"]
        kept = run_validation(self.cfg, fold=5)["cleaning"]["kept"]
        self.assertEqual(s["train_trips"] + s["validation_trips"] + s["test_trips"], kept)
        self.assertEqual(s["test_trips"], run_validation(self.cfg, fold=5)["split"]["test_trips"])
        self.assertLess(s["train_trips"], kept * 0.7)   # 60% train

    def test_selection_structure_and_determinism(self):
        sel = self.r["selected_on_validation"]
        self.assertEqual([p["period"] for p in sel["periods"]], [p[0] for p in PERIODS])
        for p in sel["periods"]:
            self.assertIsInstance(p["use_period_graph"], bool)
            if p["use_period_graph"]:       # chosen only if better on VALIDATION
                self.assertLess(p["val_mae_period_s"], p["val_mae_all_day_s"])
        self.assertGreater(sel["chain_scale_factor"], 0)
        self.assertEqual(self.r, run_selection(self.cfg))

    def test_test_metrics_share_the_same_trips(self):
        t = self.r["test"]
        n = {m["n"] for m in t.values()}
        self.assertEqual(n, {self.r["split"]["test_predictable_covered"]})

    def test_calibration_only_touches_chained_trips(self):
        c = self.r["test_chained_only"]
        t = self.r["test"]
        if c["before"]:
            self.assertNotEqual(c["before"]["bias_s"], c["after"]["bias_s"])
        # scaling chained estimates cannot change the number of covered trips
        self.assertEqual(t["all_day_df"]["n"], t["all_day_df_calibrated"]["n"])


if __name__ == "__main__":
    unittest.main()

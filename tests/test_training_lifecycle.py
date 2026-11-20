import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from overpressure.config import Project
from overpressure.model import ModelBundle
from overpressure.training import Cancelled, train


ROOT = Path(__file__).resolve().parents[1]


class TrainingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name)
        cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        cfg["data"].pop("well_files",None)
        cfg["data"].pop("well_imports",None)
        cfg["data"]["training_selection"]="named_wells"
        cfg["data"].update(well_file="wells.csv", split_method="well_holdout", development_wells=["A", "V"], validation_wells=["V"], blind_wells=["B"])
        cfg["data"]["columns"]["well"] = "well"
        cfg["data"]["depth_reference"] = {"confirmed": True, "type": "TVD", "datum": "测试基准"}
        cfg["algorithm"].update(hidden_sizes=[8], dropout=0)
        cfg["training"].update(epochs=2, batch_size=4)
        pd.DataFrame({"well": ["A", "A", "V", "V", "B", "B"], "Depth": [100, 200, 300, 400, 500, 600],
                      "AC": [100] * 6, "DEN": [2.4] * 6, "Ip": [8000] * 6, "Pp": [10, 11, 12, 13, 14, 15]}).to_csv(folder / "wells.csv", index=False)
        self.project = Project(folder / "project.yaml", cfg)

    def test_completed_model_excludes_blind_and_loads(self):
        folder = train(self.project)
        meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["preprocessing"]["fit_wells"], ["A"])
        self.assertEqual(meta["blind_wells_excluded"], ["B"])
        self.assertNotIn("blind", meta["split"])
        self.assertEqual(meta["internal_validation_metrics"]["count"], 2)
        self.assertTrue(np.isfinite(ModelBundle(folder).predict([[100, 100, 2.4, 8000]])).all())

    def test_cancelled_run_cannot_be_loaded(self):
        stop = self.project.path.parent / "cancel"; stop.touch()
        with self.assertRaises(Cancelled): train(self.project, cancel_file=stop)
        run = next((self.project.path.parent / "outputs" / "models").iterdir())
        meta = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "cancelled")
        with self.assertRaisesRegex(ValueError, "完成状态"):
            ModelBundle(run)

    def test_random_training_directly_produces_predictable_best_model(self):
        self.project.settings["data"]["depth_reference"] = {"type": "unknown", "datum": "", "confirmed": False}
        self.project.settings["data"].update(split_method="random_points", train_ratio=.8, validation_wells=[])
        folder = train(self.project)
        meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assignments = pd.read_csv(folder / "split_assignments.csv")
        self.assertEqual(meta["split_strategy"]["method"], "random_points")
        self.assertEqual(meta["split"]["train"]["rows"], 3)
        self.assertEqual(meta["split"]["validation"]["rows"], 1)
        self.assertFalse(assignments.source_row.duplicated().any())
        self.assertNotIn("B", assignments.well.tolist())
        self.assertEqual(meta["purpose"], "training")
        self.assertEqual(meta["internal_validation_metrics"]["count"], 1)
        self.assertTrue(np.isfinite(ModelBundle(folder).predict([[100, 100, 2.4, 8000]])).all())


if __name__ == "__main__":
    unittest.main()

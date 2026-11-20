import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from overpressure.config import FEATURES, Project
from overpressure.cross_validation import cross_validate
from overpressure.training import Cancelled


ROOT = Path(__file__).resolve().parents[1]


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name)
        cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        cfg["data"].pop("well_files",None)
        cfg["data"].pop("well_imports",None)
        cfg["data"]["training_selection"]="named_wells"
        cfg["data"].update(well_file="wells.csv", development_wells=["A", "V", "C"], validation_wells=["V"], blind_wells=["B"])
        cfg["data"]["columns"]["well"] = "well"
        cfg["data"]["depth_reference"] = {"confirmed": True, "type": "TVD", "datum": "test"}
        cfg["algorithm"].update(hidden_sizes=[4], dropout=0)
        cfg["training"].update(epochs=1, batch_size=8)
        self.frame = pd.DataFrame({"well": ["A", "A", "V", "V", "C", "C", "B"], "Depth": [100, 200, 300, 400, 500, 600, 99999],
                                   "AC": [100] * 7, "DEN": [2.4] * 7, "Ip": [8000] * 7, "Pp": [10, 11, 12, 13, 14, 15, 1000]})
        self.frame.to_csv(folder / "wells.csv", index=False)
        self.project = Project(folder / "project.yaml", cfg)

    def test_each_well_held_out_once_and_scaler_excludes_it_and_blind(self):
        out = cross_validate(self.project)
        report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["folds"]), 3)
        oof = pd.read_csv(out / "oof_predictions.csv")
        self.assertEqual(set(oof.well), {"A", "V", "C"}); self.assertFalse(oof.source_row.duplicated().any())
        self.assertEqual(len(oof), 6)
        for row in report["folds"]:
            meta = json.loads((Path(row["model_path"]) / "manifest.json").read_text(encoding="utf-8"))
            expected = self.frame.loc[~self.frame.well.isin(["B", row["validation_well"]]), list(FEATURES)].mean().to_numpy()
            np.testing.assert_allclose(meta["preprocessing"]["mean"], expected)
            self.assertNotIn(row["validation_well"], meta["preprocessing"]["fit_wells"])
            self.assertNotIn("B", meta["preprocessing"]["fit_wells"])

    def test_cancelled_parent_is_not_completed(self):
        cancel = self.project.path.parent / "cancel"; cancel.touch()
        with self.assertRaises(Cancelled): cross_validate(self.project, cancel_file=cancel)
        report = next((self.project.path.parent / "outputs" / "cross_validation").rglob("summary.json"))
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["status"], "cancelled")

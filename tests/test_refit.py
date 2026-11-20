from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from overpressure.config import Project
from overpressure.data import audit_wells, development_fingerprint
from overpressure.model import ModelBundle
from overpressure.refit import refit
from overpressure.training import Cancelled


ROOT = Path(__file__).resolve().parents[1]


class RefitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        cfg["data"].pop("well_files",None)
        cfg["data"].pop("well_imports",None)
        cfg["data"]["training_selection"]="named_wells"
        cfg["data"].update(well_file="wells.csv", development_wells=["A", "V"], validation_wells=["V"], blind_wells=["B"])
        cfg["data"]["columns"]["well"] = "well"
        cfg["data"]["depth_reference"] = {"confirmed": True, "type": "TVD", "datum": "test"}
        cfg["algorithm"].update(hidden_sizes=[4], dropout=0)
        cfg["training"].update(epochs=3, batch_size=8, patience=1)
        self.data = pd.DataFrame({"well": ["A", "A", "V", "V", "B"], "Depth": [100, 200, 300, 400, 99999],
                                  "AC": [100] * 5, "DEN": [2.4] * 5, "Ip": [8000] * 5, "Pp": [10, 11, 12, 13, 10000]})
        self.data.to_csv(self.folder / "wells.csv", index=False)
        self.project = Project(self.folder / "project.yaml", cfg)
        audit = audit_wells(self.project)
        self.cv = self.folder / "cv"; self.cv.mkdir()
        self.summary = {"status": "completed", "purpose": "cross_validation", "project": deepcopy(cfg),
                        "development_sha256": development_fingerprint(audit.frame, ["A", "V"]),
                        "source_sha256": audit.report["source_sha256"],
                        "folds": [{"validation_well": "A", "best_epoch": 1}, {"validation_well": "V", "best_epoch": 3}]}
        self.write_summary()

    def write_summary(self): (self.cv / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def test_all_development_rows_and_fixed_epochs_without_blind_or_validation(self):
        self.project.settings["data"]["depth_reference"] = {"type": "unknown", "datum": "", "confirmed": False}
        out = refit(self.project, self.cv)
        meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["purpose"], "final_refit")
        self.assertEqual(meta["epochs_completed"], 2)
        self.assertEqual(meta["preprocessing"]["fit_wells"], ["A", "V"])
        self.assertEqual(meta["preprocessing"]["fit_rows"], 4)
        self.assertEqual(meta["preprocessing"]["mean"][0], 250)
        self.assertEqual(set(meta["split"]), {"train"})
        self.assertNotIn("internal_validation_metrics", meta)
        self.assertNotIn("best_epoch", meta)
        history = pd.read_csv(out / "history.csv")
        self.assertNotIn("validation_data", history)
        self.assertTrue(np.isfinite(ModelBundle(out).predict([[250, 100, 2.4, 8000]])).all())

    def test_smoke_cv_and_changed_training_settings_cannot_select_formal_model(self):
        self.summary["purpose"] = "smoke_test"; self.write_summary()
        with self.assertRaisesRegex(ValueError, "小样本"): refit(self.project, self.cv)
        self.summary["purpose"] = "cross_validation"; self.write_summary()
        self.project.settings["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "training"): refit(self.project, self.cv)

    def test_blind_labels_do_not_invalidate_selection_but_development_changes_do(self):
        self.data.loc[self.data.well == "B", "Pp"] = np.nan
        self.data.to_csv(self.folder / "wells.csv", index=False)
        out = refit(self.project, self.cv, smoke=True)
        self.assertEqual(ModelBundle(out).metadata["purpose"], "smoke_test")
        self.data.loc[self.data.well == "A", "Pp"] += 1
        self.data.to_csv(self.folder / "wells.csv", index=False)
        with self.assertRaisesRegex(ValueError, "开发井数据"): refit(self.project, self.cv)

    def test_incomplete_folds_and_cancelled_run_are_rejected(self):
        original = list(self.summary["folds"])
        self.summary["folds"] = original[:1]; self.write_summary()
        with self.assertRaisesRegex(ValueError, "每口开发井"): refit(self.project, self.cv)
        self.summary["folds"] = original; self.write_summary()
        stop = self.folder / "cancel"; stop.touch()
        with self.assertRaises(Cancelled): refit(self.project, self.cv, cancel_file=stop)
        manifest = next((self.folder / "outputs" / "models").rglob("manifest.json"))
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["status"], "cancelled")

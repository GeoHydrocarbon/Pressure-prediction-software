from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml
from overpressure.config import Project, FEATURES
from overpressure.data import sha256_file
from overpressure.model import PointMLP
from overpressure.well_points import prediction_defaults
from overpressure.well_workflow import predict_new_wells


ROOT = Path(__file__).resolve().parents[1]


class WellWorkflowTests(unittest.TestCase):
    def test_batch_files_predict_and_export_each_well_with_chinese_measurements(self):
        for name,group in self.curve.groupby('name'):group.to_csv(self.folder/(name+'.csv'),index=False)
        self.points.rename(columns={'name':'井名','z':'深度','pressure':'实测压力'}).to_csv(self.folder/'measured.csv',index=False)
        cfg=self.project.settings['well_prediction'];cfg.update(well_files=['NEW.csv','TRAIN.csv'],measurement_files=['measured.csv'])
        out=self.run_prediction();files=pd.read_csv(out/'well_files.csv')
        self.assertEqual(set(files.well),{'NEW','TRAIN'});self.assertEqual(len(files),2)
        self.assertTrue(all((out/path).is_file() for path in files.predictions))
        points=pd.read_csv(out/'reference_points.csv');self.assertEqual(int(points.supervised_valid.sum()),3)
        self.assertIn('source_file_row',pd.read_csv(out/'predictions.csv'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.folder = Path(self.tmp.name)
        cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        self.project = Project(self.folder / "project.yaml", cfg)
        cfg["well_prediction"]={}
        settings = prediction_defaults(self.project)
        settings.update(well_file="new.csv", measurement_file="points.csv", independence_confirmed=True,
                        measurements_depth_confirmed=True, depth_reference={"type": "TVD", "datum": "test", "confirmed": True})
        settings["columns"]["well"] = "name"; settings["measurement_columns"] = {"well": "name", "Depth": "z", "Pp": "pressure"}
        cfg["well_prediction"] = settings
        self.curve = pd.DataFrame({"name": ["NEW"]*3 + ["TRAIN"]*3, "Depth": [100, 110, 120]*2, "AC": 100, "DEN": 2.4, "Ip": 8000})
        self.curve.to_csv(self.folder / "new.csv", index=False)
        self.points = pd.DataFrame({"name": ["NEW", "NEW", "NEW", "TRAIN"], "z": [100, 105, 125, 110], "pressure": [9, 11, 12, 10]})
        self.points.to_csv(self.folder / "points.csv", index=False)
        self.bundle = self.folder / "model"; self.bundle.mkdir(); model = PointMLP([4], 0)
        for parameter in model.parameters(): parameter.data.zero_()
        model.network[-1].bias.data.fill_(10)
        torch.save(model.state_dict(), self.bundle / "weights.pt")
        frozen = deepcopy(cfg); frozen["data"]["depth_reference"] = settings["depth_reference"]
        meta = {"schema_version": 1, "status": "completed", "algorithm": "mlp", "purpose": "training", "project": frozen,
                "features": list(FEATURES), "architecture": {"hidden_sizes": [4], "dropout": 0},
                "weights_sha256": sha256_file(self.bundle / "weights.pt"),
                "preprocessing": {"mean": [0]*4, "scale": [1]*4, "minimum": [0]*4, "maximum": [1e9]*4, "fit_wells": ["TRAIN"]}}
        (self.bundle / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")

    def run_prediction(self):
        with redirect_stdout(io.StringIO()): return predict_new_wells(self.project, self.bundle)

    def test_sparse_validation_predicts_new_wells_without_changing_training_inputs(self):
        before = deepcopy(self.project.settings["data"]); out = self.run_prediction()
        curve = pd.read_csv(out / "predictions.csv"); points = pd.read_csv(out / "reference_points.csv")
        notes = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(len(curve), 6); self.assertTrue((curve.Pp_pred == 10).all())
        self.assertEqual(notes["matched_rows"], 3); self.assertEqual(notes["unmatched_rows"], 1)
        self.assertEqual(notes["wells"]["NEW"]["count"], 2)
        self.assertEqual(notes["well_roles"]["TRAIN"], "training_review")
        self.assertEqual(notes["well_roles"]["NEW"], "independent_validation_user_confirmed")
        self.assertEqual(points.match_status.iloc[1], "matched_interpolated")
        self.assertTrue((out / "pressure_1.png").is_file())
        self.assertEqual(before, self.project.settings["data"])

    def test_prediction_only_has_no_invented_pressure_labels(self):
        self.project.settings["well_prediction"]["measurement_file"] = ""
        out = self.run_prediction(); table = pd.read_csv(out / "evaluation_metrics.csv")
        self.assertTrue(pd.read_csv(out / "reference_points.csv").empty)
        self.assertEqual(table["count"].sum(), 0)
        self.assertTrue(table.r2.isna().all())

    def test_embedded_sparse_labels_use_only_supplied_rows(self):
        self.curve["Pp"] = [9, np.nan, 12, np.nan, np.nan, np.nan]
        self.curve.to_csv(self.folder / "new.csv", index=False)
        self.project.settings["well_prediction"]["measurement_file"] = ""
        out = self.run_prediction(); points = pd.read_csv(out / "reference_points.csv")
        self.assertEqual(len(points), 2); self.assertEqual(points.Pp.tolist(), [9, 12])

    def test_units_and_depth_confirmation(self):
        cfg = self.project.settings["well_prediction"]
        self.points.z /= .3048; self.points.pressure *= 1000; self.points.to_csv(self.folder / "points.csv", index=False)
        cfg["measurement_units"] = {"Depth": "ft", "Pp": "kPa"}
        out = self.run_prediction(); points = pd.read_csv(out / "reference_points.csv")
        self.assertAlmostEqual(points.Depth.iloc[0], 100); self.assertEqual(points.Pp.iloc[0], 9)
        cfg["measurements_depth_confirmed"] = False
        cfg["depth_reference"] = {"type": "unknown", "datum": "other", "confirmed": False}
        result = self.run_prediction()
        self.assertEqual(json.loads((result / "metrics.json").read_text(encoding="utf-8"))["matched_rows"], 3)

    def test_external_points_take_precedence_over_dense_embedded_labels(self):
        self.curve["Pp"] = 999; self.curve.to_csv(self.folder / "new.csv", index=False)
        out = self.run_prediction()
        points = pd.read_csv(out / "reference_points.csv")
        self.assertEqual(len(points), 4); self.assertNotIn(999, points.Pp.tolist())

    def test_csv_well_names_keep_leading_zeros(self):
        from overpressure.well_workflow import read_well_input
        self.curve["name"] = "001"; self.curve.to_csv(self.folder / "new.csv", index=False)
        cfg = self.project.settings["well_prediction"]
        frame = read_well_input(self.folder / "new.csv", cfg["columns"], cfg["units"])
        self.assertEqual(set(frame.well), {"001"})

    def test_cancelled_plotting_is_not_published_as_completed(self):
        from overpressure.jobs import Cancelled
        stop = self.folder / "cancel"
        with patch("overpressure.charts.save_figure", side_effect=lambda *args: stop.touch()):
            with self.assertRaises(Cancelled):
                with redirect_stdout(io.StringIO()): predict_new_wells(self.project, self.bundle, cancel_file=stop)
        metadata = next((self.folder / "outputs" / "well_predictions").glob("*/metrics.json"))
        self.assertEqual(json.loads(metadata.read_text(encoding="utf-8"))["status"], "cancelled")

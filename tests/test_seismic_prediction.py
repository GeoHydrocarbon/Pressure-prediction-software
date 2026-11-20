import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import segyio
import torch
import yaml

from overpressure.config import FEATURES, Project
from overpressure.data import sha256_file
from overpressure.model import PointMLP
from overpressure.seismic import predict_seismic_trial
from overpressure.training import Cancelled
from overpressure.seismic_view import prepare_view, SeismicView


ROOT = Path(__file__).resolve().parents[1]


class SeismicPredictionTests(unittest.TestCase):
    def test_automatic_time_selection_ignores_legacy_limits_and_masks_internal_gaps(self):
        self.project.settings['time_depth'].update(valid_time_range=[50,51],extrapolation='error')
        for filename in ('density.sgy','ip.sgy'):
            with segyio.open(str(self.folder/filename),'r+',ignore_geometry=True) as f:
                for i in range(3):f.header[i][109]=1000*(i+1)
                f.trace[2]=np.zeros(4,dtype=np.float32)
                if filename=='ip.sgy':f.trace[0]=np.array([8000000,np.nan,0,8000000],dtype=np.float32)
        out=predict_seismic_trial(self.project,self.bundle,trace_count=3,block_traces=1)
        report=json.loads((out/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(report['predicted_points'],5);self.assertEqual(report['invalid_attribute_points'],7)
        np.testing.assert_allclose(report['predicted_time_range_seconds'],[1,2.003])
        with segyio.open(str(out/'pressure.sgy'),'r',ignore_geometry=True) as f:
            np.testing.assert_allclose(f.trace[0],[7,np.nan,np.nan,7],equal_nan=True)
            self.assertTrue(np.isnan(f.trace[2]).all());self.assertEqual(f.header[1][109],2000)

    def test_table_coverage_masks_points_instead_of_stopping_prediction(self):
        import pandas as pd
        pd.DataFrame(dict(time=[.001,.002],depth=[100,200])).to_csv(self.folder/'td.csv',index=False)
        self.project.settings['time_depth'].update(method='table',table_file='td.csv',valid_time_range=[50,51],extrapolation='error')
        out=predict_seismic_trial(self.project,self.bundle,trace_count=3)
        report=json.loads((out/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(report['predicted_points'],3);self.assertEqual(report['unavailable_depth_points'],6)
        with segyio.open(str(out/'pressure.sgy'),'r',ignore_geometry=True) as f:
            np.testing.assert_allclose(f.trace[0],[np.nan,np.nan,7,np.nan],equal_nan=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        cfg['seismic']['header_fields'].update(inline=21, crossline=1)
        cfg["seismic"].update(density_file="density.sgy", impedance_file="ip.sgy")
        cfg["data"]["depth_reference"] = {"confirmed": True, "type": "TVD", "datum": "test"}
        cfg["time_depth"].update(confirmed=True, depth_type="TVD", coefficients=[100, 1000], valid_time_range=[0, 1], datum="test", source="synthetic test")
        self.project = Project(self.folder / "project.yaml", cfg)
        for name, value in (("density.sgy", 2400), ("ip.sgy", 8000000)):
            spec = segyio.spec(); spec.format = 1; spec.sorting = 2; spec.samples = np.arange(4); spec.tracecount = 3
            with segyio.create(str(self.folder / name), spec) as dst:
                dst.bin[segyio.BinField.Interval] = 1000
                for i in range(3):
                    dst.header[i] = {1: i + 1, 21: 1, 73: 100 + i, 77: 200, 109: 0, 115: 4, 117: 1000}
                    dst.trace[i] = np.array([value, 0, value, value], dtype=np.float32)
        self.bundle = self.folder / "model"; self.bundle.mkdir()
        model = PointMLP([4], 0)
        for parameter in model.parameters(): parameter.data.zero_()
        model.network[-1].bias.data.fill_(7)
        torch.save(model.state_dict(), self.bundle / "weights.pt")
        self.meta = {"schema_version": 1, "status": "completed", "purpose": "training", "algorithm": "mlp", "features": list(FEATURES),
                     "project": cfg, "architecture": {"hidden_sizes": [4], "dropout": 0}, "weights_sha256": sha256_file(self.bundle / "weights.pt"),
                     "preprocessing": {"mean": [0] * 4, "scale": [1] * 4, "minimum": [0] * 4, "maximum": [1e9] * 4}}
        self.write_meta()

    def write_meta(self): (self.bundle / "manifest.json").write_text(json.dumps(self.meta), encoding="utf-8")

    def test_roundtrip_ibm_to_ieee_first_point_and_after_gap(self):
        out = predict_seismic_trial(self.project, self.bundle, trace_start=1, trace_count=2)
        report = json.loads((out / "result.json").read_text(encoding="utf-8"))
        self.assertTrue(report["round_trip_verified"]); self.assertEqual(report["source_trace_indices"], [1, 2])
        with segyio.open(str(out / "pressure.sgy"), "r", ignore_geometry=True) as pressure, segyio.open(str(out / "validity.sgy"), "r", ignore_geometry=True) as mask:
            self.assertEqual(int(pressure.format), 5); self.assertEqual(pressure.tracecount, 2)
            np.testing.assert_allclose(pressure.trace[0], [7, np.nan, 7, 7], equal_nan=True)
            np.testing.assert_array_equal(mask.trace[0], [1, 0, 1, 1])
            self.assertEqual(pressure.header[0][1], 2)
            self.assertIn(b"MPa", bytes(pressure.text[0]))
        cache = prepare_view(out)
        view = SeismicView.from_cache(cache)
        self.assertEqual(view.section("inline", 1).values.shape, (4, 2))
        np.testing.assert_equal(view.pressure[0], [7, np.nan, 7, 7])

    def test_browser_rejects_tampered_and_partial_results(self):
        out = predict_seismic_trial(self.project, self.bundle, trace_count=1)
        with segyio.open(str(out / "pressure.sgy"), "r+", ignore_geometry=True) as f:
            f.trace[0] = np.array([8, np.nan, 7, 7], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "校验失败"): prepare_view(out)
        report = json.loads((out / "result.json").read_text(encoding="utf-8")); report["status"] = "cancelled"
        (out / "result.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "已完成"): prepare_view(out)

    def test_depth_metadata_does_not_block_seismic_but_smoke_model_is_rejected(self):
        self.project.settings["time_depth"]["confirmed"] = False
        self.project.settings["time_depth"].update(depth_type="TVDSS", datum="KB", source="")
        self.project.settings["data"]["depth_reference"] = {"type": "MD", "datum": "0", "confirmed": False}
        out = predict_seismic_trial(self.project, self.bundle, trace_count=1)
        self.assertTrue(json.loads((out / "result.json").read_text(encoding="utf-8"))["round_trip_verified"])
        self.meta["purpose"] = "smoke_test"; self.write_meta()
        with self.assertRaisesRegex(ValueError, "试算模型"): predict_seismic_trial(self.project, self.bundle, trace_count=1)

    def test_cancelled_output_not_published(self):
        cancel = self.folder / "cancel"
        def cancel_after_block(event, **values):
            if event == "seismic_progress" and values.get("phase") == "predict": cancel.touch()
        with patch("overpressure.predict_volume.emit", side_effect=cancel_after_block):
            with self.assertRaises(Cancelled): predict_seismic_trial(self.project, self.bundle, trace_count=2, block_traces=1, cancel_file=cancel)
        out = next((self.folder / "outputs" / "seismic_trials").iterdir())
        self.assertFalse((out / "pressure.sgy").exists())
        self.assertEqual(json.loads((out / "result.json").read_text(encoding="utf-8"))["status"], "cancelled")

    def test_range_overrun_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "总道数"): predict_seismic_trial(self.project, self.bundle, trace_start=2, trace_count=2)

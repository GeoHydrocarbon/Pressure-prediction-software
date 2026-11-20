from contextlib import redirect_stdout
import io
import json
import os
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
from overpressure.jobs import Cancelled
from overpressure.model import PointMLP
from overpressure.seismic import predict_seismic_volume
from overpressure.volume_reader import prepare_index, VolumeReader, VolumeIndex


ROOT = Path(__file__).resolve().parents[1]


class VolumePipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.folder = Path(self.tmp.name)
        cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        cfg['seismic']['header_fields'].update(inline=21, crossline=1)
        cfg["seismic"].update(density_file="density.sgy", impedance_file="ip.sgy")
        cfg["data"]["depth_reference"] = {"confirmed": True, "type": "TVD", "datum": "synthetic"}
        cfg["time_depth"].update(confirmed=True, depth_type="TVD", coefficients=[100, 1000], valid_time_range=[0, 1], datum="synthetic", source="synthetic fixture")
        self.project = Project(self.folder / "project.yaml", cfg); self.n, self.ns = 513, 7
        for filename, value in (("density.sgy", 2400), ("ip.sgy", 8000000)):
            spec = segyio.spec(); spec.format = 1; spec.sorting = 2; spec.samples = np.arange(self.ns); spec.tracecount = self.n
            with segyio.create(str(self.folder / filename), spec) as f:
                f.bin[segyio.BinField.Interval] = 1000
                for i in range(self.n):
                    f.header[i] = {1: i + 100, 21: 1, 73: i, 77: 200, 109: 0, 115: self.ns, 117: 1000}
                    data = np.full(self.ns, value, dtype=np.float32); data[1] = 0; f.trace[i] = data
        self.bundle = self.folder / "model"; self.bundle.mkdir(); model = PointMLP([4], 0)
        for param in model.parameters(): param.data.zero_()
        model.network[-1].bias.data.fill_(7)
        torch.save(model.state_dict(), self.bundle / "weights.pt")
        meta = {"schema_version": 1, "status": "completed", "purpose": "training", "algorithm": "mlp", "features": list(FEATURES), "project": cfg,
                "architecture": {"hidden_sizes": [4], "dropout": 0}, "weights_sha256": sha256_file(self.bundle / "weights.pt"),
                "preprocessing": {"mean": [0] * 4, "scale": [1] * 4, "minimum": [0] * 4, "maximum": [1e9] * 4}}
        (self.bundle / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")

    def predict(self):
        with redirect_stdout(io.StringIO()): return predict_seismic_volume(self.project, self.bundle, block_traces=32)

    def test_line_rectangle_noncontiguous_sources_preserve_headers_and_support_slices(self):
        from overpressure.predict_volume import predict
        self.project.settings['seismic']['header_fields'].update(inline=189,crossline=193)
        for filename in ('density.sgy','ip.sgy'):
            with segyio.open(str(self.folder/filename),'r+',ignore_geometry=True) as f:
                for i in range(self.n):f.header[i].update({189:10+i%3,193:20+i//3})
        with redirect_stdout(io.StringIO()):out=predict(self.project,self.bundle,line_range=(10,10,20,22),block_traces=2)
        report=json.loads((out/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(report['shape'],[3,self.ns]); self.assertTrue(report['round_trip_verified'])
        np.testing.assert_array_equal(np.load(out/report['source_trace_indices_file']),[0,3,6])
        self.assertEqual(report['source_trace_selection']['inline'],[10,10])
        with segyio.open(str(out/'pressure.sgy'),'r',ignore_geometry=True) as f:
            self.assertEqual([f.header[i][1] for i in range(3)],[100,103,106])
            self.assertEqual([f.header[i][193] for i in range(3)],[20,21,22])
            np.testing.assert_allclose(f.trace[2],[7,np.nan,7,7,7,7,7],equal_nan=True)
        with redirect_stdout(io.StringIO()):index=prepare_index(out)
        with VolumeReader(index) as reader:
            section=reader.section('inline',10)
            np.testing.assert_array_equal(section.x,[20,21,22]); self.assertEqual(section.values.shape,(self.ns,3))

    def test_line_rectangle_can_exceed_old_512_trace_limit(self):
        from overpressure.predict_volume import predict
        with redirect_stdout(io.StringIO()):out=predict(self.project,self.bundle,line_range=(1,1,100,612))
        report=json.loads((out/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(report['shape'],[513,self.ns]); self.assertTrue(report['round_trip_verified'])

    def test_line_rectangle_rejects_reversed_empty_and_full_combination(self):
        from overpressure.predict_volume import predict
        with self.assertRaisesRegex(ValueError,'起始线号'):predict(self.project,self.bundle,line_range=(2,1,100,101))
        with self.assertRaisesRegex(ValueError,'不能同时'):predict(self.project,self.bundle,full=True,line_range=(1,1,100,101))
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'没有地震道'):predict(self.project,self.bundle,line_range=(8,9,100,101))
        self.assertFalse((self.folder/'outputs').exists())

    def test_full_volume_and_on_demand_reader_exceed_old_512_trace_limit(self):
        self.project.settings["data"].pop("depth_reference")
        self.project.settings["time_depth"].update(confirmed=False, depth_type="unknown", datum="", source="")
        with patch("overpressure.predict_volume.MAX_BLOCK_POINTS", 70): out = self.predict()
        report = json.loads((out / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(report["shape"], [self.n, self.ns]); self.assertEqual(report["purpose"], "seismic_volume")
        self.assertLessEqual(report["peak_block_points"], 70); self.assertNotIn("source_trace_indices", report)
        with redirect_stdout(io.StringIO()): index = prepare_index(out)
        self.assertLess(index.stat().st_size, 50000)
        with VolumeReader(index) as reader:
            section = reader.section("time", 0)
            self.assertEqual(section.values.shape, (1, self.n)); self.assertTrue((section.values == 7).all())
            self.assertEqual(reader.last_request["sample_values_read"], self.n)
            self.assertTrue(np.isnan(reader.section("time", 1).values).all())
            section = reader.section("inline", 1, max_cells=20)
            self.assertLessEqual(section.values.size, 20); self.assertLessEqual(reader.last_request["sample_values_read"], 20)

    def test_changes_after_index_are_detected_without_using_stale_geometry(self):
        out = self.predict()
        with redirect_stdout(io.StringIO()): index = prepare_index(out)
        stat = (out / "pressure.sgy").stat(); os.utime(out / "pressure.sgy", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, "发生变化"): VolumeIndex.load(index)

    def test_horizon_reads_at_most_two_samples_per_trace_and_keeps_invalids(self):
        with redirect_stdout(io.StringIO()): index = prepare_index(self.predict())
        with VolumeReader(index) as reader:
            self.assertIsNotNone(reader.index.x)
            section = reader.horizon_section(np.full(self.n, 2.5), "A")
            self.assertTrue((section.values == 7).all())
            self.assertEqual(reader.last_request["sample_values_read"], self.n*2)
            section = reader.horizon_section(np.full(self.n, 1.5), "A")
            self.assertTrue(np.isnan(section.values).all())
            section = reader.horizon_section(np.full(self.n, 100.), "A")
            self.assertTrue(np.isnan(section.values).all()); self.assertEqual(reader.last_request["traces_read"], 0)

    def test_variable_start_times_are_aligned_in_streamed_time_slices(self):
        for name in ("density.sgy", "ip.sgy"):
            with segyio.open(str(self.folder / name), "r+", ignore_geometry=True) as f: f.header[1][109] = 1
        out = self.predict()
        with redirect_stdout(io.StringIO()): index = prepare_index(out)
        with VolumeReader(index) as reader:
            first = reader.section("time", 0); second = reader.section("time", 1)
            self.assertEqual(first.values[0, 0], 7); self.assertTrue(np.isnan(first.values[0, 1]))
            self.assertTrue(np.isnan(second.values[0, 0])); self.assertEqual(second.values[0, 1], 7)

    def test_cancel_during_verification_does_not_publish_partial_volume(self):
        stop = self.folder / "cancel"
        def cancel(event, **values):
            if event == "seismic_progress" and values.get("phase") == "verify": stop.touch()
        with patch("overpressure.predict_volume.emit", side_effect=cancel):
            with self.assertRaises(Cancelled): predict_seismic_volume(self.project, self.bundle, block_traces=32, cancel_file=stop)
        out = next((self.folder / "outputs" / "seismic_volumes").iterdir())
        self.assertEqual(json.loads((out / "result.json").read_text(encoding="utf-8"))["status"], "cancelled")
        self.assertFalse((out / "pressure.sgy").exists()); self.assertTrue((out / "pressure.partial.sgy").exists())

    def test_cancelled_index_does_not_invalidate_completed_prediction(self):
        out = self.predict(); stop = self.folder / "cancel"
        def cancel(event, **values):
            if event == "seismic_progress" and values.get("phase") == "index": stop.touch()
        with patch("overpressure.volume_reader.emit", side_effect=cancel):
            with self.assertRaises(Cancelled): prepare_index(out, stop)
        self.assertFalse((out / "volume_index.json").exists())
        self.assertEqual(json.loads((out / "result.json").read_text(encoding="utf-8"))["status"], "completed")

    def test_insufficient_disk_space_fails_before_writing_volume(self):
        from collections import namedtuple
        usage = namedtuple("Usage", "total used free")(100, 100, 0)
        with patch("overpressure.predict_volume.shutil.disk_usage", return_value=usage), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "空间不足"): predict_seismic_volume(self.project, self.bundle)
        out = next((self.folder / "outputs" / "seismic_volumes").iterdir())
        self.assertEqual(json.loads((out / "result.json").read_text(encoding="utf-8"))["status"], "failed")
        self.assertFalse((out / "pressure.sgy").exists())

from pathlib import Path
import tempfile
import unittest

import numpy as np
import segyio
import yaml

from overpressure.config import Project
from overpressure.seismic import inspect_seismic


ROOT = Path(__file__).resolve().parents[1]


class SeismicContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        cfg['seismic']['header_fields'].update(inline=21, crossline=1, x=73, y=77)
        cfg["seismic"].update(density_file="density.sgy", impedance_file="ip.sgy")
        cfg["time_depth"].update(coefficients=[100, 1000], valid_time_range=[0, 1])
        self.project = Project(self.folder / "project.yaml", cfg)
        for name, value in (("density.sgy", 2400), ("ip.sgy", 8000000)):
            spec = segyio.spec()
            spec.format = 5
            spec.sorting = 2
            spec.samples = np.arange(4, dtype=float)
            spec.tracecount = 2
            with segyio.create(str(self.folder / name), spec) as dst:
                dst.bin[segyio.BinField.Interval] = 1000
                for i in range(2):
                    dst.header[i] = {1: i + 1, 21: 1, 73: 100 + i, 77: 200, 109: 0, 115: 4, 117: 1000}
                    dst.trace[i] = np.array([value, 0, value, value], dtype=np.float32)

    def test_matching_geometry_and_pointwise_validity(self):
        report = inspect_seismic(self.project)
        self.assertTrue(report["passed"])
        self.assertEqual(report["sample"]["prediction_valid_points"], 6)
        self.assertEqual(report["sample"]["invalid_points"], 2)
        self.assertTrue(report["warnings"])

    def test_same_shape_with_swapped_geometry_is_rejected(self):
        with segyio.open(str(self.folder / "ip.sgy"), "r+", ignore_geometry=True) as f:
            f.header[0][73] = 999
        report = inspect_seismic(self.project)
        self.assertFalse(report["passed"])
        self.assertEqual(report["header_mismatches"]["x"], 1)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import pandas as pd
from overpressure.horizons import map_horizon, overlay_line, HorizonMap, geometry_hash
from overpressure.volume_reader import scaled_coordinates
from overpressure.seismic_view import Slice


class HorizonTests(unittest.TestCase):
    def setUp(self):
        self.index = SimpleNamespace(inline=np.array([10, 10, 11, 11]), crossline=np.array([20, 21, 20, 21]),
                x=np.array([100., 110., 100., 110.]), y=np.array([200., 200., 210., 210.]), coordinate_units=np.ones(4),
                time_convention="twt", t0_ms=1000., interval_ms=2., offsets=np.array([0, 0, 1, 1]), ns=5)
        self.config = dict(coordinates="lines", columns={"inline": "il", "crossline": "xl", "time": "t", "name": "layer"},
                           time_unit="s", time_convention="twt", max_distance=2.)

    def test_line_mapping_units_layers_missing_and_conflicting_duplicates(self):
        data = pd.DataFrame({"il": [10, 10, 11, 99, 10], "xl": [20, 20, 21, 20, 21],
                             "t": [1.002, 1.004, 1.006, 1.002, 1.004], "layer": ["A", "A", "A", "A", "B"]})
        names, values, table = map_horizon(data, self.index, self.config)
        self.assertEqual(names, ["A", "B"])
        self.assertTrue(np.isnan(values[0, 0])); self.assertEqual(values[0, 3], 1006)
        self.assertEqual(values[1, 1], 1004)
        self.assertEqual(table.match_status.iloc[0], "conflicting_time")
        self.assertEqual(table.match_status.iloc[3], "trace_not_found")

    def test_xy_distances_ties_and_coordinate_scalars(self):
        np.testing.assert_array_equal(scaled_coordinates([10, 100, 3], [10, -10, 0]), [100, 10, 3])
        self.config.update(coordinates="xy", columns={"x": "x", "y": "y", "time": "t", "name": ""}, max_distance=6)
        data = pd.DataFrame({"x": [100.2, 105, 999], "y": [200, 200, 999], "t": [1.002]*3})
        _, values, table = map_horizon(data, self.index, self.config)
        self.assertEqual(table.match_status.tolist(), ["matched", "ambiguous_xy", "beyond_distance"])
        self.assertAlmostEqual(table.xy_distance.iloc[0], .2)
        self.assertEqual(values[0, 0], 1002)

    def test_overlay_breaks_missing_and_out_of_time_points(self):
        section = Slice(np.zeros((5, 2)), np.array([20, 21]), np.arange(5)+1000, "Crossline", "Time (ms)", "", True)
        times = np.array([1002, 1100, np.nan, 1006])
        line = overlay_line(self.index, times, section, "inline", 10)
        self.assertEqual(line[0], 1002); self.assertTrue(np.isnan(line[1]))
        section.x = np.array([10, 11])
        line = overlay_line(self.index, times, section, "crossline", 21)
        self.assertTrue(np.isnan(line[0])); self.assertEqual(line[1], 1006)

    def test_time_conventions_and_non_length_xy_are_rejected(self):
        data = pd.DataFrame({"il": [10], "xl": [20], "t": [1.002], "layer": ["A"]})
        self.config["time_convention"] = "owt"
        with self.assertRaisesRegex(ValueError, "单/双程"): map_horizon(data, self.index, self.config)
        self.config.update(time_convention="twt", coordinates="xy", columns={"x": "il", "y": "xl", "time": "t", "name": "layer"})
        self.index.coordinate_units[:] = 3
        with self.assertRaisesRegex(ValueError, "平面长度"): map_horizon(data, self.index, self.config)

    def test_cache_geometry_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "layer.npz"
            np.savez_compressed(path, names=np.array(["A"]), times_ms=np.full((1, 4), 1002.), geometry=np.array(geometry_hash(self.index)))
            self.assertEqual(HorizonMap.load(path, self.index).layer("A")[0], 1002)
            self.index.inline[0] = 999
            with self.assertRaisesRegex(ValueError, "几何"): HorizonMap.load(path, self.index)

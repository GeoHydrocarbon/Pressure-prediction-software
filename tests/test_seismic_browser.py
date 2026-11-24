"""Read raw data without a model, keep geometry gaps, and sample map polylines."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import segyio

from overpressure.seismic_browser import SeismicDataset, path_samples


class SeismicBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'input.sgy'
        self.pairs = [(il, xl) for il in range(10, 13) for xl in range(20, 24) if (il, xl) != (11, 21)]
        spec = segyio.spec()
        spec.format = 5
        spec.tracecount = len(self.pairs)
        spec.samples = np.arange(9) * 2 + 1000
        with segyio.create(str(self.path), spec) as f:
            f.bin[segyio.BinField.Interval] = 2000
            for i, (il, xl) in enumerate(self.pairs):
                delay = 1002 if (il, xl) == (10, 20) else 1000
                f.header[i] = {189: il, 193: xl, 181: (300000 + xl * 25) * 10,
                               185: (4000000 + il * 25) * 10, 71: -10, 89: 1,
                               109: delay, 115: 9, 117: 2000}
                f.trace[i] = np.arange(9, dtype=np.float32) + il * 100 + xl
        self.config = dict(path=str(self.path), label='Synthetic seismic', unit='Amplitude',
                           headers=dict(inline=189, crossline=193, x=181, y=185, coordinate_scalar=71, coordinate_units=89))

    def reader(self):
        reader = SeismicDataset(self.config)
        self.addCleanup(reader.close)
        return reader

    def test_map_reads_only_headers_until_time_slice_requested(self):
        reader = self.reader()
        with patch.object(reader, 'values', wraps=reader.values) as values:
            meta = reader.metadata()
            geometry = reader.geometry()
            values.assert_not_called()
            self.assertTrue(meta['xy_available'])
            np.testing.assert_allclose(geometry['x'][0], 300500)
            result, actual = reader.map_slice(1000)
            self.assertTrue(np.isnan(result[0]))
            self.assertEqual(result[1], 1021)
            self.assertEqual(actual, 1000)
            self.assertEqual(values.call_count, len(self.pairs) - 1)
            self.assertTrue(all(isinstance(c.args[1], int) for c in values.call_args_list))

    def test_fixed_section_has_actual_values_and_missing_trace_stays_blank(self):
        reader = self.reader()
        section, meta = reader.section('inline', position=11)
        np.testing.assert_array_equal(section.x, [20, 21, 22, 23])
        self.assertTrue(np.isnan(section.values[:, 1]).all())
        self.assertEqual(section.values[0, 0], 1120)
        self.assertEqual(section.values[1, 2], 1123)
        self.assertEqual(meta['position'], 11)
        crossline, _ = reader.section('crossline', position=20)
        self.assertTrue(np.isnan(crossline.values[0, 0]))
        self.assertEqual(crossline.values[1, 0], 1020)

    def test_arbitrary_xy_and_line_grid_paths_match_the_same_traces(self):
        reader = self.reader()
        section, meta = reader.section('arbitrary', vertices=[[20, 11], [23, 11]])
        self.assertEqual(meta['missing_columns'], 1)
        np.testing.assert_allclose(section.values[0], [1120, np.nan, 1122, 1123], equal_nan=True)
        xy, _ = reader.section('arbitrary', coordinate_mode='xy', vertices=[[300500, 4000275], [300575, 4000275]])
        np.testing.assert_allclose(section.values, xy.values, equal_nan=True)
        self.assertEqual(xy.x[-1], 75)

    def test_polyline_and_outside_footprint_do_not_extrapolate(self):
        reader = self.reader()
        section, _ = reader.section('polyline', vertices=[[20, 10], [23, 10], [23, 12]])
        self.assertEqual(section.x[-1], 5)
        self.assertEqual(section.values[0, -1], 1223)
        outside, _ = reader.section('arbitrary', vertices=[[100, 100], [102, 100]])
        self.assertTrue(np.isnan(outside.values).all())

    def test_invalid_paths_and_changed_files_are_rejected(self):
        for points in ([[1, 1], [1, 1]], [[1, 1], [np.nan, 2]], [[1, 1]]):
            with self.assertRaises(ValueError):
                path_samples(points, 1.)
        reader = self.reader()
        with patch('overpressure.seismic_browser.stamp', return_value=dict(size=0, mtime_ns=0)):
            with self.assertRaisesRegex(ValueError, '变化'):
                reader.section('inline', position=10)

    def test_input_attributes_mask_nonpositive_but_generic_amplitudes_keep_zero_and_negative(self):
        with segyio.open(str(self.path),'r+',ignore_geometry=True) as source:
            trace=source.trace[0].copy();trace[:3]=[0,-1,5];source.trace[0]=trace
        reader=self.reader()
        np.testing.assert_allclose(reader.values(0,slice(0,3)),[0,-1,5])
        reader.config['positive_only']=True
        np.testing.assert_allclose(reader.values(0,slice(0,3)),[np.nan,np.nan,5],equal_nan=True)

    def test_verified_pressure_results_keep_masked_samples(self):
        from test_volume_pipeline import VolumePipelineTests
        from overpressure.predict_volume import predict
        fixture = VolumePipelineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        out = predict(fixture.project, fixture.bundle, trace_count=2)
        reader = SeismicDataset(dict(path=str(out)))
        self.addCleanup(reader.close)
        self.assertEqual(reader.unit, 'MPa')
        values, _ = reader.map_slice(reader.t0_ms)
        self.assertTrue(np.isfinite(values).any())
        section, _ = reader.section('inline', position=reader.inline[0])
        self.assertTrue(np.isfinite(section.values).any())


if __name__ == '__main__':
    unittest.main()

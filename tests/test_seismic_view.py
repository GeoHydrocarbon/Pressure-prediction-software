import unittest

import numpy as np

from overpressure.charts import seismic_figure
from overpressure.seismic_view import SeismicView


class SeismicViewTests(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[0, 1, 2], [10, np.nan, 12], [20, 21, 22], [30, 31, 32]], dtype=np.float32)
        self.view = SeismicView(self.data, [10, 10, 11, 11], [20, 22, 20, 21], [0, 0, 1, 1], 1)

    def test_inline_keeps_missing_lines_and_aligns_time_without_interpolation(self):
        section = self.view.section("inline", 10)
        self.assertEqual(section.values.shape, (4, 3))
        np.testing.assert_equal(section.values[:, 0], [0, 1, 2, np.nan])
        self.assertTrue(np.isnan(section.values[:, 1]).all())
        self.assertEqual(section.values[0, 2], 10)
        self.assertTrue(np.isnan(section.values[1, 2]))

    def test_horizontal_uses_absolute_time_and_retains_valid_zero(self):
        section = self.view.section("time", 1)
        np.testing.assert_equal(section.values, [[1, np.nan, np.nan], [20, 30, np.nan]])
        self.assertIn("1 ms", section.title)
        first = self.view.section("time", 0)
        self.assertEqual(first.values[0, 0], 0)
        self.assertTrue(np.isnan(first.values[1]).all())

    def test_crossline_orientation_and_plot_mask(self):
        section = self.view.section("crossline", 20)
        np.testing.assert_equal(section.values[:, 1], [np.nan, 20, 21, 22])
        figure = seismic_figure(section, self.view.limits)
        self.assertTrue(figure.axes[0].yaxis_inverted())
        self.assertEqual(figure.axes[0].get_ylabel(), "Time (ms)")
        self.assertTrue(np.ma.isMaskedArray(figure.axes[0].collections[0].get_array()))

    def test_non_aligned_time_duplicate_geometry_and_excessive_span_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "采样网格"):
            SeismicView(self.data, [10, 10, 11, 11], [20, 22, 20, 21], [0, 0, 1, 1], 2)
        with self.assertRaisesRegex(ValueError, "重复"):
            SeismicView(self.data, [10, 10, 11, 11], [20, 20, 20, 21], [0] * 4, 1)
        with self.assertRaisesRegex(ValueError, "跨度"):
            SeismicView(self.data, [10, 10, 11, 11], [20, 99999, 20, 21], [0] * 4, 1)

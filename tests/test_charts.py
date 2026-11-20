from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from overpressure.charts import pressure_figure, training_figure, scatter_figure, save_figure, FigureStyle, export_figure
from PIL import Image


class ChartTests(unittest.TestCase):
    def test_shared_typography_separates_title_axes_ticks_legend_and_colorbar(self):
        from overpressure.charts import seismic_figure
        from overpressure.seismic_view import Slice
        section=Slice(np.ones((2,2)),np.array([10,11]),np.array([1000,1002]),'Crossline','Time (ms)','Inline 20',True)
        style=FigureStyle(font_size=11,axis_label_size=16,tick_size=12,legend_size=9,title_size=19,title='Custom slice',grid=True)
        figure=seismic_figure(section,(0,20),overlay=np.array([1000,1001]),style=style)
        ax,cbar=figure.axes
        self.assertEqual(ax.get_title(),'Custom slice');self.assertEqual(ax.title.get_fontsize(),19)
        self.assertEqual(ax.xaxis.label.get_fontsize(),16);self.assertEqual(ax.get_xticklabels()[0].get_fontsize(),12)
        self.assertEqual(ax.get_legend().get_texts()[0].get_fontsize(),9)
        self.assertEqual(cbar.get_title(),'');self.assertEqual(cbar.yaxis.label.get_fontsize(),16)
        self.assertTrue(any(line.get_visible() for line in ax.get_xgridlines()))
        style.show_title=False
        figure=seismic_figure(section,(0,20),style=style);self.assertEqual(figure.axes[0].get_title(),'')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'style.json';style.save(path);self.assertEqual(FigureStyle.load(path).tick_size,12)

    def test_scatter_has_no_reference_line_and_reports_actual_r2(self):
        frame = pd.DataFrame({"well": ["A", "B", "A"], "Pp": [10, 20, 30], "Pp_pred": [11, 19, 32]})
        ax = scatter_figure(frame).axes[0]
        self.assertEqual(len(ax.lines), 0)
        self.assertTrue(any("R² = 0.970" in text.get_text() for text in ax.texts))
        self.assertEqual(sum(len(item.get_offsets()) for item in ax.collections), 3)

    def test_custom_style_dimensions_title_and_export_roundtrip(self):
        style = FigureStyle(width_cm=15.24, height_cm=10.16, dpi=100, font_size=13, show_title=False)
        figure = training_figure(pd.DataFrame({"epoch": [1, 2], "train_data": [2, 1]}), style)
        self.assertEqual(figure.axes[0].get_title(), "")
        self.assertEqual(figure.axes[0].xaxis.label.get_fontsize(), 13)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "chart.png"; export_figure(figure, path, style)
            with Image.open(path) as raster: self.assertEqual(raster.size, (600, 400))
            style.save(Path(temp) / "style.json")
            self.assertEqual(FigureStyle.load(Path(temp) / "style.json"), style)
            export_figure(figure, Path(temp) / "chart.svg", style)
            self.assertIn("<text", (Path(temp) / "chart.svg").read_text(encoding="utf-8"))

    def test_unreasonable_export_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "像素"):
            FigureStyle(width_cm=50, height_cm=50, dpi=600).validate()

    def test_seismic_overlay_gaps_custom_limits_and_time_direction(self):
        from overpressure.charts import seismic_figure
        from overpressure.seismic_view import Slice
        section = Slice(np.ones((3, 3)), np.array([10, 11, 12]), np.array([1000, 1002, 1004]), "Crossline", "Time (ms)", "Inline 20", True)
        figure = seismic_figure(section, (0, 20), cmap="cividis", xlim=(10, 12), ylim=(1000, 1004), overlay=np.array([1001, np.nan, 1003]))
        ax = figure.axes[0]
        self.assertEqual(ax.get_ylim(), (1004, 1000)); self.assertEqual(ax.collections[0].get_clim(), (0, 20))
        self.assertTrue(np.isnan(ax.lines[0].get_ydata()[1]))
    def test_pressure_curve_breaks_at_depth_gap_and_inverts_axis(self):
        frame = pd.DataFrame({"well": ["A"] * 5, "Depth": [100, 101, 102, 120, 121], "Pp": [1, 2, np.nan, 4, 5], "Pp_pred": [1, 2, 3, 4, 5]})
        figure = pressure_figure(frame, "A"); ax = figure.axes[0]
        self.assertTrue(ax.yaxis_inverted())
        self.assertTrue(np.isnan(ax.lines[1].get_xdata()[3]))

    def test_training_plot_exports_nonempty_png(self):
        history = pd.DataFrame({"epoch": [1, 2], "train_data": [10, 5], "validation_data": [12, 8]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "curve.png"; save_figure(training_figure(history), path)
            self.assertGreater(path.stat().st_size, 1000)

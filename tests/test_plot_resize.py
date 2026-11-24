"""Mouse interaction and export contracts for the shared plot previews."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
try:
    from PySide6.QtCore import Qt,QPoint
    from PySide6.QtWidgets import QApplication,QDialogButtonBox,QCheckBox
    from PySide6.QtTest import QTest
except ImportError as exc:
    raise unittest.SkipTest("Qt unavailable in this Python environment") from exc

from overpressure.gui import configure_application
from overpressure.chart_panel import ChartPanel
from overpressure.slice_panel import SlicePanel
from overpressure.charts import FigureStyle,export_figure
from overpressure.seismic_view import SeismicView
from overpressure.ui_style import polish_layouts


class PlotResizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        configure_application(cls.app)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)
        self.charts=ChartPanel(self.folder/"style.json")
        self.charts.history=pd.DataFrame(dict(epoch=[1,2,3],train_data=[8.,5.,2.],validation_data=[10.,7.,4.]))
        self.charts.pressures=pd.DataFrame(dict(well=["Synthetic"]*12,Depth=np.arange(12)+100.,Pp=np.arange(12)+20.,Pp_pred=np.arange(12)+20.5))
        self.charts.well.addItem("All",None);self.charts.well.addItem("Synthetic","Synthetic")
        self.charts.redraw()
        self.slices=SlicePanel(lambda *args:None)
        values=np.linspace(20.,50.,4*51).reshape(4,51)
        values[1,7]=np.nan
        self.slices.view=SeismicView(values,[10,10,11,11],[20,21,20,21],[1000]*4,10)
        self.slices.update_positions()
        for panel in (self.charts,self.slices):
            polish_layouts(panel);panel.resize(1100,820);panel.show()
        self.settle()

    def tearDown(self):
        self.slices.shutdown()
        for panel in (self.charts,self.slices):
            panel.close();panel.deleteLater()
        self.settle()

    def settle(self):
        for _ in range(8):self.app.processEvents()

    def drag(self,panel,dx=-65,dy=0,corner=False):
        frame=panel.aspect_preview.frame
        start=QPoint(frame.width()-3,frame.height()-3 if corner else frame.height()//2)
        QTest.mousePress(frame,Qt.MouseButton.LeftButton,pos=start)
        QTest.mouseMove(frame,start+QPoint(dx,dy),delay=20)
        QTest.mouseRelease(frame,Qt.MouseButton.LeftButton,pos=start+QPoint(dx,dy))
        self.settle()

    def assert_bounded(self,panel):
        box=panel.aspect_preview
        self.assertTrue(box.rect().contains(box.frame.geometry()),(box.size(),box.frame.geometry()))
        self.assertEqual(panel.preview.verticalScrollBar().maximum(),0)
        self.assertEqual(panel.preview.horizontalScrollBar().maximum(),0)

    def test_edge_drag_changes_only_preview_and_keeps_zoom(self):
        for panel in (self.charts,self.slices):
            with self.subTest(panel=type(panel).__name__):
                before=(panel.canvas.width(),panel.canvas.height())
                ax=panel.canvas.figure.axes[0];ax.set_xlim(20.1,20.9)
                limits=ax.get_xlim()
                panel.toolbar.pan()
                self.drag(panel)
                self.assertLess(panel.canvas.width(),before[0]-20)
                self.assertAlmostEqual(panel.canvas.height(),before[1],delta=2)
                self.assertEqual(ax.get_xlim(),limits)
                self.assertTrue(panel.toolbar.mode)
                panel.toolbar.pan()
                self.assert_bounded(panel)

    def test_corner_lock_fit_and_restore(self):
        for panel in (self.charts,self.slices):
            with self.subTest(panel=type(panel).__name__):
                self.drag(panel)
                ratio=panel.aspect_preview.ratio
                panel.preview_controls.lock.setChecked(True)
                self.drag(panel,-35,-25,corner=True)
                self.assertAlmostEqual(panel.aspect_preview.ratio,ratio)
                self.assertAlmostEqual(panel.canvas.width()/panel.canvas.height(),ratio,delta=.025)
                panel.preview_controls.fit_button.click();self.settle()
                self.assertAlmostEqual(panel.aspect_preview.scale,1.)
                self.assertAlmostEqual(panel.aspect_preview.ratio,ratio)
                panel.preview_controls.reset_button.click();self.settle()
                configured=panel.style.width_cm/panel.style.height_cm if panel is self.charts else panel.image_width_cm/panel.image_height_cm
                self.assertAlmostEqual(panel.aspect_preview.ratio,configured)
                self.assert_bounded(panel)

    def test_preview_survives_redraw_and_slice_direction_change(self):
        pressure=self.charts.pressures.copy(deep=True)
        seismic=self.slices.view.pressure.copy()
        for panel in (self.charts,self.slices):
            self.drag(panel)
        chart_ratio=self.charts.aspect_preview.ratio
        slice_ratio=self.slices.aspect_preview.ratio
        self.charts.mode.setCurrentIndex(3);self.charts.redraw()
        self.slices.mode.setCurrentIndex(1)
        self.settle()
        self.assertAlmostEqual(self.charts.aspect_preview.ratio,chart_ratio)
        self.assertAlmostEqual(self.slices.aspect_preview.ratio,slice_ratio)
        self.charts.mode.setCurrentIndex(4);self.settle()
        self.assertFalse(self.charts.preview_controls.isVisible())
        self.charts.mode.setCurrentIndex(3);self.settle()
        self.assertTrue(self.charts.preview_controls.isVisible())
        self.assertAlmostEqual(self.charts.aspect_preview.ratio,chart_ratio)
        pd.testing.assert_frame_equal(self.charts.pressures,pressure)
        np.testing.assert_array_equal(self.slices.view.pressure,seismic)

    def test_window_resize_bounds_and_numeric_ratio_settings(self):
        for panel in (self.charts,self.slices):
            with self.subTest(panel=type(panel).__name__):
                self.drag(panel)
                ratio=panel.aspect_preview.ratio
                for w,h in ((900,700),(640,520),(1100,820)):
                    panel.resize(w,h);self.settle();self.assert_bounded(panel)
                    self.assertAlmostEqual(panel.aspect_preview.ratio,ratio)
                if panel is self.charts:panel.apply_style(FigureStyle(width_cm=12,height_cm=30),persist=False)
                else:
                    panel.image_width_cm=12;panel.image_height_cm=30;panel.refresh_display()
                self.settle()
                self.assertAlmostEqual(panel.aspect_preview.ratio,.4)
                self.assertAlmostEqual(panel.aspect_preview.scale,1.)

    def test_mouse_drag_preserves_export_dimensions_and_dpi(self):
        self.charts.apply_style(FigureStyle(width_cm=24,height_cm=15,dpi=90),persist=False)
        self.slices.image_width_cm=24;self.slices.image_height_cm=15;self.slices.image_dpi=90
        self.slices.refresh_display();self.settle()
        for panel in (self.charts,self.slices):
            self.drag(panel,dx=-100)
        chart_size=self.charts.canvas.figure.get_size_inches().copy()
        slice_size=self.slices.canvas.figure.get_size_inches().copy()
        export_figure(self.charts.canvas.figure,self.folder/"chart.png",self.charts.style)
        self.slices.save_slice_figure(self.folder/"slice.png")
        for name in ("chart.png","slice.png"):
            with Image.open(self.folder/name) as im:
                self.assertAlmostEqual(im.width,24/2.54*90,delta=2)
                self.assertAlmostEqual(im.height,15/2.54*90,delta=2)
        np.testing.assert_allclose(self.charts.canvas.figure.get_size_inches(),chart_size)
        np.testing.assert_allclose(self.slices.canvas.figure.get_size_inches(),slice_size)
        self.assertEqual((self.charts.style.width_cm,self.charts.style.height_cm,self.charts.style.dpi),(24,15,90))
        self.assertEqual((self.slices.image_width_cm,self.slices.image_height_cm,self.slices.image_dpi),(24,15,90))

    def test_canvas_backing_size_matches_device_pixels(self):
        for panel in (self.charts,self.slices):
            with self.subTest(panel=type(panel).__name__):
                self.drag(panel)
                panel.redraw();self.settle()
                canvas=panel.canvas;dpr=canvas.device_pixel_ratio
                self.assertAlmostEqual(canvas.figure.bbox.width,canvas.width()*dpr,delta=2)
                self.assertAlmostEqual(canvas.figure.bbox.height,canvas.height()*dpr,delta=2)

    def test_settings_dialog_and_preview_lock_stay_in_sync(self):
        for panel,prefix in ((self.charts,"chart"),(self.slices,"slice")):
            with self.subTest(panel=prefix):
                panel.preview_controls.lock.setChecked(True)
                dialog=panel.create_style_dialog() if panel is self.charts else panel.create_display_dialog()
                lock=dialog.findChild(QCheckBox,prefix+"_ratio_lock")
                self.assertTrue(lock.isChecked())
                lock.setChecked(False)
                dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Apply).click()
                self.settle()
                self.assertFalse(panel.preview_controls.lock.isChecked())
                self.assertFalse(panel.aspect_preview.lock_ratio)
                self.assertFalse(panel.image_ratio_locked)
                dialog.close();dialog.deleteLater()


if __name__=="__main__":unittest.main()

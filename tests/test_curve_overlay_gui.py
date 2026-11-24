"""Physical-unit grouping, pressure points, retained documents and plot modes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import tempfile
import unittest
try:
    from PySide6.QtWidgets import QApplication,QMdiArea
except ImportError as exc:
    raise unittest.SkipTest('Qt unavailable; run in desktop interpreter') from exc
import numpy as np
import pandas as pd
from overpressure.config import Project
from overpressure.well_curve_viewer import CurveViewer,CurveReader
from overpressure.workspace_views import PlotSubWindow
from overpressure.traditional_views import TraditionalResultView
from overpressure.traditional import defaults,two_point_trend,analyze_trend,predict_eaton
from overpressure.well_library import inspect_curve_file


class CurveOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.project=Project(self.folder/'project.yaml',{})
        self.viewer=CurveViewer(SimpleNamespace(project=self.project))
        self.addCleanup(self.viewer.close)
        self.addCleanup(self.viewer.stop)

    def read(self):
        path=self.folder/'curves.csv'
        pd.DataFrame(dict(Depth=[1000.,1001.,1002.,1010.],pred=[20.,21.,np.nan,24.],
                          measured=[20000.,np.nan,22000.,24000.],AC=[90.,91.,92.,93.])).to_csv(path,index=False)
        refs=[dict(name='Eaton压力',column='pred',unit='MPa',role='derived'),
              dict(name='实测压力',column='measured',unit='kPa',role='measurement'),
              dict(name='声波',column='AC',unit='us/ft')]
        refs=[dict(ref,path=str(path),depth_column='Depth',depth_unit='m') for ref in refs]
        reader=CurveReader('A',refs,self.viewer)
        data=[];errors=[];reader.loaded.connect(data.extend);reader.failed.connect(errors.append)
        reader.run();self.assertFalse(errors,errors)
        return data,refs

    def test_same_pressure_axis_uses_normalized_units_and_independent_measurement_points(self):
        data,_=self.read();self.viewer.mode.setCurrentIndex(1);self.viewer.render(data[:2])
        self.assertEqual(len(self.viewer.axes),1)
        ax=self.viewer.axes[0]
        self.assertEqual(ax.get_xlabel(),'MPa')
        self.assertEqual(len(ax.lines),1);self.assertEqual(len(ax.collections),1)
        np.testing.assert_allclose(ax.collections[0].get_offsets(),[[20.,1000.],[22.,1002.],[24.,1010.]])
        self.assertEqual(np.isfinite(ax.lines[0].get_xdata()).sum(),3)
        self.assertTrue(np.isnan(ax.lines[0].get_ydata()).any())
        self.assertGreater(ax.get_ylim()[0],ax.get_ylim()[1])
        self.assertEqual(len(ax.get_legend().get_texts()),2)
        self.assertEqual(self.viewer.dividers(),[])

    def test_integer_arrays_can_render_without_missing_interval_cast_errors(self):
        data,_=self.read()
        data[0]['x']=np.array([20,21,22,24]);data[0]['z']=np.array([1000,1001,1002,1010])
        self.viewer.render(data[:1])
        values=self.viewer.axes[0].lines[0].get_xdata()
        self.assertEqual(np.isfinite(values).sum(),4)
        self.assertTrue(np.isnan(values).any())

    def test_mixed_units_share_depth_but_keep_independent_horizontal_scales(self):
        data,_=self.read();self.viewer.mode.setCurrentIndex(1);self.viewer.render(data)
        self.assertEqual(len(self.viewer.axes),2)
        first,second=self.viewer.axes
        self.assertEqual([ax.get_xlabel() for ax in self.viewer.axes],['MPa','us/ft'])
        self.assertTrue(first.get_shared_y_axes().joined(first,second))
        first.set_ylim(1008,1001);self.assertEqual(second.get_ylim(),(1008,1001))
        self.viewer.mode.setCurrentIndex(0)
        self.assertEqual(len(self.viewer.axes),3)
        self.viewer.mode.setCurrentIndex(1)
        self.viewer.track_list.item(2).setSelected(True);self.viewer.remove_tracks()
        self.assertEqual(len(self.viewer.axes),1)
        self.assertEqual(len(self.viewer.datasets),2)

    def test_custom_tracks_mix_independent_curves_with_a_pressure_overlay(self):
        data,_=self.read()
        data.extend([dict(data[2],name='密度',unit='g/cc',x=np.array([2.3,2.4,2.5,2.4])),
                     dict(data[2],name='压力系数',unit='无量纲',x=np.array([1.2,1.3,1.4,1.5]))])
        self.viewer.render(data)
        tracks={'声波':1,'密度':2,'Eaton压力':3,'实测压力':3,'压力系数':4}
        rows=self.viewer.setting_rows()
        for row in rows:row['track']=tracks[row['name']]
        self.viewer.apply_plot_settings(rows)
        self.assertEqual(self.viewer.mode.currentText(),'自定义分道')
        self.assertEqual(len(self.viewer.base_axes),4)
        pressure=self.viewer.base_axes[2]
        self.assertEqual(len(pressure.lines),1);self.assertEqual(len(pressure.collections),1)
        self.assertEqual(len(self.viewer.base_axes[0].lines),1)
        self.assertEqual(len(self.viewer.dividers()),3)
        pressure.set_ylim(1008,1001)
        for ax in self.viewer.axes:self.assertEqual(ax.get_ylim(),(1008,1001))
        self.viewer.mode.setCurrentIndex(1);self.viewer.mode.setCurrentIndex(2)
        self.assertEqual(len(self.viewer.base_axes),4)

    def test_explicit_curve_styles_control_markers_lines_and_export(self):
        data,_=self.read();self.viewer.render(data[:2])
        from overpressure.curve_settings import CurveSettingsDialog
        dialog=CurveSettingsDialog(self.viewer,self.viewer.setting_rows())
        self.addCleanup(dialog.close)
        measured=dialog.controls[1]
        self.assertTrue(measured['markersize'].isEnabled());self.assertFalse(measured['linewidth'].isEnabled())
        measured['track'].setValue(1);measured['kind'].setCurrentIndex(measured['kind'].findData('both'))
        measured['marker'].setCurrentIndex(measured['marker'].findData('s'));measured['markersize'].setValue(9)
        measured['linewidth'].setValue(2.5);measured['linestyle'].setCurrentIndex(measured['linestyle'].findData('--'))
        dialog.set_color(measured['color'],'#d62728');dialog.set_color(measured['marker_color'],'#008800')
        predicted=dialog.controls[0]
        predicted['kind'].setCurrentIndex(predicted['kind'].findData('points'))
        predicted['marker'].setCurrentIndex(predicted['marker'].findData('D'))
        self.viewer.apply_plot_settings(dialog.values())
        ax=self.viewer.base_axes[0]
        self.assertEqual(len(ax.lines),1);self.assertEqual(len(ax.collections),2)
        self.assertEqual(ax.lines[0].get_color(),'#d62728');self.assertEqual(ax.lines[0].get_linestyle(),'--')
        self.assertEqual(ax.lines[0].get_linewidth(),2.5)
        np.testing.assert_allclose(ax.collections[1].get_sizes(),[81.])
        from matplotlib.colors import to_rgba
        np.testing.assert_allclose(ax.collections[1].get_facecolors()[0],to_rgba('#008800'))
        export=self.folder/'styles.svg';self.viewer.canvas.figure.savefig(export)
        svg=export.read_text(encoding='utf-8');self.assertIn('#d62728',svg);self.assertIn('#008800',svg)
        self.assertIn('stroke-dasharray',svg)
        self.viewer.mode.setCurrentIndex(0)
        self.assertEqual(len(self.viewer.axes[0].lines),0)
        self.assertEqual(len(self.viewer.axes[1].lines),1)

    def test_predicted_curve_named_pp_defaults_to_line_and_user_override_survives_reread(self):
        _,refs=self.read();ref=dict(refs[0],name='Pp')
        data=[];reader=CurveReader('A',[ref],self.viewer);reader.loaded.connect(data.extend);reader.run()
        self.assertFalse(data[0]['points'])
        self.viewer.render(data);self.assertEqual(len(self.viewer.axes[0].lines),1)
        rows=self.viewer.setting_rows();rows[0].update(kind='both',marker='^',markersize=7)
        self.viewer.apply_plot_settings(rows);self.viewer.render(data)
        self.assertEqual(self.viewer.curve_styles['Pp']['kind'],'both')
        self.assertEqual(len(self.viewer.axes[0].collections),1)

    def test_custom_track_divider_and_reorder_keep_group_members_and_styles(self):
        from PySide6.QtCore import QModelIndex
        data,_=self.read();self.viewer.render(data)
        rows=self.viewer.setting_rows()
        for row in rows:row['track']=2 if row['name']=='声波' else 1
        self.viewer.apply_plot_settings(rows)
        self.viewer.resize(1000,700);self.viewer.show()
        for _ in range(8):self.app.processEvents()
        self.viewer.canvas.draw()
        self.assertEqual(len(self.viewer.dividers()),1)
        x,top,bottom=self.viewer.dividers()[0];y=(top+bottom)/2
        dpr=self.viewer.canvas.device_pixel_ratio
        def event(px):return SimpleNamespace(button=1,x=px*dpr,y=(self.viewer.canvas.height()-y)*dpr)
        self.viewer.press_divider(event(x));self.viewer.move_divider(event(x+25));self.viewer.release_divider(event(x+25))
        self.assertGreater(self.viewer.weights[0],1.)
        weights=list(self.viewer.weights);styles=dict(self.viewer.curve_styles)
        self.assertTrue(self.viewer.track_list.model().moveRows(QModelIndex(),2,1,QModelIndex(),0))
        self.assertEqual(self.viewer.assignments['声波'],1)
        self.assertEqual(self.viewer.assignments['Eaton压力'],2)
        self.assertEqual(self.viewer.assignments['实测压力'],2)
        self.assertEqual(self.viewer.weights,list(reversed(weights)))
        self.assertEqual(self.viewer.curve_styles,styles)

    def test_replot_after_close_retains_visible_canvas_and_reader_completes(self):
        _,refs=self.read()
        area=QMdiArea();area.resize(850,650);area.show();self.addCleanup(area.close)
        sub=PlotSubWindow();sub.setWidget(self.viewer);area.addSubWindow(sub);sub.show()
        sub.close();sub.show();self.app.processEvents()
        self.viewer.show_curves('A',refs[:2],overlay=True)
        from PySide6.QtTest import QTest
        for _ in range(200):
            self.app.processEvents()
            if self.viewer.worker is None:break
            QTest.qWait(10)
        self.assertIsNone(self.viewer.worker)
        self.assertEqual(len(self.viewer.datasets),2)
        self.assertTrue(self.viewer.isVisible());self.assertTrue(self.viewer.canvas.isVisible())
        self.assertEqual(len(self.viewer.axes),1)
        # Leave deletion to the unittest cleanup, not QMdiArea destruction.
        area.removeSubWindow(sub);self.viewer.setParent(None)

    def test_legend_dialog_controls_each_track_without_renaming_source_curves(self):
        from overpressure.legend_settings import LegendSettingsDialog
        data,_=self.read();self.viewer.mode.setCurrentIndex(1);self.viewer.render(data[:2])
        labels={d['name']:d['name'] for d in data[:2]}
        dialog=LegendSettingsDialog(self.viewer,self.viewer.legend_rows(),labels)
        self.addCleanup(dialog.close)
        controls=dialog.controls[0]
        controls['position'].setCurrentIndex(controls['position'].findData('above'))
        controls['fontsize'].setValue(14);controls['alpha'].setValue(.35);controls['frame'].setChecked(False)
        dialog.label_edits['Eaton压力'].setText('Eaton 预测')
        self.viewer.apply_legend_settings(*dialog.values())
        legend,_,_=next(iter(self.viewer.legends.values()))
        self.assertEqual(legend.get_texts()[0].get_text(),'Eaton 预测')
        self.assertEqual(legend.get_texts()[0].get_fontsize(),14.)
        self.assertAlmostEqual(legend.get_frame().get_alpha(),.35)
        self.assertEqual(legend.get_frame().get_linewidth(),0.)
        self.assertTrue(legend.get_draggable())
        self.assertEqual(self.viewer.datasets[0]['name'],'Eaton压力')
        tracks=self.viewer.legend_rows();tracks[0]['settings']['visible']=False
        self.viewer.apply_legend_settings(tracks,labels)
        self.assertEqual(len(self.viewer.legends),0)
        self.assertEqual(len(self.viewer.base_axes[0].lines),1)

    def test_outside_legend_clears_twin_axes_decorations_and_survives_export(self):
        data,_=self.read();self.viewer.mode.setCurrentIndex(1);self.viewer.render(data)
        self.viewer.canvas.figure.set_size_inches(9,7)
        for position in ('above','below'):
            tracks=self.viewer.legend_rows();tracks[0]['settings'].update(position=position,fontsize=12)
            self.viewer.apply_legend_settings(tracks,{'Eaton压力':'Predicted','实测压力':'Measured','声波':'AC'})
            self.viewer.canvas.draw()
            legend,base,_=next(iter(self.viewer.legends.values()))
            bounds=legend.get_window_extent()
            self.assertTrue(self.viewer.canvas.figure.bbox.contains(bounds.x0,bounds.y0))
            self.assertTrue(self.viewer.canvas.figure.bbox.contains(bounds.x1,bounds.y1))
            if position=='above':
                self.assertGreater(bounds.y0,base.title.get_window_extent().y1)
                self.assertGreater(bounds.y0,self.viewer.axes[1].xaxis.label.get_window_extent().y1)
            else:self.assertLess(bounds.y1,base.xaxis.label.get_window_extent().y0)
            export=self.folder/(position+'.svg');self.viewer.canvas.figure.savefig(export)
            self.assertIn('Predicted',export.read_text(encoding='utf-8'))

    def test_mouse_dragged_legend_position_is_kept_after_redraw_and_resize(self):
        from matplotlib.backend_bases import MouseEvent
        data,_=self.read();self.viewer.mode.setCurrentIndex(1);self.viewer.render(data[:2])
        self.viewer.canvas.figure.set_size_inches(9,7);self.viewer.canvas.draw()
        key,(legend,base,_)=next(iter(self.viewer.legends.items()))
        bounds=legend.get_window_extent();x,y=bounds.x0+10,bounds.y0+10
        for name,px,py in [('button_press_event',x,y),('motion_notify_event',x-35,y-55),('button_release_event',x-35,y-55)]:
            event=MouseEvent(name,self.viewer.canvas,px,py,button=1)
            self.viewer.canvas.callbacks.process(name,event)
        options=self.viewer.legend_settings[key]
        self.assertEqual(options['position'],'custom')
        xy=np.array(options['custom_xy'])
        self.viewer.draw_tracks();self.viewer.canvas.figure.set_size_inches(10,8);self.viewer.canvas.draw()
        restored,base,_=self.viewer.legends[key]
        actual=base.transAxes.inverted().transform(restored.get_window_extent().p0)
        np.testing.assert_allclose(actual,xy,atol=1e-5)
        self.viewer.canvas.figure.savefig(self.folder/'dragged.png')

    def test_trend_and_eaton_result_contents_reopen_after_close(self):
        path=self.folder/'A.csv'
        pd.DataFrame(dict(Depth=[1000.,1001.,1002.],AC=[90.,91.,92.])).to_csv(path,index=False)
        cfg=defaults({});cfg.update(wells=['A'],analysis_well='A',analysis_min=900,analysis_max=1100,
            applied_trends={'A':dict(two_point_trend([[1000,4.5],[3000,3.8]]),confirmed=True)})
        project=Project(self.folder/'project.yaml',dict(output_dir='results',traditional_prediction=cfg,
            project_data=dict(wells=inspect_curve_file(path))))
        with redirect_stdout(io.StringIO()):paths=[analyze_trend(project),predict_eaton(project)]
        area=QMdiArea();area.resize(850,650);area.show();self.addCleanup(area.close)
        for path in paths:
            view=TraditionalResultView(path);sub=PlotSubWindow();sub.setWidget(view);area.addSubWindow(sub)
            sub.show();sub.close();sub.show();self.app.processEvents()
            self.assertTrue(view.isVisible());self.assertTrue(view.canvas.isVisible())
            view.redraw();self.assertGreater(len(view.canvas.figure.axes),0)


if __name__=='__main__':unittest.main()

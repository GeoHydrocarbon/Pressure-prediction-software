import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
import time
import unittest
from unittest.mock import patch
from pathlib import Path
try:
    from PySide6.QtWidgets import QApplication,QMenu
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
except ImportError as exc:raise unittest.SkipTest('Qt unavailable; use desktop interpreter') from exc
import pandas as pd
import yaml
from overpressure.section_overlay_dialog import SectionOverlayDialog
from overpressure.section_overlays import curve_key
from overpressure.config import load_project


class WellOverlayGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_map_workspace_gui import MapWorkspaceGuiTests
        MapWorkspaceGuiTests.setUpClass();cls.app=QApplication.instance()

    def setUp(self):
        from tests.test_map_workspace_gui import MapWorkspaceGuiTests
        fixture=MapWorkspaceGuiTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.window=w=fixture.window;self.folder=fixture.folder
        w.td_method.setCurrentText('polynomial');w.coefficients.setText('0, 1000')
        heads=self.folder/'heads.csv';pd.DataFrame(dict(well=['A','B'],X=[300525,300550],Y=[4000250,4000250])).to_csv(heads,index=False)
        track=self.folder/'track.csv';pd.DataFrame(dict(well=['A','A'],MD=[1000,1060],X=[300525,300550],Y=[4000250,4000250])).to_csv(track,index=False)
        logs=self.folder/'logs.csv';pd.DataFrame(dict(well=['A']*7,Depth=list(range(1000,1070,10)),AC=list(range(80,87)))).to_csv(logs,index=False)
        self.ref=dict(name='AC',path=str(logs),column='AC',well_column='well',well_value='A',depth_column='Depth',depth_unit='m',unit='us/ft')
        w.library.wells=[dict(name='A',curves=[self.ref]),dict(name='B',curves=[])];w.library.refresh()
        w.catalog.assets=[dict(path=str(heads),kind='coordinates',columns=dict(well='well',x='X',y='Y'),wells=['A','B']),dict(path=str(track),kind='trajectory',columns=dict(well='well',depth='MD',x='X',y='Y'),wells=['A'])]
        self.relations=[]
        for number,last_depth in [(1,1060),(2,1120)]:
            path=self.folder/f'td{number}.csv';pd.DataFrame(dict(time=[1.,1.06],depth=[1000,last_depth])).to_csv(path,index=False)
            self.relations.append(w.shell.remember_time_depth(dict(method='table',table_file=str(path),time_unit='s',time_convention='twt',depth_unit='m')))

    def wait(self,predicate):
        end=time.monotonic()+35
        while not predicate() and time.monotonic()<end:self.app.processEvents();QTest.qWait(20)
        self.assertTrue(predicate(),self.window.shell.workspace.map.status.text())

    def bind(self,index):
        with patch('overpressure.workstation.QInputDialog.getItem',return_value=(self.relations[index]['name'],True)):
            self.window.shell.apply_well_time_depth('A')

    def test_binding_menu_does_not_change_global_time_depth_and_persists(self):
        w=self.window;original=deepcopy(w.settings['time_depth']);menu=QMenu();w.shell.build_context_menu(menu,dict(well='A'))
        self.assertIn('应用时深关系…',[a.text() for a in menu.actions()])
        self.bind(0);self.assertEqual(w.settings['time_depth'],original)
        cfg=load_project(w.project.path).settings;self.assertEqual(cfg['project_data']['wells'][0]['time_depth_id'],self.relations[0]['id'])
        self.assertGreaterEqual(len(cfg['workspace']['time_depth_relations']),2)
        w.settings=cfg;w.project=load_project(w.project.path);w.populate()
        self.assertEqual(w.library.wells[0]['time_depth_id'],self.relations[0]['id'])

    def test_map_track_section_curve_styles_and_binding_refresh_keep_line(self):
        from tests.test_map_workspace_gui import synthetic_segy
        from overpressure.gui import MainWindow
        w=self.window;self.bind(0);path=self.folder/'seismic.sgy';synthetic_segy(path)
        w.shell.extra_volumes=[dict(path=str(path),label='Synthetic',unit='Amplitude')];w.shell.refresh_tree()
        panel=w.shell.workspace.map;self.wait(lambda:panel.metadata is not None)
        a=next(item for item in panel.wells if item['name']=='A');self.assertEqual(len(a['trajectory_xy']),2)
        self.assertTrue(any(len(line.get_xdata())==2 for line in panel.ax.lines))
        panel.create_line('inline',position=10);ident=panel.active_line
        self.wait(lambda:ident in w.shell.workspace.sections)
        section=w.shell.workspace.sections[ident].widget()
        self.assertIsNotNone(section.meta['well_overlays'][0]['track'])
        dialog=SectionOverlayDialog(section);self.addCleanup(dialog.close)
        child=dialog.tree.topLevelItem(0).child(0);child.setCheckState(0,Qt.CheckState.Checked);dialog.tree.setCurrentItem(child)
        dialog.color.setText('#00aa00');dialog.controls['linewidth'].setValue(3.5);dialog.distance.setValue(1)
        self.assertTrue(dialog.apply())
        self.wait(lambda:bool(section.meta['well_overlays'][0]['curve_data']))
        self.assertTrue(any(line.get_color()=='#00aa00' and line.get_linewidth()==3.5 for line in section.canvas.figure.axes[0].lines))
        self.bind(1)
        self.wait(lambda:abs(section.meta['well_overlays'][0]['track']['time'][-1]-1030)<1e-5)
        self.assertIn(ident,panel.paths);self.assertIs(w.shell.workspace.sections[ident].widget(),section)
        self.assertTrue(section.meta['well_overlays'][0]['curve_data'])
        w.shell.persist_workspace();saved=load_project(w.project.path).settings
        record=saved['workspace']['map']['sections'][0]
        self.assertEqual(record['overlay_settings']['distance'],1)
        self.assertEqual(len(record['overlay_curves']),1)
        w.close();self.app.processEvents()
        reopened=MainWindow(w.project.path);self.addCleanup(reopened.close)
        self.wait(lambda:ident in reopened.shell.workspace.sections)
        restored=reopened.shell.workspace.sections[ident].widget()
        self.assertEqual(restored.overlay_settings['distance'],1)
        self.assertTrue(restored.meta['well_overlays'][0]['curve_data'])

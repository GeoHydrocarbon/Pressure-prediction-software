"""The independent worker must integrate with project state and MDI results."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
except ImportError as exc:raise unittest.SkipTest('Use the desktop interpreter for Qt tests') from exc
import yaml
from overpressure.gui import MainWindow,configure_application

ROOT=Path(__file__).resolve().parents[1]


class RockPhysicsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);configure_application(cls.app)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name).resolve()
        cfg=yaml.safe_load((ROOT/'project.yaml').read_text(encoding='utf-8'))
        cfg['algorithm']['name']='mlp';cfg['physics']['enabled']=False
        cfg['data'].update(training_selection='well_library',development_wells=[],validation_wells=[],blind_wells=[],well_files=[],well_imports=[],features=['Depth','AC','DEN','Ip'],target_curve='Pp')
        cfg['project_data']=dict(wells=[],well_assets=[],horizons=[],pressure_result='')
        cfg['seismic'].update(density_file='',impedance_file='')
        cfg['time_depth']=dict(method='polynomial',coefficients=[0.,1000.],time_unit='s',time_convention='twt',depth_unit='m',table_file='')
        cfg['workspace']={};cfg['traditional_prediction']={};cfg['rock_physics']={};cfg['output_dir']='outputs'
        self.path=self.folder/'project.yaml';self.path.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        warning=patch('overpressure.gui.QMessageBox.warning');self.warning=warning.start();self.addCleanup(warning.stop)
        self.window=MainWindow(self.path);self.addCleanup(self.window.close);self.window.geometry_timer.stop()
        self.window.show();self.app.processEvents()

    def wait(self,predicate):
        end=time.monotonic()+40
        while not predicate() and time.monotonic()<end:self.app.processEvents();QTest.qWait(20)
        self.assertTrue(predicate(),self.window.rock_physics.status.text())

    def test_menu_is_nonmodal_and_project_switch_waits_for_own_worker(self):
        w=self.window;w.shell.module_actions['rock_physics'].trigger()
        self.assertEqual(w.pages.currentIndex(),8)
        self.assertTrue(w.shell.workspace.module_dialogs[8].isVisible())
        self.assertFalse(w.shell.workspace.module_dialogs[8].isModal())
        with patch.object(w.rock_physics,'process',object()),patch('overpressure.gui.QMessageBox.information') as info,patch('overpressure.gui.QFileDialog.getOpenFileName') as chooser:
            w.open_project();chooser.assert_not_called();info.assert_called_once()

    def test_real_worker_registers_central_result_without_collecting_training_draft(self):
        w=self.window;panel=w.rock_physics
        self.assertTrue(any(m['id']=='utils.V' for m in panel.models))
        panel.select_model('utils.V');panel.restore_defaults()
        w.hidden.setText('unfinished training draft');w.coefficients.setText('unfinished time-depth draft')
        old_model=w.bundle_path.text();panel.start_calculation()
        self.wait(lambda:panel.process is None and len(w.shell.workspace.rock_physics_documents)==1)
        self.warning.assert_not_called();self.assertEqual(w.bundle_path.text(),old_model)
        record=next(r for r in w.shell.results if r['artifact']=='rock_physics')
        path=Path(record['path']);self.assertTrue((path/'result.json').is_file())
        view=w.shell.workspace.rock_physics_documents[str(path)].widget()
        self.assertTrue(view.result['outputs']);self.assertEqual(view.result['model_id'],'utils.V')
        sub=w.shell.workspace.rock_physics_documents[str(path)];sub.close()
        reopened_view=w.shell.workspace.open_rock_physics(path)
        self.assertIs(reopened_view,view);self.assertTrue(sub.isVisible())
        cfg=yaml.safe_load(self.path.read_text(encoding='utf-8'))
        self.assertEqual(cfg['rock_physics']['model_id'],'utils.V')
        self.assertEqual(cfg['algorithm']['hidden_sizes'],w.settings['algorithm']['hidden_sizes'])
        self.assertIn('rock_physics',[r['artifact'] for r in cfg['workspace']['results']])
        w.close();self.app.processEvents()
        new_window=MainWindow(self.path);self.addCleanup(new_window.close)
        self.assertEqual(new_window.rock_physics.current_id,'utils.V')
        restored=new_window.shell.workspace.open_rock_physics(path)
        self.assertEqual(restored.result['model_id'],'utils.V')

    def add_log_well(self):
        import pandas as pd
        import numpy as np
        from overpressure.well_library import inspect_curve_file
        path=self.folder/'logs.csv'
        pd.DataFrame(dict(Depth=np.arange(100.,106.),POR=[8,10,12,14,16,18],SW=[80]*6,
            AC=[70,72,75,78,81,84],DEN=[2.5,2.48,2.46,2.44,2.42,2.40])).to_csv(path,index=False)
        self.window.library.wells=inspect_curve_file(path,forced_well='A',definitions=[
            dict(name=c,column=c,unit=u) for c,u in [('POR','%'),('SW','%'),('AC','us/ft'),('DEN','g/cc')]])
        self.window.library.refresh()
        return self.window.rock_physics.workflow

    def test_well_workflow_preview_model_writeback_and_reopen(self):
        p=self.add_log_well();w=self.window
        self.assertEqual(p.well.currentText(),'A');self.assertEqual(p.bindings['phi'][0].currentData(),'POR')
        self.assertEqual(p.bindings['phi'][2].currentText(),'%')
        p.zmin.setText('100');p.zmax.setText('105');p.dry_model.setCurrentIndex(p.dry_model.findData('cripor'))
        w.hidden.setText('unfinished');p.preview()
        self.assertIsNotNone(w.rock_physics.process,p.message.text()+' / '+w.rock_physics.status.text())
        self.wait(lambda:w.rock_physics.process is None)
        self.assertIn('有效 6',p.qc.toPlainText(),w.rock_physics.status.text());self.assertEqual(p.preview_table.rowCount(),6)
        self.assertFalse(w.shell.workspace.rock_physics_documents)
        p.start();self.wait(lambda:w.rock_physics.process is None and bool(w.shell.workspace.rock_physics_documents))
        self.warning.assert_not_called()
        key,sub=next(iter(w.shell.workspace.rock_physics_documents.items()));view=sub.widget()
        self.assertEqual(view.result['kind'],'well_log');self.assertNotIn('vs',view.result['metrics']['all'])
        self.assertGreaterEqual(len(view.viewer.datasets),6)
        groups=[{d['name'] for d in items} for _,items in view.viewer.groups()]
        self.assertIn({'实测 Vp','模型 Vp'},groups)
        view.attach({'VP_RP':'基线VP','VS_RP':'基线VS'})
        self.assertEqual(w.library.wells[0]['curves'][-1]['name'],'基线VS')
        self.assertEqual(w.library.wells[0]['curves'][-1]['role'],'derived')
        settings=view.viewer.setting_rows();settings[0]['kind']='points';settings[0]['markersize']=7.5
        view.viewer.apply_plot_settings(settings)
        before_count=len(view.viewer.datasets);view.viewer.track_list.clearSelection()
        view.viewer.track_list.item(view.viewer.track_list.count()-1).setSelected(True);view.viewer.remove_tracks()
        view.save_plot_state()
        view.edit_scheme();self.assertEqual(w.rock_physics.workflow_tabs.currentIndex(),0)
        sub.close();self.assertIs(w.shell.workspace.open_rock_physics(key),view);self.assertTrue(sub.isVisible())
        from overpressure.well_rock_physics_view import WellRockPhysicsResultView
        reopened=WellRockPhysicsResultView(key,owner=w);self.addCleanup(reopened.close)
        self.assertEqual(reopened.viewer.setting_rows()[0]['kind'],'points')
        self.assertEqual(reopened.viewer.setting_rows()[0]['markersize'],7.5)
        self.assertEqual(len(reopened.viewer.datasets),before_count-1)
        reopened.reset_tracks();self.assertEqual(len(reopened.viewer.datasets),before_count)
        cfg=yaml.safe_load(self.path.read_text(encoding='utf-8'))
        self.assertTrue(cfg['rock_physics']['workflow']['schemes'])
        self.assertEqual(cfg['project_data']['wells'][0]['curves'][-2]['name'],'基线VP')
        # QA artifacts contain only this generated test well.
        out=ROOT/'outputs'/'well_rp_qa';out.mkdir(parents=True,exist_ok=True)
        w.shell.show_page(8);w.shell.workspace.module_dialogs[8].resize(1100,850);self.app.processEvents()
        w.shell.workspace.module_dialogs[8].grab().save(str(out/'workflow.png'))
        view.resize(1400,850);view.grab().save(str(out/'result.png'))

    def test_well_scheme_drafts_and_multiple_schemes_survive_project_reload(self):
        p=self.add_log_well();p.name.setText('方案A');p.edits['alpha_soft'].setText('unfinished')
        p.save_scheme();p.copy_scheme();self.assertEqual(len(p.saved),2)
        cfg=p.collect_state();p.name.setText('temporary');p.populate(cfg)
        self.assertEqual(p.edits['alpha_soft'].text(),'unfinished');self.assertEqual(len(p.saved),2)
        self.window.shell.persist_workspace()
        restored=MainWindow(self.path);self.addCleanup(restored.close)
        self.assertEqual(restored.rock_physics.workflow.edits['alpha_soft'].text(),'unfinished')
        self.assertEqual(len(restored.rock_physics.workflow.saved),2)

    def test_visible_workbench_auto_preview_never_registers_or_saves_artifacts(self):
        p=self.add_log_well();w=self.window;before=self.path.read_bytes()
        w.shell.show_page(8)
        self.wait(lambda:p.live.result.result is not None)
        self.assertFalse(w.shell.workspace.rock_physics_documents);self.assertFalse(w.shell.results)
        self.assertEqual(self.path.read_bytes(),before);self.assertFalse((self.folder/'outputs/rock_physics').exists())
        self.assertEqual(p.live.result.result['preview']['sampled_rows'],6)
        self.wait(lambda:p.live.result.viewer.plot_box.height()>=180)
        out=ROOT/'outputs/live_rp_qa';out.mkdir(parents=True,exist_ok=True)
        w.shell.workspace.module_dialogs[8].grab().save(str(out/'module_window.png'))
        previous=p.live.result.result;w.shell.workspace.module_dialogs[8].hide()
        self.assertIsNone(p.live.process);self.assertFalse(p.live.timer.isActive())
        w.shell.show_page(8);self.wait(lambda:p.live.result.result is not previous)
        self.assertTrue(w.can_switch_project())

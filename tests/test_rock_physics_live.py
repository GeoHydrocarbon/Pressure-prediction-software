"""Latest-input live preview, progressive UI and no implicit persistence."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QWidget
from PySide6.QtTest import QTest

from overpressure.config import Project
from overpressure.gui import configure_application
from overpressure.well_library import inspect_curve_file
from overpressure.rock_physics_panel import RockPhysicsPanel


class RockPhysicsLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);configure_application(cls.app)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        path=self.root/'logs.csv';z=np.linspace(100,200,501)
        pd.DataFrame(dict(Depth=z,POR=np.linspace(5,20,len(z)),VP=4200-z,VS=2500-z,DEN=2.5-z/10000)).to_csv(path,index=False)
        self.owner=QWidget();self.owner.project=Project(self.root/'project.yaml',{});self.owner.settings=dict(output_dir='outputs')
        self.owner.library=SimpleNamespace(wells=inspect_curve_file(path,forced_well='A',definitions=[dict(name=c,column=c,unit=u) for c,u in [('POR','%'),('VP','m/s'),('VS','m/s'),('DEN','g/cc')]]))
        self.panel=RockPhysicsPanel(self.owner);self.p=self.panel.workflow;self.live=self.p.live
        self.p.dry_model.setCurrentIndex(self.p.dry_model.findData('cripor'))
        self.addCleanup(self.owner.close);self.addCleanup(self.panel.close);self.addCleanup(self.panel.stop)
        self.command=patch('overpressure.rock_physics_live.worker_command',return_value=(sys.executable,['-u','-m','overpressure.rock_physics.worker']))
        self.worker=self.command.start();self.addCleanup(self.command.stop)

    def wait(self,predicate):
        end=time.monotonic()+25
        while not predicate() and time.monotonic()<end:self.app.processEvents();QTest.qWait(20)
        self.assertTrue(predicate(),self.live.result.status.text())

    def show(self):
        self.panel.resize(1450,900);self.panel.show();self.app.processEvents()

    def test_minimal_inputs_render_automatically_and_adjustments_update_without_files(self):
        self.show();self.wait(lambda:self.live.result.result is not None)
        result=self.live.result.result
        self.assertEqual(result['preview']['total_rows'],501);self.assertEqual(result['preview']['sampled_rows'],240)
        self.assertFalse((self.root/'outputs').exists());self.assertEqual(len(self.owner.library.wells[0]['curves']),5)
        self.assertEqual(result['parameters']['scheme']['bindings']['phi']['curve'],'POR')
        self.assertEqual(float(self.p.zmin.text()),100);self.assertEqual(float(self.p.zmax.text()),200)
        original=np.array(self.live.result.numeric['VP_RP']);self.p.edits['critical_porosity'].setText('.55')
        self.wait(lambda:self.live.result.result is not result)
        self.assertFalse(np.allclose(original,self.live.result.numeric['VP_RP']))
        self.assertFalse((self.root/'outputs').exists())
        rows=self.live.result.viewer.setting_rows();rows[0].update(kind='points',markersize=7.)
        self.live.result.viewer.apply_plot_settings(rows);self.live.result.viewer.track_list.clearSelection()
        self.live.result.viewer.track_list.item(self.live.result.viewer.track_list.count()-1).setSelected(True)
        self.live.result.viewer.remove_tracks();previous=self.live.result.result
        self.p.edits['critical_porosity'].setText('.6');self.wait(lambda:self.live.result.result is not previous)
        self.assertEqual(len(self.live.result.viewer.datasets),5)
        self.assertEqual(self.live.result.viewer.setting_rows()[0]['kind'],'points')
        self.live.result.reset_plot();self.assertEqual(len(self.live.result.viewer.datasets),6)
        self.assertTrue(self.live.sections['basic'][0].isChecked());self.assertFalse(self.live.sections['matrix'][0].isChecked())
        self.assertFalse(self.p.mapping.isRowHidden(0));self.assertTrue(self.p.mapping.isRowHidden(1))
        # In-progress calibration edits do not prevent forward preview.
        self.p.edits['cal_min'].setText('unfinished');self.assertEqual(self.p.scheme(for_preview=True)['calibration']['mode'],'none')
        previous=self.live.result.result;self.live.sections['matrix'][0].setChecked(True);self.p.add_mineral();self.app.processEvents()
        self.wait(lambda:self.live.result.result is not previous)
        out=Path(__file__).resolve().parents[1]/'outputs/live_rp_qa';out.mkdir(parents=True,exist_ok=True)
        self.live.sections['matrix'][0].setChecked(False);self.app.processEvents();QTest.qWait(100)
        self.assertGreater(self.live.result.viewer.plot_box.height(),200)
        self.panel.grab().save(str(out/'workbench.png'))

    def test_debounce_and_late_results_never_replace_newer_input(self):
        # Slow results deliberately carry only a marker; capture display calls.
        script="import sys,json,time; q=json.load(sys.stdin); time.sleep(.35); print(json.dumps({'event':'live_preview','result':{'name':q['scheme']['name']}}),flush=True)"
        shown=[]
        with patch('overpressure.rock_physics_live.worker_command',return_value=(sys.executable,['-u','-c',script])) as command,patch.object(self.live.result,'display',side_effect=lambda r:shown.append(r['name'])):
            self.show();self.p.name.setText('old');self.wait(lambda:self.live.process is not None)
            self.p.name.setText('intermediate');self.p.name.setText('newest')
            self.wait(lambda:shown==['newest']);self.assertEqual(command.call_count,2)
            self.assertNotIn('old',shown);self.assertNotIn('intermediate',shown)

    def test_incomplete_input_pause_hide_and_starting_cancel_are_safe(self):
        self.show();self.p.edits['alpha_hard'].setText('')
        self.wait(lambda:'请完善输入' in self.live.result.status.text())
        self.assertIsNone(self.live.process);self.assertIsNone(self.live.result.result)
        self.p.edits['alpha_hard'].setText('.15');self.live.enabled.setChecked(False)
        self.assertFalse(self.live.timer.isActive());self.assertIsNone(self.live.process)
        with patch('overpressure.rock_physics_live.worker_command',return_value=(sys.executable,['-u','-c','import time; time.sleep(20)'])):
            self.live.changed(force=True);self.live.timer.stop();self.live.calculate();self.assertIsNotNone(self.live.process)
            self.panel.stop();self.app.processEvents();self.assertIsNone(self.live.process)
        self.panel.hide();self.assertFalse(self.live.timer.isActive())

    def test_live_layout_preferences_survive_state_and_reload_clears_old_result(self):
        self.live.enabled.setChecked(False);self.live.sections['matrix'][0].setChecked(True)
        state=self.panel.collect_state();self.live.sections['matrix'][0].setChecked(False);self.panel.populate(state)
        self.assertFalse(self.live.enabled.isChecked());self.assertTrue(self.live.sections['matrix'][0].isChecked())
        self.assertIsNone(self.live.result.result);self.assertIsNone(self.live.process)


if __name__=='__main__':unittest.main()

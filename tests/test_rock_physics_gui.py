import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
try:
    from PySide6.QtWidgets import QApplication,QWidget
    from PySide6.QtTest import QTest,QSignalSpy
except ImportError as exc:raise unittest.SkipTest('Qt unavailable; use desktop interpreter') from exc
from overpressure.config import Project
from overpressure.gui import configure_application
from overpressure.rock_physics_panel import RockPhysicsPanel
from overpressure.rock_physics_view import RockPhysicsResultView


MODELS=[dict(id='EM.VRH',category='矿物混合',title='Voigt-Reuss-Hill',description='Input volumes, moduli in GPa.',validated_example=True,
    parameters=[dict(name='volumes',default=[.5,.5],required=True,description='体积分数'),dict(name='M',default=[37,20],required=True,unit='GPa')],example=dict(volumes=[.5,.5],M=[37,20])),
    dict(id='Fluid.test',category='流体',title='流体模型',description='Test only.',parameters=[dict(name='K',default=None,required=True,unit='GPa'),dict(name='kind',default='water',required=False)],example=dict(K=2.2,kind='water'))]


class RockPhysicsGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);configure_application(cls.app)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name).resolve()
        self.owner=QWidget();self.owner.project=Project(self.folder/'project.yaml',{});self.owner.settings=dict(output_dir='outputs');self.addCleanup(self.owner.close)
        with patch('overpressure.rock_physics_panel.available_models',return_value=deepcopy(MODELS)):
            self.panel=RockPhysicsPanel(self.owner)
        self.addCleanup(self.panel.close);self.addCleanup(self.panel.stop)

    def wait(self,predicate):
        end=time.monotonic()+15
        while not predicate() and time.monotonic()<end:self.app.processEvents();QTest.qWait(20)
        self.assertTrue(predicate(),self.panel.status.text())

    def test_switch_preserves_drafts_restore_and_project_state(self):
        p=self.panel;self.assertEqual(p.parameter_values()['M'],[37,20])
        p.parameters.item(0,1).setText('[0.2, 0.8]');p.select_model('Fluid.test');self.assertEqual(p.parameter_values()['kind'],'water')
        p.parameters.item(0,1).setText('unfinished')
        state=p.collect_state();p.select_model('EM.VRH');self.assertEqual(p.parameter_values()['volumes'],[.2,.8])
        p.populate(state);self.assertEqual(p.current_id,'Fluid.test');self.assertEqual(p.parameters.item(0,1).text(),'unfinished')
        with self.assertRaises(ValueError):p.parameter_values()
        p.restore_defaults();self.assertEqual(p.parameter_values()['K'],2.2)
        p.parameters.item(0,1).setText('');p.run_model();self.assertIsNone(p.process);self.assertIn('必填',p.status.text())

    def test_mineral_curve_field_opens_popup_on_body_click_and_preserves_missing_choice(self):
        from PySide6.QtCore import Qt
        from overpressure.well_rock_physics_panel import CurveSourceCombo
        class ClickProbe(CurveSourceCombo):
            def __init__(self):super().__init__();self.popup_calls=0
            def showPopup(self):self.popup_calls+=1
        # Keep the real Qt body-click dispatch while substituting only the OS
        # popup window, which can terminate Qt on unattended Windows runners.
        picker=ClickProbe();picker.resize(240,36);picker.set_curves(['QUAZ','LIME'],'QUAZ');picker.show();self.addCleanup(picker.close)
        self.app.processEvents();QTest.mouseClick(picker,Qt.MouseButton.LeftButton,pos=picker.rect().center());self.app.processEvents()
        self.assertEqual(picker.popup_calls,1);self.assertFalse(picker.isEditable())
        picker.setCurrentIndex(2)
        self.assertEqual(picker.currentText(),'LIME')
        picker.set_curves(['QUAZ'],picker.currentText());self.assertEqual(picker.currentText(),'LIME')
        picker.setCurrentIndex(0);self.assertEqual(picker.currentText(),'')

    def test_common_mineral_presets_and_custom_properties_roundtrip(self):
        p=self.panel.workflow;p.mineral_preset.setCurrentText('方解石');p.add_mineral()
        row=p.minerals.rowCount()-1
        self.assertEqual(p.minerals.item(row,0).text(),'方解石');self.assertEqual(float(p.minerals.item(row,5).text()),76.8)
        count=p.minerals.rowCount();p.add_mineral();self.assertEqual(p.minerals.rowCount(),count)
        p.mineral_preset.setCurrentText('自定义矿物');p.add_mineral();row=p.minerals.rowCount()-1
        self.assertEqual(p.minerals.item(row,4).text(),'')
        with self.assertRaisesRegex(ValueError,'密度'):p.scheme()
        for col,text in [(0,'自定义组分'),(4,'2.8'),(5,'50'),(6,'25')]:p.minerals.item(row,col).setText(text)
        draft=p.collect_state();p.populate(draft)
        self.assertEqual(p.scheme()['minerals'][-1]['name'],'自定义组分')
        self.assertEqual(p.scheme()['minerals'][-1]['K'],50)

    def test_worker_success_is_independent_of_training(self):
        p=self.panel;spy=QSignalSpy(p.resultReady)
        script="import json,sys; q=json.load(sys.stdin); print(json.dumps({'event':'completed','path':q['output_dir'],'result':{'status':'completed','model_id':q['model_id'],'title':'Result','parameters':q['parameters'],'outputs':[{'name':'K','values':25.,'shape':[]}],'table':{'columns':['K'],'rows':[[25.]]},'warnings':[]}}),flush=True)"
        with patch('overpressure.rock_physics_panel.worker_command',return_value=(sys.executable,['-u','-c',script])):
            p.run_model();self.assertIsNotNone(p.process);self.wait(lambda:p.process is None)
        self.assertEqual(spy.count(),1);self.assertEqual(spy.at(0)[1]['parameters']['M'],[37,20])

    def test_worker_error_returns_without_result(self):
        p=self.panel;spy=QSignalSpy(p.resultReady)
        error="import json,sys; json.load(sys.stdin); print(json.dumps({'event':'error','message':'参数超出范围'}),flush=True); sys.exit(1)"
        with patch('overpressure.rock_physics_panel.worker_command',return_value=(sys.executable,['-u','-c',error])):
            p.run_model();self.wait(lambda:p.process is None)
        self.assertIn('参数超出范围',p.status.text());self.assertEqual(spy.count(),0)

    def test_worker_cancel_during_starting_is_safe(self):
        p=self.panel;results=[];p.resultReady.connect(lambda path,result:results.append(result))
        with patch('overpressure.rock_physics_panel.worker_command',return_value=(sys.executable,['-u','-c','import time; time.sleep(20)'])):
            p.run_model();p.stop();self.app.processEvents()
        self.assertIsNone(p.process);self.assertIn('取消',p.status.text());self.assertTrue(p.run_button.isEnabled());self.assertEqual(results,[])

    def result(self,outputs,columns,rows,parameters=None):
        return dict(model_id='Test.model',title='Synthetic model',status='completed',parameters=parameters or {},outputs=outputs,table=dict(columns=columns,rows=rows),warnings=[])

    def test_scalar_view_and_data_export(self):
        result=self.result([dict(name='K',values=25.,shape=[],unit='GPa')],['K'],[[25.]])
        view=RockPhysicsResultView(self.folder,result);self.addCleanup(view.close)
        self.assertIn('25',view.figure.axes[0].texts[0].get_text());self.assertFalse(view.figure.axes[0].lines)
        csv=self.folder/'result.csv';view.export_data(csv);self.assertIn('25',csv.read_text(encoding='utf-8-sig'))
        target=self.folder/'result.json';view.export_data(target);self.assertEqual(json.loads(target.read_text(encoding='utf-8'))['model_id'],'Test.model')
        restored=RockPhysicsResultView(self.folder);self.addCleanup(restored.close);self.assertEqual(restored.table.item(0,0).text(),'25')

    def test_vector_axes_styles_and_figure_export(self):
        result=self.result([dict(name='K',values=[20.,18.,15.],shape=[3],unit='GPa')],['input.phi','K'],[[0.,20.],[.1,18.],[.2,15.]],dict(phi=[0.,.1,.2]))
        view=RockPhysicsResultView(self.folder,result);self.addCleanup(view.close)
        self.assertEqual(view.x.currentData(),'input.phi');self.assertEqual(view.y.currentData(),'K')
        view.color.setText('#c14436');view.width.setValue(3.5);view.redraw();line=view.figure.axes[0].lines[0]
        self.assertEqual(line.get_color(),'#c14436');self.assertEqual(line.get_linewidth(),3.5)
        view.mode.setCurrentIndex(1);self.assertEqual(len(view.figure.axes[0].collections),1)
        view.style.dpi=72;target=self.folder/'model.png';view.export_plot(target);self.assertGreater(target.stat().st_size,1000)

    def test_matrix_view_and_bad_color_does_not_destroy_plot(self):
        values=[[1.,2.],[3.,4.]];result=self.result([dict(name='C',values=values,shape=[2,2],unit='GPa')],['C[0]','C[1]'],values)
        view=RockPhysicsResultView(self.folder,result);self.addCleanup(view.close)
        self.assertEqual(view.mode.currentData(),'matrix');self.assertEqual(len(view.figure.axes[0].images),1)
        self.assertEqual(view.figure.axes[-1].get_ylabel(),'GPa')
        old=view.figure;view.color.setText('invalid value');view.redraw();self.assertIs(view.figure,old);self.assertIn('颜色无效',view.status.text())

    def test_index_column_is_not_selected_as_physical_y_output(self):
        result=self.result([dict(name='K_dry',values=[30.,25.,20.],shape=[3],unit='GPa'),dict(name='porosity',values=[0.,.1,.2],shape=[3],unit='fraction')],
            ['index','K_dry [GPa]','porosity [fraction]'],[[0,30.,0.],[1,25.,.1],[2,20.,.2]])
        view=RockPhysicsResultView(self.folder,result);self.addCleanup(view.close)
        self.assertEqual(view.y.currentData(),'K_dry [GPa]');self.assertIsNone(view.x.currentData())
        self.assertEqual(list(view.figure.axes[0].lines[0].get_ydata()),[30.,25.,20.])

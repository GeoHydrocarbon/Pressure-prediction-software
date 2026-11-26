import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    from PySide6.QtWidgets import QApplication,QDialog,QDialogButtonBox
except ImportError as exc:raise unittest.SkipTest('Qt unavailable; use desktop interpreter') from exc
import yaml
from overpressure.gui import MainWindow,configure_application
from overpressure.project_catalog import AssetDialog
from overpressure.well_spatial import read_asset

ROOT=Path(__file__).resolve().parents[1]


class SpatialImportGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);configure_application(cls.app)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name).resolve()
        cfg=yaml.safe_load((ROOT/'project.yaml').read_text(encoding='utf-8'))
        cfg['algorithm']['name']='mlp';cfg['workspace']={};cfg['traditional_prediction']={}
        cfg['project_data']=dict(wells=[],well_assets=[],horizons=[],pressure_result='')
        cfg['data'].update(training_selection='well_library',development_wells=[],validation_wells=[],blind_wells=[],well_files=[],well_imports=[])
        cfg['seismic'].update(density_file='',impedance_file='')
        project=self.folder/'project.yaml';project.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        self.window=MainWindow(project);self.addCleanup(self.window.close);self.window.geometry_timer.stop()

    def test_preview_row_selection_header_mapping_and_edit_roundtrip(self):
        path=self.folder/'heads.txt';path.write_text('说明\n井名\tX\tY\n单位\n001\t100\t200\n002\t300\t400\n',encoding='gb18030')
        before=path.read_bytes();dialog=AssetDialog(self.window,path,'coordinates');self.addCleanup(dialog.close)
        self.assertEqual(dialog.source_preview.item(1,1).text(),'井名\tX\tY')
        self.assertFalse(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled())
        dialog.header_row.setValue(2);dialog.data_row.setValue(4)
        self.assertEqual(dialog.table_preview.horizontalHeaderItem(0).text(),'井名')
        self.assertEqual(dialog.table_preview.verticalHeaderItem(0).text(),'4')
        self.assertEqual(dialog.columns['well'].currentText(),'井名')
        dialog.accept();self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        config=dialog.config();self.assertEqual(config['table_import']['header_row'],2)
        reopened=AssetDialog(self.window,path,'coordinates',initial=config);self.addCleanup(reopened.close)
        self.assertEqual(reopened.config(),config);self.assertEqual(path.read_bytes(),before)

    def test_survey_mode_validation_and_explicit_initial_segment(self):
        path=self.folder/'survey.txt';path.write_text('MD AZI INC\n100 90 30\n200 90 30\n',encoding='utf-8')
        dialog=AssetDialog(self.window,path,'trajectory',forced_well='A');self.addCleanup(dialog.close)
        dialog.trajectory_mode.setCurrentIndex(1)
        self.assertTrue(dialog.columns['x'].isHidden());self.assertFalse(dialog.columns['azimuth'].isHidden())
        self.assertEqual(dialog.columns['well'].currentText(),'')
        dialog.accept();self.assertNotEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertIn('首个测点',dialog.status.text())
        dialog.initial_segment.setCurrentIndex(1);dialog.accept();self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        config=dialog.config();self.assertTrue(config['relative']);self.assertNotIn('x',config['columns'])
        _,frame=read_asset(config);self.assertEqual(frame.well.tolist(),['A','A']);self.assertAlmostEqual(frame.x.iloc[-1],100.)
        reopened=AssetDialog(self.window,path,'trajectory',forced_well='A',initial=config);self.addCleanup(reopened.close)
        self.assertEqual(reopened.config(),config)

    def test_actual_import_registers_names_options_and_project_persistence(self):
        path=self.folder/'heads.txt';path.write_text('说明\nwell X Y\n单位\nA 100 200\nB 300 400\n',encoding='utf-8')
        def configure(dialog):
            dialog.header_row.setValue(2);dialog.data_row.setValue(4);dialog.accept();return dialog.result()
        with patch('overpressure.well_library_ui.QFileDialog.getOpenFileName',return_value=(str(path),'')),patch.object(AssetDialog,'exec',new=configure):
            self.window.library.import_spatial('coordinates')
        self.assertEqual({w['name'] for w in self.window.library.wells},{'A','B'})
        self.assertEqual(self.window.catalog.assets[0]['wells'],['A','B'])
        self.window.shell.persist_workspace()
        cfg=yaml.safe_load(self.window.project.path.read_text(encoding='utf-8'))
        saved=cfg['project_data']['well_assets'][0];self.assertEqual(saved['table_import']['data_row'],4)
        self.assertEqual(read_asset(saved)[1].source_row.tolist(),[4,5])
        def cancel(dialog):return QDialog.DialogCode.Rejected
        with patch('overpressure.well_library_ui.QFileDialog.getOpenFileName',return_value=(str(path),'')),patch.object(AssetDialog,'exec',new=cancel):
            self.window.library.import_spatial('coordinates')
        self.assertEqual(len(self.window.catalog.assets),1)

"""Require per-file settings before import; validate and persist that exact mapping."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
try:
    from PySide6.QtWidgets import QApplication,QDialog
    from PySide6.QtTest import QTest
except ImportError as exc:raise unittest.SkipTest('Qt unavailable; run in desktop interpreter') from exc
import yaml
from overpressure.gui import MainWindow,configure_application
from overpressure.seismic_import import SeismicImportDialog,STANDARD,LEGACY
from test_seismic_import import legacy_segy

ROOT=Path(__file__).resolve().parents[1]


class SeismicImportGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);configure_application(cls.app)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name).resolve();self.path=self.folder/'legacy.sgy';legacy_segy(self.path)
        cfg=yaml.safe_load((ROOT/'project.yaml').read_text(encoding='utf-8'))
        cfg['workspace']={};cfg['traditional_prediction']={}
        cfg['project_data']=dict(wells=[],well_assets=[],horizons=[],pressure_result='')
        cfg['data'].update(training_selection='well_library',development_wells=[],validation_wells=[],blind_wells=[])
        cfg['seismic'].update(density_file='',impedance_file='',header_fields=STANDARD.copy())
        project=self.folder/'project.yaml';project.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        self.window=MainWindow(project);self.window.geometry_timer.stop();self.addCleanup(self.window.close)

    def wait(self,predicate):
        deadline=time.monotonic()+25
        while not predicate() and time.monotonic()<deadline:self.app.processEvents();QTest.qWait(20)
        self.assertTrue(predicate())

    def test_precheck_blocks_wrong_headers_then_accepts_legacy_and_invalidates_changes(self):
        dialog=SeismicImportDialog(self.window,self.path,dict(headers=STANDARD));dialog.show();self.addCleanup(dialog.reject)
        self.wait(lambda:dialog.process is None and '未通过' in dialog.summary.toPlainText())
        self.assertFalse(dialog.ok_button.isEnabled());self.assertIn('0 ～ 0',dialog.summary.toPlainText())
        dialog.presets.setCurrentIndex(2);dialog.preview_button.click()
        self.wait(lambda:dialog.process is None and dialog.checked is not None)
        self.assertTrue(dialog.ok_button.isEnabled());self.assertEqual(dialog.values()['headers'],LEGACY)
        dialog.headers['inline'].setValue(189);self.assertFalse(dialog.ok_button.isEnabled())
        dialog.accept();self.assertNotEqual(dialog.result(),QDialog.DialogCode.Accepted)

    def test_import_opens_settings_before_registration_and_preserves_per_file_mapping(self):
        shell=self.window.shell;before=self.path.read_bytes()
        def configure(dialog):
            self.assertEqual(shell.extra_volumes,[])
            dialog.presets.setCurrentIndex(2);dialog.show()
            self.wait(lambda:dialog.process is None and dialog.checked is not None)
            dialog.accept();return dialog.result()
        self.window.hidden.setText('unfinished unrelated training edit')
        self.window.den_path.setText(str(self.path));self.window.geometry_timer.stop()
        with patch('overpressure.workstation.QFileDialog.getOpenFileNames',return_value=([str(self.path)],'')),patch.object(SeismicImportDialog,'exec',new=configure):
            shell.import_seismic()
        self.assertEqual(len(shell.extra_volumes),1);self.assertEqual(shell.extra_volumes[0]['headers'],LEGACY)
        self.assertEqual(shell.map_sources()[0]['headers'],LEGACY)
        self.assertEqual({k:v.value() for k,v in self.window.header_edits.items()},STANDARD)
        self.wait(lambda:shell.workspace.map.metadata is not None)
        self.assertEqual(shell.workspace.map.metadata['inline_range'],[10,11])
        saved=yaml.safe_load(self.window.project.path.read_text(encoding='utf-8'))
        self.assertEqual(saved['workspace']['seismic_volumes'][0]['headers'],LEGACY)
        self.assertEqual(self.path.read_bytes(),before)

    def test_cancel_does_not_register_or_start_a_delayed_precheck(self):
        def cancel(dialog):dialog.reject();return dialog.result()
        with patch('overpressure.workstation.QFileDialog.getOpenFileNames',return_value=([str(self.path)],'')),patch.object(SeismicImportDialog,'exec',new=cancel):
            self.window.shell.import_seismic()
        self.app.processEvents()
        self.assertEqual(self.window.shell.extra_volumes,[])
        for dialog in self.window.findChildren(SeismicImportDialog):self.assertIsNone(dialog.process)


if __name__=='__main__':unittest.main()

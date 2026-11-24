"""Map-centred workflow, live worker and project isolation regression checks."""
from copy import deepcopy
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PySide6.QtWidgets import QApplication, QMdiArea
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
except ImportError as exc:
    raise unittest.SkipTest('Qt unavailable; run GUI checks in the desktop interpreter') from exc
import numpy as np
import yaml
from matplotlib.backend_bases import MouseEvent

from overpressure.gui import MainWindow, configure_application

ROOT = Path(__file__).resolve().parents[1]


def synthetic_segy(path):
    binary = bytearray(400)
    for offset, value in ((16, 2000), (20, 31), (24, 5), (54, 1)):
        struct.pack_into('>h', binary, offset, value)
    with Path(path).open('wb') as f:
        f.write(b' ' * 3200)
        f.write(binary)
        for il in range(10, 14):
            for xl in range(20, 25):
                if (il, xl) == (11, 21):
                    continue
                header = bytearray(240)
                for offset, value in ((188, il), (192, xl), (180, 300000 + xl * 25), (184, 4000000 + il * 25)):
                    struct.pack_into('>i', header, offset, value)
                for offset, value in ((70, 1), (88, 1), (108, 1000), (114, 31), (116, 2000)):
                    struct.pack_into('>h', header, offset, value)
                f.write(header)
                f.write((np.sin(np.arange(31) / 4) * 20 + il + xl).astype('>f4').tobytes())


class MapWorkspaceGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_application(cls.app)

    def setUp(self):
        warnings = patch('overpressure.gui.QMessageBox.warning')
        self.warning = warnings.start()
        self.addCleanup(warnings.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cfg = yaml.safe_load((ROOT / 'project.yaml').read_text(encoding='utf-8'))
        self.cfg['data'].update(training_selection='well_library', development_wells=[], validation_wells=[], blind_wells=[],
                                well_files=[], well_imports=[], features=['Depth', 'AC', 'DEN', 'Ip'],
                                feature_units={}, target_curve='Pp', split_method='random_points')
        self.cfg['physics']['enabled'] = False
        self.cfg['seismic'].update(density_file='', impedance_file='')
        self.cfg['seismic']['header_fields'].update(inline=189, crossline=193, x=181, y=185, coordinate_scalar=71, coordinate_units=89)
        self.cfg['project_data'] = dict(wells=[], well_assets=[], horizons=[], pressure_result='')
        self.cfg['workspace'] = {}
        self.project_path = self.folder / 'project.yaml'
        self.write_project(self.project_path)
        self.window = MainWindow(self.project_path)
        self.addCleanup(self.window.close)
        self.window.show()
        self.app.processEvents()

    def write_project(self, path, cfg=None):
        path.write_text(yaml.safe_dump(cfg or self.cfg, allow_unicode=True), encoding='utf-8')

    def wait_until(self, predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(20)
        self.assertTrue(predicate(), self.window.shell.workspace.map.status.text())

    def test_modules_are_nonmodal_windows_and_central_workspace_stays_map(self):
        views = self.window.shell.workspace
        self.assertIsInstance(self.window.centralWidget(), QMdiArea)
        self.assertTrue(views.map_window.isVisible())
        self.assertFalse(any(sub.isVisible() for sub in views.documents.values()))
        for index in (1, 3, 5):
            self.window.shell.show_page(index)
            self.assertTrue(views.module_dialogs[index].isVisible())
            self.assertFalse(views.module_dialogs[index].isModal())
            self.assertTrue(views.map_window.isVisible())
            button={1:self.window.training_button,3:self.window.well_prediction.start_button,5:self.window.seismic_full_button}[index]
            self.assertTrue(button.isVisible())
            views.module_dialogs[index].close()
            self.assertIsNone(self.window.process)

    def test_closed_plot_documents_restore_their_contents(self):
        views=self.window.shell.workspace
        for index in (0,4,6):
            self.window.shell.show_page(index)
            views.tile()
            sub=views.documents[index]
            sub.close()
            self.app.processEvents()
            self.assertFalse(sub.isVisible())
            self.window.shell.show_page(index)
            self.app.processEvents()
            self.assertTrue(sub.isVisible())
            self.assertTrue(sub.widget().isVisible(), f'Page {index} contents remained closed')
        views.map_window.close()
        views.show_map()
        self.assertTrue(views.map.isVisible())

    def test_model_and_result_associations_are_saved_and_restored(self):
        self.window.bundle_path.setText('saved-model')
        self.window.seismic_bundle.setText('saved-model')
        self.window.refit_cv.setText('saved-cv')
        self.window.shell.register_result(dict(path='saved-result', artifact='predict-wells'))
        self.assertTrue(self.window.save_settings())
        second = MainWindow(self.project_path)
        self.addCleanup(second.close)
        self.assertEqual(second.bundle_path.text(), 'saved-model')
        self.assertEqual(second.seismic_bundle.text(), 'saved-model')
        self.assertEqual(second.refit_cv.text(), 'saved-cv')
        self.assertEqual(second.shell.results[0]['path'], 'saved-result')

    def test_switching_project_clears_models_and_previous_views(self):
        self.window.bundle_path.setText('old-model')
        self.window.seismic_bundle.setText('old-model')
        self.window.refit_cv.setText('old-cv')
        self.window.slices.result_dir.setText('old-volume')
        self.window.charts.directory.setText('old-charts')
        self.window.shell.workspace.map.metadata={'label':'old map'}
        self.window.shell.workspace.map.geometry=dict(inline=np.array([1]),crossline=np.array([1]),x=np.array([]),y=np.array([]))
        other = self.folder / 'other.yaml'
        self.write_project(other)
        with patch('overpressure.gui.QFileDialog.getOpenFileName', return_value=(str(other), '')):
            self.window.open_project()
        for widget in (self.window.bundle_path, self.window.seismic_bundle, self.window.refit_cv,
                       self.window.slices.result_dir, self.window.charts.directory):
            self.assertEqual(widget.text(), '')
        self.assertTrue(self.window.shell.workspace.map_window.isVisible())
        self.assertIsNone(self.window.shell.workspace.map.metadata)
        self.assertEqual(self.window.shell.workspace.map.geometry,{})

    def test_prediction_configuration_is_independent_of_training_draft(self):
        self.window.hidden.setText('invalid unfinished edit')
        for i in range(self.window.library.feature_list.count()):
            self.window.library.feature_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        cfg = self.window.collect_settings('prediction')
        self.assertEqual(cfg['algorithm'], self.cfg['algorithm'])
        self.assertEqual(cfg['data']['features'], self.cfg['data']['features'])
        with self.assertRaises(ValueError):
            self.window.collect_settings()

    def test_actual_worker_map_arbitrary_section_drag_and_time_slice(self):
        path = self.folder / 'synthetic.sgy'
        synthetic_segy(path)
        shell = self.window.shell
        shell.extra_volumes.append(dict(path=str(path), label='Synthetic seismic', unit='Amplitude'))
        shell.refresh_tree()
        panel = shell.workspace.map
        self.wait_until(lambda: panel.metadata is not None)
        self.assertTrue(panel.metadata['xy_available'])
        self.assertEqual(panel.metadata['traces'], 19)
        panel.coordinates.setCurrentIndex(0)
        panel.tool.setCurrentIndex(panel.tool.findData('arbitrary'))
        self.app.processEvents()
        panel.canvas.draw()
        def event(name, point, button=1):
            x, y = panel.ax.transData.transform(point)
            e = MouseEvent(name, panel.canvas, x, y, button=button)
            panel.canvas.callbacks.process(name, e)
        event('button_press_event', [20, 10])
        event('motion_notify_event', [24, 13])
        event('button_release_event', [24, 13])
        self.wait_until(lambda: len(shell.workspace.sections) == 1)
        ident = panel.active_line
        sub = shell.workspace.sections[ident]
        self.assertTrue(np.isfinite(sub.widget().section.values).any())
        self.assertEqual(sub.widget().meta['unit'], 'Amplitude')
        sub.close()
        self.app.processEvents()
        self.assertFalse(sub.isVisible())
        shell.workspace.section_ready(ident,sub.widget().section,sub.widget().meta)
        self.assertTrue(sub.widget().isVisible())
        panel.tool.setCurrentIndex(0)
        self.app.processEvents()
        panel.canvas.draw()
        event('button_press_event', [24, 13])
        event('motion_notify_event', [24, 12])
        event('button_release_event', [24, 12])
        self.wait_until(lambda: abs(sub.widget().meta['vertices'][-1][1] - 12) < .01)
        panel.canvas.draw()
        event('button_press_event', [22,11])
        event('motion_notify_event', [22,10.5])
        event('button_release_event', [22,10.5])
        self.wait_until(lambda:abs(sub.widget().meta['vertices'][-1][1]-11.5)<.01)
        self.assertAlmostEqual(sub.widget().meta['vertices'][0][1],9.5,places=2)
        panel.attribute.setChecked(True)
        self.wait_until(lambda: panel.map_values is not None)
        self.assertEqual(len(panel.map_values), 19)
        self.assertTrue(self.window.save_settings())
        saved = yaml.safe_load(self.project_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['workspace']['map']['source'], str(path))
        self.assertEqual(len(saved['workspace']['map']['sections']), 1)
        self.assertEqual(saved['workspace']['seismic_volumes'][0]['path'], str(path))
        second=MainWindow(self.project_path)
        self.addCleanup(second.close)
        self.wait_until(lambda:len(second.shell.workspace.sections)==1)
        restored=next(iter(second.shell.workspace.sections.values())).widget()
        self.assertEqual(restored.meta['source'],str(path))
        self.assertAlmostEqual(restored.meta['vertices'][-1][1],11.5,places=2)
        self.assertIn('view_id',second.settings['workspace']['map']['sections'][0])

    def test_new_project_and_save_as_keep_data_references_correct(self):
        self.window.bundle_path.setText('models/current')
        self.window.shell.extra_volumes.append(dict(path='input.sgy',label='Input',unit='Amplitude'))
        destination=self.folder/'nested';destination.mkdir()
        copied=destination/'copied.yaml'
        with patch('overpressure.gui.QFileDialog.getSaveFileName',return_value=(str(copied),'')):
            self.window.save_project_as()
        self.assertEqual(self.window.project.path,copied)
        saved=yaml.safe_load(copied.read_text(encoding='utf-8'))
        self.assertEqual(saved['workspace']['seismic_volumes'][0]['path'],str(self.folder/'input.sgy'))
        self.assertEqual(saved['workspace']['well_model'],str(self.folder/'models'/'current'))
        fresh=destination/'fresh.yaml'
        with patch('overpressure.gui.QFileDialog.getSaveFileName',return_value=(str(fresh),'')):
            self.window.new_project()
        self.assertEqual(self.window.project.path,fresh)
        self.assertEqual(self.window.shell.extra_volumes,[])
        self.assertEqual(self.window.bundle_path.text(),'')
        self.assertEqual(self.window.library.wells,[])


if __name__ == '__main__':
    unittest.main()

from copy import deepcopy
import codecs
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication,QToolBar
except ImportError as exc:
    raise unittest.SkipTest("该环境的 Qt 运行库不可用；GUI 测试请在 Anaconda base 环境运行") from exc
import yaml
import pandas as pd
import numpy as np

from overpressure.gui import MainWindow
from overpressure.charts import FigureStyle


ROOT = Path(__file__).resolve().parents[1]


class GuiContractTests(unittest.TestCase):
    def test_business_module_entries_open_dialogs_without_duplicate_toolbar_actions(self):
        shell=self.window.shell
        self.assertEqual(list(shell.module_actions),['training','traditional','wells','seismic','rock_physics'])
        self.assertEqual(self.window.pages.tabText(5),'三维地震预测')
        self.assertEqual(list(shell.training_actions),['mlp','rf','xgboost'])
        for name,page in [('traditional',7),('wells',3),('seismic',5),('rock_physics',8)]:
            action=shell.module_actions[name]
            self.assertIsNone(action.menu())
            action.trigger()
            self.assertEqual(self.window.pages.currentIndex(),page)
            self.assertTrue(shell.workspace.module_dialogs[page].isVisible())
            self.assertIsNone(self.window.process)
            shell.workspace.module_dialogs[page].hide()
        for name in ('rf','xgboost'):
            index=self.window.algorithm.findData(name)
            self.assertTrue(self.window.algorithm.model().item(index).isEnabled())
            shell.training_actions[name].trigger()
            self.assertEqual(self.window.collect_settings()['algorithm']['name'],name)
            self.assertTrue(shell.workspace.module_dialogs[1].isVisible())
        shell.training_actions['mlp'].trigger()
        self.assertEqual(self.window.collect_settings()['algorithm']['name'],'mlp')
        toolbar=self.window.findChild(QToolBar,'mainTools')
        self.assertFalse({'模型训练','井预测','地震预测'} & {action.text() for action in toolbar.actions()})

    def test_sparse_pressure_column_is_available_even_when_preview_has_no_values(self):
        from overpressure.well_library_ui import CurveImportDialog
        path=Path(self.tmp.name)/'sparse.csv'
        pd.DataFrame(dict(Depth=np.arange(20)+100.,Pp=[np.nan]*19+[20.])).to_csv(path,index=False)
        dialog=CurveImportDialog(self.window,[str(path)],measured=True)
        self.addCleanup(dialog.close)
        definitions=dialog.definitions();self.assertEqual(len(definitions),1)
        self.assertEqual(definitions[0]['column'],'Pp');self.assertEqual(definitions[0]['role'],'measurement')

    def test_well_library_selection_is_the_training_source(self):
        for name in ('well_path','development','blind','validation'):self.assertFalse(hasattr(self.window,name))
        from overpressure.well_library import inspect_curve_file
        for name in ('N876','N18'):
            path=Path(self.tmp.name)/(name+'.csv');pd.DataFrame(dict(Depth=[100,101],AC=[90,91],DEN=[2.4,2.4],Ip=[8000,8001],Pp=[20,21])).to_csv(path,index=False)
            self.window.library.receive(inspect_curve_file(path))
        cfg=self.window.collect_settings()
        self.assertEqual(set(cfg['data']['development_wells']),{'N876','N18'})
        self.assertEqual(cfg['data']['blind_wells'],[]);self.assertNotIn('well_file',cfg['data'])
        self.assertEqual(cfg['data']['training_selection'],'well_library')
        self.assertEqual(len(cfg['project_data']['wells']),2)
        self.assertEqual(self.window.pages.tabText(0),'井曲线')
        self.assertFalse(hasattr(self.window.well_prediction,'well_file'));self.assertFalse(hasattr(self.window.well_prediction,'measurement_file'))
        self.assertTrue(hasattr(self.window.well_prediction,'log_files'));self.assertTrue(hasattr(self.window.well_prediction,'point_files'))

    def test_custom_training_percentage_is_editable_saved_and_not_used_by_leave_one(self):
        self.assertEqual(self.window.train_percent.value(),80);self.assertFalse(self.window.train_percent.isReadOnly())
        self.window.train_percent.setValue(65)
        cfg=self.window.collect_settings();self.assertEqual(cfg['data']['train_ratio'],.65)
        self.assertIn('65% / 35%',self.window.training_button.text())
        self.window.settings=cfg;self.window.populate();self.assertEqual(self.window.train_percent.value(),65)
        self.window.split_method.setCurrentIndex(1);self.assertFalse(self.window.train_percent.isEnabled())
        self.window.split_method.setCurrentIndex(0);self.assertTrue(self.window.train_percent.isEnabled())

    def test_two_training_modes_dispatch_and_do_not_raise_qt_slot_errors(self):
        with patch('sys.excepthook') as errors,patch.object(self.window,'start_task') as task:
            self.window.split_method.setCurrentIndex(1);self.window.start_training_mode()
            self.assertEqual(task.call_args.args[0],'cross-validate')
            self.assertEqual(self.window.collect_settings()['data']['split_method'],'leave_one_well_out')
            self.window.split_method.setCurrentIndex(0);self.window.start_training_mode(True)
            self.assertEqual(task.call_args.args,('train',['--smoke']));errors.assert_not_called()

    def test_time_depth_uses_automatic_input_times_without_manual_range_controls(self):
        self.assertFalse(hasattr(self.window,'td_range'));self.assertFalse(hasattr(self.window,'td_extrapolation'))
        cfg=self.window.collect_settings()
        self.assertNotIn('valid_time_range',cfg['time_depth']);self.assertNotIn('extrapolation',cfg['time_depth'])

    def test_input_geometry_extents_show_on_both_pages_and_clear_after_path_change(self):
        self.window.geometry_timer.stop()
        self.window.display_seismic_geometry(dict(files=dict(density=dict(inline=[2871,4037],crossline=[6050,6094],traces=51582)),geometry_matches=False))
        label=self.window.prediction_geometry_label.text()
        self.assertIn('2871～4037',label);self.assertIn('6050～6094',label);self.assertIn('不一致',label)
        self.assertEqual(label,self.window.data_geometry_label.text())
        self.window.den_path.setText('changed.sgy');self.assertNotIn('2871～4037',self.window.prediction_geometry_label.text())
        self.window.geometry_timer.stop();self.window.read_seismic_geometry()
        self.assertIn('尚未找到',self.window.prediction_geometry_label.text())

    def test_custom_seismic_range_is_saved_and_passed_as_line_numbers(self):
        values=dict(inline_start=2871,inline_end=2900,crossline_start=6050,crossline_end=6060)
        for key,value in values.items():self.window.prediction_range_edits[key].setValue(value)
        self.assertEqual(self.window.collect_settings()['seismic']['prediction_range'],values)
        with patch.object(self.window,'start_task') as start:
            self.window.predict_custom_range()
            command,args=start.call_args.args; self.assertEqual(command,'predict-seismic')
            self.assertEqual(args[args.index('--inline-start')+1],'2871')
            self.assertEqual(args[args.index('--crossline-end')+1],'6060')
            self.assertNotIn('--trace-count',args); self.assertNotIn('--full',args)
        self.window.prediction_range_edits['inline_end'].setValue(2800)
        with patch.object(self.window,'start_task') as start, patch('overpressure.gui.QMessageBox.warning') as warning:
            self.window.predict_custom_range(); start.assert_not_called(); warning.assert_called_once()

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        cfg['algorithm']['name']='mlp';cfg['workspace']={}
        cfg['seismic'].update(density_file='',impedance_file='')
        cfg.setdefault('project_data',{}).update(well_assets=[],horizons=[],pressure_result='')
        cfg.setdefault('project_data',{})['wells']=[]
        cfg['data'].update(training_selection='well_library',development_wells=[],validation_wells=[],blind_wells=[],features=['Depth','AC','DEN','Ip'],feature_units={},target_curve='Pp',well_files=[],well_imports=[],split_method='random_points',train_ratio=.8)
        path = Path(self.tmp.name) / "project.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        self.window = MainWindow(path)
        self.addCleanup(self.window.close)

    def test_editing_physics_changes_saved_configuration(self):
        self.window.physical_enabled.setChecked(True)
        self.window.physics_edits["weight"].setValue(.25)
        self.window.physics_edits["nac_a"].setValue(-.0004)
        cfg = self.window.collect_settings()
        self.assertTrue(cfg["physics"]["enabled"])
        self.assertEqual(cfg["physics"]["weight"], .25)
        self.assertEqual(cfg["physics"]["nac_a"], -.0004)
        self.assertEqual(cfg["data"]["blind_wells"], [])
        self.assertFalse(hasattr(self.window,'blind'))

    def test_project_data_tabs_and_spatial_assets_do_not_change_training_input(self):
        self.assertEqual([self.window.data_tabs.tabText(i) for i in range(3)],['井数据','地震数据','层位数据'])
        before=self.window.collect_settings()['data']
        asset=dict(kind='coordinates',path='heads.csv',columns=dict(well='well',x='X',y='Y'))
        self.window.catalog.assets=[asset]; self.window.pressure_result.setText('outputs/pressure')
        cfg=self.window.collect_settings()
        self.assertEqual(before,cfg['data']); self.assertEqual(cfg['project_data']['well_assets'],[asset])
        self.window.settings=deepcopy(cfg); self.window.populate()
        self.assertEqual(self.window.catalog.assets,[asset]); self.assertEqual(self.window.pressure_result.text(),'outputs/pressure')

    def test_seismic_header_controls_are_hidden_until_advanced_is_enabled(self):
        self.window.show(); self.app.processEvents()
        self.window.data_tabs.setCurrentIndex(1); self.app.processEvents()
        self.assertFalse(self.window.header_edits['coordinate_scalar'].isVisible())

    def test_default_random_workflow_and_advanced_tools(self):
        self.assertEqual(self.window.split_method.currentIndex(), 0)
        self.assertEqual(self.window.train_percent.value(), 80)
        self.assertFalse(hasattr(self.window,'validation'))
        self.assertTrue(self.window.advanced_training.isHidden())
        self.window.advanced_toggle.setChecked(True)
        self.assertFalse(self.window.advanced_training.isHidden())
        self.window.seed.setText("123")
        cfg = self.window.collect_settings()
        self.assertEqual(cfg["data"]["split_method"], "random_points")
        self.assertEqual(cfg["data"]["train_ratio"], .8)
        self.assertEqual(cfg["training"]["seed"], 123)

    def test_depth_metadata_features_are_removed_not_hidden(self):
        for name in ("depth_type", "depth_datum", "depth_confirmed", "td_depth_type", "td_datum", "td_source", "td_confirmed"):
            self.assertFalse(hasattr(self.window, name), name)
        for name in ("depth_type", "datum", "depth_confirmed", "points_confirmed", "model_info"):
            self.assertFalse(hasattr(self.window.well_prediction, name), name)
        cfg = self.window.collect_settings()
        self.assertNotIn("depth_reference", cfg["data"])
        for name in ("depth_type", "datum", "source", "confirmed"):
            self.assertNotIn(name, cfg["time_depth"])
        self.assertNotIn("depth_reference", cfg["well_prediction"])

    def test_prediction_inputs_do_not_replace_training_inputs_or_require_example_wells(self):
        original = self.window.collect_settings()['data']
        panel = self.window.well_prediction
        panel.log_files.set_files(['new_wells.csv']);panel.point_files.set_files(['few_points.csv'])
        settings = self.window.collect_settings()
        self.assertEqual(settings['data'],original)
        self.assertEqual(settings["data"]["blind_wells"], [])
        self.assertTrue(settings['well_prediction']['well_files'][0].endswith('new_wells.csv'))
        self.assertTrue(settings['well_prediction']['measurement_files'][0].endswith('few_points.csv'))
        self.assertNotIn('well_file',settings['well_prediction']);self.assertNotIn('measurement_file',settings['well_prediction'])
        self.assertNotIn("depth_reference", settings["well_prediction"])

    def test_sparse_pressure_overlay_metrics_and_match_table(self):
        folder = Path(self.tmp.name) / "sparse"; folder.mkdir()
        pd.DataFrame({"well": ["NEW"]*3, "Depth": [100, 110, 120], "Pp_pred": [10, 20, 30]}).to_csv(folder / "predictions.csv", index=False)
        pd.DataFrame({"well": ["NEW"]*2, "source_row": [2, 3], "Depth": [105, 125], "Pp": [15, 32], "Pp_pred": [15, np.nan], "supervised_valid": [True, False],
                      "match_status": ["matched_interpolated", "outside_range"], "error_MPa": [0, np.nan], "match_reason": ["插值", "超范围"]}).to_csv(folder / "reference_points.csv", index=False)
        panel = self.window.charts; panel.load_directory(folder)
        self.assertEqual(len(panel.canvas.figure.axes[0].lines), 1)
        self.assertEqual(len(panel.canvas.figure.axes[0].collections), 2)
        self.assertEqual(panel.current_metrics().iloc[0]["count"], 1)
        panel.mode.setCurrentIndex(3)
        self.assertEqual(sum(len(item.get_offsets()) for item in panel.canvas.figure.axes[0].collections), 1)
        panel.mode.setCurrentIndex(5); self.assertEqual(panel.table.rowCount(), 2)
        self.assertEqual(panel.table_export.text(), "导出匹配明细")

    def test_roundtrip_and_available_algorithms(self):
        before = self.window.collect_settings()
        after = self.window.collect_settings()
        for key in ("data", "algorithm", "physics", "training"):
            self.assertEqual(after[key], before[key])
        self.assertTrue(self.window.algorithm.model().item(1).isEnabled())
        self.assertTrue(self.window.algorithm.model().item(2).isEnabled())

    def test_algorithm_parameters_switch_save_and_preserve_mlp_physics(self):
        w=self.window;w.physical_enabled.setChecked(True);w.lr.setValue(.002)
        w.shell.open_training('rf')
        self.assertTrue(w.hidden.isHidden());self.assertTrue(w.training_physics_card.isHidden())
        self.assertFalse(w.tree_edits['rf']['n_estimators'].isHidden())
        w.tree_edits['rf']['n_estimators'].setValue(123)
        w.hidden.setText('unfinished MLP edit')
        cfg=w.collect_settings('training');self.assertFalse(cfg['physics']['enabled'])
        self.assertEqual(cfg['algorithm']['rf']['n_estimators'],123)
        w.settings=cfg;w.populate();w.shell.open_training('xgboost')
        w.tree_edits['xgboost']['learning_rate'].setValue(.07)
        cfg=w.collect_settings('training');self.assertEqual(cfg['algorithm']['xgboost']['learning_rate'],.07)
        self.assertEqual(cfg['algorithm']['rf']['n_estimators'],123)
        w.shell.open_training('mlp');cfg=w.collect_settings('training')
        self.assertTrue(cfg['physics']['enabled']);self.assertEqual(cfg['training']['learning_rate'],.002)
        self.assertFalse(w.hidden.isHidden());self.assertTrue(w.tree_edits['rf']['n_estimators'].isHidden())

    def test_model_rename_updates_tree_both_selectors_and_survives_reload(self):
        from PySide6.QtWidgets import QMenu
        from overpressure.config import load_project
        w=self.window;shell=w.shell;paths=[]
        for algorithm in ('rf','xgboost'):
            folder=Path(self.tmp.name)/algorithm;folder.mkdir();paths.append(str(folder.resolve()))
            (folder/'manifest.json').write_text(json.dumps(dict(algorithm=algorithm,display_name=algorithm+' test',status='completed',purpose='training')),encoding='utf-8')
            shell.register_result(dict(path=str(folder),artifact='final_model',smoke=False))
        w.bundle_path.setText(paths[0]);w.seismic_bundle.setText(paths[1])
        original=(Path(paths[0])/'manifest.json').read_bytes()
        menu=QMenu();shell.build_context_menu(menu,dict(model=True,path=paths[0]))
        self.assertIn('重命名…',[a.text() for a in menu.actions()])
        with patch('overpressure.workstation.QInputDialog.getText',return_value=('沙三段 RF',True)):
            shell.rename_model(paths[0])
        self.assertEqual(w.bundle_path.text(),paths[0]);self.assertEqual(w.seismic_bundle.text(),paths[1])
        for choice in (w.well_model_choice,w.seismic_model_choice):
            self.assertEqual(choice.itemText(choice.findData(paths[0])),'沙三段 RF [RF]')
        self.assertTrue(any(label=='沙三段 RF [RF]' for label,_ in shell.entries()['模型']))
        self.assertEqual((Path(paths[0])/'manifest.json').read_bytes(),original)
        w.project=load_project(w.project.path);w.settings=deepcopy(w.project.settings);w.populate()
        self.assertEqual(shell.model_aliases[paths[0]],'沙三段 RF')
        w.well_model_choice.setCurrentIndex(w.well_model_choice.findData(paths[1]))
        self.assertEqual(w.bundle_path.text(),paths[1])
        self.assertEqual(w.seismic_bundle.text(),paths[1])
        with patch('overpressure.workstation.QInputDialog.getText',return_value=('沙三段 RF',True)),patch('overpressure.workstation.QMessageBox.warning') as warning:
            shell.rename_model(paths[1]);warning.assert_called_once()
        self.assertNotIn(paths[1],shell.model_aliases)

    def test_smoke_completion_does_not_select_formal_model(self):
        self.window.handle_line('{"event":"completed","path":"smoke","smoke":true}')
        self.assertEqual(self.window.bundle_path.text(), "")
        self.window.handle_line('{"event":"completed","path":"formal","smoke":false}')
        self.assertEqual(self.window.bundle_path.text(), "formal")

    def test_cross_validation_does_not_replace_selected_formal_model(self):
        self.window.bundle_path.setText("original")
        self.window.task_command = "cross-validate"
        self.window.handle_line('{"event":"completed","path":"fold","smoke":false}')
        self.assertEqual(self.window.bundle_path.text(), "original")

    def test_pressure_chart_load_and_seismic_settings(self):
        folder = Path(self.tmp.name) / "predictions"; folder.mkdir()
        pd.DataFrame({"well": ["A", "A", "B"], "Depth": [100, 200, 300], "Pp": [10, 20, 30], "Pp_pred": [11, 21, 31]}).to_csv(folder / "predictions.csv", index=False)
        self.window.charts.load_directory(folder)
        self.window.charts.well.setCurrentText("B")
        self.window.charts.canvas.draw()
        self.assertTrue(self.window.charts.canvas.figure.axes[0].yaxis_inverted())
        self.window.den_unit.setCurrentText("g/cc")
        self.window.block_traces.setValue(64)
        self.window.header_edits["inline"].setValue(189)
        cfg = self.window.collect_settings()
        self.assertEqual(cfg["seismic"]["density_unit"], "g/cc")
        self.assertEqual(cfg["seismic"]["header_fields"]["inline"], 189)
        self.assertEqual(cfg["seismic"]["block_traces"], 64)

    def test_live_epoch_updates_training_chart(self):
        self.window.handle_line('{"event":"epoch","epoch":1,"epochs":3,"train_data":10,"validation_data":12}')
        self.assertEqual(len(self.window.charts.history), 1)
        self.assertEqual(self.window.charts.mode.currentIndex(), 0)

    def test_scatter_table_style_and_csv_share_the_same_pairs(self):
        folder = Path(self.tmp.name) / "evaluation"; folder.mkdir()
        pd.DataFrame({"well": ["A", "A", "B"], "Depth": [100, 200, 300], "Pp": [0, 10, 20], "Pp_pred": [1, 12, 18]}).to_csv(folder / "predictions.csv", index=False)
        panel = self.window.charts; panel.load_directory(folder); panel.mode.setCurrentIndex(3); panel.well.setCurrentIndex(0)
        self.assertEqual(len(panel.canvas.figure.axes[0].lines), 0)
        panel.mode.setCurrentIndex(4); self.assertEqual(panel.table.rowCount(), 3)
        self.assertEqual(panel.current_metrics().iloc[0]["count"], 3)
        panel.save_metrics(folder / "metrics.csv")
        self.assertEqual(pd.read_csv(folder / "metrics.csv").iloc[0]["mape_count"], 2)
        self.assertTrue((folder / "metrics.notes.json").is_file())
        panel.settings_path = folder / "style.json"; panel.mode.setCurrentIndex(3)
        panel.apply_style(FigureStyle(title="Custom evaluation", width_cm=16, height_cm=12))
        self.assertEqual(panel.canvas.figure.axes[0].get_title(), "Custom evaluation")
        self.assertEqual(FigureStyle.load(panel.settings_path).width_cm, 16)
        from PySide6.QtWidgets import QFontComboBox
        dialog = panel.create_style_dialog()
        self.assertEqual(dialog.findChild(QFontComboBox).currentText(), "Arial")
        dialog.close()

    def test_final_refit_epoch_has_no_fabricated_validation_curve(self):
        self.window.task_command = "refit"
        self.window.handle_line('{"event":"epoch","epoch":1,"epochs":2,"train_data":8}')
        self.assertNotIn("validation_data", self.window.charts.history)
        self.assertEqual(len(self.window.charts.canvas.figure.axes[0].lines), 1)
        self.window.handle_line('{"event":"completed","artifact":"final_model","path":"final","smoke":false}')
        self.assertEqual(self.window.bundle_path.text(), "final")

    def test_slice_panel_switches_axes_and_masks_missing_geometry(self):
        path = Path(self.tmp.name) / "view.npz"
        np.savez_compressed(path, pressure=np.array([[0, 1, 2], [3, np.nan, 5]], dtype=np.float32), inline=np.array([10, 11]), crossline=np.array([20, 20]), delay_ms=np.array([0, 0]), interval_ms=np.array(1.))
        self.window.slices.load_cache(path)
        self.window.slices.mode.setCurrentIndex(2)
        self.window.slices.position.setCurrentIndex(1)
        self.window.slices.canvas.draw()
        self.assertEqual(self.window.slices.canvas.figure.axes[0].get_xlabel(), "Crossline")
        self.assertIn("1 ms", self.window.slices.canvas.figure.axes[0].get_title())
        self.window.slices.mode.setCurrentIndex(0)
        self.assertTrue(self.window.slices.canvas.figure.axes[0].yaxis_inverted())

    def test_stale_slice_response_cannot_overwrite_latest_selection(self):
        panel = self.window.slices
        first, latest = "a" * 32, "b" * 32
        class Reader:
            output = b""
            writes = []
            def readAllStandardOutput(self): value, self.output = self.output, b""; return value
            def write(self, value): self.writes.append(value)
        proc = Reader(); panel.reader_process = proc; panel.reader_ready = True; panel.reader_busy = True
        panel.reader_decoder = codecs.getincrementaldecoder("utf-8")("replace"); panel.reader_buffer = ""
        panel.base_context = "test"; panel.stopping = False; panel.latest_id = latest
        panel.pending_request = {"id": latest, "mode": "time", "value": 1}
        try:
            proc.output = (json.dumps({"event": "slice", "id": first, "path": "does-not-exist.npz"}) + "\n").encode()
            panel.read_reader()
            self.assertEqual(json.loads(proc.writes[0])["id"], latest)
            path = Path(self.tmp.name) / "latest.npz"
            np.savez_compressed(path, values=np.array([[7.]], dtype=np.float32), x=np.array([1]), y=np.array([2]), xlabel=np.array("Crossline"), ylabel=np.array("Inline"), title=np.array("Latest slice"), invert_y=np.array(False), limits=np.array([6., 8.]), x_stride=np.array(2), y_stride=np.array(1))
            proc.output = (json.dumps({"event": "slice", "id": latest, "path": str(path)}) + "\n").encode()
            panel.read_reader()
            self.assertEqual(panel.canvas.figure.axes[0].get_title(), "Latest slice")
            self.assertIn("抽稀", panel.context.text())
        finally:
            panel.reader_process = None

    def test_slice_display_ranges_colorbar_and_horizon_column_mapping(self):
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QComboBox, QDialogButtonBox
        from overpressure.seismic_view import Slice
        from overpressure.horizon_dialog import HorizonImportDialog
        panel = self.window.slices
        panel.render(Slice(np.array([[1., 2.], [3., 4.]]), np.array([20, 21]), np.array([1000, 1002]), "Crossline", "Time (ms)", "Inline 10", True), (1, 4))
        dialog = panel.create_display_dialog(); boxes = dialog.findChildren(QCheckBox)
        for box in boxes: box.setChecked(False)
        names=['slice_axis_0_0','slice_axis_0_1','slice_axis_1_0','slice_axis_1_1','slice_color_min','slice_color_max']
        for name,value in zip(names,[19,22,999,1003,0,10]):dialog.findChild(QDoubleSpinBox,name).setValue(value)
        dialog.findChild(QComboBox,'slice_cmap').setCurrentText("cividis")
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Apply).click()
        ax = panel.canvas.figure.axes[0]
        self.assertEqual(ax.get_xlim(), (19, 22)); self.assertEqual(ax.get_ylim(), (1003, 999))
        self.assertEqual(ax.collections[0].get_clim(), (0, 10)); self.assertEqual(ax.collections[0].cmap.name, "cividis")
        dialog.close()
        path = Path(self.tmp.name) / "horizon.csv"
        pd.DataFrame({"X": [100], "Y": [200], "Time": [1.1], "Horizon": ["A"]}).to_csv(path, index=False)
        importer = HorizonImportDialog(panel); importer.path.setText(str(path)); importer.read_headers(); importer.coordinates.setCurrentIndex(1)
        self.assertEqual(importer.config()["columns"]["x"], "X")
        self.assertEqual(importer.config()["columns"]["name"], "Horizon")
        importer.close()

    def test_progress_message_split_inside_utf8_character_is_preserved(self):
        raw = (json.dumps({"event": "stage", "message": "中文路径校验中"}, ensure_ascii=False) + "\n").encode("utf-8")
        cut = raw.index("中".encode("utf-8")) + 1
        class Process:
            chunks = [raw[:cut], raw[cut:]]
            def readAllStandardOutput(self): return self.chunks.pop(0)
        self.window.process = Process(); self.window.output_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            self.window.read_output(); self.window.read_output()
            self.assertIn("中文路径校验中", self.window.log.toPlainText())
        finally: self.window.process = None


if __name__ == "__main__":
    unittest.main()

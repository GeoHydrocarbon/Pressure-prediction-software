from __future__ import annotations

from copy import deepcopy
import codecs
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import yaml
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl, QSize, QTimer
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget, QTabWidget,QSizePolicy,
)

from . import __version__
from .config import FEATURES, UNITS, load_project, validate_settings, strip_depth_metadata
from .time_depth import TimeDepth
from .runtime import worker_command, user_data_root, default_project
from .ui_style import Card, polish_layouts, QSS, PAGE_PADDING, SPACING,PageScroll,WindowBoundsGuard
from .batch_files import BatchFilesPanel


ROOT = user_data_root()


from .ui_modules.training import TrainingModuleUI
from .ui_modules.wells import WellModuleUI
from .ui_modules.seismic import SeismicModuleUI


class MainWindow(TrainingModuleUI, WellModuleUI, SeismicModuleUI, QMainWindow):
    def __init__(self, project_path=None):
        super().__init__()
        self.project = load_project(project_path or default_project())
        self.settings = deepcopy(self.project.settings)
        self.process = None
        self.pending = ""
        self.cancel_path = None
        self.run_path = None
        self.cancel_requested = False
        self.task_command = None
        self.epoch_records = []
        self.cv_fold = 0; self.cv_folds = 0
        self.setWindowTitle(f"超压预测软件 · 开发版 {__version__}")
        self.resize(1140, 860)
        self.actions = []
        self.geometry_process = None; self.geometry_signature = None; self.geometry_generation = 0
        self.geometry_timer = QTimer(self); self.geometry_timer.setSingleShot(True); self.geometry_timer.setInterval(400)
        self.geometry_timer.timeout.connect(self.read_seismic_geometry)
        from .workstation import DocumentTabs, WorkstationShell
        # Keep the index dispatcher for existing task and catalog callbacks.
        self.nav = QListWidget(self); self.nav.hide()
        self.nav.addItems(DocumentTabs.TITLES)
        self.pages = DocumentTabs()
        self.build_data()
        from .well_library_ui import WellLibrary
        self.library=WellLibrary(self)
        self.build_training(); self.build_time_depth(); self.build_results()
        from .chart_panel import ChartPanel
        self.charts = ChartPanel();self.charts_page=self.charts; self.pages.addWidget(self.charts_page)
        self.build_seismic()
        from .slice_panel import SlicePanel
        self.slices = SlicePanel(self.start_task);self.slices_page=self.slices; self.pages.addWidget(self.slices_page); self.actions.append(self.slices.load_button)
        from .traditional_panel import TraditionalPanel
        self.traditional=TraditionalPanel(self)
        self.traditional_page=self.page('传统压力预测','正常压实趋势标定与 Eaton 井压力预测。',show_heading=False)
        self.traditional_page.addWidget(self.traditional)
        self.slices.well_config_provider = self.catalog.well_config
        self.slices.horizon_config_provider = self.catalog.horizon_configs
        self.slices.interpolation_registered = self.catalog.register_interpolated_horizon
        self.nav.currentRowChanged.connect(self.navigate_to_page)
        self.pages.currentChanged.connect(self.nav.setCurrentRow)
        self.shell = WorkstationShell(self)
        self.nav.setCurrentRow(0)
        self.populate()
        for edit in (self.den_path,self.ip_path): edit.textChanged.connect(self.invalidate_seismic_geometry)
        for key in ('inline','crossline'): self.header_edits[key].valueChanged.connect(self.invalidate_seismic_geometry)
        self.nav.currentRowChanged.connect(lambda index: self.geometry_timer.start() if index in (0,5) else None)
        self.invalidate_seismic_geometry()
        polish_layouts(self)
        self.screen_guard=WindowBoundsGuard(self)
        for document in self.shell.workspace.documents.values():document.hide()
        self.shell.workspace.show_map()

    def navigate_to_page(self,index):
        if index==2 and hasattr(self,'shell'):
            self.shell.open_time_depth()
            self.nav.blockSignals(True);self.nav.setCurrentRow(self.pages.currentIndex());self.nav.blockSignals(False)
        else:
            self.pages.setCurrentIndex(index)
            if hasattr(self,'shell'):self.shell.workspace.show_page(index)

    def page(self, title, description, *, scrollable=True,show_heading=True):
        body = QWidget(); body.setObjectName('pageBody'); layout = QVBoxLayout(body)
        heading = QLabel(title); heading.setObjectName("heading"); layout.addWidget(heading);heading.setVisible(show_heading)
        text = QLabel(description); text.setObjectName('hint'); text.setWordWrap(True); layout.addWidget(text)
        if scrollable:
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(body);self.pages.addWidget(scroll)
        else:self.pages.addWidget(body)
        return layout

    def path_field(self, layout, label, *, directory=False):
        row = QWidget(); box = QHBoxLayout(row); box.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(); btn = QPushButton("选择")
        def choose():
            path = QFileDialog.getExistingDirectory(self, label, str(self.project.path.parent)) if directory else QFileDialog.getOpenFileName(self, label, str(self.project.path.parent), "所有文件 (*)")[0]
            if path:
                edit.setText(path)
        btn.clicked.connect(choose); box.addWidget(edit, 1); box.addWidget(btn)
        layout.addRow(label, row)
        return edit

    def combo(self, choices):
        box = QComboBox(); box.addItems(choices); return box

    def number(self, minimum, maximum, decimals=4):
        spin = QDoubleSpinBox(); spin.setDecimals(decimals); spin.setRange(minimum, maximum); return spin

    def button(self, layout, text, callback):
        button = QPushButton(text); button.clicked.connect(callback); layout.addWidget(button); self.actions.append(button)
        if text.startswith(('开始','保存并检查')):button.setProperty('role','primary')
        return button

    def build_data(self):
        page = self.page("项目与数据", "原始文件保留。训练井由批量清单勾选决定：勾选文件中的井参与训练，不勾选则排除。缺失标签仅影响监督训练。",scrollable=False)
        self.data_tabs = QTabWidget(); page.addWidget(self.data_tabs)
        well_body = QWidget(); page = QVBoxLayout(well_body)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(well_body); self.data_tabs.addTab(scroll, "井数据")
        from .project_catalog import ProjectCatalog
        self.catalog = ProjectCatalog(self)
        group = Card("训练井文件"); form = QFormLayout(group)
        self.train_files=BatchFilesPanel(self,lambda:{k:e.text().strip() for k,e in self.column_edits.items()},('well',*FEATURES,'Pp'),'训练文件批量导入')
        form.addRow(self.train_files)
        self.train_files.status.setText('勾选文件中的全部井参与训练；支持只添加一个文件，不再使用开发井或盲井名单筛选。')
        self.split_method = self.combo(["全部参与井随机打乱：按设置比例划分", "留一井验证：每轮留一口井，其余井训练"])
        self.train_percent = self.number(.01,99.99,2); self.train_percent.setValue(80)
        page.addWidget(group)
        fields = Card("字段与源单位"); mapping = QFormLayout(fields)
        self.column_edits, self.unit_edits = {}, {}
        for field in ("well", *FEATURES, "Pp"):
            row = QWidget(); line = QHBoxLayout(row); line.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(); line.addWidget(edit); self.column_edits[field] = edit
            if field != "well":
                combo = self.combo(list(UNITS[field])); line.addWidget(combo); self.unit_edits[field] = combo
            mapping.addRow("井名" if field == "well" else field, row)
        page.addWidget(fields)
        self.button(page, "检查井数据与地震对应关系", lambda: self.start_task("audit", []))
        page.addWidget(self.catalog)
        page.addStretch()
        seismic_body = QWidget(); seismic_layout = QVBoxLayout(seismic_body)
        seismic_scroll = QScrollArea(); seismic_scroll.setWidgetResizable(True); seismic_scroll.setWidget(seismic_body)
        self.data_tabs.addTab(seismic_scroll, "地震数据")
        seismic_card = Card('地震输入与压力成果'); form = QFormLayout(seismic_card); seismic_layout.addWidget(seismic_card)
        self.den_path = self.path_field(form, "输入：密度 SEG-Y")
        self.ip_path = self.path_field(form, "输入：波阻抗 SEG-Y")
        self.seismic_data_form = form; self.seismic_data_layout = seismic_layout
        self.data_geometry_label = QLabel(); self.data_geometry_label.setWordWrap(True); form.addRow('输入地震线号范围',self.data_geometry_label)
        self.pressure_result = self.path_field(form, "输出：压力成果目录", directory=True)
        self.button(seismic_layout, "载入压力成果到地震切片", self.load_catalog_pressure)
        self.data_tabs.addTab(self.catalog.horizon_widget, "层位数据")

    def load_catalog_pressure(self):
        value = self.pressure_result.text().strip()
        self.slices.result_dir.setText(str(self.project.resolve(value)) if value else ''); self.slices.load_result()

    def build_time_depth(self):
        page = self.page("时深关系", "设置项目使用的公式或时间深度表，并核对时间单位、单/双程时间及输出深度单位。",show_heading=False)
        card=Card('时间与深度转换参数'); form = QFormLayout(card)
        self.td_method = self.combo(["polynomial", "table"]); form.addRow("方式（多项式 / 对照表）", self.td_method)
        self.coefficients = QLineEdit(); form.addRow("系数（常数、一次、二次、三次）", self.coefficients)
        self.td_file = self.path_field(form, "时间深度表（time, depth）")
        self.td_unit = self.combo(["s", "ms"]); form.addRow("关系式/表格的时间单位", self.td_unit)
        self.td_convention = self.combo(["twt", "owt"]); form.addRow("时间类型（双程 / 单程）", self.td_convention)
        self.td_depth_unit = self.combo(["m", "ft"]); form.addRow("输出深度单位", self.td_depth_unit)
        note=QLabel('自动使用每道起始时间和采样间隔；仅预测密度、波阻抗及转换深度均有效的点。公式在实际有效时间区间内需单调递增，时深表覆盖外不外推。');note.setWordWrap(True);note.setObjectName('hint');form.addRow(note)
        page.addWidget(card)
        self.td_check_button=self.button(page, "保存并检查时深设置", self.check_time_depth)
        self.td_method.currentTextChanged.connect(self.update_time_depth_controls);self.update_time_depth_controls()
        page.addStretch()

    def update_time_depth_controls(self,*_):
        table=self.td_method.currentText()=='table'
        self.coefficients.setEnabled(not table);self.td_file.parentWidget().setEnabled(table)

    def set_geometry_text(self,text):
        self.data_geometry_label.setText(text); self.prediction_geometry_label.setText(text)

    def invalidate_seismic_geometry(self,*_):
        self.geometry_generation += 1; self.geometry_signature = None
        self.set_geometry_text('正在更新输入文件的 Inline / Crossline 范围……')
        self.geometry_timer.start()

    def geometry_config(self):
        files={key:str(self.project.resolve(edit.text().strip())) for key,edit in [('density',self.den_path),('impedance',self.ip_path)] if edit.text().strip()}
        headers={key:self.header_edits[key].value() for key in ('inline','crossline')}
        stamps=[]
        for key,path in files.items():
            try:
                stat=Path(path).stat(); stamps.append((key,path,stat.st_size,stat.st_mtime_ns))
            except OSError: stamps.append((key,path,None,None))
        return dict(files=files,headers=headers), (str(self.project.path),tuple(stamps),tuple(headers.values()))

    def read_seismic_geometry(self):
        if self.geometry_process is not None:return
        config,signature=self.geometry_config()
        if signature==self.geometry_signature:return
        self.geometry_signature=signature
        if not config['files'] or not any(Path(p).is_file() for p in config['files'].values()):
            self.set_geometry_text('尚未找到预测输入 SEG-Y 文件，请在“数据 → 导入数据 → 地震输入与单位”选择文件。');return
        generation=self.geometry_generation
        self.set_geometry_text('正在读取 SEG-Y 道头范围（不读取属性体）……')
        proc=QProcess(self); self.geometry_process=proc
        env=QProcessEnvironment.systemEnvironment(); env.insert('PYTHONIOENCODING','utf-8'); proc.setProcessEnvironment(env); proc.setWorkingDirectory(str(ROOT))
        def send():
            proc.write(json.dumps(config).encode('utf-8')); proc.closeWriteChannel()
        def finish(*_):
            if self.geometry_process is not proc:return
            output=bytes(proc.readAllStandardOutput()).decode('utf-8',errors='replace')
            errors=bytes(proc.readAllStandardError()).decode('utf-8',errors='replace')
            self.geometry_process=None; proc.deleteLater()
            if generation!=self.geometry_generation or signature!=self.geometry_config()[1]:
                self.geometry_signature=None; self.geometry_timer.start();return
            try:
                result=json.loads(output)
                if 'error' in result:raise ValueError(result['error'])
                self.display_seismic_geometry(result)
            except Exception as exc:self.set_geometry_text('地震线号范围读取失败：'+(errors.strip() or str(exc)))
        proc.started.connect(send); proc.finished.connect(finish)
        proc.errorOccurred.connect(lambda error: finish() if error==QProcess.ProcessError.FailedToStart else None)
        proc.start(*worker_command('overpressure.seismic_geometry'))

    def display_seismic_geometry(self,result):
        lines=[]
        for key,label in [('density','密度'),('impedance','波阻抗')]:
            item=result['files'].get(key)
            if item is None:lines.append(label+'：未选择文件');continue
            if 'error' in item:lines.append(label+'：无法读取 — '+item['error']);continue
            il,xl=item['inline'],item['crossline']
            lines.append(f'{label}：Inline {il[0]}～{il[1]}；Crossline {xl[0]}～{xl[1]}（{item["traces"]} 道）')
        if result.get('geometry_matches') is False:lines.append('密度与波阻抗的逐道线号不一致，请核对文件或道头设置。')
        lines.append('以上为道头最小～最大线号；范围内可能存在缺失道。')
        self.set_geometry_text('\n'.join(lines))

    def populate(self):
        cfg = self.settings; data = cfg["data"]; td = cfg["time_depth"]
        self.library.populate(cfg)
        self.catalog.populate(cfg.get("project_data", {})); self.pressure_result.setText(cfg.get("project_data", {}).get("pressure_result", ""))
        self.train_files.set_files(data.get('well_imports',data.get('well_files',[])),data.get('well_files'),active=True)
        self.den_path.setText(cfg["seismic"]["density_file"]); self.ip_path.setText(cfg["seismic"]["impedance_file"])
        self.den_unit.setCurrentText(cfg["seismic"]["density_unit"]); self.ip_unit.setCurrentText(cfg["seismic"]["impedance_unit"])
        self.block_traces.setValue(cfg["seismic"].get("block_traces", 32))
        for key,edit in self.prediction_range_edits.items():edit.setValue(cfg['seismic'].get('prediction_range',{}).get(key,0))
        self.seismic_convention.setCurrentText(cfg["seismic"]["time_convention"])
        for key, edit in self.header_edits.items(): edit.setValue(cfg["seismic"]["header_fields"][key])
        self.split_method.setCurrentIndex(0 if data.get("split_method", "well_holdout") == "random_points" else 1)
        self.train_percent.setValue(data.get('train_ratio',.8)*100)
        self.seed.setText(str(cfg["training"]["seed"]))
        for field, edit in self.column_edits.items():
            edit.setText(data["columns"][field])
        for field, edit in self.unit_edits.items():
            edit.setCurrentText(data["units"][field])
        index=self.algorithm.findData(cfg['algorithm']['name'])
        if index < 0:raise ValueError('项目包含未知训练算法')
        self.algorithm.setCurrentIndex(index)
        self.hidden.setText(", ".join(map(str, cfg["algorithm"]["hidden_sizes"])))
        self.dropout.setValue(cfg["algorithm"]["dropout"])
        for attr, key in (("epochs", "epochs"), ("batch", "batch_size"), ("lr", "learning_rate"), ("patience", "patience")):
            getattr(self, attr).setValue(cfg["training"][key])
        self.physical_enabled.setChecked(cfg["physics"]["enabled"])
        for key, edit in self.physics_edits.items():
            edit.setValue(cfg["physics"][key]); edit.setEnabled(self.physical_enabled.isChecked())
        self.td_method.setCurrentText(td["method"]); self.coefficients.setText(", ".join(map(str, td.get("coefficients", []))))
        self.td_file.setText(td.get("table_file", ""))
        for attr, key in (("td_unit", "time_unit"), ("td_convention", "time_convention"), ("td_depth_unit", "depth_unit")):
            getattr(self, attr).setCurrentText(td[key])
        self.project_label.setText(self.project.path.name)
        if hasattr(self,"shell"): self.shell.project_changed()
        self.well_prediction.populate(self.project)
        self.well_prediction.refresh_library(cfg.get('well_prediction',{}).get('library_wells',[]))
        self.traditional.populate(cfg)
        self.library.migrate_legacy(cfg)
        self.invalidate_seismic_geometry()

    def collect_settings(self, scope='all'):
        cfg = deepcopy(self.settings); data = cfg["data"]
        independent_task=scope in ('prediction','seismic','traditional')
        cfg["project_data"] = self.catalog.collect(); cfg["project_data"]["pressure_result"] = self.pressure_result.text().strip()
        cfg['project_data']['wells']=deepcopy(self.library.wells)
        split = lambda text: [v.strip() for v in text.replace("，", ",").split(",") if v.strip()]
        data.pop('well_file',None);data.pop('blind_well_candidates',None)
        data.update(training_selection='checked_files',well_files=self.train_files.files(),well_imports=self.train_files.all_files(),
                    development_wells=self.train_files.names() if self.train_files.ready() else [],validation_wells=[],blind_wells=[])
        data.update(split_method="random_points" if self.split_method.currentIndex() == 0 else "leave_one_well_out", train_ratio=self.train_percent.value()/100)
        if self.library.active:
            data.update(training_selection='well_library',development_wells=self.library.checked(self.library.well_list),blind_wells=[],validation_wells=[],features=self.library.checked(self.library.feature_list),target_curve=self.library.target.currentText() or 'Pp',feature_units=self.library.selected_units())
        data['validation_wells']=[]
        if not independent_task:cfg["training"]["seed"] = int(self.seed.text().strip())
        cfg["well_prediction"] = self.well_prediction.collect()
        for field, edit in self.column_edits.items(): data["columns"][field] = edit.text().strip()
        for field, edit in self.unit_edits.items(): data["units"][field] = edit.currentText()
        cfg["seismic"].update(density_file=self.den_path.text().strip(), impedance_file=self.ip_path.text().strip())
        cfg["seismic"].update(density_unit=self.den_unit.currentText(), impedance_unit=self.ip_unit.currentText(), time_convention=self.seismic_convention.currentText(), header_fields={key: edit.value() for key, edit in self.header_edits.items()})
        cfg["seismic"]["block_traces"] = self.block_traces.value()
        cfg['seismic']['prediction_range']={key:edit.value() for key,edit in self.prediction_range_edits.items()}
        if not independent_task:
            cfg["algorithm"].update(name=self.algorithm.currentData(), hidden_sizes=[int(v) for v in split(self.hidden.text())], dropout=self.dropout.value())
            cfg["training"].update(epochs=self.epochs.value(), batch_size=self.batch.value(), learning_rate=self.lr.value(), patience=self.patience.value())
            cfg["physics"]["enabled"] = self.physical_enabled.isChecked()
            for key, edit in self.physics_edits.items(): cfg["physics"][key] = edit.value()
        else:
            cfg['data']=deepcopy(self.settings['data'])
        try:coefficients=[float(v) for v in split(self.coefficients.text())] if scope not in ('prediction','training','traditional') else deepcopy(cfg['time_depth'].get('coefficients',[]))
        except ValueError:
            if self.td_method.currentText()!='table':raise
            coefficients=deepcopy(cfg['time_depth'].get('coefficients',[]))
        cfg["time_depth"].update(method=self.td_method.currentText(), coefficients=coefficients,
            table_file=self.td_file.text().strip(), time_unit=self.td_unit.currentText(), time_convention=self.td_convention.currentText(),
            depth_unit=self.td_depth_unit.currentText())
        if not cfg["algorithm"]["hidden_sizes"] or any(v <= 0 for v in cfg["algorithm"]["hidden_sizes"]):
            raise ValueError("隐藏层必须为正整数列表")
        if hasattr(self,'shell'):cfg['workspace']=deepcopy(self.shell.workspace_settings())
        cfg['traditional_prediction']=self.traditional.collect()
        strip_depth_metadata(cfg); validate_settings(cfg)
        return cfg

    def save_settings(self, scope='all'):
        try:
            cfg = self.collect_settings(scope)
            temp = self.project.path.with_suffix(".yaml.tmp")
            temp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            temp.replace(self.project.path)
            self.settings = cfg; self.project = load_project(self.project.path)
            self.status.setText("项目设置已保存。")
            return True
        except Exception as exc:
            QMessageBox.warning(self, "设置需要修改", str(exc)); return False

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", str(self.project.path.parent), "项目文件 (*.yaml *.yml)")
        if path:
            try:
                project = load_project(path)
                from .algorithms.registry import get_algorithm
                get_algorithm(project.settings["algorithm"]["name"]).require_available()
                self.project = project; self.settings = deepcopy(project.settings); self.populate();self.shell.workspace.show_map()
            except Exception as exc: QMessageBox.warning(self, "无法打开项目", str(exc))

    def new_project(self):
        path,_=QFileDialog.getSaveFileName(self,'新建超压预测项目',str(self.project.path.parent/'新项目.yaml'),'项目文件 (*.yaml)')
        if not path:return
        try:
            cfg=deepcopy(self.settings);cfg['name']=Path(path).stem;cfg['output_dir']='outputs'
            cfg['data'].update(training_selection='well_library',well_files=[],well_imports=[],development_wells=[],validation_wells=[],blind_wells=[],features=list(FEATURES),target_curve='Pp',feature_units={},split_method='random_points')
            cfg['data'].pop('well_file',None)
            cfg['project_data']=dict(wells=[],well_assets=[],horizons=[],pressure_result='',max_distance=25.,vertical_fallback=True)
            cfg['workspace']={};cfg['traditional_prediction']={};cfg['physics']['enabled']=False
            cfg['seismic'].update(density_file='',impedance_file='',prediction_range={k:0 for k in self.prediction_range_edits})
            cfg['time_depth']=dict(method='polynomial',coefficients=[0.,1000.],time_unit='s',time_convention='twt',depth_unit='m',table_file='')
            cfg['well_prediction'].update(well_files=[],well_imports=[],measurement_files=[],measurement_imports=[],library_wells=[],use_library=True,independence_confirmed=False)
            target=Path(path).resolve();target.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
            self.project=load_project(target);self.settings=deepcopy(self.project.settings);self.populate();self.shell.workspace.show_map()
        except Exception as exc:QMessageBox.warning(self,'无法新建项目',str(exc))

    def save_project_as(self):
        path,_=QFileDialog.getSaveFileName(self,'项目另存为',str(self.project.path),'项目文件 (*.yaml)')
        if not path:return
        try:
            cfg=self.collect_settings()
            # Preserve references when the project file moves to another directory.
            keys={'path','output_dir','pressure_result','well_model','seismic_model','refit_cv','table_file','density_file','impedance_file','well_file','measurement_file','source_report','result'}
            lists={'well_files','well_imports','measurement_files','measurement_imports'}
            def absolute(value):
                if isinstance(value,dict):
                    return {k:str(self.project.resolve(v)) if k in keys and isinstance(v,str) and v else
                            [str(self.project.resolve(p)) for p in v] if k in lists and isinstance(v,list) else absolute(v) for k,v in value.items()}
                if isinstance(value,list):return [absolute(item) for item in value]
                return value
            cfg=absolute(cfg);target=Path(path).resolve();cfg['name']=target.stem
            target.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
            self.project=load_project(target);self.settings=deepcopy(self.project.settings);self.populate();self.shell.workspace.show_map()
        except Exception as exc:QMessageBox.warning(self,'无法另存项目',str(exc))

    def check_time_depth(self):
        if self.save_settings():
            try:
                TimeDepth(self.project)
                self.status.setText("时深设置已保存，结构检查通过。预测时自动读取有效时间，并检查公式单调性；表格覆盖外留空。")
            except Exception as exc: QMessageBox.warning(self, "时深设置需要修改", str(exc))

    def open_outputs(self):
        path = self.project.resolve(self.settings["output_dir"]); path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def start_task(self, command, extra):
        if self.process is not None: return
        if command in ('train','cross-validate','audit','refit','traditional-trend','traditional-predict') and self.library.worker is not None:
            QMessageBox.information(self,'正在导入井数据','请等待井库导入完成。');return
        if command in ('train','cross-validate','audit','refit') and not self.library.active and self.train_files.active and not self.train_files.ready():
            QMessageBox.information(self,'先检查批量文件','请先检查批量文件并等待完成；缺少字段或无法读取的文件请修正或取消勾选。');return
        scope='prediction' if command=='predict-wells' else 'seismic' if command=='predict-seismic' else 'all'
        if command in ('train','cross-validate','refit') or command=='audit' and '--wells-only' in extra:scope='training'
        if command in ('traditional-trend','traditional-predict'):scope='traditional'
        if command != "prepare-seismic-view" and not self.save_settings(scope): return
        if command == "refit" and not self.refit_cv.text().strip():
            QMessageBox.information(self, "选择验证记录", "请先选择完成的留一井验证成果目录。"); return
        if command == "predict-wells" and not self.bundle_path.text().strip():
            QMessageBox.information(self, "选择模型", "请先选择完整训练的模型目录。"); return
        self.log.clear(); self.pending = ""; self.run_path = None; self.cancel_requested = False
        self.output_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.task_command = command; self.epoch_records = []
        self.presentation_failed = False
        self.cv_fold = 0; self.cv_folds = 0
        self.shell.task_dock.show()
        self.progress.setRange(0, 0)
        for button in self.actions: button.setEnabled(False)
        self.cancel_button.setEnabled(command in ("train", "cross-validate", "refit", "predict-seismic", "prepare-seismic-view", "predict-wells",'traditional-trend','traditional-predict'))
        process = QProcess(self); self.process = process
        environment = QProcessEnvironment.systemEnvironment(); environment.insert("PYTHONIOENCODING", "utf-8")
        cache = ROOT / "outputs" / "matplotlib_cache"; cache.mkdir(parents=True, exist_ok=True)
        environment.insert("MPLCONFIGDIR", str(cache))
        process.setProcessEnvironment(environment); process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.cancel_path = ROOT / "outputs" / "tasks" / f"{uuid4().hex}.cancel"
        self.cancel_path.parent.mkdir(parents=True, exist_ok=True)
        if command in ("train", "cross-validate", "refit", "predict-seismic", "prepare-seismic-view", "predict-wells",'traditional-trend','traditional-predict'): extra = [*extra, "--cancel-file", str(self.cancel_path)]
        process.readyReadStandardOutput.connect(self.read_output)
        process.finished.connect(self.task_finished)
        process.errorOccurred.connect(self.process_error)
        self.status.setText("任务运行中……")
        process.start(*worker_command("overpressure", [command, "--project", str(self.project.path), *extra]))

    def read_output(self):
        if self.process is None: return
        self.pending += self.output_decoder.decode(bytes(self.process.readAllStandardOutput()))
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.handle_line(line.rstrip())

    def handle_line(self, line):
        try: event = json.loads(line)
        except (ValueError, TypeError):
            self.log.appendPlainText(line); return
        if not isinstance(event, dict): self.log.appendPlainText(line); return
        kind = event.get("event")
        if kind == "run_started":
            self.epoch_records = []
            self.run_path = event["path"]
            self.log.appendPlainText(f"模型目录：{self.run_path}\n计算设备：{event['device']}\n试算：{'是' if event['smoke'] else '否'}")
        elif kind == "epoch":
            self.epoch_records.append(event); self.charts.live_history(self.epoch_records)
            if self.task_command == "cross-validate" and self.cv_folds:
                self.progress.setRange(0, self.cv_folds * event["epochs"])
                self.progress.setValue((self.cv_fold - 1) * event["epochs"] + event["epoch"])
            else:
                self.progress.setRange(0, event["epochs"]); self.progress.setValue(event["epoch"])
            message = f"第 {event['epoch']} 轮  训练 MSE={event['train_data']:.4f}"
            message += f"  内部验证 MSE={event['validation_data']:.4f}" if "validation_data" in event else "  固定轮数最终重训（无验证/盲井参与）"
            self.log.appendPlainText(message)
        elif kind == "completed":
            self.log.appendPlainText("成果目录：" + event["path"])
            self.log.appendPlainText(json.dumps(event.get("validation", event.get("metrics", {})), ensure_ascii=False, indent=2))
            artifact = event.get("artifact")
            self.shell.register_result(event)
            if artifact == "cross_validation":
                self.refit_cv.setText(event["path"]);self.advanced_toggle.setChecked(True)
            if artifact in ("seismic_trial", "seismic_volume"):
                self.slices.result_dir.setText(event["path"]); self.pressure_result.setText(event["path"])
            if artifact == "seismic_index":
                try:
                    self.slices.load_index(event["path"]); self.nav.setCurrentRow(6)
                except Exception as exc:
                    self.presentation_failed = True; self.log.appendPlainText(f"索引载入失败：{exc}")
            if artifact == "seismic_view":
                try:
                    self.slices.load_cache(event["path"]); self.nav.setCurrentRow(6)
                except Exception as exc:
                    self.presentation_failed = True; self.log.appendPlainText(f"切片载入失败：{exc}")
            if (self.task_command in (None, "train", "refit")) and not event.get("smoke", True) and artifact in (None, "final_model"):
                self.bundle_path.setText(event["path"]); self.seismic_bundle.setText(event["path"])
            if artifact in ('traditional_trend','traditional_prediction'):
                try:self.traditional.load_result(event['path'],artifact)
                except Exception as exc:
                    self.presentation_failed=True;self.log.appendPlainText('传统方法成果载入失败：'+str(exc))
            try:
                if artifact not in ("seismic_trial", "seismic_view", "seismic_volume", "seismic_index",'traditional_trend','traditional_prediction'): self.charts.load_directory(event["path"])
            except (ValueError, OSError): pass
            if artifact not in ('seismic_index','seismic_view'):
                try:self.shell.persist_workspace();self.shell.refresh_map_sources()
                except OSError as exc:self.log.appendPlainText('成果已生成，但项目关联保存失败：'+str(exc))
        elif kind == "fold_started":
            self.cv_fold = event["fold"]; self.cv_folds = event["folds"]
            self.log.appendPlainText(f"留一井验证 {event['fold']}/{event['folds']}：留出 {event['validation_well']}")
        elif kind == 'traditional_progress':
            self.progress.setRange(0,event['total']);self.progress.setValue(event['completed'])
        elif kind == "seismic_progress":
            self.progress.setRange(0, event["total"]); self.progress.setValue(event["completed"])
            phase = {"predict": "预测", "verify": "输出校验", "index": "成果索引"}.get(event.get("phase"), "处理")
            speed = f"，{event['traces_per_second']:.1f} 道/秒，预计剩余 {event['eta_seconds'] / 60:.1f} 分钟" if "traces_per_second" in event else ""
            self.status.setText(f"{phase}：{event['completed']}/{event['total']}{speed}")
        elif kind == "stage": self.log.appendPlainText(event["message"]); self.status.setText(event["message"])
        else: self.log.appendPlainText(line)

    def task_finished(self, code, exit_status=None):
        self.read_output()
        if self.pending: self.handle_line(self.pending); self.pending = ""
        if self.process is not None: self.process.deleteLater()
        self.process = None
        if getattr(self, "presentation_failed", False): code = 1
        self.progress.setRange(0, 100); self.progress.setValue(100 if code == 0 else 0)
        self.status.setText("任务已取消，未完成成果不可用于预测。" if self.cancel_requested else ("任务完成，详细结果见日志和成果目录。" if code == 0 else "任务失败，请查看日志。"))
        self.cancel_button.setEnabled(False)
        for button in self.actions: button.setEnabled(True)
        self.traditional.update_trend()
        if self.cancel_path and self.cancel_path.exists(): self.cancel_path.unlink()

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText("后台任务无法启动：" + self.process.errorString()); self.task_finished(1)

    def cancel_task(self):
        if self.process is not None and self.cancel_path:
            self.cancel_path.write_text("cancel", encoding="ascii"); self.cancel_requested = True
            self.cancel_button.setEnabled(False); self.status.setText("正在取消，将在下一个计算批次停止……")

    def closeEvent(self, event):
        if self.process is not None:
            self.cancel_task()
            self.status.setText("请等待当前任务结束后关闭窗口。"); event.ignore()
        else:
            self.shell.timer.stop()
            self.shell.workspace.shutdown()
            self.library.stop();self.shell.curve_viewer.stop()
            self.train_files.stop_worker();self.well_prediction.log_files.stop_worker();self.well_prediction.point_files.stop_worker()
            self.geometry_timer.stop()
            proc=self.geometry_process; self.geometry_process=None
            if proc is not None:
                if proc.state()!=QProcess.ProcessState.NotRunning:proc.kill();proc.waitForFinished(1000)
                proc.deleteLater()
            self.slices.shutdown(); event.accept()


def configure_application(app):
    if not QFontDatabase.families() and sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("msyh.ttc", "msyhbd.ttc"):
            if (fonts / name).is_file():
                QFontDatabase.addApplicationFont(str(fonts / name))
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
def main():
    app = QApplication(sys.argv)
    configure_application(app)
    window = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

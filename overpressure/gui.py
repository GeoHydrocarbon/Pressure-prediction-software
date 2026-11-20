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


class MainWindow(QMainWindow):
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

    def navigate_to_page(self,index):
        if index==2 and hasattr(self,'shell'):
            self.shell.open_time_depth()
            self.nav.blockSignals(True);self.nav.setCurrentRow(self.pages.currentIndex());self.nav.blockSignals(False)
        else:self.pages.setCurrentIndex(index)

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
        self.button(page, "检查井数据", lambda: self.start_task("audit", ["--wells-only"]))
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

    def build_training(self):
        page = self.page("模型训练", "勾选的井全部参与所选训练模式。随机划分默认 80% 训练 / 20% 验证，训练比例可自定义，剩余部分用于验证；也可选择留一井验证。")
        page.addWidget(self.library.training_card)
        group = Card("算法与训练参数");self.training_algorithm_card=group; form = QFormLayout(group)
        form.addRow('训练模式',self.split_method);form.addRow('随机模式训练比例（%，默认 80）',self.train_percent)
        self.algorithm = self.combo(["MLP", "RF（后续接入）", "XGBoost（后续接入）"])
        for i in (1, 2):
            self.algorithm.model().item(i).setEnabled(False)
        form.addRow("算法", self.algorithm)
        self.hidden = QLineEdit(); form.addRow("隐藏层宽度", self.hidden)
        self.dropout = self.number(0, .95, 2); form.addRow("Dropout", self.dropout)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 100000); form.addRow("最多训练轮数", self.epochs)
        self.batch = QSpinBox(); self.batch.setRange(1, 65536); form.addRow("批量大小", self.batch)
        self.lr = self.number(.0000001, 1, 7); form.addRow("学习率", self.lr)
        self.patience = QSpinBox(); self.patience.setRange(1, 10000); form.addRow("早停耐心轮数", self.patience)
        self.seed = QLineEdit(); form.addRow("随机种子（划分与训练）", self.seed)
        page.addWidget(group)
        physical = Card("可选物理约束");self.training_physics_card=physical; physform = QFormLayout(physical)
        self.physical_enabled = QCheckBox("训练中加入 Eaton 软物理损失")
        physform.addRow(self.physical_enabled)
        self.physics_edits = {}
        for key, caption, low, high, decimals in (
            ("weight", "物理损失权重", 0, .9999, 4), ("eaton_exponent", "Eaton 指数", .01, 20, 4),
            ("fluid_density", "流体密度（g/cc）", .01, 10, 4), ("rho_rock", "岩石密度（g/cc）", .01, 10, 4),
            ("gravity", "重力加速度（m/s²）", .01, 100, 4), ("nac_a", "正常压实趋势 a", -100, 100, 9),
            ("nac_b", "正常压实趋势 b", -100, 100, 6)):
            edit = self.number(low, high, decimals); self.physics_edits[key] = edit; physform.addRow(caption, edit)
        self.physical_enabled.toggled.connect(lambda on: [edit.setEnabled(on) for edit in self.physics_edits.values()])
        page.addWidget(physical)
        self.button(page, "检查当前训练模式（小样本）", lambda: self.start_training_mode(True))
        self.training_button=self.button(page, "开始随机划分训练", lambda:self.start_training_mode(False))
        self.split_method.currentIndexChanged.connect(self.update_training_mode_controls)
        self.train_percent.valueChanged.connect(self.update_training_mode_controls)
        self.update_training_mode_controls()
        self.advanced_toggle = QCheckBox("留一井完成后：使用全部参与井训练最终模型")
        page.addWidget(self.advanced_toggle)
        self.advanced_training = QWidget(); advanced = QVBoxLayout(self.advanced_training)
        hint = QLabel("留一井验证输出各井验证结果和汇总指标；预测新井或地震体时，可在这里根据验证记录训练最终模型。随机划分模式无需再次重训。"); hint.setWordWrap(True); advanced.addWidget(hint)
        refit_form = QFormLayout(); self.refit_cv = self.path_field(refit_form, "留一井验证成果目录", directory=True); advanced.addLayout(refit_form)
        self.button(advanced, "验证全部参与井重训流程（小样本）", lambda: self.start_task("refit", ["--cv", self.refit_cv.text(), "--smoke"]))
        self.button(advanced, "使用全部参与井重训最终模型", lambda: self.start_task("refit", ["--cv", self.refit_cv.text()]))
        page.addWidget(self.advanced_training); self.advanced_training.hide()
        self.advanced_toggle.toggled.connect(self.advanced_training.setVisible)
        page.addStretch()

    def start_training_mode(self,smoke=False):
        self.start_task('cross-validate' if self.split_method.currentIndex() else 'train',['--smoke'] if smoke else [])

    def update_training_mode_controls(self,*_):
        leave_one=bool(self.split_method.currentIndex());self.train_percent.setEnabled(not leave_one)
        ratio=self.train_percent.value()
        self.training_button.setText('开始留一井验证' if leave_one else f'开始随机划分训练（{ratio:g}% / {100-ratio:g}%）')

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

    def build_results(self):
        page = self.page('井预测与验证','选择模型与待预测井，设置实测点匹配后开始预测；数据从左侧井对象导入。',show_heading=False)
        page.setAlignment(Qt.AlignmentFlag.AlignTop)
        form = QFormLayout()
        self.bundle_path = self.path_field(form,'模型目录',directory=True)
        from .well_panel import WellPredictionPanel
        self.well_prediction = WellPredictionPanel(self,model_form=form); page.addWidget(self.well_prediction)
        self.bundle_path.textChanged.connect(lambda: self.well_prediction.independent.setChecked(False))
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(240); page.addWidget(self.log, 1)

    def build_seismic(self):
        page = self.page("地震压力预测", "填写 Inline、Crossline 起止线号，预测矩形范围内已有的全部道（包含边界）；也可预测全体。按道分块计算，保留原始时间采样与道头，缺失交点不补道。")
        card=Card('预测模型与空间范围'); form = QFormLayout(card)
        self.seismic_bundle = self.path_field(form, "正式模型目录", directory=True)
        self.prediction_geometry_label = QLabel(); self.prediction_geometry_label.setWordWrap(True); form.addRow('输入地震线号范围',self.prediction_geometry_label)
        refresh = QPushButton('刷新输入地震线号范围'); refresh.clicked.connect(self.invalidate_seismic_geometry); form.addRow(refresh)
        self.prediction_range_edits = {}
        for key,label in [('inline_start','Inline 起始线号'),('inline_end','Inline 结束线号'),('crossline_start','Crossline 起始线号'),('crossline_end','Crossline 结束线号')]:
            edit=QSpinBox(); edit.setRange(-2147483648,2147483647); edit.setKeyboardTracking(False)
            self.prediction_range_edits[key]=edit; form.addRow(label,edit)
        self.block_traces = QSpinBox(); self.block_traces.setRange(1, 256); self.block_traces.setValue(32); form.addRow("每块道数（可按内存调整）", self.block_traces)
        page.addWidget(card)
        form = self.seismic_data_form
        self.den_unit = self.combo(list(UNITS["DEN"])); form.addRow("地震密度源单位", self.den_unit)
        self.ip_unit = self.combo(list(UNITS["Ip"])); form.addRow("地震波阻抗源单位", self.ip_unit)
        self.seismic_convention = self.combo(["twt", "owt"]); form.addRow("地震时间类型（双程 / 单程）", self.seismic_convention)
        advanced = QCheckBox("显示 SEG-Y 道头设置（通常保留默认值）"); self.seismic_data_layout.addWidget(advanced)
        header_widget = QWidget(); form = QFormLayout(header_widget); self.seismic_data_layout.addWidget(header_widget)
        header_widget.hide(); advanced.toggled.connect(header_widget.setVisible)
        self.seismic_data_layout.addStretch()
        self.header_edits = {}
        for key, title in (("inline", "Inline 道头位置"), ("crossline", "Crossline 道头位置"), ("x", "X 坐标道头位置"), ("y", "Y 坐标道头位置"), ("coordinate_scalar", "坐标缩放道头位置"), ("coordinate_units", "坐标单位道头位置")):
            edit = QSpinBox(); edit.setRange(1, 240); self.header_edits[key] = edit; form.addRow(title, edit)
        self.seismic_range_button=self.button(page, "开始自定义范围 SEG-Y 预测", self.predict_custom_range)
        self.seismic_full_button=self.button(page, "开始全体 SEG-Y 预测", lambda: self.start_task("predict-seismic", ["--bundle", self.seismic_bundle.text(), "--full", "--block-traces", str(self.block_traces.value())]))
        page.addStretch()

    def predict_custom_range(self):
        values={key:edit.value() for key,edit in self.prediction_range_edits.items()}
        if values['inline_start'] > values['inline_end'] or values['crossline_start'] > values['crossline_end']:
            QMessageBox.warning(self,'预测范围需要修改','Inline、Crossline 的起始线号必须小于或等于结束线号。'); return
        extra=['--bundle',self.seismic_bundle.text(),'--block-traces',str(self.block_traces.value())]
        for key,value in values.items():extra.extend(['--'+key.replace('_','-'),str(value)])
        self.start_task('predict-seismic',extra)

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
            self.set_geometry_text('尚未找到输入 SEG-Y 文件，请在“项目与数据 → 地震数据”选择文件。');return
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
        self.library.migrate_legacy(cfg)
        self.invalidate_seismic_geometry()

    def collect_settings(self):
        cfg = deepcopy(self.settings); data = cfg["data"]
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
        cfg["training"]["seed"] = int(self.seed.text().strip())
        cfg["well_prediction"] = self.well_prediction.collect()
        for field, edit in self.column_edits.items(): data["columns"][field] = edit.text().strip()
        for field, edit in self.unit_edits.items(): data["units"][field] = edit.currentText()
        cfg["seismic"].update(density_file=self.den_path.text().strip(), impedance_file=self.ip_path.text().strip())
        cfg["seismic"].update(density_unit=self.den_unit.currentText(), impedance_unit=self.ip_unit.currentText(), time_convention=self.seismic_convention.currentText(), header_fields={key: edit.value() for key, edit in self.header_edits.items()})
        cfg["seismic"]["block_traces"] = self.block_traces.value()
        cfg['seismic']['prediction_range']={key:edit.value() for key,edit in self.prediction_range_edits.items()}
        cfg["algorithm"].update(name="mlp", hidden_sizes=[int(v) for v in split(self.hidden.text())], dropout=self.dropout.value())
        cfg["training"].update(epochs=self.epochs.value(), batch_size=self.batch.value(), learning_rate=self.lr.value(), patience=self.patience.value())
        cfg["physics"]["enabled"] = self.physical_enabled.isChecked()
        for key, edit in self.physics_edits.items(): cfg["physics"][key] = edit.value()
        try:coefficients=[float(v) for v in split(self.coefficients.text())]
        except ValueError:
            if self.td_method.currentText()!='table':raise
            coefficients=deepcopy(cfg['time_depth'].get('coefficients',[]))
        cfg["time_depth"].update(method=self.td_method.currentText(), coefficients=coefficients,
            table_file=self.td_file.text().strip(), time_unit=self.td_unit.currentText(), time_convention=self.td_convention.currentText(),
            depth_unit=self.td_depth_unit.currentText())
        if not cfg["algorithm"]["hidden_sizes"] or any(v <= 0 for v in cfg["algorithm"]["hidden_sizes"]):
            raise ValueError("隐藏层必须为正整数列表")
        strip_depth_metadata(cfg); validate_settings(cfg)
        return cfg

    def save_settings(self):
        try:
            cfg = self.collect_settings()
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
                if project.settings["algorithm"]["name"] != "mlp":
                    raise ValueError("当前界面仅支持 MLP 项目")
                self.project = project; self.settings = deepcopy(project.settings); self.populate()
            except Exception as exc: QMessageBox.warning(self, "无法打开项目", str(exc))

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
        if command in ('train','cross-validate','audit','refit') and self.library.worker is not None:
            QMessageBox.information(self,'正在导入井数据','请等待井库导入完成。');return
        if command in ('train','cross-validate','audit','refit') and not self.library.active and self.train_files.active and not self.train_files.ready():
            QMessageBox.information(self,'先检查批量文件','请先检查批量文件并等待完成；缺少字段或无法读取的文件请修正或取消勾选。');return
        if command != "prepare-seismic-view" and not self.save_settings(): return
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
        self.cancel_button.setEnabled(command in ("train", "cross-validate", "refit", "predict-seismic", "prepare-seismic-view", "predict-wells"))
        process = QProcess(self); self.process = process
        environment = QProcessEnvironment.systemEnvironment(); environment.insert("PYTHONIOENCODING", "utf-8")
        cache = ROOT / "outputs" / "matplotlib_cache"; cache.mkdir(parents=True, exist_ok=True)
        environment.insert("MPLCONFIGDIR", str(cache))
        process.setProcessEnvironment(environment); process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.cancel_path = ROOT / "outputs" / "tasks" / f"{uuid4().hex}.cancel"
        self.cancel_path.parent.mkdir(parents=True, exist_ok=True)
        if command in ("train", "cross-validate", "refit", "predict-seismic", "prepare-seismic-view", "predict-wells"): extra = [*extra, "--cancel-file", str(self.cancel_path)]
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
            try:
                if artifact not in ("seismic_trial", "seismic_view", "seismic_volume", "seismic_index"): self.charts.load_directory(event["path"])
            except (ValueError, OSError): pass
        elif kind == "fold_started":
            self.cv_fold = event["fold"]; self.cv_folds = event["folds"]
            self.log.appendPlainText(f"留一井验证 {event['fold']}/{event['folds']}：留出 {event['validation_well']}")
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

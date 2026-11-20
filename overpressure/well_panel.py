"""Simple prediction inputs, with optional measurement points and advanced mapping."""
from pathlib import Path
import re
from PySide6.QtCore import Qt,QSize
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QMessageBox,QListWidget,QListWidgetItem,QSizePolicy,QListView)
from .config import FEATURES, UNITS
from .data import read_table
from .well_points import prediction_defaults
from .ui_style import Card
from .batch_files import BatchFilesPanel
from .well_io import infer_columns


class WellPredictionPanel(QWidget):
    def __init__(self, owner,model_form=None):
        super().__init__(); self.owner = owner
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop);self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Maximum)
        basic = Card();self.basic_card=basic;form = QVBoxLayout(basic);form.setAlignment(Qt.AlignmentFlag.AlignTop)
        if model_form is not None:form.addLayout(model_form)
        self.library_box=QWidget();library_layout=QVBoxLayout(self.library_box);library_layout.setContentsMargins(0,0,0,0)
        library_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        selection_row=QHBoxLayout();selection_row.addWidget(QLabel('待预测井'));self.selection_count=QLabel();self.selection_count.setObjectName('hint');selection_row.addStretch();selection_row.addWidget(self.selection_count);library_layout.addLayout(selection_row)
        self.library_list=QListWidget();self.library_list.setMinimumHeight(95);self.library_list.setMaximumHeight(140);self.library_list.setToolTip('所需输入曲线及顺序由模型决定；在左侧井对象中导入和管理曲线。');library_layout.addWidget(self.library_list);form.addWidget(self.library_box);self.library_box.hide()
        self.library_list.itemChanged.connect(self.update_selection_count)
        self.library_list.setFlow(QListView.Flow.LeftToRight);self.library_list.setWrapping(True);self.library_list.setResizeMode(QListView.ResizeMode.Adjust);self.library_list.setGridSize(QSize(165,34))
        self.log_files=BatchFilesPanel(owner,lambda:{k:e.text().strip() for k,e in self.columns.items()},('well',*FEATURES),'新井批量预测文件')
        form.addWidget(self.log_files)
        self.point_files=BatchFilesPanel(owner,lambda:{k:e.text().strip() for k,e in self.point_columns.items()},('well','Depth','Pp'),'实测压力点文件（可选，可包含多口井）')
        form.addWidget(self.point_files)
        hint = QLabel('有单独实测点时仅用这些点验证，否则使用井中压力曲线；无标签时只预测。');hint.setObjectName('hint')
        hint.setWordWrap(True); form.addWidget(hint)
        self.independent = QCheckBox('确认实测点未参与训练或调参');self.independent.setToolTip('用于声明验证测点的独立性；参与训练的井仍根据模型记录标为复核。'); form.addWidget(self.independent)
        layout.addWidget(basic)
        toggle = QCheckBox("字段、单位与深度匹配设置（默认映射不符时展开）");self.mapping_toggle=toggle; layout.addWidget(toggle)
        self.advanced = QWidget(); advanced = QVBoxLayout(self.advanced); advanced.setContentsMargins(0, 0, 0, 0)
        self.columns, self.units, self.point_columns, self.point_units = {}, {}, {}, {}
        for measured, fields, edits, unit_edits, title in ((False, ("well", *FEATURES, "Pp"), self.columns, self.units, "测井数据列名与源单位"),
                (True, ("well", "Depth", "Pp"), self.point_columns, self.point_units, "单独实测点列名与源单位")):
            group = Card(title); fields_form = QFormLayout(group)
            for field in fields:
                row = QWidget(); line = QHBoxLayout(row); line.setContentsMargins(0, 0, 0, 0)
                edit = QLineEdit(); edits[field] = edit; line.addWidget(edit)
                if field != "well":
                    combo = QComboBox(); combo.addItems(list(UNITS[field])); unit_edits[field] = combo; line.addWidget(combo)
                fields_form.addRow("井名" if field == "well" else field + ("（可选）" if field == "Pp" and not measured else ""), row)
            button = QPushButton("从文件识别常见列名")
            button.clicked.connect(lambda _, measured=measured: self.detect_columns(measured)); fields_form.addRow(button)
            advanced.addWidget(group)
        gap_form = QFormLayout(); self.gap = QDoubleSpinBox(); self.gap.setRange(0, 100000); self.gap.setDecimals(3); self.gap.setSpecialValueText("自动：5 倍中位采样间隔")
        gap_form.addRow("插值最大深度间隔（m）", self.gap); advanced.addLayout(gap_form)
        note = QLabel("只在同井相邻有效预测点之间插值，不外推、不跨无效点或大间断。自动阈值不等于地质标定；采样很稀疏时建议手动设置。两类数据先按各自源单位转为 m / MPa。")
        note.setWordWrap(True); advanced.addWidget(note)
        layout.addWidget(self.advanced); self.advanced.hide(); toggle.toggled.connect(self.advanced.setVisible)
        footer=QHBoxLayout();layout.addLayout(footer)
        template = QPushButton('导出输入模板…');template.setToolTip('导出测井数据和实测压力点的空白模板'); template.clicked.connect(self.export_templates);footer.addWidget(template);footer.addStretch()
        self.start_button=owner.button(footer,'开始预测与验证',lambda: owner.start_task('predict-wells',['--bundle',owner.bundle_path.text(),'--generic']))
        self.library_gap=QDoubleSpinBox();self.library_gap.setRange(0,100000);self.library_gap.setDecimals(3);self.library_gap.setSpecialValueText('自动')
        self.library_gap.setMaximumWidth(220)
        self.library_gap.valueChanged.connect(self.gap.setValue);self.gap.valueChanged.connect(self.library_gap.setValue)
        library_form=QFormLayout();library_form.addRow('最大匹配间隔（m）',self.library_gap);library_layout.addLayout(library_form)
        self.library_gap.setToolTip('0 为自动：使用 5 倍中位深度采样间隔。仅匹配同井相邻有效预测点，不外推、不跨无效区或过大间隔。')
        note=QLabel('自动值为 5 倍中位采样间隔；测点匹配不外推、不跨无效区。');note.setObjectName('hint');note.setWordWrap(True);library_layout.addWidget(note)
        self.log_files.changed.connect(lambda:self.independent.setChecked(False));self.point_files.changed.connect(lambda:self.independent.setChecked(False))

    def update_selection_count(self,*_):
        selected=sum(self.library_list.item(i).checkState()==Qt.CheckState.Checked for i in range(self.library_list.count()))
        self.selection_count.setText(f'已选 {selected} / {self.library_list.count()} 口井')

    def populate(self, project):
        cfg = prediction_defaults(project)
        self.log_files.set_files(cfg.get('well_imports',cfg.get('well_files',[])),cfg.get('well_files'),active=True)
        self.point_files.set_files(cfg.get('measurement_imports',cfg.get('measurement_files',[])),cfg.get('measurement_files'),active=True)
        self.independent.setChecked(cfg["independence_confirmed"]); self.gap.setValue(cfg["max_gap_m"])
        for edits, values in ((self.columns, cfg["columns"]), (self.point_columns, cfg["measurement_columns"])):
            for field, edit in edits.items(): edit.setText(values[field])
        for edits, values in ((self.units, cfg["units"]), (self.point_units, cfg["measurement_units"])):
            for field, edit in edits.items(): edit.setCurrentText(values[field])

    def collect(self):
        cfg={
                "columns": {key: edit.text().strip() for key, edit in self.columns.items()}, "units": {key: edit.currentText() for key, edit in self.units.items()},
                "measurement_columns": {key: edit.text().strip() for key, edit in self.point_columns.items()},
                "measurement_units": {key: edit.currentText() for key, edit in self.point_units.items()},
                "independence_confirmed": self.independent.isChecked(), "max_gap_m": self.gap.value()}
        cfg.update(well_files=self.log_files.files(),well_imports=self.log_files.all_files())
        cfg.update(measurement_files=self.point_files.files(),measurement_imports=self.point_files.all_files())
        if hasattr(self.owner,'library') and self.owner.library.active:
            cfg.update(use_library=True,library_wells=[self.library_list.item(i).text() for i in range(self.library_list.count()) if self.library_list.item(i).checkState()==Qt.CheckState.Checked])
        return cfg

    def refresh_library(self,selected=None):
        if not hasattr(self.owner,'library'):return
        active=self.owner.library.active
        self.library_box.setVisible(active);self.log_files.setVisible(not active);self.point_files.setVisible(not active)
        self.mapping_toggle.setVisible(not active)
        if active:self.advanced.hide()
        selected=selected if selected is not None else [self.library_list.item(i).text() for i in range(self.library_list.count()) if self.library_list.item(i).checkState()==Qt.CheckState.Checked]
        self.library_list.clear()
        for well in self.owner.library.wells:
            item=QListWidgetItem(well['name']);item.setToolTip(well['name']);item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable);item.setCheckState(Qt.CheckState.Checked if well['name'] in selected else Qt.CheckState.Unchecked);self.library_list.addItem(item)
        self.update_selection_count()

    def detect_columns(self, measured=False):
        panel=self.point_files if measured else self.log_files
        edits = self.point_columns if measured else self.columns
        try:
            if not panel.files():raise ValueError('请先在文件清单中添加并勾选文件')
            columns = infer_columns(read_table(self.owner.project.resolve(panel.files()[0]),nrows=0).columns, {key: edit.text() for key, edit in edits.items()})
            for key, edit in edits.items(): edit.setText(columns[key])
            QMessageBox.information(self, "列名已识别", "已填写唯一匹配的常见列名；请核对源单位，未唯一识别的列保持原值。")
        except Exception as exc: QMessageBox.warning(self, "无法识别列名", str(exc))

    def export_templates(self):
        path = QFileDialog.getExistingDirectory(self, "选择模板保存目录")
        if path:
            try:
                folder = Path(path)
                # Keep templates blank so demonstration values cannot be mistaken for observations.
                from datetime import datetime
                target = folder / ("井数据模板_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")); target.mkdir()
                (target / "测井数据模板.csv").write_text("井,Depth,AC,DEN,Ip,Pp\n", encoding="utf-8-sig")
                (target / "实测压力点模板.csv").write_text("井,Depth,Pp\n", encoding="utf-8-sig")
                (target / "填写说明.txt").write_text("每行一个采样点，井名按文本填写并保持一致。测井表 Pp 可留空或删除，不用零代替未知压力。实测点表只填真实测点，不插补压力。请设置各列源单位。单独实测点文件存在时仅用它验证。无实测点时只预测，不生成虚构指标。", encoding="utf-8")
                QMessageBox.information(self, "模板已导出", str(target))
            except Exception as exc: QMessageBox.warning(self, "模板导出失败", str(exc))

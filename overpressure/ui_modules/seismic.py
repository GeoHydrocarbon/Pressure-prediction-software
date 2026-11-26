"""SeismicModuleUI controls and interactions."""
from PySide6.QtWidgets import QWidget, QFormLayout, QCheckBox, QSpinBox, QLabel, QPushButton, QMessageBox
from ..ui_style import Card
from ..config import UNITS

class SeismicModuleUI:
    def build_seismic(self):
        page = self.page("地震压力预测", "填写 Inline、Crossline 起止线号，预测矩形范围内已有的全部道（包含边界）；也可预测全体。按道分块计算，保留原始时间采样与道头，缺失交点不补道。")
        card=Card('预测模型与空间范围'); form = QFormLayout(card)
        self.seismic_model_choice = self.combo([]); form.addRow('项目模型', self.seismic_model_choice)
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


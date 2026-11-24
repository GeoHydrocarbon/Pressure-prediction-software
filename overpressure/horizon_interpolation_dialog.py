"""Parameters for a derived horizon; computation runs in the SEG-Y reader."""
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QLabel,QComboBox,QDoubleSpinBox,QLineEdit,QDialogButtonBox,QMessageBox
from .ui_style import polish_layouts,WindowBoundsGuard

class HorizonInterpolationDialog(QDialog):
    def __init__(self,parent,entries,selected=None):
        super().__init__(parent);self.entries=entries;self.setWindowTitle('层位空间插值');self.resize(540,430)
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.layer=QComboBox();self.layer.setObjectName('interpolation_source')
        for key,entry in entries.items():self.layer.addItem(entry['label'],key)
        if selected in entries:self.layer.setCurrentIndex(self.layer.findData(selected))
        form.addRow('原始层位',self.layer)
        self.name=QLineEdit();self.name.setObjectName('interpolation_name');form.addRow('新层位名称',self.name)
        form.addRow('插值方法',QLabel('线性插值（三角网）'))
        self.coordinates=QComboBox();self.coordinates.setObjectName('interpolation_coordinates');self.coordinates.addItems(['Inline / Crossline 网格','地震 XY 平面坐标']);form.addRow('计算距离使用的坐标',self.coordinates)
        self.span=QDoubleSpinBox();self.span.setObjectName('interpolation_max_span');self.span.setRange(.01,1e9);self.span.setDecimals(2);self.span.setValue(10);self.span.setKeyboardTracking(False)
        self.span_label=QLabel('最大插值跨度（线号网格距离）');form.addRow(self.span_label,self.span)
        self.span.setToolTip('用于补点的三角形，每条边都不能超过此值；跨度越小，允许填补的缺口越小。')
        note=QLabel('仅在原始层位点的凸包内补点，不向外外推；最大跨度限制可跨越的缺口。只生成已有地震道上的层位时间，不补地震道，也不插值压力。\n\n原始层位保留，结果生成独立层位，可切换比较。本版不自动识别断层，请避免使用过大的跨度跨越断层或真实边界。');note.setWordWrap(True);layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.button(QDialogButtonBox.StandardButton.Ok).setText('开始插值');buttons.accepted.connect(self.validate_accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        self.layer.currentIndexChanged.connect(self.update_name);self.coordinates.currentIndexChanged.connect(self.update_coordinates)
        self.update_name();polish_layouts(self);self.screen_guard=WindowBoundsGuard(self)
    def update_name(self):
        base=self.entries[self.layer.currentData()]['name']+'（插值）';name=base;number=2;existing={entry['name'] for entry in self.parent().horizon_entries.values()}
        while name in existing:name=f'{base} {number}';number+=1
        self.name.setText(name)
    def update_coordinates(self):
        self.span_label.setText('最大插值跨度（地震 XY 坐标单位）' if self.coordinates.currentIndex() else '最大插值跨度（线号网格距离）');self.span.setValue(100 if self.coordinates.currentIndex() else 10)
    def parameters(self):
        entry=self.entries[self.layer.currentData()]
        return dict(horizon_path=entry['path'],horizon_name=entry['name'],new_name=self.name.text().strip(),max_span=self.span.value(),coordinates='xy' if self.coordinates.currentIndex() else 'lines')
    def validate_accept(self):
        if not self.name.text().strip() or self.name.text().strip() in {entry['name'] for entry in self.parent().horizon_entries.values()}:
            QMessageBox.warning(self,'检查新层位名称','请输入未使用的新层位名称，以保留原始层位。');return
        self.accept()

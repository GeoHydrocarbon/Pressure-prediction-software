"""Preview source rows before mapping wellheads or directional surveys."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QFormLayout,QLabel,QComboBox,QCheckBox,
    QDialogButtonBox,QSpinBox,QTableWidget,QTableWidgetItem,QTabWidget,QScrollArea,QWidget,QAbstractItemView)
from .spatial_table import DEFAULT_IMPORT,preview_source,read_spatial_table
from .ui_style import polish_layouts,WindowBoundsGuard


class AssetDialog(QDialog):
    def __init__(self,parent,path,kind,forced_well='',initial=None):
        super().__init__(parent);self.path=str(Path(path).resolve());self.kind=kind;self.forced_well=forced_well
        initial=initial or {};options=dict(DEFAULT_IMPORT,**initial.get('table_import',{}))
        self.setWindowTitle('导入井口坐标' if kind=='coordinates' else '导入井轨迹' if kind=='trajectory' else '导入井对比曲线')
        self.resize(880,820);layout=QVBoxLayout(self);filename=QLabel(self.path);filename.setWordWrap(True);layout.addWidget(filename)
        body=QWidget();form=QFormLayout(body);self.form=form
        self.header_row=QSpinBox();self.header_row.setRange(0,1000000);self.header_row.setSpecialValueText('无表头（自动生成列名）');self.header_row.setValue(options['header_row'])
        self.data_row=QSpinBox();self.data_row.setRange(1,1000000);self.data_row.setValue(options['data_row'])
        self.header_row.setKeyboardTracking(False);self.data_row.setKeyboardTracking(False)
        form.addRow('表头所在行（原始文件行号）',self.header_row);form.addRow('数据起始行（原始文件行号）',self.data_row)
        self.delimiter=QComboBox()
        for title,value in [('自动识别','auto'),('空格 / 连续空白','whitespace'),('Tab','\t'),('逗号',','),('分号',';'),('竖线','|')]:self.delimiter.addItem(title,value)
        self.delimiter.setCurrentIndex(self.delimiter.findData(options['delimiter']));form.addRow('文本分隔符',self.delimiter)
        self.encoding=QComboBox()
        for title,value in [('自动（UTF-8 / GB18030）','auto'),('UTF-8','utf-8-sig'),('GB18030 / GBK','gb18030')]:self.encoding.addItem(title,value)
        self.encoding.setCurrentIndex(self.encoding.findData(options['encoding']));form.addRow('文本编码',self.encoding)
        excel=Path(path).suffix.lower()=='.xlsx';self.delimiter.setEnabled(not excel);self.encoding.setEnabled(not excel)
        self.trajectory_mode=QComboBox();self.trajectory_mode.addItem('深度 + X / Y（现有方式）','xy');self.trajectory_mode.addItem('MD + 方位角 + 井斜角（最小曲率法）','survey')
        self.trajectory_mode.setCurrentIndex(max(0,self.trajectory_mode.findData(initial.get('trajectory_mode','xy'))))
        if kind=='trajectory':form.addRow('轨迹导入方式',self.trajectory_mode)
        self.columns={}
        keys=['well']+([] if kind=='coordinates' else ['depth'])+(['x','y'] if kind!='curves' else [])+(['azimuth','inclination'] if kind=='trajectory' else [])
        titles=dict(well='井名列',depth='深度 / 测量井深 MD 列',x='X / 东向偏移列',y='Y / 北向偏移列',azimuth='方位角列',inclination='井斜角列')
        for key in keys:
            combo=QComboBox();self.columns[key]=combo;form.addRow(titles[key],combo)
        self.depth_unit=QComboBox();self.depth_unit.addItems(['m','ft']);self.depth_unit.setCurrentText(initial.get('depth_unit','m'))
        if kind!='coordinates':form.addRow('深度单位',self.depth_unit)
        self.relative=QCheckBox('轨迹 XY 是相对井口的偏移量');self.relative.setChecked(initial.get('relative',False))
        self.angle_unit=QComboBox();self.angle_unit.addItems(['deg','rad']);self.angle_unit.setCurrentText(initial.get('angle_unit','deg'))
        self.initial_segment=QComboBox();self.initial_segment.addItem('要求每口井首个测点 MD = 0','require_zero');self.initial_segment.addItem('首段沿首测点方向（从井口延伸，属于假设）','first_direction')
        self.initial_segment.setCurrentIndex(max(0,self.initial_segment.findData(initial.get('initial_segment','require_zero'))))
        if kind=='trajectory':
            form.addRow(self.relative);form.addRow('角度单位',self.angle_unit);form.addRow('首个测点之前的轨迹',self.initial_segment)
        self.time=QComboBox();self.time_unit=QComboBox();self.time_unit.addItems(['ms','s']);self.time_unit.setCurrentText(initial.get('time_unit','ms'))
        self.convention=QComboBox();self.convention.addItems(['twt','owt']);self.convention.setCurrentText(initial.get('time_convention','twt'))
        self.pressure=QComboBox();self.pressure_unit=QComboBox();self.pressure_unit.addItems(['MPa','kPa','psi']);self.pressure_unit.setCurrentText(initial.get('pressure_unit','MPa'))
        if kind=='curves':
            form.addRow('时间列（留空使用项目时深关系）',self.time);form.addRow('时间单位',self.time_unit);form.addRow('单 / 双程时间',self.convention)
            form.addRow('压力列（可不选）',self.pressure);form.addRow('压力单位',self.pressure_unit)
        self.hint=QLabel();self.hint.setWordWrap(True);form.addRow(self.hint)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(body);layout.addWidget(scroll,1)
        tabs=QTabWidget();self.preview_tabs=tabs;self.source_preview=QTableWidget(0,2);self.source_preview.setHorizontalHeaderLabels(['原始行号','文件内容'])
        self.table_preview=QTableWidget();tabs.addTab(self.source_preview,'原始文件（前 60 行）');tabs.addTab(self.table_preview,'解析后的数据（前 20 条）')
        for table in (self.source_preview,self.table_preview):table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);table.setMinimumHeight(130)
        tabs.setMaximumHeight(250);layout.addWidget(tabs)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText('确认导入');self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject);layout.addWidget(self.buttons)
        self.initial=initial;self.pending_columns=dict(initial.get('columns',{}));self.last_headers=None;self.parsed=None
        for edit in (self.header_row,self.data_row):edit.valueChanged.connect(self.refresh_preview)
        for combo in (self.delimiter,self.encoding):combo.currentIndexChanged.connect(self.refresh_preview)
        for combo in self.columns.values():combo.currentIndexChanged.connect(self.validate_preview)
        self.trajectory_mode.currentIndexChanged.connect(self.update_mode)
        for combo in (self.depth_unit,self.angle_unit,self.initial_segment):combo.currentIndexChanged.connect(self.validate_preview)
        self.update_mode();self.refresh_preview();polish_layouts(self);self.screen_guard=WindowBoundsGuard(self)

    def import_options(self):
        return dict(header_row=self.header_row.value(),data_row=self.data_row.value(),delimiter=self.delimiter.currentData(),encoding=self.encoding.currentData())

    def update_mode(self,*_):
        survey=self.kind=='trajectory' and self.trajectory_mode.currentData()=='survey'
        if self.kind=='trajectory':
            for key in ('x','y'):self.form.setRowVisible(self.columns[key],not survey)
            for key in ('azimuth','inclination'):self.form.setRowVisible(self.columns[key],survey)
            self.form.setRowVisible(self.relative,not survey)
            self.form.setRowVisible(self.angle_unit,survey);self.form.setRowVisible(self.initial_segment,survey)
        self.hint.setText(('MD 为沿井眼深度，每口井须严格递增；井斜角 0° 为垂直向下；方位角从北顺时针，须与平面坐标北向一致。X=东、Y=北，计算偏移单位为 m，井口与地震 XY 也须为 m。按 MD 关联测井；不自动更改项目时深关系。' if survey else '文件可包含多口井，按井名关联。XY 必须与地震缩放后的平面坐标及单位一致。轨迹覆盖外不外推。')+(' 当前关联井：'+self.forced_well+'；单井文件无井名列时可留空。' if self.forced_well else ''))
        self.validate_preview()

    def refresh_preview(self,*_):
        self.parsed=None
        try:
            rows=preview_source(self.path,self.encoding.currentData());self.source_preview.setRowCount(len(rows))
            for i,(number,text) in enumerate(rows):
                self.source_preview.setItem(i,0,QTableWidgetItem(str(number)));self.source_preview.setItem(i,1,QTableWidgetItem(text))
            self.source_preview.setColumnWidth(0,80);self.source_preview.horizontalHeader().setStretchLastSection(True)
            frame=read_spatial_table(self.path,self.import_options(),nrows=20);names=list(frame.columns)
            if names!=self.last_headers:
                aliases=dict(well=['well','井名','井','Well'],depth=['MD','Depth','depth','测深','测量井深','深度'],x=['X','x','Easting','东坐标'],y=['Y','y','Northing','北坐标'],azimuth=['Azimuth','AZIM','AZI','Azi','azimuth','方位角','方位'],inclination=['Inclination','INC','Inc','inclination','井斜角','井斜'])
                for key,combo in self.columns.items():
                    old=self.pending_columns.get(key,combo.currentText());combo.blockSignals(True);combo.clear()
                    combo.addItems(([''] if key=='well' and self.forced_well else [])+names)
                    explicitly_blank=key in self.pending_columns and old=='' and key=='well' and self.forced_well
                    chosen=old if old in names or explicitly_blank else next((a for a in aliases[key] if a in names),'' if key=='well' and self.forced_well else names[0] if names else '')
                    combo.setCurrentText(chosen);combo.blockSignals(False)
                for key,combo in (('time_column',self.time),('pressure_column',self.pressure)):
                    old=self.initial.get(key,combo.currentText());combo.clear();combo.addItems(['']+names);combo.setCurrentText(old if old in names else '')
                self.last_headers=names;self.pending_columns={}
            self.parsed=frame;self.table_preview.setColumnCount(len(names));self.table_preview.setHorizontalHeaderLabels(names);self.table_preview.setRowCount(len(frame))
            self.table_preview.setVerticalHeaderLabels([str(row) for row in frame.attrs['source_rows']])
            for i,row in enumerate(frame.itertuples(index=False,name=None)):
                for j,value in enumerate(row):self.table_preview.setItem(i,j,QTableWidgetItem('' if value is None else str(value)))
            self.validate_preview()
        except Exception as exc:
            self.table_preview.setRowCount(0);self.status.setText('无法解析：'+str(exc));self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def config(self):
        survey=self.kind=='trajectory' and self.trajectory_mode.currentData()=='survey'
        keys=['well']+([] if self.kind=='coordinates' else ['depth'])+(['azimuth','inclination'] if survey else ['x','y'] if self.kind!='curves' else [])
        return dict(path=self.path,kind=self.kind,columns={key:self.columns[key].currentText() for key in keys},well_name=self.forced_well,
                    table_import=self.import_options(),trajectory_mode='survey' if survey else 'xy',depth_unit=self.depth_unit.currentText(),
                    relative=survey or self.relative.isChecked(),angle_unit=self.angle_unit.currentText(),initial_segment=self.initial_segment.currentData(),
                    time_column=self.time.currentText(),time_unit=self.time_unit.currentText(),time_convention=self.convention.currentText(),pressure_column=self.pressure.currentText(),pressure_unit=self.pressure_unit.currentText())

    def validate_preview(self,*_):
        if not hasattr(self,'parsed') or self.parsed is None:return
        try:
            cfg=self.config();selected=[v for v in cfg['columns'].values() if v]
            if len(selected)!=len(set(selected)):raise ValueError('各字段不能选择同一列，请核对字段对应')
            if self.parsed.empty:raise ValueError('数据起始行之后没有可导入记录')
            self.status.setText('预览已更新；解析数据左侧为原始行号。确认时检查完整文件。')
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        except Exception as exc:
            self.status.setText(str(exc));self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def accept(self):
        try:
            from .well_spatial import read_asset
            _,frame=read_asset(self.config())
            if frame.empty or not frame.well.notna().any():raise ValueError('没有可关联的井名或数据')
            fields=['x','y'] if self.kind=='coordinates' else ['depth','x','y'] if self.kind=='trajectory' else ['depth']
            valid=frame.well.notna().to_numpy() & np.isfinite(frame[fields].to_numpy()).all(axis=1)
            if 'depth' in fields:valid &= frame.depth.to_numpy()>=0
            if self.kind in ('coordinates','trajectory') and not valid.all():
                rows=frame.loc[~valid,'source_row'].astype(str).tolist()
                raise ValueError('原始第 '+', '.join(rows[:8])+' 行包含空井名或无效坐标 / 深度，请检查字段、起始行和文件内容')
            if not valid.any():raise ValueError('没有有效的坐标 / 轨迹数据，请检查字段和起始行')
            super().accept()
        except Exception as exc:self.status.setText('不能导入：'+str(exc))

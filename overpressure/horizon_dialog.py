"""Column mapping and explicit time/XY settings for a horizon file."""
import re
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QLineEdit,
    QPushButton, QFileDialog, QComboBox, QLabel, QDoubleSpinBox, QDialogButtonBox, QMessageBox,QSpinBox,QWidget,QTableWidget,QTableWidgetItem,QAbstractItemView)
from .horizon_io import read_horizon_table
from .ui_style import polish_layouts,PageScroll,WindowBoundsGuard


class HorizonImportDialog(QDialog):
    def __init__(self, parent, convention="twt"):
        super().__init__(parent); self.setWindowTitle("导入层位 · CSV / Excel / TXT / UDF"); self.resize(720, 720)
        layout = QVBoxLayout(self);body=QWidget();form = QFormLayout(body);layout.addWidget(PageScroll(body),1)
        row = QHBoxLayout(); self.path = QLineEdit(); row.addWidget(self.path)
        choose = QPushButton("选择文件"); choose.clicked.connect(self.choose); row.addWidget(choose); form.addRow("层位文件", row)
        self.separator=QComboBox();self.separator.addItems(['自动识别','空格 / 连续空白','逗号','制表符 Tab','分号','竖线 |']);form.addRow('TXT / UDF 分隔符',self.separator)
        self.header_mode=QComboBox();self.header_mode.addItems(['自动识别','首行是列名','没有列名（按第 1、2…列）']);form.addRow('TXT / UDF 列名',self.header_mode)
        self.skip_rows=QSpinBox();self.skip_rows.setRange(0,100000);self.skip_rows.setKeyboardTracking(False);form.addRow('跳过开头说明行数',self.skip_rows)
        self.preview=QTableWidget();self.preview.setMaximumHeight(155);self.preview.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);form.addRow('前 5 行预览',self.preview)
        self._preview_valid=False
        self.coordinates = QComboBox(); self.coordinates.addItems(["Inline / Xline / 时间", "X / Y / 时间"]); form.addRow("坐标形式", self.coordinates)
        self.columns = {}
        self.column_labels = {}
        for key, label in (("inline", "Inline 列"), ("crossline", "Xline 列"), ("x", "X 坐标列"), ("y", "Y 坐标列"), ("time", "层位时间列"), ("name", "层位名称列（可选）")):
            combo = QComboBox(); combo.setEditable(True); self.columns[key] = combo; form.addRow(label, combo); self.column_labels[key] = form.labelForField(combo)
        self.unit = QComboBox(); self.unit.addItems(["ms", "s"]); form.addRow("层位时间单位", self.unit)
        self.convention = QComboBox(); self.convention.addItems(["twt", "owt"]); self.convention.setCurrentText(convention); form.addRow("双程 / 单程时间", self.convention)
        self.distance = QDoubleSpinBox(); self.distance.setRange(0, 1e9); self.distance.setDecimals(4); self.distance.setValue(25)
        form.addRow("XY 最大匹配距离（坐标单位）", self.distance); self.distance_label = form.labelForField(self.distance)
        note = QLabel("TXT / 文本表格型 UDF 支持有列名或无列名；无列名时请根据预览对应各列，特别注意 Inline / Crossline 的顺序。#、!、// 开头的整行注释会忽略。名称列留空时使用文件名作为层位名称。XY 必须与地震使用相同平面坐标系和长度单位，不自动转换坐标系；不进行空间插值补洞。")
        note.setWordWrap(True); form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("导入")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept_checked); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.coordinates.currentIndexChanged.connect(self.toggle); self.toggle()
        self.path.editingFinished.connect(lambda: self.read_headers() if self.path.text().strip() else None)
        self.path.textChanged.connect(lambda *_:setattr(self,'_preview_valid',False))
        for widget in (self.separator,self.header_mode):widget.currentIndexChanged.connect(lambda *_:self.read_headers() if self.path.text().strip() else None)
        self.skip_rows.valueChanged.connect(lambda *_:self.read_headers() if self.path.text().strip() else None)
        polish_layouts(self)
        self.screen_guard=WindowBoundsGuard(self)

    def toggle(self):
        xy = self.coordinates.currentIndex() == 1
        for key in ("inline", "crossline", "x", "y"):
            show = xy if key in ("x", "y") else not xy
            self.columns[key].setVisible(show); self.column_labels[key].setVisible(show)
        self.distance.setVisible(xy); self.distance_label.setVisible(xy)

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择层位文件", "", "层位数据 (*.csv *.xlsx *.txt *.udf);;文本层位 (*.txt *.udf);;表格 (*.csv *.xlsx)")
        if path: self.path.setText(path); self.read_headers()

    def read_headers(self):
        self._preview_valid=False
        try:
            from pathlib import Path
            text=Path(self.path.text().strip()).suffix.lower() in ('.txt','.udf')
            for widget in (self.separator,self.header_mode,self.skip_rows):widget.setEnabled(text)
            data=read_horizon_table(Path(self.path.text().strip()),self.text_options(),nrows=5)
            if data.empty:raise ValueError('层位文件没有数据行')
            headers=[str(value) for value in data.columns]
            self.preview.clear();self.preview.setRowCount(len(data));self.preview.setColumnCount(len(headers));self.preview.setHorizontalHeaderLabels(headers)
            for row,values in enumerate(data.itertuples(index=False,name=None)):
                for column,value in enumerate(values):self.preview.setItem(row,column,QTableWidgetItem(str(value)))
            self.preview.resizeColumnsToContents()
            aliases = {"inline": ["inline", "iline", "il", "主测线"], "crossline": ["crossline", "xline", "xl", "联络线"],
                       "x": ["x", "easting", "xcoordinate", "x坐标"], "y": ["y", "northing", "ycoordinate", "y坐标"],
                       "time": ["time", "t", "twt", "owt", "时间", "层位时间", "timems"], "name": ["name", "horizon", "layer", "层位", "层位名称"]}
            normalize = lambda value: re.sub(r"[\s_\-()]+", "", value).lower()
            for key, combo in self.columns.items():
                combo.clear(); combo.addItems(([""] if key == "name" else []) + headers)
                found = [header for header in headers if normalize(header) in aliases[key]]
                combo.setCurrentText(found[0] if len(found) == 1 else "")
            if headers and headers[0]=='第1列' and len(headers)==3:
                for key,index in [('inline',0),('crossline',1),('x',0),('y',1),('time',2)]:self.columns[key].setCurrentText(headers[index])
            elif headers and headers[0]=='第1列' and len(headers)==5:
                for key,index in [('x',2),('y',3),('time',4)]:self.columns[key].setCurrentText(headers[index])
            self._preview_valid=True
        except Exception as exc: QMessageBox.warning(self, "无法读取层位文件", str(exc))

    def text_options(self):
        return dict(delimiter=('auto','whitespace','comma','tab','semicolon','pipe')[self.separator.currentIndex()],
                    header=('auto','present','none')[self.header_mode.currentIndex()],skip_rows=self.skip_rows.value())

    def config(self):
        from pathlib import Path
        return {"path": self.path.text().strip(), "coordinates": "xy" if self.coordinates.currentIndex() else "lines",
                "columns": {key: combo.currentText().strip() for key, combo in self.columns.items()},
                "time_unit": self.unit.currentText(), "time_convention": self.convention.currentText(), "max_distance": self.distance.value(),'text_options':self.text_options(),'default_name':Path(self.path.text().strip()).stem}

    def accept_checked(self):
        if not self._preview_valid:
            self.read_headers()
            if not self._preview_valid:return
        cfg = self.config(); fields = ("x", "y", "time") if cfg["coordinates"] == "xy" else ("inline", "crossline", "time")
        if not cfg["path"] or any(not cfg["columns"][key] for key in fields):
            QMessageBox.information(self, "需要选择字段", "请选择文件并填写坐标和时间列名。"); return
        self.accept()

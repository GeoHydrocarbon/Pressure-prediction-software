"""Per-curve track assignment and explicit line/point styles."""
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QLabel,QTableWidget,QTableWidgetItem,
    QSpinBox,QDoubleSpinBox,QComboBox,QPushButton,QColorDialog,QDialogButtonBox,QHeaderView)
from .ui_style import WindowBoundsGuard

KINDS=[('线','line'),('点','points'),('线＋点','both')]
LINES=[('实线','-'),('虚线','--'),('点划线','-.'),('点线',':')]
MARKERS=[('圆形','o'),('方形','s'),('上三角','^'),('下三角','v'),('菱形','D'),('加号','+'),('叉号','x'),('星形','*')]


class CurveSettingsDialog(QDialog):
    def __init__(self,parent,rows):
        super().__init__(parent)
        self.setWindowTitle('分道与曲线样式');self.resize(1120,570)
        self.rows=deepcopy(rows)
        layout=QVBoxLayout(self)
        hint=QLabel('相同道号的曲线叠加，不同道号独立显示；道号从左到右排列。例如 AC=1、密度=2、预测压力=3、实测压力=3。\n每条曲线可独立选择线、点或线＋点。点大小单位为 pt；同道不同单位仍使用独立横轴。')
        hint.setWordWrap(True);layout.addWidget(hint)
        self.table=QTableWidget(len(rows),9)
        self.table.setHorizontalHeaderLabels(['曲线','道号','绘制方式','线颜色','线型','线宽（pt）','点颜色','点形状','点大小（pt）'])
        self.table.verticalHeader().hide();self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.setAlternatingRowColors(True)
        self.controls=[]
        for index,row in enumerate(rows):
            name=QTableWidgetItem(row['name']);name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(index,0,name)
            track=QSpinBox();track.setRange(1,max(1,len(rows)));track.setValue(row['track'])
            kind=self.combo(KINDS,row['kind']);color=self.color_button(row['color'])
            line=self.combo(LINES,row['linestyle']);width=self.number(.1,12,row['linewidth'])
            point_color=self.color_button(row['marker_color']);marker=self.combo(MARKERS,row['marker'])
            size=self.number(1,40,row['markersize'])
            controls=dict(track=track,kind=kind,color=color,linestyle=line,linewidth=width,
                          marker_color=point_color,marker=marker,markersize=size)
            self.controls.append(controls)
            for col,key in enumerate(controls,1):self.table.setCellWidget(index,col,controls[key])
            kind.currentIndexChanged.connect(lambda _,r=index:self.update_enabled(r))
            self.update_enabled(index)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        for col in range(1,9):self.table.setColumnWidth(col,110 if col in (3,6) else 100)
        layout.addWidget(self.table,1)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('确定')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.screen_guard=WindowBoundsGuard(self)

    @staticmethod
    def combo(options,current):
        box=QComboBox()
        for label,value in options:box.addItem(label,value)
        box.setCurrentIndex(max(0,box.findData(current)))
        return box

    @staticmethod
    def number(low,high,value):
        spin=QDoubleSpinBox();spin.setRange(low,high);spin.setDecimals(1);spin.setValue(value)
        return spin

    def color_button(self,value):
        button=QPushButton();self.set_color(button,value)
        def choose():
            color=QColorDialog.getColor(QColor(button.property('curveColor')),self,'选择曲线颜色')
            if color.isValid():self.set_color(button,color.name())
        button.clicked.connect(choose)
        return button

    @staticmethod
    def set_color(button,value):
        button.setProperty('curveColor',value);button.setText(value)
        button.setStyleSheet(f'QPushButton {{ border: 2px solid {value}; }}')

    def update_enabled(self,row):
        controls=self.controls[row];kind=controls['kind'].currentData()
        for key in ('color','linestyle','linewidth'):controls[key].setEnabled(kind!='points')
        for key in ('marker_color','marker','markersize'):controls[key].setEnabled(kind!='line')

    def values(self):
        rows=deepcopy(self.rows)
        for row,controls in zip(rows,self.controls):
            for key,widget in controls.items():
                row[key]=(widget.currentData() if isinstance(widget,QComboBox) else
                          widget.property('curveColor') if isinstance(widget,QPushButton) else widget.value())
        return rows

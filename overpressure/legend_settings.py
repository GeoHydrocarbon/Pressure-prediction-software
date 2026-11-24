"""Legend settings for each visible well-log track, without renaming source curves."""
from copy import deepcopy
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QLabel,QTableWidget,QTableWidgetItem,
    QCheckBox,QComboBox,QDoubleSpinBox,QSpinBox,QDialogButtonBox,QHeaderView,QTabWidget,QWidget,QLineEdit)
from PySide6.QtCore import Qt
from .ui_style import WindowBoundsGuard

POSITIONS=[('自动位置','best'),('道内右上','upper right'),('道内左上','upper left'),
           ('道内右下','lower right'),('道内左下','lower left'),
           ('图框外上方','above'),('图框外下方','below'),('自定义（鼠标拖动）','custom')]


def legend_defaults(count):
    return dict(visible=count>1,position='best',fontsize=10.,alpha=.85,frame=True,columns=1,draggable=True,custom_xy=[.02,.75])


class LegendSettingsDialog(QDialog):
    def __init__(self,parent,tracks,labels):
        super().__init__(parent)
        self.setWindowTitle('图例设置');self.resize(1100,530)
        self.tracks=deepcopy(tracks)
        layout=QVBoxLayout(self)
        hint=QLabel('每个道可单独设置图例。建议选择“图框外上方”避免遮挡曲线；勾选“允许拖动”后，可按住图例移动。\n“图例名称”页只修改显示文字，不改变井库曲线名。导出保留图例位置与样式。')
        hint.setWordWrap(True);layout.addWidget(hint)
        tabs=QTabWidget();layout.addWidget(tabs,1)
        page=QWidget();body=QVBoxLayout(page)
        self.table=QTableWidget(len(tracks),8)
        self.table.setHorizontalHeaderLabels(['测井道','显示','位置','字号（pt）','背景不透明度','边框','列数','允许拖动'])
        self.table.verticalHeader().hide();self.table.verticalHeader().setDefaultSectionSize(42)
        self.controls=[]
        for index,track in enumerate(tracks):
            title=QTableWidgetItem(track['title']);title.setFlags(title.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(index,0,title)
            options=track['settings']
            visible=QCheckBox();visible.setChecked(options['visible'])
            position=QComboBox()
            for name,value in POSITIONS:position.addItem(name,value)
            position.setCurrentIndex(position.findData(options['position']))
            font=QDoubleSpinBox();font.setRange(5,36);font.setDecimals(1);font.setValue(options['fontsize'])
            alpha=QDoubleSpinBox();alpha.setRange(0,1);alpha.setDecimals(2);alpha.setSingleStep(.1);alpha.setValue(options['alpha'])
            alpha.setToolTip('0 完全透明，1 完全不透明；取消边框仍保留背景填充。')
            frame=QCheckBox();frame.setChecked(options['frame'])
            columns=QSpinBox();columns.setRange(1,max(1,track['count']));columns.setValue(options['columns'])
            draggable=QCheckBox();draggable.setChecked(options['draggable'])
            controls=dict(visible=visible,position=position,fontsize=font,alpha=alpha,frame=frame,columns=columns,draggable=draggable)
            self.controls.append(controls)
            for col,widget in enumerate(controls.values(),1):self.table.setCellWidget(index,col,widget)
        self.table.setColumnWidth(0,250);self.table.setColumnWidth(2,170)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        body.addWidget(self.table);tabs.addTab(page,'逐道图例')
        names=QWidget();name_layout=QVBoxLayout(names)
        self.names=QTableWidget(len(labels),2);self.names.setHorizontalHeaderLabels(['井库曲线名','图例显示名称'])
        self.names.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.names.verticalHeader().hide();self.names.verticalHeader().setDefaultSectionSize(42)
        self.label_edits={}
        for index,(name,label) in enumerate(labels.items()):
            item=QTableWidgetItem(name);item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable);self.names.setItem(index,0,item)
            edit=QLineEdit(label);edit.setPlaceholderText(name);self.names.setCellWidget(index,1,edit);self.label_edits[name]=edit
        name_layout.addWidget(self.names);tabs.addTab(names,'图例名称')
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('确定');buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.screen_guard=WindowBoundsGuard(self)

    def values(self):
        tracks=deepcopy(self.tracks)
        for track,controls in zip(tracks,self.controls):
            for name,widget in controls.items():
                track['settings'][name]=(widget.isChecked() if isinstance(widget,QCheckBox) else
                                         widget.currentData() if isinstance(widget,QComboBox) else widget.value())
        return tracks,{name:edit.text().strip() or name for name,edit in self.label_edits.items()}

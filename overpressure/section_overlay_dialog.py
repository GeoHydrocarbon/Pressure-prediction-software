"""Extensible per-object appearance settings for a section's overlay layers."""
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QTreeWidget,QTreeWidgetItem,
    QLabel,QDoubleSpinBox,QComboBox,QLineEdit,QPushButton,QCheckBox,QDialogButtonBox,QColorDialog,QMessageBox,QScrollArea,QWidget)
from matplotlib.colors import is_color_like
from .section_overlays import default_style,well_key,curve_key,selected_curves
from .ui_style import WindowBoundsGuard


class SectionOverlayDialog(QDialog):
    def __init__(self,panel):
        super().__init__(panel);self.panel=panel;self.state=deepcopy(panel.overlay_settings);self.styles=self.state.setdefault('styles',{});self.current=None;self.loading=False
        self.setWindowTitle('剖面叠加显示');self.resize(780,760);layout=QVBoxLayout(self)
        info=QLabel('按井口或轨迹到剖面的最近距离筛选井，入选后投影显示范围内的整条轨迹。勾选井或曲线控制显示；选择对象设置样式。曲线使用该井已应用的时深关系，按需读取。');info.setWordWrap(True);layout.addWidget(info)
        form=QFormLayout();self.distance=QDoubleSpinBox();self.distance.setRange(0,1e9);self.distance.setDecimals(3);self.distance.setValue(self.state.get('distance',25.))
        form.addRow('井到剖面最大距离（'+panel.meta.get('well_distance_unit','XY 坐标单位')+'）',self.distance)
        self.legend=QCheckBox('显示曲线图例');self.legend.setChecked(self.state.get('legend',True));form.addRow(self.legend);layout.addLayout(form)
        self.tree=QTreeWidget();self.tree.setHeaderLabels(['叠加对象']);self.tree.setMinimumHeight(150);layout.addWidget(self.tree,1)
        self.objects={}
        for well in panel.meta.get('well_overlays',[]):
            key=well_key(well['name']);self.styles.setdefault(key,default_style('well'));style=self.styles[key]
            root=QTreeWidgetItem([well['name']]);root.setData(0,Qt.ItemDataRole.UserRole,key);root.setCheckState(0,Qt.CheckState.Checked if style.get('visible',True) else Qt.CheckState.Unchecked);self.tree.addTopLevelItem(root);self.objects[key]=('well',root)
            for ref in well.get('curves',[]):
                key=curve_key(ref['id']);self.styles.setdefault(key,default_style('curve',ref.get('mode','line')));style=self.styles[key]
                child=QTreeWidgetItem([ref['name']+(' ['+ref['unit']+']' if ref.get('unit') else '')]);child.setData(0,Qt.ItemDataRole.UserRole,key)
                child.setCheckState(0,Qt.CheckState.Checked if style.get('visible',False) else Qt.CheckState.Unchecked);root.addChild(child);self.objects[key]=('curve',child)
            root.setExpanded(True)
        body=QWidget();self.form=QFormLayout(body);self.controls={}
        self.color=QLineEdit();color_button=QPushButton('选择颜色…');color_button.clicked.connect(self.choose_color);row=QHBoxLayout();row.addWidget(self.color);row.addWidget(color_button);self.form.addRow('颜色',row)
        for key,label,low,high,value in [('linewidth','线宽',.1,20,1.5),('alpha','透明度',0,1,1),('markersize','点大小（pt）',.1,40,4),('width','曲线宽度（横轴范围 %）',.01,100,8),('scale','幅值缩放',.001,100,1),('offset','横向偏移（横轴范围 %）',-100,100,0),('minimum','幅值下限',-1e15,1e15,0),('maximum','幅值上限',-1e15,1e15,1)]:
            edit=QDoubleSpinBox();edit.setRange(low,high);edit.setDecimals(4);edit.setValue(value);self.controls[key]=edit;self.form.addRow(label,edit)
        for key,label,choices in [('linestyle','线型',[('实线','-'),('虚线','--'),('点线',':'),('点划线','-.')]),('mode','绘制方式',[('线','line'),('点','points'),('线 + 点','both')]),('marker','点形状',[(k,k) for k in ('o','s','^','v','D','+','x','*','.')])]:
            combo=QComboBox()
            for title,data in choices:combo.addItem(title,data)
            self.controls[key]=combo;self.form.addRow(label,combo)
        for key,label in [('auto_range','自动幅值范围（1%～99%）'),('show_name','显示井名'),('show_track','显示井线')]:
            edit=QCheckBox(label);self.controls[key]=edit;self.form.addRow(edit)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(body);layout.addWidget(scroll,2)
        self.controls['auto_range'].toggled.connect(lambda enabled:[self.controls[key].setEnabled(not enabled) for key in ('minimum','maximum')])
        self.tree.currentItemChanged.connect(self.select_object)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Apply)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('确定');buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消');buttons.button(QDialogButtonBox.StandardButton.Apply).setText('应用')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply);layout.addWidget(buttons)
        self.screen_guard=WindowBoundsGuard(self)
        if self.tree.topLevelItemCount():self.tree.setCurrentItem(self.tree.topLevelItem(0))
        else:body.setEnabled(False);self.color.setEnabled(False)

    def choose_color(self):
        color=QColorDialog.getColor(parent=self)
        if color.isValid():self.color.setText(color.name())

    def save_object(self):
        if self.current is None:return
        style=self.styles[self.current];style['color']=self.color.text().strip()
        for key,edit in self.controls.items():
            style[key]=edit.isChecked() if isinstance(edit,QCheckBox) else edit.currentData() if isinstance(edit,QComboBox) else edit.value()

    def select_object(self,item,*_):
        self.save_object()
        if item is None:return
        self.current=item.data(0,Qt.ItemDataRole.UserRole);kind=self.objects[self.current][0];style=self.styles[self.current]
        self.color.setText(style['color'])
        for key,edit in self.controls.items():
            if isinstance(edit,QCheckBox):edit.setChecked(style[key])
            elif isinstance(edit,QComboBox):edit.setCurrentIndex(max(0,edit.findData(style[key])))
            else:edit.setValue(style[key])
            curve_control=key in ('width','scale','offset','minimum','maximum','auto_range','mode','marker','markersize')
            self.form.setRowVisible(edit,kind=='curve' if curve_control else kind=='well' if key in ('show_name','show_track') else True)

    def apply(self):
        self.save_object()
        for key,(_,item) in self.objects.items():self.styles[key]['visible']=item.checkState(0)==Qt.CheckState.Checked
        self.state.update(distance=self.distance.value(),legend=self.legend.isChecked())
        for style in self.styles.values():
            if not is_color_like(style['color']):QMessageBox.warning(self,'颜色无效','请输入有效颜色，例如 #243b53');return False
            if not style['auto_range'] and style['maximum']<=style['minimum']:QMessageBox.warning(self,'幅值范围无效','幅值上限必须大于下限');return False
        if len(selected_curves(self.state,self.panel.meta.get('well_overlays',[])))>64:
            QMessageBox.warning(self,'曲线过多','每个剖面最多同时叠加 64 条井曲线');return False
        self.panel.set_overlay_settings(deepcopy(self.state));return True

    def accept(self):
        if self.apply():super().accept()

"""Well-centric baseline workflow editor; calculation runs in the RP worker."""
from copy import deepcopy
import json

from PySide6.QtCore import Qt,Signal,QPointF
from PySide6.QtGui import QPainter,QPolygonF,QColor
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QComboBox,
    QLineEdit,QPushButton,QTabWidget,QTableWidget,QTableWidgetItem,QHeaderView,QCheckBox,
    QPlainTextEdit,QFileDialog,QMessageBox,QScrollArea)

from .rock_physics.well_model import FIELDS,PARAMETERS,default_scheme,validate_scheme
from .rock_physics.minerals import MINERAL_PRESETS


class CurveSourceCombo(QComboBox):
    """Selection-only curve picker: clicking the field opens the popup."""
    def __init__(self):
        super().__init__();self.setEditable(False);self.setMaxVisibleItems(18)
        self.view().setMinimumWidth(240)
        self.setToolTip('点击选择当前井的矿物比例曲线；空白项表示使用右侧比例常数。')

    def paintEvent(self,event):
        super().paintEvent(event)
        # The shared stylesheet customises drop-downs without an arrow image.
        painter=QPainter(self);painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#355b73' if self.isEnabled() else '#96a4b0'))
        x=self.width()-16;y=self.height()/2
        painter.drawPolygon(QPolygonF([QPointF(x-4,y-2),QPointF(x+4,y-2),QPointF(x,y+3)]))

    def set_curves(self,names,current=''):
        self.clear();self.addItem('');self.addItems(names)
        if current and current not in names:
            self.addItem(current);self.setItemData(self.count()-1,'当前井中缺少此曲线，请重新选择',Qt.ItemDataRole.ToolTipRole)
        self.setCurrentText(current)


ALIASES=dict(phi=['POR','PHI','PHIE','PHIT','孔隙度'],sw=['SW','含水饱和度'],toc=['TOC'],
    vp=['VP','DTC','AC'],vs=['VS','DTS'],rho=['DEN','RHOB','密度'],pressure=['Pp','PRESSURE'],temperature=['TEMP','TEMPERATURE'])


class WellRockPhysicsPanel(QWidget):
    requested=Signal(dict,str)

    def __init__(self,owner):
        super().__init__();self.owner=owner;self.loading=False;self.saved=[];self.bindings={};self.edits={};self.pending_draft=None
        layout=QVBoxLayout(self)
        note=QLabel('选择井与曲线 → 检查输入 → 构建基质/骨架/流体 → 正演或标定 → 对比与保存成果。参数初值仅为示例，请按工区资料核对。')
        note.setWordWrap(True);layout.addWidget(note)
        row=QHBoxLayout();self.saved_combo=QComboBox();row.addWidget(self.saved_combo,1)
        for title,slot in [('载入方案',self.load_saved),('保存方案',self.save_scheme),('复制方案',self.copy_scheme),('删除方案',self.remove_scheme)]:
            button=QPushButton(title);button.clicked.connect(slot);row.addWidget(button)
        layout.addLayout(row)
        form=QFormLayout();self.well=QComboBox();self.name=QLineEdit();form.addRow('建模井',self.well);form.addRow('方案名称',self.name)
        interval=QHBoxLayout();self.zmin=QLineEdit('0');self.zmax=QLineEdit('10000')
        interval.addWidget(self.zmin);interval.addWidget(QLabel('至'));interval.addWidget(self.zmax);form.addRow('建模井段（m）',interval);layout.addLayout(form)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        page,box=self.page('1  曲线与质控')
        self.mapping=QTableWidget(len(FIELDS),4);self.mapping.setHorizontalHeaderLabels(['角色','曲线 / 常数来源','常数值','输入单位'])
        for row,(key,(label,units,constant)) in enumerate(FIELDS.items()):
            item=QTableWidgetItem(label+('（参考，可缺省）' if key in ('vp','vs','rho') else ''))
            item.setToolTip(item.text())
            item.setFlags(item.flags()&~Qt.ItemFlag.ItemIsEditable);self.mapping.setItem(row,0,item)
            combo=CurveSourceCombo();combo.setToolTip('点击选择当前井曲线；选择“常数 / 不使用”可填写常数或留空。');combo.addItem('常数 / 不使用','');edit=QLineEdit('' if constant is None else str(constant));unit=QComboBox();unit.addItems(units)
            self.bindings[key]=(combo,edit,unit)
            for col,widget in enumerate((combo,edit,unit),1):self.mapping.setCellWidget(row,col,widget)
            combo.currentIndexChanged.connect(lambda _,k=key:self.source_changed(k))
        self.mapping.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.mapping.setMinimumHeight(300);box.addWidget(self.mapping)
        self.interpolate=QCheckBox('允许线性插值（仅相邻有效点，不外推、不跨越显式缺失点）');box.addWidget(self.interpolate)
        form=QFormLayout();self.edit(form,'max_gap_m','允许插值的最大采样间隔（m）');self.edit(form,'null_below','小于等于此值视为缺失')
        box.addLayout(form);self.qc_button=QPushButton('预览输入与质控');self.qc_button.clicked.connect(self.preview);box.addWidget(self.qc_button)
        self.qc=QPlainTextEdit();self.qc.setReadOnly(True);self.qc.setMaximumHeight(150);box.addWidget(self.qc)
        self.preview_table=QTableWidget();self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers);box.addWidget(self.preview_table)
        page,box=self.page('2  基质组成')
        form=QFormLayout();self.basis=QComboBox();self.basis.addItem('无机矿物体积分数','volume');self.basis.addItem('无机矿物质量分数','weight');form.addRow('组成类型',self.basis)
        self.mixing=QComboBox();self.mixing.addItems(['hill','voigt','reuss']);form.addRow('基质模量混合',self.mixing);box.addLayout(form)
        self.normalize=QCheckBox('将有效无机矿物组成归一化');self.clip_negative=QCheckBox('将负矿物比例置零（原曲线保留，结果记录处理数量）')
        box.addWidget(self.normalize);box.addWidget(self.clip_negative)
        self.minerals=QTableWidget(0,7);self.minerals.setHorizontalHeaderLabels(['矿物名称','比例来源曲线（空=常数）','比例常数','比例单位','密度 g/cm³','K GPa','G GPa'])
        self.minerals.setMinimumHeight(220)
        self.minerals.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);box.addWidget(self.minerals,1)
        row=QHBoxLayout();self.mineral_preset=CurveSourceCombo()
        self.mineral_preset.addItems(list(MINERAL_PRESETS)+['自定义矿物']);self.mineral_preset.setCurrentText('方解石')
        self.mineral_preset.setToolTip('选择常见矿物后点击添加；预设密度和模量均可修改。黏土为混合组分示例。')
        row.addWidget(self.mineral_preset)
        for title,slot in [('添加矿物',lambda:self.add_mineral()),('删除所选矿物',self.remove_minerals)]:
            button=QPushButton(title);button.clicked.connect(slot);row.addWidget(button)
        box.addLayout(row);self.kerogen=QCheckBox('启用 TOC 转干酪根（TOC 为干固体中的碳质量分数）');box.addWidget(self.kerogen)
        form=QFormLayout()
        for key,label in [('carbon_fraction','干酪根碳质量比例'),('kerogen_density','干酪根密度（g/cm³）'),('kerogen_K','干酪根 K（GPa）'),('kerogen_G','干酪根 G（GPa）')]:self.edit(form,key,label)
        box.addLayout(form)
        page,box=self.page('3  骨架与流体')
        form=QFormLayout();self.dry_model=QComboBox()
        for label,value in [('双孔隙 DEM（硬孔＋软孔）','dual_dem'),('单孔隙 DEM','single_dem'),('临界孔隙度模型','cripor')]:self.dry_model.addItem(label,value)
        form.addRow('干岩骨架',self.dry_model);self.order=QComboBox();self.order.addItem('先硬孔后软孔','hard_then_soft');self.order.addItem('先软孔后硬孔','soft_then_hard');form.addRow('双孔隙加孔顺序',self.order)
        for key,(label,*_) in PARAMETERS.items():self.edit(form,key,label)
        box.addLayout(form);box.addWidget(QLabel('分层覆盖：每行覆盖一个参数；同一井段可设置多个不同参数。范围为左闭右开，包含建模终点。'))
        self.layers=QTableWidget(0,4);self.layers.setHorizontalHeaderLabels(['起点 m','终点 m','参数','覆盖值']);self.layers.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);box.addWidget(self.layers)
        row=QHBoxLayout()
        for title,slot in [('添加分层参数',self.add_layer),('删除所选分层',lambda:self.remove_rows(self.layers))]:
            button=QPushButton(title);button.clicked.connect(slot);row.addWidget(button)
        box.addLayout(row);form=QFormLayout()
        for key,label in [('salinity_ppm','地层水矿化度（ppm）'),('oil_density','原油参考密度（g/cm³）'),('gas_gravity','天然气相对密度（活油）'),('gor','气油比（L/L；0 为死油）')]:self.edit(form,key,label)
        box.addLayout(form);hint=QLabel('流体体系：油水两相，体积模量按 Reuss 混合。温度、压力和 Sw 在“基础输入”的补充输入中设置；本阶段采用 Gassmann 低频各向同性基线。');hint.setWordWrap(True);box.addWidget(hint)
        page,box=self.page('4  敏感性与标定')
        form=QFormLayout();self.cal_mode=QComboBox()
        for label,value in [('仅正演，不标定','none'),('参数扫描（1～2 个参数）','scan'),('自动标定（1～3 个参数）','optimize')]:self.cal_mode.addItem(label,value)
        form.addRow('计算方式',self.cal_mode)
        for key,label in [('cal_min','标定起点（m）'),('cal_max','标定终点（m）'),('validation_min','独立验证起点（m）'),('validation_max','独立验证终点（m；等于起点则不启用）'),
            ('max_samples','标定最大样本数（沿深度均匀抽样）'),('max_evaluations','自动标定最大迭代数'),('weight_vp','Vp 相对误差权重'),('weight_vs','Vs 相对误差权重'),('weight_rho','密度相对误差权重')]:self.edit(form,key,label)
        box.addLayout(form)
        self.cal_table=QTableWidget(len(PARAMETERS),4);self.cal_table.setHorizontalHeaderLabels(['参与标定的参数','下限','上限','扫描点数'])
        for row,(key,(label,value,low,high)) in enumerate(PARAMETERS.items()):
            item=QTableWidgetItem(label);item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable);item.setCheckState(Qt.CheckState.Unchecked);item.setData(Qt.ItemDataRole.UserRole,key);self.cal_table.setItem(row,0,item)
            for col,text in enumerate((max(low,value/2),min(high,value*2),7),1):self.cal_table.setItem(row,col,QTableWidgetItem(str(text)))
        self.cal_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);box.addWidget(self.cal_table)
        hint=QLabel('采用相对误差并按每条曲线有效样本数归一化。未映射的参考曲线不参与。分层已覆盖的参数不能同时做全井标定。验证井段必须与标定井段分开。');hint.setWordWrap(True);box.addWidget(hint);box.addStretch()
        row=QHBoxLayout();self.run=QPushButton('运行建模 / 标定');self.run.clicked.connect(self.start);row.addWidget(self.run)
        self.export=QPushButton('导出方案…');self.export.clicked.connect(self.export_scheme);row.addWidget(self.export)
        self.import_button=QPushButton('导入方案…');self.import_button.clicked.connect(self.import_scheme);row.addWidget(self.import_button);layout.addLayout(row)
        self.message=QLabel('');self.message.setWordWrap(True);layout.addWidget(self.message)
        self.well.currentTextChanged.connect(self.well_changed);self.set_scheme(default_scheme());self.refresh_library()
        from .rock_physics_live import install_workbench
        install_workbench(self)

    def live_changed(self):
        if hasattr(self,'live'):self.live.wire();self.live.changed()

    def stop_live(self):
        if hasattr(self,'live'):self.live.shutdown()

    def page(self,title):
        widget=QWidget();layout=QVBoxLayout(widget)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(widget);scroll.setMinimumSize(0,0)
        self.tabs.addTab(scroll,title);return widget,layout

    def edit(self,form,key,label):
        widget=QLineEdit();self.edits[key]=widget;form.addRow(label,widget)

    def well_record(self):
        return next((w for w in getattr(getattr(self.owner,'library',None),'wells',[]) if w['name']==self.well.currentText()),None)

    def resolved_well(self):
        well=deepcopy(self.well_record())
        if well is None:raise ValueError('请先向项目井库导入测井曲线并选择井')
        for ref in well['curves']:ref['path']=str(self.owner.project.resolve(ref['path']))
        return well

    def refresh_library(self):
        if self.loading:return
        selected=self.well.currentText();self.well.blockSignals(True);self.well.clear()
        self.well.addItems([w['name'] for w in getattr(getattr(self.owner,'library',None),'wells',[])])
        if selected:self.well.setCurrentText(selected)
        self.well.blockSignals(False);self.refresh_sources(bool(self.well.currentText()) and selected!=self.well.currentText())
        if not selected and self.well.currentText():self.use_well_interval()
        if self.pending_draft and self.pending_draft.get('well') in [self.well.itemText(i) for i in range(self.well.count())]:
            draft=self.pending_draft;self.pending_draft=None;self.restore_draft(draft)
        self.live_changed()

    def refresh_sources(self,auto):
        refs=(self.well_record() or {}).get('curves',[]);names=[r['name'] for r in refs if r['name']!='Depth']
        for key,(combo,edit,unit) in self.bindings.items():
            current=combo.currentData();combo.blockSignals(True);combo.clear();combo.addItem('常数 / 不使用','')
            for name in names:combo.addItem(name,name)
            if current:combo.addItem('缺失：'+current,current) if current not in names else None
            target=current or (next((n for alias in ALIASES.get(key,[]) for n in names if n.upper()==alias.upper()),'') if auto else '')
            combo.setCurrentIndex(max(0,combo.findData(target)));combo.blockSignals(False);self.source_changed(key,update_unit=auto)
        for row in range(self.minerals.rowCount()):
            combo=self.minerals.cellWidget(row,1);combo.set_curves(names,combo.currentText())

    def source_changed(self,key,update_unit=True):
        combo,edit,unit=self.bindings[key];edit.setEnabled(not bool(combo.currentData()))
        if update_unit and combo.currentData():
            ref=next((r for r in (self.well_record() or {}).get('curves',[]) if r['name']==combo.currentData()),{})
            value=ref.get('unit','');value={'g/cm³':'g/cm3','m/s':'m/s','%':'%'}.get(value,value)
            if unit.findText(value)>=0:unit.setCurrentText(value)

    def well_changed(self,*_):
        if self.loading:return
        self.pending_draft=None
        self.use_well_interval()
        self.refresh_sources(True)

    def use_well_interval(self):
        refs=(self.well_record() or {}).get('curves',[])
        lows=[float(r['depth_min']) for r in refs if r.get('depth_min') is not None]
        highs=[float(r['depth_max']) for r in refs if r.get('depth_max') is not None]
        if lows and highs and max(highs)>min(lows):self.zmin.setText(str(min(lows)));self.zmax.setText(str(max(highs)))

    def add_mineral(self,mineral=None):
        user_added=not isinstance(mineral,dict)
        if not isinstance(mineral,dict):
            name=self.mineral_preset.currentText();existing={self.minerals.item(r,0).text() for r in range(self.minerals.rowCount())}
            if name in MINERAL_PRESETS:
                if name in existing:
                    self.message.setText('已存在该矿物，请修改现有行，或选择其它矿物。');return
                mineral=dict(name=name,constant=0.,unit='fraction',curve='',**MINERAL_PRESETS[name])
            else:
                name='自定义矿物';index=2
                while name in existing:name='自定义矿物'+str(index);index+=1
                mineral=dict(name=name,constant=0.,unit='fraction',density='',K='',G='',curve='')
        row=self.minerals.rowCount();self.minerals.insertRow(row)
        for col,key in [(0,'name'),(2,'constant'),(4,'density'),(5,'K'),(6,'G')]:self.minerals.setItem(row,col,QTableWidgetItem(str(mineral[key])))
        source=CurveSourceCombo();source.set_curves([r['name'] for r in (self.well_record() or {}).get('curves',[]) if r['name']!='Depth'],mineral.get('curve',''));self.minerals.setCellWidget(row,1,source)
        unit=QComboBox();unit.addItems(['fraction','%']);unit.setCurrentText(mineral['unit']);self.minerals.setCellWidget(row,3,unit)
        if user_added:self.message.setText('已添加“'+mineral['name']+'”；请选择比例来源，并核对或填写物性参数。')
        self.live_changed()

    def remove_minerals(self):self.remove_rows(self.minerals)

    @staticmethod
    def remove_rows(table):
        for row in sorted({i.row() for i in table.selectedIndexes()},reverse=True):table.removeRow(row)

    def add_layer(self,layer=None,key=None,value=None):
        if not isinstance(layer,dict):layer={'from':self.zmin.text(),'to':self.zmax.text()};key='soft_fraction';value=.15
        row=self.layers.rowCount();self.layers.insertRow(row)
        for col,text in [(0,layer['from']),(1,layer['to']),(3,value)]:self.layers.setItem(row,col,QTableWidgetItem(str(text)))
        combo=QComboBox()
        for name,(label,*_) in PARAMETERS.items():combo.addItem(label,name)
        combo.setCurrentIndex(max(0,combo.findData(key)));self.layers.setCellWidget(row,2,combo)
        self.live_changed()

    def raw_draft(self):
        return dict(name=self.name.text(),well=self.well.currentText(),zmin=self.zmin.text(),zmax=self.zmax.text(),
            bindings={k:dict(curve=c.currentData() or '',constant=e.text(),unit=u.currentText()) for k,(c,e,u) in self.bindings.items()},
            edits={k:w.text() for k,w in self.edits.items()},basis=self.basis.currentData(),mixing=self.mixing.currentText(),
            normalize=self.normalize.isChecked(),clip_negative=self.clip_negative.isChecked(),kerogen=self.kerogen.isChecked(),
            interpolate=self.interpolate.isChecked(),dry_model=self.dry_model.currentData(),order=self.order.currentData(),cal_mode=self.cal_mode.currentData(),
            minerals=[[self.minerals.cellWidget(r,c).currentText() if c in (1,3) else self.minerals.item(r,c).text() if self.minerals.item(r,c) else '' for c in range(7)] for r in range(self.minerals.rowCount())],
            layers=[[self.layers.cellWidget(r,c).currentData() if c==2 else self.layers.item(r,c).text() if self.layers.item(r,c) else '' for c in range(4)] for r in range(self.layers.rowCount())],
            calibration=[[self.cal_table.item(r,0).checkState()==Qt.CheckState.Checked]+[self.cal_table.item(r,c).text() for c in range(1,4)] for r in range(self.cal_table.rowCount())])

    def restore_draft(self,draft):
        d=deepcopy(draft);self.loading=True
        self.name.setText(d['name']);index=self.well.findText(d.get('well',''))
        if index>=0:self.well.setCurrentIndex(index)
        elif d.get('well'):self.pending_draft=d
        self.zmin.setText(d['zmin']);self.zmax.setText(d['zmax'])
        self.refresh_sources(False)
        for key,b in d['bindings'].items():
            c,e,u=self.bindings[key];idx=c.findData(b['curve'])
            if idx<0:c.addItem('缺失：'+b['curve'],b['curve']);idx=c.count()-1
            c.blockSignals(True);c.setCurrentIndex(idx);c.blockSignals(False);e.setText(b['constant']);u.setCurrentText(b['unit']);self.source_changed(key,False)
        for key,value in d['edits'].items():self.edits[key].setText(value)
        for combo,key in [(self.basis,'basis'),(self.dry_model,'dry_model'),(self.order,'order'),(self.cal_mode,'cal_mode')]:combo.setCurrentIndex(max(0,combo.findData(d[key])))
        self.mixing.setCurrentText(d['mixing'])
        for widget,key in [(self.normalize,'normalize'),(self.clip_negative,'clip_negative'),(self.kerogen,'kerogen'),(self.interpolate,'interpolate')]:widget.setChecked(d[key])
        self.minerals.setRowCount(0)
        for values in d['minerals']:self.add_mineral(dict(zip(['name','curve','constant','unit','density','K','G'],values)))
        self.layers.setRowCount(0)
        for a,b,key,value in d['layers']:self.add_layer({'from':a,'to':b},key,value)
        for row,values in enumerate(d['calibration']):
            self.cal_table.item(row,0).setCheckState(Qt.CheckState.Checked if values[0] else Qt.CheckState.Unchecked)
            for col,value in enumerate(values[1:],1):self.cal_table.item(row,col).setText(value)
        self.loading=False

        self.live_changed()

    def scheme(self,for_preview=False):
        s=default_scheme();s.update(name=self.name.text().strip(),well=self.well.currentText(),depth_min=float(self.zmin.text()),depth_max=float(self.zmax.text()),
            mineral_basis=self.basis.currentData(),normalize_minerals=self.normalize.isChecked(),clip_negative=self.clip_negative.isChecked(),
            mixing=self.mixing.currentText(),kerogen=self.kerogen.isChecked(),interpolate=self.interpolate.isChecked(),dry_model=self.dry_model.currentData(),dem_order=self.order.currentData())
        for key,widget in self.edits.items():
            if key in PARAMETERS:s['parameters'][key]=float(widget.text())
            elif key in s:s[key]=float(widget.text())
        for key,(combo,edit,unit) in self.bindings.items():s['bindings'][key]=dict(curve=combo.currentData() or '',constant=None if combo.currentData() or not edit.text().strip() else float(edit.text()),unit=unit.currentText())
        s['minerals']=[]
        for values in self.raw_draft()['minerals']:
            name,curve,constant,unit,density,k,g=values
            if any(not v.strip() for v in (density,k,g)):raise ValueError('请填写矿物“'+name+'”的密度、K 和 G')
            s['minerals'].append(dict(name=name,curve=curve.strip(),constant=0. if curve.strip() else float(constant),unit=unit,density=float(density),K=float(k),G=float(g)))
        layers={}
        for a,b,key,value in self.raw_draft()['layers']:
            pair=(float(a),float(b));layer=layers.setdefault(pair,dict(**{'from':pair[0],'to':pair[1]},parameters={}))
            if key in layer['parameters']:raise ValueError('同一分层参数重复')
            layer['parameters'][key]=float(value)
        s['layers']=list(layers.values());c=s['calibration'];c['mode']=self.cal_mode.currentData()
        if for_preview:
            c['mode']='none';c['parameters']=[]
            return validate_scheme(s)
        for key,source in [('depth_min','cal_min'),('depth_max','cal_max'),('validation_min','validation_min'),('validation_max','validation_max'),('max_samples','max_samples'),('max_evaluations','max_evaluations')]:c[key]=float(self.edits[source].text())
        c['weights']={k:float(self.edits['weight_'+k].text()) for k in ('vp','vs','rho')};c['parameters']=[]
        for row in range(self.cal_table.rowCount()):
            if self.cal_table.item(row,0).checkState()==Qt.CheckState.Checked:
                c['parameters'].append(dict(name=self.cal_table.item(row,0).data(Qt.ItemDataRole.UserRole),min=float(self.cal_table.item(row,1).text()),max=float(self.cal_table.item(row,2).text()),steps=int(self.cal_table.item(row,3).text())))
        return validate_scheme(s)

    def set_scheme(self,s):
        self.loading=True;self.name.setText(s['name']);self.zmin.setText(str(s['depth_min']));self.zmax.setText(str(s['depth_max']))
        if self.well.findText(s['well'])>=0:self.well.setCurrentText(s['well'])
        self.refresh_sources(False)
        for key,b in s['bindings'].items():
            combo,edit,unit=self.bindings[key];idx=combo.findData(b['curve'])
            if idx<0:combo.addItem('缺失：'+b['curve'],b['curve']);idx=combo.count()-1
            combo.blockSignals(True);combo.setCurrentIndex(idx);combo.blockSignals(False);edit.setText('' if b['constant'] is None else str(b['constant']));unit.setCurrentText(b['unit']);self.source_changed(key,False)
        for key,edit in self.edits.items():
            if key in PARAMETERS:edit.setText(str(s['parameters'][key]))
            elif key in s:edit.setText(str(s[key]))
        for combo,key in [(self.basis,'mineral_basis'),(self.dry_model,'dry_model'),(self.order,'dem_order')]:combo.setCurrentIndex(max(0,combo.findData(s[key])))
        self.mixing.setCurrentText(s['mixing'])
        for widget,key in [(self.normalize,'normalize_minerals'),(self.clip_negative,'clip_negative'),(self.kerogen,'kerogen'),(self.interpolate,'interpolate')]:widget.setChecked(s[key])
        self.minerals.setRowCount(0)
        for mineral in s['minerals']:self.add_mineral(mineral)
        self.layers.setRowCount(0)
        for layer in s.get('layers',[]):
            for key,value in layer['parameters'].items():self.add_layer(layer,key,value)
        c=s['calibration'];self.cal_mode.setCurrentIndex(max(0,self.cal_mode.findData(c['mode'])))
        for key,source in [('depth_min','cal_min'),('depth_max','cal_max'),('validation_min','validation_min'),('validation_max','validation_max'),('max_samples','max_samples'),('max_evaluations','max_evaluations')]:self.edits[source].setText(str(c[key]))
        for key in ('vp','vs','rho'):self.edits['weight_'+key].setText(str(c['weights'].get(key,1)))
        for row,key in enumerate(PARAMETERS):
            b=next((b for b in c['parameters'] if b['name']==key),None);self.cal_table.item(row,0).setCheckState(Qt.CheckState.Checked if b else Qt.CheckState.Unchecked)
            if b:
                for col,value in enumerate((b['min'],b['max'],b.get('steps',7)),1):self.cal_table.item(row,col).setText(str(value))
        self.loading=False
        self.live_changed()

    def start(self):
        try:
            s=self.scheme();well=self.resolved_well();self.save_scheme(persist=False)
            self.requested.emit(dict(operation='well_log',scheme=s,well=well),'测井岩石物理 · '+s['name'])
        except Exception as exc:self.message.setText('不能运行：'+str(exc))

    def preview(self):
        # Uses the same isolated process as modelling; no numerical work on the UI thread.
        try:self.requested.emit(dict(operation='well_preview',scheme=self.scheme(),well=self.resolved_well()),'输入质控')
        except Exception as exc:self.message.setText('不能预览：'+str(exc))

    def show_preview(self,result):
        self.qc.setPlainText(result['message']);table=result['table'];self.preview_table.setColumnCount(len(table['columns']));self.preview_table.setHorizontalHeaderLabels(table['columns']);self.preview_table.setRowCount(len(table['rows']))
        for r,values in enumerate(table['rows']):
            for c,value in enumerate(values):self.preview_table.setItem(r,c,QTableWidgetItem('' if value is None else str(value)))
        self.tabs.setCurrentIndex(0)
        if hasattr(self,'live'):self.live.sections['quality'][0].setChecked(True)

    def collect_state(self):return dict(draft=self.pending_draft or self.raw_draft(),schemes=deepcopy(self.saved),live=self.live.collect_state())

    def populate(self,cfg):
        self.stop_live()
        self.saved=deepcopy(cfg.get('schemes',[]));self.pending_draft=None;self.refresh_saved();self.refresh_library()
        if cfg.get('draft'):self.restore_draft(cfg['draft'])
        else:
            self.set_scheme(default_scheme());self.refresh_sources(True);self.use_well_interval()
        self.live.populate(cfg.get('live',{}))

    def refresh_saved(self):
        self.saved_combo.clear()
        for d in self.saved:self.saved_combo.addItem(d['well']+' · '+d['name'])

    def save_scheme(self,*_,persist=True):
        try:
            draft=self.raw_draft()
            if not draft['name'].strip():raise ValueError('请填写方案名称')
            index=next((i for i,d in enumerate(self.saved) if (d['well'],d['name'])==(draft['well'],draft['name'])),len(self.saved))
            if index==len(self.saved):self.saved.append(draft)
            else:self.saved[index]=draft
            self.refresh_saved();self.saved_combo.setCurrentIndex(index)
            if persist and hasattr(self.owner,'shell'):self.owner.shell.persist_workspace()
            self.message.setText('方案已保存；载入后可修改井、输入曲线或模型参数。')
        except Exception as exc:self.message.setText('保存失败：'+str(exc))

    def load_saved(self):
        index=self.saved_combo.currentIndex()
        if index>=0:self.restore_draft(self.saved[index])

    def copy_scheme(self):
        self.name.setText(self.name.text()+' 副本');self.save_scheme()

    def remove_scheme(self):
        index=self.saved_combo.currentIndex()
        if index>=0:
            self.saved.pop(index);self.refresh_saved()
            if hasattr(self.owner,'shell'):self.owner.shell.persist_workspace()

    def export_scheme(self):
        path,_=QFileDialog.getSaveFileName(self,'导出建模方案','','JSON (*.json)')
        if not path:return
        try:
            from pathlib import Path
            Path(path).write_text(json.dumps(self.scheme(),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        except Exception as exc:QMessageBox.warning(self,'导出失败',str(exc))

    def import_scheme(self):
        path,_=QFileDialog.getOpenFileName(self,'导入建模方案','','JSON (*.json)')
        if not path:return
        try:
            from pathlib import Path
            self.set_scheme(validate_scheme(json.loads(Path(path).read_text(encoding='utf-8'))))
        except Exception as exc:QMessageBox.warning(self,'导入失败',str(exc))

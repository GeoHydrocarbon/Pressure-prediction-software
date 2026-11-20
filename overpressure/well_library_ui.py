"""Well object import, curve inspection and explicit training selection widgets."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt,QThread,Signal,QTimer
from PySide6.QtWidgets import (QWidget,QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QComboBox,
    QLineEdit,QPushButton,QFileDialog,QMessageBox,QTableWidget,QTableWidgetItem,QDialogButtonBox,
    QListWidget,QListWidgetItem,QCheckBox,QScrollArea,QAbstractItemView,QHeaderView)
from .config import FEATURES,UNITS,CANONICAL_UNITS,selected_features,unit_factor
from .data import read_table
from .well_io import infer_columns
from .well_library import inspect_curve_file,merge_wells,curve_key
from .ui_style import Card,WindowBoundsGuard,polish_layouts


class CurveImportDialog(QDialog):
    def __init__(self,parent,paths,well='',measured=False):
        super().__init__(parent);self.paths=paths;self.forced_well=well;self.measured=measured
        self.setWindowTitle('导入实测压力点' if measured else '导入井与测井曲线');self.resize(760,600)
        layout=QVBoxLayout(self);label=QLabel(f'导入 {len(paths)} 个文件'+(f'，归入井 {well}' if well else '，按井名归类；无井名列时使用文件名'))
        label.setWordWrap(True);layout.addWidget(label)
        raw=read_table(Path(paths[0]),nrows=12);headers=list(map(str,raw.columns));mapping=infer_columns(headers,{'well':'井','Depth':'Depth'})
        form=QFormLayout();self.well_column=QComboBox();self.well_column.addItems(['']+headers);self.well_column.setCurrentText(mapping['well'] if mapping['well'] in headers else '')
        self.depth=QComboBox();self.depth.addItems(headers);self.depth.setCurrentText(mapping['Depth'])
        self.depth_unit=QComboBox();self.depth_unit.addItems(list(UNITS['Depth']));form.addRow('井名列（留空使用所选井 / 文件名）',self.well_column);form.addRow('深度列',self.depth);form.addRow('深度单位',self.depth_unit);layout.addLayout(form)
        hint=QLabel('勾选导入的曲线，设置统一的曲线名称和源单位。同一井的同名曲线重新导入时替换引用，原始文件保留。已知曲线支持单位换算；其它曲线请填写准确且一致的单位。')
        hint.setWordWrap(True);layout.addWidget(hint)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['导入','源列','曲线名称','源单位'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for column in headers:
            if column in (self.well_column.currentText(),self.depth.currentText()):continue
            key=curve_key(column);numeric=pd.to_numeric(raw[column],errors='coerce').notna().any()
            row=self.table.rowCount();self.table.insertRow(row)
            checked=QTableWidgetItem();checked.setFlags(Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEnabled);checked.setCheckState(Qt.CheckState.Checked if (not measured and (numeric or key in UNITS)) or measured and key=='Pp' else Qt.CheckState.Unchecked)
            self.table.setItem(row,0,checked);source=QTableWidgetItem(column);source.setFlags(Qt.ItemFlag.ItemIsEnabled);self.table.setItem(row,1,source)
            self.table.setItem(row,2,QTableWidgetItem('实测压力' if measured and key=='Pp' else key))
            self.table.setItem(row,3,QTableWidgetItem(CANONICAL_UNITS.get(key,'')))
        layout.addWidget(self.table,1)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(self.validate_accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.screen_guard=WindowBoundsGuard(self)
    def definitions(self):
        result=[]
        for row in range(self.table.rowCount()):
            if self.table.item(row,0).checkState()!=Qt.CheckState.Checked:continue
            column=self.table.item(row,1).text()
            if column in (self.depth.currentText(),self.well_column.currentText()):continue
            result.append(dict(column=column,name=self.table.item(row,2).text().strip(),unit=self.table.item(row,3).text().strip(),role='measurement' if self.measured else 'curve'))
        return result
    def validate_accept(self):
        definitions=self.definitions();names=[d['name'] for d in definitions]
        if not names or any(not n or n in ('well','Depth','source_row','Pp_pred') for n in names) or len(names)!=len(set(names)):
            QMessageBox.warning(self,'检查曲线名称','请至少选择一条曲线，名称不能重复或使用保留字段。');return
        if self.measured and (len(names)!=1 or definitions[0]['unit'] not in UNITS['Pp']):
            QMessageBox.warning(self,'检查压力单位','实测点每次选择一条压力列，单位必须为 MPa、kPa 或 psi。');return
        for item in definitions:
            if item['name'] in UNITS and item['unit'] not in UNITS[item['name']]:
                QMessageBox.warning(self,'检查单位',f'{item["name"]} 支持的单位：{", ".join(UNITS[item["name"]])}');return
        self.accept()
    def specs(self):
        return [dict(path=str(p),well_column=self.well_column.currentText(),depth_column=self.depth.currentText(),depth_unit=self.depth_unit.currentText(),forced_well=self.forced_well,definitions=self.definitions()) for p in self.paths]


class ImportWorker(QThread):
    imported=Signal(list);problem=Signal(str)
    def __init__(self,specs,parent):super().__init__(parent);self.specs=specs
    def run(self):
        for spec in self.specs:
            if self.isInterruptionRequested():return
            try:
                spec=deepcopy(spec);path=spec.pop('path')
                training_default=spec.pop('training_default',True)
                headers=read_table(Path(path),nrows=0).columns
                mapping=infer_columns(headers,{'well':spec['well_column'],'Depth':spec['depth_column']})
                if spec['well_column'] and mapping['well'] not in headers:
                    if 'default_units' in spec:spec['well_column']=''
                    else:raise ValueError('缺少井名列，请单独导入并设置映射')
                spec['well_column']=mapping['well'] if spec['well_column'] else '';spec['depth_column']=mapping['Depth']
                for definition in spec.get('definitions') or []:
                    if definition['column'] not in headers:definition['column']=infer_columns(headers,{definition['name']:definition['column']})[definition['name']]
                legacy_columns=spec.pop('legacy_columns',None)
                if legacy_columns:
                    inferred=infer_columns(headers,legacy_columns);spec['logical_names']={source:key for key,source in inferred.items()}
                wells=inspect_curve_file(path,**spec)
                for well in wells:well['_training_default']=training_default
                self.imported.emit(wells)
            except Exception as exc:self.problem.emit(f'{Path(spec.get("path",path)).name}：{exc}')


from .well_curve_viewer import CurveViewer,CurveReader


class WellLibrary:
    def __init__(self,owner):
        self.owner=owner;self.wells=[];self.worker=None;self.errors=[];self.active=False;self.loading=False;self.generation=0
        self.training_card=Card('训练数据：选择井、输入曲线与压力目标');layout=QVBoxLayout(self.training_card)
        row=QHBoxLayout();self.well_list=QListWidget();self.well_list.setMaximumHeight(140);self.feature_list=QListWidget();self.feature_list.setMaximumHeight(140)
        for title,widget in [('参与训练的井',self.well_list),('模型输入曲线（可多选）',self.feature_list)]:
            box=QVBoxLayout();box.addWidget(QLabel(title));box.addWidget(widget);row.addLayout(box,1)
        layout.addLayout(row);form=QFormLayout();self.target=QComboBox();form.addRow('压力目标曲线（独立于输入）',self.target);layout.addLayout(form)
        self.note=QLabel('从左侧“井”节点右键导入数据。不同文件曲线按相同深度匹配，不自动插值；训练会检查所选井是否缺少曲线。');self.note.setWordWrap(True);layout.addWidget(self.note)
        self.well_list.itemChanged.connect(self.selection_changed);self.feature_list.itemChanged.connect(self.selection_changed);self.target.currentTextChanged.connect(self.selection_changed)
        self.training_card.hide()
    def checked(self,widget):return [widget.item(i).text() for i in range(widget.count()) if widget.item(i).checkState()==Qt.CheckState.Checked]
    def populate(self,cfg):
        self.generation+=1;self.stop()
        self.wells=deepcopy(cfg.get('project_data',{}).get('wells',[]));self.active=True
        self.refresh(cfg['data'].get('development_wells',[]),cfg['data'].get('features',list(FEATURES)),cfg['data'].get('target_curve','Pp'))
    def refresh(self,selected=None,features=None,target=None):
        selected=self.checked(self.well_list) if selected is None else selected;features=self.checked(self.feature_list) if features is None else features;target=target or self.target.currentText() or 'Pp'
        names=list(dict.fromkeys(c['name'] for well in self.wells for c in well['curves'] if c.get('role')!='measurement'))
        self.loading=True;self.well_list.clear();self.feature_list.clear();self.target.clear()
        well_names={well['name'] for well in self.wells}
        for name in dict.fromkeys([*selected,*[well['name'] for well in self.wells]]):
            if name not in well_names:continue
            item=QListWidgetItem(name);item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable);item.setCheckState(Qt.CheckState.Checked if name in selected else Qt.CheckState.Unchecked);self.well_list.addItem(item)
        for name in dict.fromkeys([*features,*names]):
            if name=='Pp':continue
            item=QListWidgetItem(name);item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable);item.setCheckState(Qt.CheckState.Checked if name in features else Qt.CheckState.Unchecked);self.feature_list.addItem(item)
        pressure_names=list(dict.fromkeys(c['name'] for well in self.wells for c in well['curves'] if c.get('role')!='measurement' and c.get('unit') in UNITS['Pp']))
        self.target.addItems(pressure_names);self.target.setCurrentText(target)
        self.target.setPlaceholderText('请先导入压力目标曲线（MPa / kPa / psi）')
        self.loading=False;self.training_card.setVisible(self.active)
        self.selection_changed()
        if hasattr(self.owner,'well_prediction'):self.owner.well_prediction.refresh_library()
        if hasattr(self.owner,'shell'):self.owner.shell.signature=None;self.owner.shell.refresh_tree()
    def selection_changed(self,*_):
        if self.loading:return
        self.loading=True
        for i in range(self.feature_list.count()):
            item=self.feature_list.item(i);flags=item.flags()|Qt.ItemFlag.ItemIsEnabled
            if item.text()==self.target.currentText():
                item.setCheckState(Qt.CheckState.Unchecked);flags &= ~Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)
        self.loading=False
        self.note.setText('不同曲线按相同深度匹配，不自动插值；输入按清单顺序保存到模型。压力目标不能勾选为输入。Eaton 约束需要 Depth 和 AC。')
    def selected_units(self):
        units={}
        selected=set(self.checked(self.well_list))
        for well in self.wells:
            if well['name'] in selected:
                for curve in well['curves']:units.setdefault(curve['name'],CANONICAL_UNITS.get(curve['name'],curve.get('unit','')))
        return units
    def import_files(self,well='',measured=False):
        if self.worker is not None:QMessageBox.information(self.owner,'正在导入','请等待当前批次导入结束。');return
        paths,_=QFileDialog.getOpenFileNames(self.owner,'导入实测压力点' if measured else '导入井曲线',str(self.owner.project.path.parent),'井数据 (*.xlsx *.csv)')
        if not paths:return
        try:
            dialog=CurveImportDialog(self.owner,paths,well,measured)
            if dialog.exec()==QDialog.DialogCode.Accepted:self.import_specs(dialog.specs())
        except Exception as exc:QMessageBox.warning(self.owner,'无法读取文件',str(exc))
    def import_specs(self,specs):
        if self.worker is not None:return
        self.errors=[];worker=ImportWorker(specs,self.owner);self.worker=worker
        generation=self.generation
        worker.imported.connect(lambda wells:self.receive(wells) if generation==self.generation else None)
        worker.problem.connect(lambda message:self.errors.append(message) if generation==self.generation else None)
        worker.finished.connect(lambda:self.finished(worker) if generation==self.generation else worker.deleteLater());worker.start();self.owner.status.setText('正在导入井曲线并检查井名、深度与字段…')
    def receive(self,wells):
        old={w['name'] for w in self.wells};selected=self.checked(self.well_list)
        self.wells=merge_wells(self.wells,wells);self.active=True
        selected.extend(w['name'] for w in wells if w['name'] not in old and w.get('_training_default',True) and any(c['name']=='Pp' and c.get('role')!='measurement' for c in w['curves']))
        self.refresh(selected,self.checked(self.feature_list) or list(FEATURES))
    def finished(self,worker):
        self.worker=None;worker.deleteLater();self.owner.status.setText(f'井库共 {len(self.wells)} 口井；导入结束'+(f'，{len(self.errors)} 个文件失败' if self.errors else ''))
        if self.errors:QMessageBox.warning(self.owner,'部分文件未导入','\n'.join(self.errors))
    def stop(self):
        if self.worker is not None:self.worker.requestInterruption();self.worker.wait();self.worker=None

    def migrate_legacy(self,cfg):
        if self.wells or self.worker is not None or cfg['data'].get('training_selection')=='well_library':return
        paths=cfg['data'].get('well_imports',cfg['data'].get('well_files',[]))
        if not paths:return
        specs=[]
        for value in paths:
            path=self.owner.project.resolve(value)
            if not path.is_file():continue
            # Infer numeric curves in the worker; old projects remain readable.
            specs.append(dict(path=str(path),well_column=cfg['data']['columns']['well'],depth_column=cfg['data']['columns']['Depth'],depth_unit=cfg['data']['units']['Depth'],definitions=None,default_units=cfg['data']['units'],legacy_columns=cfg['data']['columns']))
        generation=self.generation
        if specs:QTimer.singleShot(0,lambda:self.import_specs(specs) if generation==self.generation else None)

    def import_spatial(self,kind,well=''):
        from .project_catalog import AssetDialog
        from .well_spatial import read_asset
        path,_=QFileDialog.getOpenFileName(self.owner,'导入井口坐标' if kind=='coordinates' else '导入井轨迹',str(self.owner.project.path.parent),'井数据 (*.xlsx *.csv)')
        if not path:return
        try:
            dialog=AssetDialog(self.owner,path,kind,forced_well=well)
            if dialog.exec()!=QDialog.DialogCode.Accepted:return
            asset=dialog.config();raw,frame=read_asset(asset)
            names=frame.well.dropna().unique().tolist()
            if not names:raise ValueError('没有可关联的井名')
            asset['wells']=names;self.owner.catalog.assets.append(asset);self.owner.catalog.refresh()
            self.wells=merge_wells(self.wells,[dict(name=str(name),curves=[]) for name in names]);self.refresh()
        except Exception as exc:QMessageBox.warning(self.owner,'无法导入空间数据',str(exc))


"""Project spatial assets; independent from the training table mapping."""
from copy import deepcopy
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QDialog, QFormLayout, QCheckBox, QDialogButtonBox, QFileDialog, QMessageBox, QDoubleSpinBox)
from .ui_style import Card
from .spatial_asset_dialog import AssetDialog


class ProjectCatalog(QWidget):
    def __init__(self, owner):
        super().__init__(); self.owner=owner; self.assets=[]; self.horizons=[]
        outer=QVBoxLayout(self);outer.setContentsMargins(0,0,0,0)
        card=Card('井空间数据与剖面对比曲线');outer.addWidget(card);layout=QVBoxLayout(card)
        hint=QLabel('与训练表独立，可添加多个文件；双击文件修改字段对应。');hint.setObjectName('hint');hint.setWordWrap(True);layout.addWidget(hint)
        self.list=QListWidget(); self.list.setMinimumHeight(115);self.list.setMaximumHeight(150); layout.addWidget(self.list)
        self.list.setToolTip('双击可查看或修改该文件的字段对应'); self.list.itemDoubleClicked.connect(self.edit_asset)
        row=QHBoxLayout()
        for title, kind in [('添加测井曲线 / 井预测结果','curves'),('添加井口坐标','coordinates'),('添加井轨迹','trajectory')]:
            b=QPushButton(title); b.clicked.connect(lambda _, k=kind:self.add_asset(k)); row.addWidget(b)
        layout.addLayout(row); b=QPushButton('移除选中的井文件'); b.clicked.connect(self.remove); layout.addWidget(b)
        self.distance=QDoubleSpinBox(); self.distance.setRange(0,1e9); self.distance.setValue(25); self.distance.setSuffix('（地震坐标单位）')
        form=QFormLayout(); form.addRow('井到最近地震道最大距离',self.distance); layout.addLayout(form)
        self.vertical=QCheckBox('缺少轨迹时按井口垂直井定位'); self.vertical.setChecked(True); layout.addWidget(self.vertical)
        self.horizon_widget=QWidget(); host=QVBoxLayout(self.horizon_widget);card=Card('层位文件管理');host.addWidget(card);hl=QVBoxLayout(card)
        text=QLabel('分别登记层位文件；载入地震压力成果后，选择层位并发送到切片。支持线号或 XY＋时间。'); text.setWordWrap(True); hl.addWidget(text)
        self.hlist=QListWidget(); hl.addWidget(self.hlist)
        for label, callback in [('添加层位文件',self.add_horizon),('移除选中的层位文件',self.remove_horizon),('将选中层位载入地震切片',self.use_horizon)]:
            b=QPushButton(label); b.clicked.connect(callback); hl.addWidget(b)
        host.addStretch()

    def refresh(self):
        self.list.clear(); self.hlist.clear()
        labels={'curves':'曲线','coordinates':'井口坐标','trajectory':'轨迹'}
        self.list.addItems([labels[a['kind']]+' | '+a['path'] for a in self.assets])
        self.hlist.addItems([a.get('display_name',a['path']) for a in self.horizons])

    def add_asset(self,kind):
        path,_=QFileDialog.getOpenFileName(self,'选择井数据','','表格 (*.csv *.xlsx *.txt)')
        if not path:return
        try:
            dialog=AssetDialog(self,path,kind)
            if dialog.exec()==QDialog.DialogCode.Accepted:
                from .well_spatial import read_asset
                asset=dialog.config();_,frame=read_asset(asset);asset['wells']=frame.well.dropna().unique().tolist()
                self.assets.append(asset);self.refresh()
        except Exception as exc:QMessageBox.warning(self,'读取失败',str(exc))

    def remove(self):
        i=self.list.currentRow()
        if i>=0:self.assets.pop(i); self.refresh()

    def edit_asset(self,*_):
        i=self.list.currentRow()
        if i<0:return
        asset=self.assets[i]
        try:
            dialog=AssetDialog(self,str(self.owner.project.resolve(asset['path'])),asset['kind'],forced_well=asset.get('well_name',''),initial=asset)
            if dialog.exec()==QDialog.DialogCode.Accepted:
                from .well_spatial import read_asset
                updated=dialog.config();updated['excluded_wells']=asset.get('excluded_wells',[])
                _,frame=read_asset(updated);updated['wells']=frame.well.dropna().unique().tolist()
                self.assets[i]=updated;self.refresh()
        except Exception as exc:QMessageBox.warning(self,'读取失败',str(exc))

    def add_horizon(self):
        from .horizon_dialog import HorizonImportDialog
        dialog=HorizonImportDialog(self,self.owner.seismic_convention.currentText())
        if dialog.exec()==QDialog.DialogCode.Accepted:
            self.horizons.append(dialog.config());self.refresh()
            if self.owner.slices.mode.currentIndex()==3:self.owner.slices.ensure_project_horizons()

    def remove_horizon(self):
        i=self.hlist.currentRow()
        if i>=0:self.horizons.pop(i); self.refresh()

    def use_horizon(self):
        i=self.hlist.currentRow()
        if i>=0:
            config=deepcopy(self.horizons[i]); config['path']=str(self.owner.project.resolve(config['path']))
            self.owner.slices.import_horizon_config(config); self.owner.nav.setCurrentRow(6)

    def horizon_configs(self):
        configs=deepcopy(self.horizons)
        for config in configs:config['path']=str(self.owner.project.resolve(config['path']))
        return configs

    def register_interpolated_horizon(self,config):
        existing=next((i for i,item in enumerate(self.horizons) if str(self.owner.project.resolve(item['path']))==str(self.owner.project.resolve(config['path']))),None)
        if existing is None:self.horizons.append(deepcopy(config))
        else:self.horizons[existing]=deepcopy(config)
        self.refresh()

    def populate(self,cfg):
        self.assets=deepcopy(cfg.get('well_assets',[])); self.horizons=deepcopy(cfg.get('horizons',[]))
        self.distance.setValue(cfg.get('max_distance',25)); self.vertical.setChecked(cfg.get('vertical_fallback',True)); self.refresh()

    def collect(self):
        return dict(well_assets=deepcopy(self.assets),horizons=deepcopy(self.horizons),max_distance=self.distance.value(),vertical_fallback=self.vertical.isChecked())

    def well_config(self):
        cfg=self.owner.collect_settings('seismic'); catalog=cfg['project_data']; assets=deepcopy(catalog['well_assets'])
        for a in assets:a['path']=str(self.owner.project.resolve(a['path']))
        for well in self.owner.library.wells:
            for ref in well['curves']:
                if ref['name']=='Depth':continue
                asset=dict(path=str(self.owner.project.resolve(ref['path'])),kind='curves',columns=dict(well=ref.get('well_column',''),depth=ref['depth_column']),well_name=well['name'],depth_unit=ref['depth_unit'],curve_column=ref['column'],curve_name=ref['name'],curve_unit=ref['unit'])
                if ref['name']=='Pp' or ref.get('role')=='measurement':asset.update(pressure_column=ref['column'],pressure_unit=ref['unit'])
                assets.append(asset)
        return dict(assets=assets, project_path=str(self.owner.project.path),time_depth=cfg['time_depth'],max_distance=catalog['max_distance'],vertical_fallback=catalog['vertical_fallback'])

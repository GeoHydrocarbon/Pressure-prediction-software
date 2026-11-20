"""Reusable batch file picker with asynchronous per-file inspection."""
from pathlib import Path
from PySide6.QtCore import Qt,QThread,Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QListWidget,QListWidgetItem,QPushButton,QLabel,QFileDialog
from .data import read_table
from .well_io import infer_columns


class FileInspector(QThread):
    record=Signal(dict)
    def __init__(self,paths,columns,required,parent):
        super().__init__(parent);self.paths=paths;self.columns=columns;self.required=required
    def run(self):
        for path in self.paths:
            if self.isInterruptionRequested():break
            result=dict(path=path,wells=[],rows=0,valid=False)
            try:
                headers=read_table(Path(path),nrows=0).columns;mapping=infer_columns(headers,self.columns)
                raw=read_table(Path(path),dtype={mapping['well']:'string'} if mapping['well'] in headers else None)
                names=raw[mapping['well']].astype('string').str.strip().dropna().unique().tolist() if mapping['well'] in raw else [Path(path).stem]
                missing=[mapping[k] for k in self.required if k!='well' and mapping[k] not in raw]
                result.update(wells=names,rows=len(raw),valid=not missing and bool(len(raw)),status='可用' if not missing and len(raw) else '缺少列：'+','.join(missing) if missing else '没有数据行')
            except Exception as exc:result['status']=str(exc)
            self.record.emit(result)


class BatchFilesPanel(QWidget):
    changed=Signal()
    inspected=Signal()
    def __init__(self,owner,columns,required,title):
        super().__init__();self.owner=owner;self.columns_provider=columns;self.required=required;self.worker=None;self.records={};self.active=False
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        label=QLabel(title+'；勾选参与文件，井名列缺失时采用文件名。');label.setWordWrap(True);layout.addWidget(label)
        self.list=QListWidget();self.list.setMinimumHeight(115);self.list.setMaximumHeight(190);layout.addWidget(self.list)
        row=QHBoxLayout();layout.addLayout(row)
        for text,callback in [('批量添加文件',self.choose_files),('导入整个文件夹',self.choose_folder),('移除选中文件',self.remove),('检查文件 / 更新井清单',self.inspect)]:
            b=QPushButton(text);b.clicked.connect(callback);row.addWidget(b)
        self.status=QLabel('可添加一个或多个 XLSX / CSV 文件，勾选需要使用的文件。');self.status.setWordWrap(True);self.status.setObjectName('hint');layout.addWidget(self.status)
        self.list.itemChanged.connect(lambda _:self.changed.emit())

    def all_files(self):return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
    def files(self):return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count()) if self.list.item(i).checkState()==Qt.CheckState.Checked]
    def names(self):return list(dict.fromkeys(name for p in self.files() for name in self.records.get(p,{}).get('wells',[])))
    def ready(self):return bool(self.files()) and all(self.records.get(p,{}).get('valid',False) for p in self.files())

    def set_files(self,paths,selected=None,active=True):
        self.stop_worker();self.active=active;self.records={};self.list.blockSignals(True);self.list.clear()
        for value in paths:
            path=str(self.owner.project.resolve(value));item=QListWidgetItem(Path(path).name+' | 未检查');item.setToolTip(path);item.setData(Qt.ItemDataRole.UserRole,path)
            item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable);item.setCheckState(Qt.CheckState.Checked if selected is None or value in selected or path in selected else Qt.CheckState.Unchecked);self.list.addItem(item)
        self.list.blockSignals(False)

    def add_files(self,paths):
        previous=self.all_files();merged=list(dict.fromkeys(previous+[str(Path(p).resolve()) for p in paths]));selected=self.files()+[p for p in merged if p not in previous]
        self.set_files(merged,selected);self.changed.emit();self.inspect()

    def choose_files(self):
        paths,_=QFileDialog.getOpenFileNames(self,'批量导入井文件',str(self.owner.project.path.parent),'表格 (*.xlsx *.csv)')
        if paths:self.add_files(paths)

    def choose_folder(self):
        path=QFileDialog.getExistingDirectory(self,'导入文件夹',str(self.owner.project.path.parent))
        if path:self.add_files(sorted(str(p) for p in Path(path).iterdir() if p.suffix.lower() in ('.xlsx','.csv') and not p.name.startswith('~$')))

    def remove(self):
        item=self.list.takeItem(self.list.currentRow())
        if item:self.records.pop(item.data(Qt.ItemDataRole.UserRole),None);self.active=True;self.changed.emit()

    def inspect(self):
        if self.worker is not None:return
        if not self.all_files():return
        self.status.setText('正在后台检查文件，请稍候；缺字段的文件请修正映射或取消勾选。')
        worker=FileInspector(self.all_files(),self.columns_provider(),self.required,self);self.worker=worker
        worker.record.connect(lambda record:self.receive(record) if self.worker is worker else None)
        def done():
            if self.worker is not worker:worker.deleteLater();return
            self.worker=None
            self.status.setText(f'已检查 {len(self.records)} 个文件；当前勾选 {len(self.files())} 个。请核对井名和字段状态。')
            worker.deleteLater();self.inspected.emit();self.changed.emit()
        worker.finished.connect(done);worker.start()

    def receive(self,record):
        self.records[record['path']]=record
        for i in range(self.list.count()):
            item=self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole)==record['path']:
                item.setText(f"{Path(record['path']).name} | {', '.join(record['wells'])} | {record['rows']} 行 | {record['status']}")

    def stop_worker(self):
        worker=self.worker
        if worker is not None:
            worker.requestInterruption();worker.wait();self.worker=None

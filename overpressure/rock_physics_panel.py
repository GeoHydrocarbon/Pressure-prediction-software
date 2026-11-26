"""Rock physics parameter editor; numerical work stays in an isolated process."""
from copy import deepcopy
import codecs
import json
from pathlib import Path

from PySide6.QtCore import QProcess,QProcessEnvironment,Qt,Signal
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QComboBox,
    QTableWidget,QTableWidgetItem,QPlainTextEdit,QPushButton,QHeaderView,QAbstractItemView,QTabWidget,QScrollArea,QSizePolicy)

from .runtime import worker_command,application_root
from .ui_style import Card


def available_models():
    from .rock_physics import catalog
    return catalog()


class RockPhysicsPanel(QWidget):
    resultReady=Signal(str,dict)

    def __init__(self,owner):
        super().__init__();self.owner=owner;self.process=None;self.models=[];self.drafts={}
        self.current_id=None;self.completed=None;self.cancelled=False;self.worker_error='';self.pending=''
        self.setMinimumSize(0,0)
        root=QVBoxLayout(self);self.workflow_tabs=QTabWidget();root.addWidget(self.workflow_tabs,1)
        self.workflow_tabs.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        from .well_rock_physics_panel import WellRockPhysicsPanel
        self.workflow=WellRockPhysicsPanel(owner);self.workflow.requested.connect(self.run_workflow_request)
        self.workflow_tabs.addTab(self.workflow,'测井岩石物理建模')
        toolbox=QWidget();layout=QVBoxLayout(toolbox);layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        toolbox_scroll=QScrollArea();toolbox_scroll.setWidgetResizable(True);toolbox_scroll.setWidget(toolbox)
        self.workflow_tabs.addTab(toolbox_scroll,'模型工具箱')
        hint=QLabel('选择模型，核对参数与单位后计算。结果作为独立图件在中央绘图窗口打开；不需要训练压力模型。')
        hint.setWordWrap(True);layout.addWidget(hint)
        card=Card('岩石物理模型与参数');form=QFormLayout(card)
        self.category=QComboBox();self.model=QComboBox();form.addRow('模型分类',self.category);form.addRow('计算模型',self.model)
        self.source=QLabel('计算内核：RockPhyPy；各模型的单位和适用条件见下方说明。');self.source.setWordWrap(True);form.addRow(self.source)
        layout.addWidget(card)
        self.parameters=QTableWidget(0,4);self.parameters.setHorizontalHeaderLabels(['参数','数值','单位','说明'])
        self.parameters.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.parameters.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents)
        self.parameters.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch)
        self.parameters.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.ResizeToContents)
        self.parameters.horizontalHeader().setSectionResizeMode(3,QHeaderView.ResizeMode.Stretch)
        self.parameters.setMinimumHeight(210);layout.addWidget(self.parameters,2)
        self.input_hint=QLabel('数值示例：37；数组：[0.1, 0.2, 0.3]；矩阵：[[1, 0], [0, 1]]。字符串保留引号；可选参数留空使用模型默认值。')
        self.input_hint.setWordWrap(True);layout.addWidget(self.input_hint)
        row=QHBoxLayout();self.example_button=QPushButton('恢复示例 / 默认参数');self.run_button=QPushButton('开始计算');self.cancel_button=QPushButton('取消计算')
        self.cancel_button.setEnabled(False)
        row.addWidget(self.example_button);row.addStretch();row.addWidget(self.run_button);row.addWidget(self.cancel_button);layout.addLayout(row)
        self.help=QPlainTextEdit();self.help.setReadOnly(True);self.help.setMinimumHeight(110);self.help.setMaximumHeight(210);layout.addWidget(self.help,1)
        self.status=QLabel('');self.status.setWordWrap(True);layout.addWidget(self.status)
        # Worker status and cancellation stay visible while either tab is active.
        root.addWidget(self.cancel_button);root.addWidget(self.status)
        self.status.setMaximumHeight(60);self.status.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Maximum)
        def footer(index):
            self.cancel_button.setVisible(index==1 or self.process is not None)
            self.status.setVisible(index==1 or self.process is not None)
        self.workflow_tabs.currentChanged.connect(footer);footer(0)
        self.category.currentIndexChanged.connect(self.category_changed);self.model.currentIndexChanged.connect(self.model_changed)
        self.example_button.clicked.connect(self.restore_defaults);self.run_button.clicked.connect(self.run_model);self.cancel_button.clicked.connect(self.stop)
        try:
            self.models=available_models()
            self.category.addItems(list(dict.fromkeys(spec['category'] for spec in self.models)))
            self.category_changed()
        except Exception as exc:
            self.status.setText('岩石物理模型目录暂不可用：'+str(exc));self.run_button.setEnabled(False)

    def specification(self):
        ident=self.model.currentData()
        return next((spec for spec in self.models if spec['id']==ident),None)

    def save_draft(self):
        if self.current_id:
            self.drafts[self.current_id]={self.parameters.item(row,0).text():self.parameters.item(row,1).text()
                for row in range(self.parameters.rowCount())}

    def category_changed(self,*_):
        self.save_draft();self.model.blockSignals(True);self.model.clear()
        for spec in self.models:
            if spec['category']==self.category.currentText():self.model.addItem(spec['title'],spec['id'])
        self.model.blockSignals(False);self.model_changed()

    def model_changed(self,*_):
        self.save_draft();spec=self.specification();self.current_id=spec['id'] if spec else None
        self.parameters.setRowCount(0)
        if spec is None:self.run_button.setEnabled(False);return
        draft=self.drafts.get(spec['id']);example=spec.get('example',{})
        self.parameters.setRowCount(len(spec['parameters']))
        for row,param in enumerate(spec['parameters']):
            name=param['name'];value=example.get(name,param.get('default'))
            text=draft.get(name,'') if draft is not None else '' if value is None else json.dumps(value,ensure_ascii=False,allow_nan=False)
            description=param.get('description','')
            if param.get('required'):description='必填。'+description
            for column,content in enumerate((name,text,param.get('unit',''),description)):
                item=QTableWidgetItem(str(content));item.setToolTip(str(content))
                if column!=1:item.setFlags(item.flags()&~Qt.ItemFlag.ItemIsEditable)
                self.parameters.setItem(row,column,item)
        self.help.setPlainText(spec.get('description',''))
        self.status.setText('已载入可计算示例；请按自己的数据修改参数。' if spec.get('validated_example') else '此模型请根据说明填写输入；参数留空不代表零。')
        self.run_button.setEnabled(self.process is None)

    def restore_defaults(self):
        if self.current_id:self.drafts.pop(self.current_id,None)
        # Avoid immediately saving the old cells back over the reset.
        self.current_id=None;self.model_changed()

    def show_category(self,name):
        spec=next((m for m in self.models if m['id']==name),None)
        category=spec['category'] if spec else name
        index=self.category.findText(category)
        if index>=0:self.category.setCurrentIndex(index)
        if spec:
            index=self.model.findData(spec['id'])
            if index>=0:self.model.setCurrentIndex(index)

    def select_model(self,model_id):
        self.workflow_tabs.setCurrentIndex(1)
        self.show_category(model_id)

    def start_calculation(self):
        self.run_model()

    def collect_state(self):
        self.save_draft()
        return dict(model_id=self.current_id,category=self.category.currentText(),drafts=deepcopy(self.drafts),workflow=self.workflow.collect_state(),active_tab=self.workflow_tabs.currentIndex())

    def populate(self,cfg):
        if self.process is not None:return
        self.workflow.populate(cfg.get('workflow',{}))
        self.workflow_tabs.setCurrentIndex(cfg.get('active_tab',0))
        self.current_id=None
        drafts=cfg.get('drafts',{}) if isinstance(cfg,dict) else {}
        self.drafts={str(key):{str(name):str(value) for name,value in values.items()} for key,values in drafts.items() if isinstance(values,dict)} if isinstance(drafts,dict) else {}
        ident=cfg.get('model_id') if isinstance(cfg,dict) else None
        spec=next((m for m in self.models if m['id']==ident),None)
        category=spec['category'] if spec else cfg.get('category','') if isinstance(cfg,dict) else ''
        self.category.blockSignals(True)
        index=self.category.findText(category)
        self.category.setCurrentIndex(max(0,index));self.category.blockSignals(False);self.category_changed()
        if spec:
            self.current_id=None;self.model.setCurrentIndex(self.model.findData(ident));self.model_changed()

    def parameter_values(self):
        spec=self.specification()
        if spec is None:raise ValueError('请先选择计算模型')
        values={}
        for row,param in enumerate(spec['parameters']):
            text=self.parameters.item(row,1).text().strip();name=param['name']
            if not text:
                if param.get('required'):raise ValueError(name+' 为必填参数')
                continue
            try:value=json.loads(text,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('不支持 '+value)))
            except (ValueError,TypeError) as exc:raise ValueError(name+' 格式不正确，请使用数值、带引号的字符串或数组') from exc
            values[name]=value
        return values

    def set_busy(self,busy):
        for control in (self.category,self.model,self.parameters,self.example_button,self.run_button):control.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.cancel_button.setVisible(busy or self.workflow_tabs.currentIndex()==1)
        if busy:self.status.show()
        self.workflow.setEnabled(not busy)
        self.workflow.live.manual_busy=busy
        if busy:self.workflow.stop_live()
        elif self.workflow.isVisible():self.workflow.live_changed()

    def run_model(self):
        if self.process is not None:return
        try:
            values=self.parameter_values();spec=self.specification()
            folder=self.owner.project.resolve(self.owner.settings['output_dir'])/'rock_physics';folder.mkdir(parents=True,exist_ok=True)
            request=dict(model_id=spec['id'],parameters=values,output_dir=str(folder))
            command=worker_command('overpressure.rock_physics.worker')
        except Exception as exc:self.status.setText('不能开始计算：'+str(exc));return
        self.launch_request(request,spec['title'],command)

    def run_workflow_request(self,request,title):
        if self.process is not None:return
        try:
            folder=self.owner.project.resolve(self.owner.settings['output_dir'])/'rock_physics'
            request=dict(request,output_dir=str(folder))
            self.launch_request(request,title,worker_command('overpressure.rock_physics.worker'))
        except Exception as exc:self.status.setText('不能开始计算：'+str(exc))

    def launch_request(self,request,title,command):
        self.save_draft();self.completed=None;self.cancelled=False;self.worker_error='';self.pending=''
        self.decoder=codecs.getincrementaldecoder('utf-8')('replace');self.stderr=''
        process=QProcess(self);self.process=process;self.set_busy(True);self.status.setText('正在计算 '+title+'…')
        environment=QProcessEnvironment.systemEnvironment();environment.insert('PYTHONIOENCODING','utf-8');process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(application_root()))
        process.started.connect(lambda:self.send_request(process,request))
        process.readyReadStandardOutput.connect(lambda:self.read_output(process))
        process.readyReadStandardError.connect(lambda:self.read_error(process))
        process.finished.connect(lambda code,*_:self.finished(process,code))
        process.errorOccurred.connect(lambda error:self.failed_start(process) if error==QProcess.ProcessError.FailedToStart else None)
        try:process.start(*command)
        except Exception as exc:self.worker_error=str(exc);self.finished(process,1)

    def send_request(self,process,request):
        if self.process is not process or self.cancelled:return
        process.write(json.dumps(request,ensure_ascii=False,allow_nan=False).encode('utf-8'));process.closeWriteChannel()

    def read_error(self,process):
        self.stderr+=bytes(process.readAllStandardError()).decode('utf-8',errors='replace')

    def read_output(self,process):
        if self.process is not process:return
        self.pending+=self.decoder.decode(bytes(process.readAllStandardOutput()))
        while '\n' in self.pending:
            line,self.pending=self.pending.split('\n',1);self.handle_line(line)

    def handle_line(self,line):
        try:event=json.loads(line)
        except (ValueError,TypeError):return
        if not isinstance(event,dict):return
        if event.get('event') in ('completed','preview'):self.completed=event
        elif event.get('event')=='error':self.worker_error=event.get('message') or event.get('error') or '岩石物理计算失败'
        elif event.get('event')=='stage':self.status.setText(event.get('message','正在计算…'))

    def failed_start(self,process):
        if self.process is process:self.worker_error=process.errorString();self.finished(process,1)

    def finished(self,process,code):
        if self.process is not process:return
        self.read_output(process);self.read_error(process)
        if self.pending.strip():self.handle_line(self.pending);self.pending=''
        self.process=None;self.set_busy(False);process.blockSignals(True);process.deleteLater()
        if self.cancelled:self.status.setText('计算已取消。');return
        if code==0 and self.completed and self.completed.get('event')=='preview':
            self.workflow.show_preview(self.completed['result']);self.status.setText('输入质控预览完成。');return
        if code==0 and self.completed and isinstance(self.completed.get('result'),dict) and self.completed.get('path'):
            event=self.completed
            self.status.setText('计算完成：'+event['result'].get('title','')+'。成果已在中央绘图窗口打开。' if event['result'].get('kind')=='well_log' else '计算完成：'+event['path'])
            self.status.setToolTip(event['path']);self.resultReady.emit(event['path'],event['result'])
        else:
            detail=self.worker_error
            if not detail and self.stderr.strip():detail=self.stderr.strip().splitlines()[-1]
            self.status.setText('计算失败：'+(detail or '后台没有返回完整结果'))

    def stop(self):
        self.workflow.stop_live()
        process=self.process
        if process is None:return
        self.cancelled=True;self.process=None
        process.blockSignals(True)
        if process.state()==QProcess.ProcessState.Starting:process.waitForStarted(1000)
        if process.state()!=QProcess.ProcessState.NotRunning:process.kill();process.waitForFinished(1000)
        # Retain the stopped QObject under this panel until the panel is destroyed.
        self.set_busy(False);self.completed=None;self.status.setText('计算已取消。')

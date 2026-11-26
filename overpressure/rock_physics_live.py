"""Progressive well editor and disposable, latest-input-only live previews."""
from copy import deepcopy
import codecs
import json

import numpy as np
from PySide6.QtCore import QObject,QEvent,QTimer,QProcess,QProcessEnvironment,Qt,QSize
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,
    QScrollArea,QSplitter,QToolButton,QMenu,QCheckBox,QPushButton,QComboBox,QTabWidget,
    QPlainTextEdit)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT

from .runtime import application_root,worker_command
from .well_curve_viewer import CurveViewer
from .charts import FigureStyle,apply_typography
from .rock_physics.well_model import FIELDS,PARAMETERS
from .flow_layout import FlowLayout


class LiveResultPane(QWidget):
    def __init__(self,owner):
        super().__init__();self.result=None;self.numeric={};self.available_names=set();self.fit_dimensions=None
        self.fit_timer=QTimer(self);self.fit_timer.setSingleShot(True);self.fit_timer.setInterval(40);self.fit_timer.timeout.connect(self.fit_plot)
        layout=QVBoxLayout(self)
        self.heading=QLabel('选择井与孔隙度，开始建模');self.heading.setWordWrap(True);self.heading.setStyleSheet('font-size: 16px; font-weight: 600;');layout.addWidget(self.heading)
        self.status=QLabel('实测速度和密度用于对比，可以稍后补充。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.statistics=QLabel('');self.statistics.setWordWrap(True);layout.addWidget(self.statistics)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        self.viewer=CurveViewer(owner);self.viewer.list_toggle.setChecked(False);self.tabs.addTab(self.viewer,'曲线对比')
        self.compact_plot_controls();self.viewer.plot_box.setMinimumHeight(180);self.viewer.plot_box.installEventFilter(self)
        cross=QWidget();box=QVBoxLayout(cross);row=QHBoxLayout();row.addWidget(QLabel('实测与模型'))
        self.cross_property=QComboBox();self.cross_property.addItems(['Vp','Vs','密度']);self.cross_property.currentIndexChanged.connect(self.draw_cross);row.addWidget(self.cross_property);row.addStretch();box.addLayout(row)
        self.canvas=FigureCanvasQTAgg(Figure(layout='constrained'));box.addWidget(NavigationToolbar2QT(self.canvas,self));box.addWidget(self.canvas,1);self.tabs.addTab(cross,'交会图')
        self.assumptions=QPlainTextEdit();self.assumptions.setReadOnly(True);self.tabs.addTab(self.assumptions,'本次输入与假设')

    def compact_plot_controls(self):
        v=self.viewer;old=v.layout().takeAt(1).layout();hidden=QWidget(v);hidden.hide();self.hidden_controls=hidden
        while old.count():
            item=old.takeAt(0)
            if item.widget():item.widget().setParent(hidden)
        v.caption.hide();row=FlowLayout();v.layout().insertLayout(1,row)
        v.mode.setMaximumWidth(150);row.addWidget(v.mode);v.mode.show()
        menu=QMenu(v)
        for text,slot in [('分道与曲线样式…',v.plot_settings),('图例设置…',v.edit_legends),('恢复图框布局',v.restore_layout)]:menu.addAction(text,slot)
        curves=menu.addAction('显示曲线列表');curves.setCheckable(True);curves.toggled.connect(v.list_toggle.setChecked)
        lock=menu.addAction('锁定图框比例');lock.setCheckable(True);lock.toggled.connect(v.lock.setChecked)
        button=QToolButton();button.setText('绘图设置');button.setMenu(menu);button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup);row.addWidget(button)
        v.fit_button.setText('适应窗口');row.addWidget(v.fit_button);v.fit_button.show();v.toolbar.setIconSize(QSize(18,18))

    def eventFilter(self,watched,event):
        if event.type()==QEvent.Type.Resize and watched.size().toTuple()!=self.fit_dimensions:self.fit_timer.start()
        return False

    def fit_plot(self):
        box=self.viewer.plot_box
        if not box.lock_ratio and box.width()>20 and box.height()>20:
            self.fit_dimensions=box.size().toTuple();ratio=(box.width()-16)/(box.height()-16)
            if abs(box.ratio-ratio)>1e-6:box.ratio=ratio;box.refresh_size()

    def pending(self,text,well=''):
        if self.result and well and self.result['well']!=well:self.clear()
        self.status.setText(text+('；图中暂为上一次结果。' if self.result else ''))
        self.status.setStyleSheet('color: #976018;')

    def clear(self):
        self.result=None;self.numeric={};self.available_names=set();self.statistics.clear();self.viewer.reset();self.assumptions.clear()
        self.canvas.figure.clear();self.canvas.draw_idle();self.heading.setText('选择井与孔隙度，开始建模')

    def display(self,result):
        ratio=self.viewer.plot_box.ratio if self.result else 1.75
        scale=self.viewer.plot_box.scale if self.result else 1.
        hidden=self.available_names-{d['name'] for d in self.viewer.datasets} if self.result and self.result['well']==result['well'] else set()
        self.result=result;table=result['table'];self.numeric={}
        for i,column in enumerate(table['columns']):
            try:self.numeric[column]=np.asarray([r[i] for r in table['rows']],float)
            except (ValueError,TypeError):pass
        self.heading.setText(result['title'])
        p=result['preview'];sampled=p['sampled_rows']<p['total_rows']
        self.status.setText(f"实时正演 · {p['sampled_rows']} / {p['total_rows']} 个样点"+('（抽样预览）' if sampled else '')+' · 尚未保存；完整成果请点击“计算并保存成果”。')
        self.status.setStyleSheet('color: #28648a;')
        stats=[]
        for key,value in result['metrics']['all'].items():
            stats.append(f"{key.upper()} RMSE {value['rmse']:.4g} {'g/cc' if key=='rho' else 'm/s'} (N={value['count']})")
        self.statistics.setText('预览样点误差：'+'；'.join(stats) if stats else '尚未添加实测参考曲线；当前显示模型正演结果。')
        old={r['name']:r for r in self.viewer.setting_rows()};datasets=[];tracks={};z=self.numeric['Depth'];self.available_names=set()
        for track,(obs,model,label,unit) in enumerate([('vp','VP_RP','Vp','m/s'),('vs','VS_RP','Vs','m/s'),('rho','DEN_RP','密度','g/cc')],1):
            for column,title,color in [(obs,'实测 '+label,'#263746'),(model,'模型 '+label,'#d4573e')]:
                x=self.numeric[column]
                if not np.isfinite(x).any():continue
                self.available_names.add(title)
                if title in hidden:continue
                breaks=np.asarray(p['breaks'].get('model' if column==model else obs,[]),int)
                xx=np.insert(x,breaks,np.nan);zz=np.insert(z,breaks,np.nan)
                datasets.append(dict(name=title,x=xx,z=zz,unit=unit,well=result['well'],source='interactive:'+result['well'],column=column,points=False))
                tracks[title]=(track,color)
        self.viewer.render(datasets);rows=self.viewer.setting_rows()
        for row in rows:
            if row['name'] in old:row.update(old[row['name']])
            else:row.update(track=tracks[row['name']][0],color=tracks[row['name']][1])
        self.viewer.apply_plot_settings(rows);self.viewer.plot_box.ratio=ratio;self.viewer.plot_box.scale=scale;self.viewer.plot_box.refresh_size()
        self.fit_plot()
        self.viewer.caption.setText('实测与模型同道 · 可调整分道、样式和图例')
        self.assumptions.setPlainText('本图使用的输入与参数（自动预览仅正演）\n'+json.dumps(result['parameters']['adopted_scheme'],ensure_ascii=False,indent=2)+'\n\n'+'\n'.join(result['warnings']))
        self.draw_cross()

    def reset_plot(self):
        if self.result:
            self.available_names=set();self.viewer.datasets=[];self.viewer.curve_styles={};self.viewer.legend_settings={};self.display(self.result)

    def draw_cross(self,*_):
        figure=self.canvas.figure;figure.clear();ax=figure.add_subplot(111)
        obs,model,label=[('vp','VP_RP','Vp (m/s)'),('vs','VS_RP','Vs (m/s)'),('rho','DEN_RP','Density (g/cc)')][self.cross_property.currentIndex()]
        if obs in self.numeric:
            x=self.numeric[obs];y=self.numeric[model];valid=np.isfinite(x)&np.isfinite(y)
            if valid.any():
                dots=ax.scatter(x[valid],y[valid],c=self.numeric['Depth'][valid],s=14,alpha=.8);figure.colorbar(dots,ax=ax,label='Depth (m)')
                lo=min(x[valid].min(),y[valid].min());hi=max(x[valid].max(),y[valid].max());ax.plot([lo,hi],[lo,hi],'--',color='gray')
            else:ax.text(.5,.5,'补充相应实测曲线后显示交会图',ha='center',transform=ax.transAxes)
        ax.set(xlabel='Measured '+label,ylabel='Model '+label);ax.grid(alpha=.2);apply_typography(figure,FigureStyle());self.canvas.draw_idle()


class LiveWorkbench(QObject):
    def __init__(self,panel):
        super().__init__(panel);self.panel=panel;self.revision=0;self.process=None;self.manual_busy=False
        self.timer=QTimer(self);self.timer.setSingleShot(True);self.timer.setInterval(500);self.timer.timeout.connect(self.calculate)
        self.sections={};self.build();self.wire();panel.installEventFilter(self)

    def build(self):
        p=self.panel
        self.legacy=QWidget(p);self.legacy.setLayout(p.layout());self.legacy.hide()
        root=QVBoxLayout(p);root.setContentsMargins(0,0,0,0)
        self.splitter=QSplitter(Qt.Orientation.Horizontal);root.addWidget(self.splitter,1)
        editor=QWidget();body=QVBoxLayout(editor);body.setContentsMargins(6,6,6,6)
        row=QHBoxLayout();title=QLabel('建模设置');title.setStyleSheet('font-size: 16px; font-weight: 600;');row.addWidget(title);row.addStretch()
        menu=QMenu(p)
        for text,slot in [('保存方案',p.save_scheme),('复制方案',p.copy_scheme),('删除所选方案',p.remove_scheme),('导入方案…',p.import_scheme),('导出方案…',p.export_scheme)]:menu.addAction(text,slot)
        manager=QToolButton();manager.setText('方案管理');manager.setMenu(menu);manager.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup);row.addWidget(manager);body.addLayout(row)
        row=QHBoxLayout();row.addWidget(p.saved_combo);p.saved_combo.show();load=QPushButton('载入');load.clicked.connect(p.load_saved);row.addWidget(load);body.addLayout(row)
        basic=QWidget();box=QVBoxLayout(basic);form=QFormLayout();form.addRow('井',p.well);form.addRow('方案',p.name)
        interval=QHBoxLayout();interval.addWidget(p.zmin);interval.addWidget(QLabel('至'));interval.addWidget(p.zmax);form.addRow('井段（m）',interval);box.addLayout(form)
        for widget in (p.well,p.name,p.zmin,p.zmax):widget.show()
        box.addWidget(p.mapping);p.mapping.show()
        self.more=QCheckBox('补充 Sw、TOC 和温压输入');self.more.toggled.connect(self.mapping_rows);box.addWidget(self.more)
        self.defaults=QLabel('');self.defaults.setWordWrap(True);box.addWidget(self.defaults)
        self.add_section(body,'basic','基础输入',basic,True)
        for index,key,title in [(1,'matrix','基质组成'),(2,'model','骨架与流体'),(0,'quality','深度匹配与质控'),(3,'calibration','敏感性与标定')]:
            page=p.tabs.widget(index).takeWidget();self.add_section(body,key,title,page,False)
        body.addStretch();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(editor);scroll.setMinimumWidth(380);self.editor=editor
        self.splitter.addWidget(scroll)
        right=QWidget();layout=QVBoxLayout(right);layout.setContentsMargins(4,4,4,4)
        row=FlowLayout();self.enabled=QCheckBox('自动更新');self.enabled.setChecked(True);self.enabled.toggled.connect(self.auto_changed);row.addWidget(self.enabled)
        refresh=QPushButton('刷新预览');refresh.clicked.connect(lambda:self.changed(force=True));row.addWidget(refresh)
        self.result=LiveResultPane(p.owner);reset=QPushButton('恢复默认图');reset.clicked.connect(self.result.reset_plot);row.addWidget(reset);row.addWidget(QLabel('预览 ≤240 点 · 不自动保存'));layout.addLayout(row)
        layout.addWidget(self.result,1);self.splitter.addWidget(right);self.splitter.setStretchFactor(1,1);self.splitter.setSizes([440,850])
        row=QHBoxLayout();p.run.setText('计算并保存成果');row.addWidget(p.run);p.run.show();row.addStretch();root.addLayout(row)
        root.addWidget(p.message);p.message.show();p.message.setMaximumHeight(45)
        self.mapping_rows();p.cal_mode.currentIndexChanged.connect(self.run_title)

    def add_section(self,layout,key,title,body,expanded):
        frame=QWidget();box=QVBoxLayout(frame);box.setContentsMargins(0,0,0,0)
        button=QToolButton();button.setText(title);button.setCheckable(True);button.setChecked(expanded)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon);button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        box.addWidget(button);box.addWidget(body);body.setVisible(expanded)
        button.toggled.connect(lambda checked:(body.setVisible(checked),button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)))
        self.sections[key]=(button,body);layout.addWidget(frame)

    def mapping_rows(self,*_):
        p=self.panel;labels=dict(phi='孔隙度',sw='Sw',toc='TOC',temperature='温度',pressure='压力',vp='Vp / 时差',vs='Vs / 时差',rho='实测密度')
        p.mapping.verticalHeader().hide()
        for row,key in enumerate(FIELDS):p.mapping.setRowHidden(row,key not in ('phi','vp','vs','rho') and not self.more.isChecked());p.mapping.item(row,0).setText(labels[key]);p.mapping.setRowHeight(row,38)
        p.mapping.setMinimumHeight(340 if self.more.isChecked() else 190);p.mapping.setMaximumHeight(340 if self.more.isChecked() else 190)
        p.mapping.setHorizontalHeaderLabels(['输入 / 参考','曲线来源','常数','单位'])

    def run_title(self,*_):self.panel.run.setText('运行标定并保存成果' if self.panel.cal_mode.currentData()!='none' else '计算并保存成果')

    def wire(self):
        p=self.panel
        controls=[p.well,p.name,p.zmin,p.zmax,p.basis,p.mixing,p.normalize,p.clip_negative,p.kerogen,p.interpolate,p.dry_model,p.order]
        controls.extend(w for key,w in p.edits.items() if key in PARAMETERS or key in ('max_gap_m','null_below','carbon_fraction','kerogen_density','kerogen_K','kerogen_G','salinity_ppm','oil_density','gas_gravity','gor'))
        controls.extend(w for triple in p.bindings.values() for w in triple)
        controls.extend(p.minerals.cellWidget(r,c) for r in range(p.minerals.rowCount()) for c in (1,3))
        controls.extend(p.layers.cellWidget(r,2) for r in range(p.layers.rowCount()))
        for widget in controls:
            if widget is None:continue
            if widget.property('liveConnected'):continue
            widget.setProperty('liveConnected',True)
            signal=widget.currentIndexChanged if isinstance(widget,QComboBox) else widget.toggled if isinstance(widget,QCheckBox) else widget.textChanged
            signal.connect(self.changed)
        for table in (p.minerals,p.layers):
            if not table.property('liveConnected'):
                table.setProperty('liveConnected',True);table.itemChanged.connect(self.changed);table.model().rowsRemoved.connect(self.changed)

    def eventFilter(self,watched,event):
        if event.type()==QEvent.Type.Resize:self.reflow()
        elif event.type()==QEvent.Type.Show:self.reflow();QTimer.singleShot(0,self.changed)
        elif event.type()==QEvent.Type.Hide:self.stop()
        return False

    def reflow(self):
        orientation=Qt.Orientation.Vertical if self.panel.width()<1050 else Qt.Orientation.Horizontal
        self.result.heading.setVisible(orientation==Qt.Orientation.Horizontal)
        if self.splitter.orientation()!=orientation:
            self.splitter.setOrientation(orientation)
            self.splitter.setSizes([260,440] if orientation==Qt.Orientation.Vertical else [440,850])

    def auto_changed(self,enabled):
        if enabled:self.changed()
        else:self.stop();self.result.pending('自动更新已暂停',self.panel.well.currentText())

    def changed(self,*_,force=False):
        p=self.panel
        if p.loading:return
        self.revision+=1;self.timer.stop();self.cancel()
        self.update_defaults();self.run_title()
        if self.manual_busy or not p.isVisible() or (not self.enabled.isChecked() and not force):return
        self.result.pending('等待输入稳定后更新',p.well.currentText());self.timer.start()

    def update_defaults(self):
        p=self.panel;items=[]
        for key in ('phi','sw','temperature','pressure'):
            combo,edit,unit=p.bindings[key]
            if not combo.currentData():items.append(FIELDS[key][0]+'='+edit.text()+' '+unit.currentText())
        for r in range(p.minerals.rowCount()):
            if p.minerals.cellWidget(r,1) is None or p.minerals.cellWidget(r,3) is None or p.minerals.item(r,0) is None or p.minerals.item(r,2) is None:continue
            if not p.minerals.cellWidget(r,1).currentText():items.append(p.minerals.item(r,0).text()+'='+p.minerals.item(r,2).text()+' '+p.minerals.cellWidget(r,3).currentText())
        self.defaults.setText('当前常数 / 初值：'+'；'.join(items)+'。可在下方设置中补充或修改。')

    def calculate(self):
        p=self.panel
        if p.loading or self.manual_busy or not p.isVisible():return
        try:
            request=dict(operation='well_live',scheme=p.scheme(for_preview=True),well=p.resolved_well(),preview_limit=240)
            command=worker_command('overpressure.rock_physics.worker')
        except Exception as exc:self.result.pending('请完善输入：'+str(exc),p.well.currentText());return
        revision=self.revision;process=QProcess(self);self.process=process;buffer=[''];completed=[None];errors=[''];decoder=codecs.getincrementaldecoder('utf-8')('replace')
        env=QProcessEnvironment.systemEnvironment();env.insert('PYTHONIOENCODING','utf-8');process.setProcessEnvironment(env);process.setWorkingDirectory(str(application_root()))
        def started():
            if self.process is not process:process.kill();return
            process.write(json.dumps(request,ensure_ascii=False,allow_nan=False).encode('utf-8'));process.closeWriteChannel()
        def read():
            if self.process is not process:return
            buffer[0]+=decoder.decode(bytes(process.readAllStandardOutput()))
            while '\n' in buffer[0]:
                line,buffer[0]=buffer[0].split('\n',1)
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('event')=='live_preview':completed[0]=event['result']
                elif event.get('event')=='error':errors[0]=event.get('message','计算失败')
        def finish(code,*_):
            if self.process is not process:return
            read();self.process=None;process.blockSignals(True);process.deleteLater()
            if revision!=self.revision:return
            if code==0 and completed[0]:self.result.display(completed[0])
            else:self.result.pending('预览未更新：'+(errors[0] or '计算进程未返回结果'),p.well.currentText())
        process.started.connect(started);process.readyReadStandardOutput.connect(read)
        process.readyReadStandardError.connect(lambda:process.readAllStandardError())
        process.finished.connect(finish)
        process.errorOccurred.connect(lambda error:finish(1) if error==QProcess.ProcessError.FailedToStart else None)
        self.result.pending('正在计算最新输入…',p.well.currentText());process.start(*command)

    def cancel(self):
        process=self.process;self.process=None
        if process is not None and process.state()!=QProcess.ProcessState.NotRunning:process.kill()
        # Cancelled Starting QProcesses are retained under this QObject; their
        # guarded started handler kills late starts without native Qt deletion races.

    def stop(self):
        self.revision+=1;self.timer.stop();self.cancel()

    def shutdown(self):
        self.stop()
        for process in self.findChildren(QProcess):
            if process.state()==QProcess.ProcessState.Starting:process.waitForStarted(500)
            if process.state()!=QProcess.ProcessState.NotRunning:process.kill();process.waitForFinished(500)

    def collect_state(self):return dict(enabled=self.enabled.isChecked(),expanded=[key for key,(button,_) in self.sections.items() if button.isChecked()])

    def populate(self,cfg):
        self.shutdown();self.result.clear();self.enabled.blockSignals(True);self.enabled.setChecked(cfg.get('enabled',True));self.enabled.blockSignals(False)
        for key,(button,_) in self.sections.items():button.setChecked(key in cfg.get('expanded',['basic']))
        self.wire();self.changed()


def install_workbench(panel):
    panel.live=LiveWorkbench(panel)

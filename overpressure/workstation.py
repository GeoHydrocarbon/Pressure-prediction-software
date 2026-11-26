"""Desktop workstation composition. Computation stays in the existing panels."""
from pathlib import Path
import json
from PySide6.QtCore import Qt, QTimer, QUrl, QSize,QItemSelectionModel
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QTabWidget,QDockWidget,QTreeWidget,QTreeWidgetItem,QStackedWidget,QScrollArea,
    QProgressBar,QToolBar,QToolButton,QStyle,QSizePolicy,QMenu,QMessageBox,QBoxLayout,QFormLayout,QDialog,QDialogButtonBox,QAbstractItemView,QFileDialog,QLineEdit,QSpinBox,QComboBox,QInputDialog)
from .modules.catalog import MODULES, module_for_page


class DocumentTabs(QTabWidget):
    TITLES=['井曲线','模型训练','时深关系','井预测与验证','成果图表','三维地震预测','地震切片','传统压力预测','岩石物理建模']
    def __init__(self):
        super().__init__();self.setObjectName('documents');self.setDocumentMode(True)
        self.setUsesScrollButtons(True);self.setMinimumSize(0,0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
    def addWidget(self,widget):
        index=self.addTab(widget,self.TITLES[self.count()])
        module=module_for_page(index)
        if module:self.setTabToolTip(index,module.title+'模块 · '+self.TITLES[index])
        return index


class ParameterColumns(QWidget):
    """Two parameter groups share a row on wide screens and stack on small ones."""
    def __init__(self,first,second):
        super().__init__();self.row=QHBoxLayout(self);self.row.setContentsMargins(0,0,0,0)
        self.row.addWidget(first,1,Qt.AlignmentFlag.AlignTop);self.row.addWidget(second,1,Qt.AlignmentFlag.AlignTop)
        self.setMinimumWidth(0)
    def resizeEvent(self,event):
        self.row.setDirection(QBoxLayout.Direction.LeftToRight if self.width()>=850 else QBoxLayout.Direction.TopToBottom)
        super().resizeEvent(event)


def vertical_controls(widget):
    """Reflow action rows inside a narrow dock, preserving controls and signals."""
    for layout in widget.findChildren(QHBoxLayout):
        if layout.property('keepHorizontal'):continue
        layout.setDirection(QBoxLayout.Direction.TopToBottom)
        layout.setSpacing(6)
        for i in range(layout.count()):layout.setStretch(i,0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    for layout in widget.findChildren(QVBoxLayout):layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    for form in widget.findChildren(QFormLayout):
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    for tab in widget.findChildren(QTabWidget):
        tab.setUsesScrollButtons(True)
        tab.tabBar().setExpanding(False)
    for label in widget.findChildren(QLabel):label.setWordWrap(True)
    for child in widget.findChildren(QWidget):
        if child.minimumWidth()>160:child.setMinimumWidth(0)


class WorkstationShell:
    def __init__(self,window):
        self.w=window;self.results=[];self.signature=None;self.parameter_user_visible=True
        self.extra_volumes=[];self.map_signature=None;self.saved_map_state={};self.model_aliases={};self.time_depth_relations=[]
        for panel in (window.train_files,window.well_prediction.log_files,window.well_prediction.point_files):
            panel.list.setMaximumHeight(135)
        self.project_key=str(window.project.path.resolve())
        from .well_library_ui import CurveViewer
        window.pages.blockSignals(True)
        self.data_host=window.pages.widget(0);window.pages.removeTab(0);self.data_host.setParent(window);self.data_host.hide()
        self.curve_viewer=CurveViewer(window);window.pages.insertTab(0,self.curve_viewer,'井曲线')
        window.pages.setCurrentIndex(0);window.pages.blockSignals(False)
        self.data_dialog=QDialog(window);self.data_dialog.setWindowTitle('地震与层位数据属性');self.data_dialog.resize(800,700)
        dialog_layout=QVBoxLayout(self.data_dialog);dialog_layout.addWidget(window.data_tabs)
        window.data_tabs.setTabVisible(0,False)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close);buttons.rejected.connect(self.data_dialog.hide);dialog_layout.addWidget(buttons)
        from .ui_style import WindowBoundsGuard
        self.data_dialog.screen_guard=WindowBoundsGuard(self.data_dialog)
        # Keep task routing indices stable, while moving the complete editor
        # out of the document area into the project object's properties dialog.
        time_depth_slot=window.pages.widget(2);content=time_depth_slot.takeWidget()
        window.pages.setTabVisible(2,False)
        self.time_depth_dialog=QDialog(window);self.time_depth_dialog.setWindowTitle('项目时深关系');self.time_depth_dialog.resize(650,560)
        td_layout=QVBoxLayout(self.time_depth_dialog);td_scroll=QScrollArea();td_scroll.setWidgetResizable(True);td_scroll.setWidget(content);td_layout.addWidget(td_scroll,1)
        content.layout().removeWidget(window.td_check_button)
        td_footer=QHBoxLayout();td_footer.addWidget(window.td_check_button);td_footer.addStretch()
        td_close=QPushButton('关闭');td_close.clicked.connect(self.time_depth_dialog.hide);td_footer.addWidget(td_close);td_layout.addLayout(td_footer)
        self.time_depth_dialog.screen_guard=WindowBoundsGuard(self.time_depth_dialog)
        window.setCentralWidget(window.pages)
        window.setDockOptions(window.DockOption.AllowTabbedDocks | window.DockOption.AllowNestedDocks)
        self.project_dock=self.dock('项目对象','projectObjects',Qt.DockWidgetArea.LeftDockWidgetArea)
        self.tree=QTreeWidget();self.tree.setHeaderHidden(True);self.tree.setIndentation(16)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setMinimumWidth(150);self.tree.setUniformRowHeights(True)
        self.tree.itemDoubleClicked.connect(self.open_item)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.context_menu)
        self.project_dock.setWidget(self.tree)
        self.parameter_dock=self.dock('显示参数','displayParameters',Qt.DockWidgetArea.RightDockWidgetArea)
        self.parameters=QStackedWidget();self.parameters.setMinimumSize(0,0)
        self.parameters.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        self.parameter_dock.setWidget(self.parameters)
        self.install_plot_parameters()
        first,second=window.training_algorithm_card,window.training_physics_card
        training_layout=first.parentWidget().layout();position=training_layout.indexOf(first)
        training_layout.removeWidget(first);training_layout.removeWidget(second)
        training_layout.insertWidget(position,ParameterColumns(first,second))
        self.task_dock=self.dock('任务与日志','taskLog',Qt.DockWidgetArea.BottomDockWidgetArea)
        task=QWidget();layout=QVBoxLayout(task);layout.setContentsMargins(8,6,8,6);layout.setSpacing(6)
        row=QHBoxLayout();window.progress=QProgressBar();window.progress.setRange(0,100);window.progress.setValue(0)
        row.addWidget(window.progress,1);window.cancel_button=QPushButton('取消任务');window.cancel_button.setEnabled(False)
        window.cancel_button.clicked.connect(window.cancel_task);row.addWidget(window.cancel_button);layout.addLayout(row)
        window.log.setMinimumHeight(45);layout.addWidget(window.log,1);self.task_dock.setWidget(task)
        window.status=QLabel('就绪');window.status.setObjectName('statusMessage');window.status.setMinimumWidth(0)
        window.status.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred);window.statusBar().addWidget(window.status,1)
        window.project_label=QLabel(window.project.path.name);window.project_label.setObjectName('projectPath');window.project_label.setToolTip(str(window.project.path))
        window.statusBar().addPermanentWidget(window.project_label)
        from .workspace_views import WorkspaceViews
        self.workspace=WorkspaceViews(self)
        for choice,edit in ((window.well_model_choice,window.bundle_path),(window.seismic_model_choice,window.seismic_bundle)):
            choice.currentIndexChanged.connect(lambda index,c=choice,e=edit:e.setText(c.currentData() or '') if index >= 0 else None)
            edit.textChanged.connect(self.refresh_model_choices)
        self.build_menus()
        window.pages.currentChanged.connect(self.page_changed)
        self.parameter_dock.visibilityChanged.connect(self.parameter_visibility)
        self.timer=QTimer(window);self.timer.setInterval(700);self.timer.timeout.connect(self.refresh_tree);self.timer.start()
        window.resize(1440,940)
        self.reset_layout();self.refresh_tree();self.page_changed(window.pages.currentIndex())

    def dock(self,title,name,area):
        dock=QDockWidget(title,self.w);dock.setObjectName(name)
        # Keep panels docked so closing or dragging never strands controls off-screen.
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setMinimumSize(0,0);self.w.addDockWidget(area,dock);return dock

    def install_plot_parameters(self):
        w=self.w
        scroll=w.slices.settings_scroll;w.slices.page_content_layout.removeWidget(scroll)
        scroll.setMinimumHeight(0);scroll.setMaximumHeight(16777215)
        w.slices.control_tabs.currentChanged.disconnect(w.slices.update_settings_height)
        vertical_controls(scroll.widget());self.parameters.addWidget(scroll)
        w.charts.layout().removeWidget(w.charts.settings_card)
        vertical_controls(w.charts.settings_card)
        chart_scroll=QScrollArea();chart_scroll.setWidgetResizable(True);chart_scroll.setWidget(w.charts.settings_card)
        self.parameters.addWidget(chart_scroll)
        # Headings are supplied by the document tab; keep plots close to their toolbar.
        for panel in (w.charts,w.slices):
            for label in panel.findChildren(QLabel):
                if label.objectName()=='heading':label.hide()
        self.parameter_pages={6:0,4:1}

    def action(self,menu,text,callback,shortcut=None,icon=None,guard=False):
        act=QAction(text,self.w)
        if shortcut:act.setShortcut(shortcut)
        if icon is not None:act.setIcon(self.w.style().standardIcon(icon))
        act.triggered.connect(lambda checked=False:callback());menu.addAction(act)
        if guard:self.w.actions.append(act)
        return act

    def show_page(self,index,tab=None):
        if index==2:self.open_time_depth();return
        if index==0 and tab in (1,2):
            self.w.data_tabs.setCurrentIndex(tab);self.data_dialog.show();self.data_dialog.raise_()
            if tab==1:self.w.geometry_timer.start()
            return
        self.w.nav.setCurrentRow(index)
        self.w.pages.setCurrentIndex(index)
        self.workspace.show_page(index)
        if tab is not None:self.w.data_tabs.setCurrentIndex(tab)

    def open_time_depth(self,method=None):
        if method is not None:self.w.td_method.setCurrentText(method)
        self.time_depth_dialog.show();self.time_depth_dialog.raise_();self.time_depth_dialog.activateWindow()

    def load_time_depth(self):
        path,_=QFileDialog.getOpenFileName(self.w,'加载时深关系表',str(self.w.project.path.parent),'时深关系表 (*.xlsx *.csv)')
        if not path:return
        self.capture_time_depth()
        self.w.td_file.setText(str(Path(path).resolve()));self.open_time_depth('table')
        self.w.status.setText('已选择时深表；请核对单位和单/双程时间，并保存检查。')
        self.signature=None;self.refresh_tree()

    def remember_time_depth(self,config):
        from .time_depth_library import relation
        entry=relation(self.w.project,config)
        if not any(r['id']==entry['id'] for r in self.time_depth_relations):self.time_depth_relations.append(entry)
        return entry

    def capture_time_depth(self):
        try:
            cfg=dict(method=self.w.td_method.currentText(),table_file=self.w.td_file.text().strip(),
                     coefficients=[float(v.strip()) for v in self.w.coefficients.text().replace('，',',').split(',') if v.strip()] if self.w.td_method.currentText()=='polynomial' else [],
                     time_unit=self.w.td_unit.currentText(),time_convention=self.w.td_convention.currentText(),depth_unit=self.w.td_depth_unit.currentText())
            return self.remember_time_depth(cfg)
        except (ValueError,OSError,KeyError):return None

    def apply_well_time_depth(self,name):
        well=next((w for w in self.w.library.wells if w['name']==name),None)
        if well is None:return
        self.capture_time_depth()
        entries=self.time_depth_relations
        labels=['不应用时深关系']+[entry['name'] for entry in entries]
        for i,entry in enumerate(entries,1):
            if labels.count(entry['name'])>1:labels[i]=entry['name']+' · '+entry['config'].get('table_file',entry['id'])
        index=next((i+1 for i,r in enumerate(entries) if r['id']==well.get('time_depth_id')),0)
        label,ok=QInputDialog.getItem(self.w,'应用时深关系 · '+name,'选择已加载的时深关系',labels,index,False)
        if not ok:return
        previous=well.get('time_depth_id');selected=labels.index(label)
        if selected:well['time_depth_id']=entries[selected-1]['id']
        else:well.pop('time_depth_id',None)
        try:self.persist_workspace()
        except OSError as exc:
            if previous:well['time_depth_id']=previous
            else:well.pop('time_depth_id',None)
            QMessageBox.warning(self.w,'关系未保存',str(exc));return
        self.signature=None;self.refresh_tree();self.w.status.setText(name+'：'+label)

    def open_saved_time_depth(self,ident):
        entry=next((r for r in self.time_depth_relations if r['id']==ident),None)
        if entry is None:return
        self.capture_time_depth();cfg=entry['config'];w=self.w
        w.td_method.setCurrentText(cfg['method']);w.td_file.setText(cfg.get('table_file',''))
        if 'coefficients' in cfg:w.coefficients.setText(', '.join(map(str,cfg['coefficients'])))
        w.td_unit.setCurrentText(cfg['time_unit']);w.td_convention.setCurrentText(cfg['time_convention']);w.td_depth_unit.setCurrentText(cfg['depth_unit'])
        self.open_time_depth()

    def import_batch(self,panel,page):
        self.show_page(page);panel.choose_files();self.refresh_tree()

    def import_asset(self,kind):
        self.w.library.import_spatial(kind);self.refresh_tree()

    def build_menus(self):
        w=self.w;bar=w.menuBar()
        project=bar.addMenu('项目(&P)');data=bar.addMenu('数据(&D)')
        self.module_actions={}
        for spec in MODULES:
            if spec.key=='training':
                training=bar.addMenu(spec.title);self.module_actions[spec.key]=training.menuAction()
                from .algorithms.registry import ALGORITHM_SPECS
                self.training_actions={}
                for algorithm in ALGORITHM_SPECS.values():
                    self.training_actions[algorithm.name]=self.action(training,algorithm.title,lambda name=algorithm.name:self.open_training(name),guard=True)
            elif spec.key=='results':
                results=bar.addMenu(spec.title)
            else:
                self.module_actions[spec.key]=self.action(bar,spec.title,lambda page=spec.pages[0]:self.show_page(page))
        view=bar.addMenu('视图(&V)')
        self.action(project,'新建项目…',w.new_project,'Ctrl+N',guard=True)
        opened=self.action(project,'打开项目…',w.open_project,'Ctrl+O',QStyle.StandardPixmap.SP_DirOpenIcon,True)
        saved=self.action(project,'保存项目设置',w.save_settings,'Ctrl+S',QStyle.StandardPixmap.SP_DialogSaveButton,True)
        self.action(project,'项目另存为…',w.save_project_as,'Ctrl+Shift+S',guard=True)
        outputs=self.action(project,'成果目录',w.open_outputs,icon=QStyle.StandardPixmap.SP_DirIcon)
        project.addSeparator();self.action(project,'退出',w.close,'Alt+F4')
        imports=data.addMenu('导入数据')
        self.action(imports,'地震体 SEG-Y（显示 / 拉剖面）…',self.import_seismic,guard=True)
        self.action(imports,'井与测井曲线…',w.library.import_files,guard=True)
        self.action(imports,'实测压力点…',lambda:w.library.import_files(measured=True),guard=True)
        for title,kind in [('井口坐标…','coordinates'),('井轨迹…','trajectory'),('井震对比曲线…','curves')]:
            self.action(imports,title,lambda k=kind:self.import_asset(k),guard=True)
        imports.addSeparator()
        self.action(imports,'地震输入与单位…',lambda:self.show_page(0,1))
        data.addSeparator()
        self.action(data,'层位文件管理…',lambda:self.show_page(0,2))
        self.action(data,'检查井数据与地震对应关系',lambda:w.start_task('audit',[]),guard=True)
        self.action(results,'井曲线',lambda:self.show_page(0),'Ctrl+1')
        self.action(results,'成果图表',lambda:self.show_page(4),'Ctrl+5')
        self.action(results,'地震切片',lambda:self.show_page(6),'Ctrl+6')
        self.action(results,'项目底图 Map',self.workspace.show_map,'Ctrl+M')
        windows=bar.addMenu('窗口(&W)')
        self.action(windows,'平铺绘图窗口',self.workspace.tile)
        self.action(windows,'层叠绘图窗口',self.workspace.area.cascadeSubWindows)
        view.addAction(self.project_dock.toggleViewAction());view.addAction(self.task_dock.toggleViewAction())
        self.parameter_action=QAction('显示参数面板',w);self.parameter_action.setCheckable(True);self.parameter_action.setChecked(True)
        self.parameter_action.triggered.connect(self.toggle_parameters);view.addAction(self.parameter_action)
        self.action(view,'恢复默认布局',self.reset_layout)
        toolbar=QToolBar('常用操作',w);toolbar.setObjectName('mainTools');toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18,18));toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon);w.addToolBar(toolbar)
        toolbar.addAction(opened);toolbar.addAction(saved);toolbar.addSeparator()
        import_button=QToolButton();import_button.setText('导入数据');import_button.setIcon(w.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        import_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon);import_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup);import_button.setMenu(imports)
        toolbar.addWidget(import_button);toolbar.addAction(outputs)
        toolbar.addSeparator();self.action(toolbar,'底图 Map',self.workspace.show_map)
        self.action(toolbar,'平铺窗口',self.workspace.tile)

    def open_training(self,name):
        self.w.algorithm.setCurrentIndex(self.w.algorithm.findData(name));self.show_page(1)

    def page_changed(self,index):
        if index in self.parameter_pages:
            self.parameters.setCurrentIndex(self.parameter_pages[index])
            self.parameter_dock.setWindowTitle('地震切片参数' if index==6 else '成果图表参数')
            self._switching=True;self.parameter_dock.setVisible(self.parameter_user_visible);self._switching=False
        else:
            self._switching=True;self.parameter_dock.hide();self._switching=False

    def parameter_visibility(self,visible):
        if not getattr(self,'_switching',False) and self.w.pages.currentIndex() in self.parameter_pages:
            self.parameter_user_visible=visible;self.parameter_action.setChecked(visible)

    def toggle_parameters(self,visible):
        self.parameter_user_visible=visible;self.page_changed(self.w.pages.currentIndex())

    def reveal_parameters(self):
        if self.w.pages.currentIndex() not in self.parameter_pages:self.show_page(6)
        self.parameter_user_visible=True;self.parameter_action.setChecked(True);self.page_changed(self.w.pages.currentIndex())

    def well_matching_settings(self):
        dialog=QDialog(self.w);dialog.setWindowTitle('井震空间匹配设置');layout=QVBoxLayout(dialog)
        form=QFormLayout();distance=self.w.catalog.distance;vertical=self.w.catalog.vertical
        form.addRow('井到地震道最大距离',distance);form.addRow(vertical);layout.addLayout(form)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close);buttons.rejected.connect(dialog.accept);layout.addWidget(buttons)
        # Restore ownership after closing; these are the original project controls.
        dialog.exec();distance.setParent(self.w.catalog);vertical.setParent(self.w.catalog);distance.hide();vertical.hide()

    def reset_layout(self):
        w=self.w
        for dock,area in [(self.project_dock,Qt.DockWidgetArea.LeftDockWidgetArea),(self.parameter_dock,Qt.DockWidgetArea.RightDockWidgetArea),(self.task_dock,Qt.DockWidgetArea.BottomDockWidgetArea)]:
            w.addDockWidget(area,dock)
        self.project_dock.show();self.task_dock.show();self.parameter_user_visible=True
        if hasattr(self,'parameter_action'):self.parameter_action.setChecked(True)
        w.resizeDocks([self.project_dock,self.parameter_dock],[220,320],Qt.Orientation.Horizontal)
        w.resizeDocks([self.task_dock],[140],Qt.Orientation.Vertical)
        self.page_changed(w.pages.currentIndex())

    def project_changed(self):
        self.w.project_label.setToolTip(str(self.w.project.path))
        self.w.setWindowTitle(f'{self.w.settings.get("name",self.w.project.path.stem)} · 超压预测工作站')
        key=str(self.w.project.path.resolve())
        if key!=self.project_key:
            self.results=[];self.project_key=key
            self.curve_viewer.reset()
            self.workspace.reset_project()
        state=self.w.settings.get('workspace',{})
        self.results=list(state.get('results',[]))
        self.model_aliases=dict(state.get('model_aliases',{}))
        self.time_depth_relations=list(state.get('time_depth_relations',[]))
        self.extra_volumes=list(state.get('seismic_volumes',[]))
        self.saved_map_state=dict(state.get('map',{}))
        for attr,key in [('bundle_path','well_model'),('seismic_bundle','seismic_model'),('refit_cv','refit_cv')]:
            getattr(self.w,attr).setText(state.get(key,''))
        self.map_signature=None
        self.signature=None;self.refresh_tree()

    def register_result(self,event):
        if self.w.task_command=='cross-validate' and event.get('artifact') is None:return
        artifact=event.get('artifact') or ('final_model' if self.w.task_command in ('train','refit',None) else self.w.task_command)
        if artifact in ('audit','seismic_index','seismic_view'):return
        record={'path':event['path'],'artifact':artifact,'smoke':event.get('smoke',False)}
        if record not in self.results:self.results.append(record)
        self.signature=None;self.refresh_tree()

    def workspace_settings(self):
        panel=self.workspace.map
        map_state=dict(source=(panel.dataset.currentData() or {}).get('path',''),coordinates=panel.coordinates.currentData(),
                       time_ms=panel.time.value(),time_slice=panel.attribute.isChecked(),equal_scale=panel.equal_scale.isChecked(),
                       sections=[dict(record,view_id=key) for key,record in list(panel.paths.items())[:64]])
        return dict(results=self.results,seismic_volumes=self.extra_volumes,model_aliases=self.model_aliases,time_depth_relations=self.time_depth_relations,
                    well_model=self.w.bundle_path.text().strip(),seismic_model=self.w.seismic_bundle.text().strip(),
                    refit_cv=self.w.refit_cv.text().strip(),map=map_state)

    def persist_workspace(self):
        """Persist completed artifacts without collecting partially edited parameters."""
        from copy import deepcopy
        import yaml
        from .config import Project
        cfg=deepcopy(self.w.settings);cfg['workspace']=deepcopy(self.workspace_settings())
        cfg.setdefault('project_data',{})['pressure_result']=self.w.pressure_result.text().strip()
        cfg['project_data']['wells']=deepcopy(self.w.library.wells)
        cfg['project_data']['well_assets']=deepcopy(self.w.catalog.assets)
        cfg['rock_physics']=self.w.rock_physics.collect_state()
        cfg['traditional_prediction']=self.w.traditional.collect()
        temp=self.w.project.path.with_suffix('.yaml.tmp')
        temp.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        temp.replace(self.w.project.path)
        self.w.settings=cfg;self.w.project=Project(self.w.project.path,cfg)

    def map_sources(self):
        w=self.w;headers={key:edit.value() for key,edit in w.header_edits.items()}
        sources=[]
        for name,edit,unit in [('密度',w.den_path,w.den_unit.currentText()),('波阻抗',w.ip_path,w.ip_unit.currentText())]:
            if edit.text().strip():sources.append(dict(label=name+' · '+Path(edit.text().strip()).name,path=str(w.project.resolve(edit.text().strip())),unit=unit,positive_only=True))
        sources.extend(dict(item,path=str(w.project.resolve(item['path']))) for item in self.extra_volumes)
        for result in self.results:
            if result['artifact'] in ('seismic_volume','seismic_trial'):
                sources.append(dict(path=result['path'],label='预测压力 · '+Path(result['path']).name,unit='MPa'))
        if w.pressure_result.text().strip():
            path=str(w.project.resolve(w.pressure_result.text().strip()))
            sources.append(dict(path=path,label='预测压力 · '+Path(path).name,unit='MPa'))
        unique={}
        assets=[]
        for asset in w.catalog.assets:
            if asset.get('kind') in ('coordinates','trajectory'):assets.append(dict(asset,path=str(w.project.resolve(asset['path']))))
        from copy import deepcopy
        library=deepcopy(w.library.wells)
        for well in library:
            for ref in well.get('curves',[]):ref['path']=str(w.project.resolve(ref['path']))
        for item in sources:
            item=dict(item);item.setdefault('headers',headers);item.setdefault('time_convention',w.seismic_convention.currentText())
            item['well_assets']=assets;item['well_max_distance']=w.catalog.distance.value()
            item['well_library']=library;item['time_depth_relations']=deepcopy(self.time_depth_relations);item['project_path']=str(w.project.path)
            unique.setdefault(item['path'],item)
        # A per-file import mapping must win over a prediction-input reference
        # to the same file; otherwise the user's header edits are silently lost.
        for entry in self.extra_volumes:
            path=str(w.project.resolve(entry['path']))
            if path in unique:
                unique[path].update({key:entry[key] for key in ('headers','time_convention','label','unit') if key in entry})
        return list(unique.values())

    def refresh_map_sources(self):
        sources=self.map_sources();signature=json.dumps(sources,sort_keys=True,ensure_ascii=False)
        if signature!=self.map_signature:
            self.map_signature=signature
            selected={'path':self.saved_map_state.get('source','')} if self.workspace.map.desired_config is None and self.saved_map_state else None
            self.workspace.map.set_datasets(sources,selected)

    def restore_map_state(self,metadata):
        state=self.saved_map_state
        if not state or str(self.w.project.resolve(state.get('source','')))!=metadata['path']:return
        self.saved_map_state={};panel=self.workspace.map
        if state.get('coordinates')=='lines':panel.coordinates.setCurrentIndex(0)
        panel.equal_scale.setChecked(state.get('equal_scale',False))
        panel.time.setValue(state.get('time_ms',panel.time.value()))
        panel.attribute.setChecked(state.get('time_slice',False))
        from copy import deepcopy
        for record in deepcopy(state.get('sections',[])[:64]):
            if record.get('mode') not in ('inline','crossline','arbitrary','polyline'):continue
            ident=record.pop('view_id',None)
            if ident:
                panel.paths[ident]=record;panel.active_line=ident;panel.request_active_section()

    def edit_seismic_source(self,record):
        if not record.get('extra_volume'):
            self.show_page(0,1);return
        entry=next((item for item in self.extra_volumes if item['path']==record['path']),None)
        if entry is None:return
        from .seismic_import import SeismicImportDialog
        initial=dict(headers={key:edit.value() for key,edit in self.w.header_edits.items()},time_convention=self.w.seismic_convention.currentText())
        initial.update(entry)
        dialog=SeismicImportDialog(self.w,self.w.project.resolve(entry['path']),initial)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            entry.update(dialog.values())
            self.refresh_tree()
            self.persist_workspace()

    def import_seismic(self):
        paths,_=QFileDialog.getOpenFileNames(self.w,'导入地震体用于底图与剖面',str(self.w.project.path.parent),'SEG-Y (*.sgy *.segy);;所有文件 (*)')
        if not paths:return
        from .seismic_import import SeismicImportDialog
        imported=[]
        for path in paths:
            target=str(Path(path).resolve())
            existing=next((item for item in self.extra_volumes if str(self.w.project.resolve(item['path']))==target),None)
            initial=dict(headers={key:edit.value() for key,edit in self.w.header_edits.items()},time_convention=self.w.seismic_convention.currentText())
            if existing:initial.update(existing)
            dialog=SeismicImportDialog(self.w,target,initial)
            if dialog.exec()!=QDialog.DialogCode.Accepted:break
            if existing is not None:existing.update(dialog.values())
            else:self.extra_volumes.append(dialog.values())
            imported.append(target)
        if not imported:return
        self.signature=None;self.refresh_tree();self.workspace.show_map()
        self.open_map_source(imported[-1])
        self.persist_workspace()

    def open_map_source(self,path):
        self.refresh_map_sources();panel=self.workspace.map;target=str(self.w.project.resolve(path))
        for i in range(panel.dataset.count()):
            if panel.dataset.itemData(i)['path']==target:
                panel.dataset.setCurrentIndex(i);break
        self.workspace.show_map()

    def entries(self):
        w=self.w
        groups={key:[] for key in ('井','地震输入','时深关系','层位','模型','预测与验证成果')}
        table=w.td_method.currentText()=='table';unit=w.td_unit.currentText();depth=w.td_depth_unit.currentText();convention='双程' if w.td_convention.currentText()=='twt' else '单程'
        path=w.td_file.text().strip() if table else ''
        label=('表：'+(Path(path).name if path else '未选择文件')) if table else f'公式（{unit} → {depth}）'
        details=(str(w.project.resolve(path)) if path else '未选择时深表') if table else '公式系数（常数、一次、二次、三次）：'+w.coefficients.text()
        details+=f'\n时间：{unit}，{convention}；输出深度：{depth}'
        groups['时深关系'].append((label,dict(page=2,time_depth=True,path=path,details=details)))
        for entry in self.time_depth_relations:
            cfg=entry['config'];details=cfg.get('table_file','公式系数：'+str(cfg.get('coefficients',[])))
            groups['时深关系'].append(('已加载 · '+entry['name'],dict(page=2,time_depth=True,relation_id=entry['id'],path=cfg.get('table_file',''),details=details)))
        for well in w.library.wells:
            children=[]
            entry=next((r for r in self.time_depth_relations if r['id']==well.get('time_depth_id')),None)
            if entry:children.append(('时深关系 · '+entry['name'],dict(well=well['name'],well_time_depth=True,page=0)))
            for curve in well['curves']:
                label=curve['name']+(' · 实测点' if curve.get('role')=='measurement' else '')
                children.append((label,dict(page=0,well=well['name'],curve=curve['name'],path=curve['path'])))
            for index,asset in enumerate(w.catalog.assets):
                if well['name'] not in asset.get('wells',[asset.get('well_name','')]):continue
                label={'curves':'剖面对比曲线','coordinates':'井口坐标','trajectory':'井轨迹'}[asset['kind']]
                children.append((label+' · '+Path(asset['path']).name,dict(page=0,well=well['name'],path=asset['path'],asset=index)))
            groups['井'].append((well['name'],dict(page=0,well=well['name'],children=children)))
        for name,edit in [('密度',w.den_path),('波阻抗',w.ip_path)]:
            if edit.text().strip():groups['地震输入'].append((name+' · '+Path(edit.text()).name,dict(path=edit.text().strip(),seismic_source=True)))
        for item in self.extra_volumes:
            groups['地震输入'].append((item['label'],dict(path=item['path'],seismic_source=True,extra_volume=True)))
        loaded_sources={str(Path(entry['source']).resolve()) for entry in w.slices.horizon_entries.values()}
        for i,cfg in enumerate(w.catalog.horizons):
            if str(w.project.resolve(cfg['path'])) not in loaded_sources:groups['层位'].append((cfg.get('display_name',Path(cfg['path']).name),dict(page=0,tab=2,path=cfg['path'],horizon=i)))
        for key,cfg in w.slices.horizon_entries.items():groups['层位'].append((cfg['label'],dict(page=6,path=cfg['source'],layer=key)))
        for path in self.model_paths():
            groups['模型'].append((self.model_label(path),dict(page=3,path=path,model=True)))
        paths={}
        if w.pressure_result.text().strip():paths[w.pressure_result.text().strip()]='seismic_volume'
        for result in self.results:
            if result['artifact']=='final_model':
                continue
            if result['artifact'] not in ('seismic_index','seismic_view'):paths[result['path']]=result['artifact']
        for path,kind in paths.items():
            if kind=='final_model':continue
            prefix={'traditional_trend':'正常压实趋势 · ','traditional_prediction':'Eaton 预测 · ','rock_physics':'岩石物理 · '}.get(kind,'')
            groups['预测与验证成果'].append((prefix+Path(path).name,dict(page=6 if kind in ('seismic_volume','seismic_trial') else 4,path=path,result=kind)))
        return groups

    def refresh_tree(self):
        self.refresh_model_choices()
        self.refresh_map_sources()
        groups=self.entries();signature=json.dumps(groups,ensure_ascii=False,sort_keys=True)
        if signature==self.signature:return
        self.signature=signature
        expanded={self.tree.topLevelItem(i).text(0).split('  (')[0]:self.tree.topLevelItem(i).isExpanded() for i in range(self.tree.topLevelItemCount())}
        selected=self.tree.currentItem();selected_data=selected.data(0,Qt.ItemDataRole.UserRole) if selected else None
        selection=[item.data(0,Qt.ItemDataRole.UserRole) for item in self.tree.selectedItems()]
        expanded_nodes={}
        def remember(node):
            key=json.dumps(node.data(0,Qt.ItemDataRole.UserRole) or {},sort_keys=True,ensure_ascii=False)
            expanded_nodes[key]=node.isExpanded()
            for i in range(node.childCount()):remember(node.child(i))
        for i in range(self.tree.topLevelItemCount()):remember(self.tree.topLevelItem(i))
        self.tree.clear()
        routes={'井':(0,None),'地震输入':(0,1),'时深关系':(2,None),'层位':(0,2),'模型':(1,None),'预测与验证成果':(4,None)}
        def append(parent,label,record):
            children=record.get('children',[]);data={key:value for key,value in record.items() if key!='children'}
            child=QTreeWidgetItem([label]);child.setData(0,Qt.ItemDataRole.UserRole,data);child.setToolTip(0,record.get('details',record.get('path','')));parent.addChild(child)
            child.setIcon(0,self.w.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon if children else QStyle.StandardPixmap.SP_FileIcon))
            if data==selected_data:self.tree.setCurrentItem(child,0,QItemSelectionModel.SelectionFlag.NoUpdate)
            if data in selection:child.setSelected(True)
            for name,item in children:append(child,name,item)
            child.setExpanded(expanded_nodes.get(json.dumps(data,sort_keys=True,ensure_ascii=False),False))
        for title,items in groups.items():
            parent=QTreeWidgetItem([f'{title}  ({len(items)})']);page,tab=routes[title]
            parent.setData(0,Qt.ItemDataRole.UserRole,dict(page=page,tab=tab,group=title));parent.setIcon(0,self.w.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon));self.tree.addTopLevelItem(parent)
            parent.setExpanded(expanded.get(title,True))
            for label,record in items:
                append(parent,label,record)

    def open_item(self,item,*_):
        record=item.data(0,Qt.ItemDataRole.UserRole) or {}
        if record.get('relation_id'):self.open_saved_time_depth(record['relation_id']);return
        if record.get('well_time_depth'):self.apply_well_time_depth(record['well']);return
        if record.get('result')=='rock_physics':
            try:self.workspace.open_rock_physics(self.w.project.resolve(record['path']))
            except Exception as exc:QMessageBox.warning(self.w,'无法查看岩石物理成果',str(exc))
            return
        if record.get('seismic_source'):
            self.open_map_source(record['path']);return
        if record.get('result') in ('seismic_volume','seismic_trial'):
            self.open_map_source(record['path']);return
        if record.get('result') in ('traditional_trend','traditional_prediction'):
            try:self.workspace.open_traditional(self.w.project.resolve(record['path']))
            except Exception as exc:QMessageBox.warning(self.w,'无法查看传统方法成果',str(exc))
            return
        self.show_page(record.get('page',0),record.get('tab'))
        if 'curve' in record:
            well=next((well for well in self.w.library.wells if well['name']==record['well']),None)
            if well is not None:
                curve=next((curve for curve in well['curves'] if curve['name']==record['curve']),None)
                if curve:self.curve_viewer.show_curve(well['name'],curve)
        if 'asset' in record:
            index=record['asset']
            if index<len(self.w.catalog.assets):
                self.w.catalog.list.setCurrentRow(index);self.w.catalog.edit_asset()
        if record.get('model'):
            path=str(self.w.project.resolve(record['path']));self.w.bundle_path.setText(path);self.w.seismic_bundle.setText(path)
            self.w.status.setText('已选择预测模型：'+self.model_label(path))
        if 'horizon' in record:self.w.catalog.hlist.setCurrentRow(record['horizon'])
        if 'layer' in record:
            self.w.slices.control_tabs.setCurrentIndex(1)
            for i in range(self.w.slices.horizon_list.count()):
                candidate=self.w.slices.horizon_list.item(i)
                if candidate.data(Qt.ItemDataRole.UserRole)==record['layer']:self.w.slices.horizon_list.setCurrentRow(i)
        if 'result' in record:
            path=str(self.w.project.resolve(record['path']))
            if record['page']==6:
                self.w.slices.result_dir.setText(path);self.w.slices.load_result()
            else:
                try:self.w.charts.load_directory(path)
                except Exception as exc:QMessageBox.warning(self.w,'无法查看成果',str(exc))

    def context_menu(self,pos):
        item=self.tree.itemAt(pos)
        if item is None:return
        record=item.data(0,Qt.ItemDataRole.UserRole) or {};menu=QMenu(self.tree)
        self.build_context_menu(menu,record,item)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def build_context_menu(self,menu,record,item=None):
        well=record.get('well','')
        time_depth=record.get('group')=='时深关系' or record.get('time_depth')
        if time_depth:
            action=menu.addAction('加载时深关系…',self.load_time_depth);action.setEnabled(self.w.process is None)
            menu.addAction('设置转换公式…',lambda:self.open_time_depth('polynomial'))
            menu.addAction('查看 / 修改参数…',lambda:self.open_saved_time_depth(record['relation_id']) if record.get('relation_id') else self.open_time_depth())
        if 'curve' in record:
            curves=[dict(selected.data(0,Qt.ItemDataRole.UserRole)) for selected in self.tree.selectedItems() if 'curve' in (selected.data(0,Qt.ItemDataRole.UserRole) or {})]
            if len(curves)>1:
                action=menu.addAction(f'绘制所选曲线（{len(curves)} 条）',lambda records=curves:self.plot_curve_objects(records))
                action.setEnabled(len({entry['well'] for entry in curves})==1)
                action.setToolTip('第一版支持同一口井的多条曲线，共用深度轴。')
                overlay=menu.addAction(f'叠加所选曲线（{len(curves)} 条）',lambda records=curves:self.plot_curve_objects(records,overlay=True))
                overlay.setEnabled(len({entry['well'] for entry in curves})==1)
                overlay.setToolTip('同井曲线叠加在一个图框；相同单位共用横轴，不同单位使用独立横轴。')
                menu.addSeparator()
        if record.get('group')=='井' or well:
            menu.addAction('导入测井曲线…' if well else '批量导入井与测井曲线…',lambda:self.w.library.import_files(well=well))
            menu.addAction('导入实测压力点…',lambda:self.w.library.import_files(well=well,measured=True))
            menu.addAction('导入井口坐标…',lambda:self.w.library.import_spatial('coordinates',well))
            menu.addAction('导入井轨迹…',lambda:self.w.library.import_spatial('trajectory',well))
            if well:menu.addAction('应用时深关系…',lambda:self.apply_well_time_depth(well))
            menu.addSeparator();menu.addAction('选择训练井与曲线',lambda:self.show_page(1))
            menu.addAction('批量井预测与验证',lambda:self.show_page(3));menu.addSeparator()
            menu.addAction('传统压力预测',lambda:self.open_traditional_for_well(well))
            menu.addAction('井震空间匹配设置…',self.well_matching_settings)
        if record.get('group')=='地震输入':
            menu.addAction('导入地震体 SEG-Y…',self.import_seismic)
            menu.addAction('预测属性与道头设置…',lambda:self.show_page(0,1))
        if record.get('seismic_source'):
            menu.addAction('显示底图 / 拉剖面',lambda:self.open_map_source(record['path']))
            menu.addAction('地震属性与道头设置…',lambda:self.edit_seismic_source(record))
        if record.get('result') in ('seismic_volume','seismic_trial'):
            menu.addAction('显示底图 / 拉剖面',lambda:self.open_map_source(record['path']))
            menu.addAction('层位切片与井震对比…',lambda:self.open_pressure_tools(record['path']))
        if record.get('group')=='层位':menu.addAction('导入层位…',self.w.catalog.add_horizon)
        if 'layer' in record or 'horizon' in record:
            menu.addAction('空间插值…',lambda:self.interpolate_horizon_object(record))
            report_path=self.w.slices.horizon_entries.get(record.get('layer'),{}).get('interpolation_report','')
            if 'horizon' in record:report_path=self.w.catalog.horizons[record['horizon']].get('interpolation_report','')
            if report_path:menu.addAction('查看插值报告',lambda path=report_path:self.show_horizon_interpolation_report(path))
        if item is not None and not time_depth:menu.addAction('打开 / 管理',lambda:self.open_item(item))
        if record.get('path'):
            path=self.w.project.resolve(record['path'])
            menu.addAction('打开所在文件夹',lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent))))
            menu.addAction('查看路径',lambda:QMessageBox.information(self.w,'对象路径',str(path)))
        if record.get('model'):
            menu.addAction('重命名…',lambda:self.rename_model(record['path']))
            menu.addAction('用于井预测',lambda:self.select_model(record['path'],3))
            menu.addAction('用于地震预测',lambda:self.select_model(record['path'],5))
            menu.addAction('查看训练图表',lambda:self.open_model_charts(record['path']))
        if 'curve' in record:menu.addAction('曲线属性',lambda:self.curve_properties(record))
        if any(key in record for key in ('batch','asset','horizon','layer','well','extra_volume')):
            menu.addAction('移除引用（保留原始文件）',lambda:self.remove_reference(record))

    def open_model_charts(self,path):
        try:self.w.charts.load_directory(self.w.project.resolve(path));self.show_page(4)
        except Exception as exc:QMessageBox.warning(self.w,'无法查看训练图表',str(exc))

    def model_paths(self):
        paths=[r['path'] for r in self.results if r['artifact']=='final_model']
        paths += [self.w.bundle_path.text().strip(), self.w.seismic_bundle.text().strip()]
        return list(dict.fromkeys(str(self.w.project.resolve(path)) for path in paths if path))

    def model_label(self,path):
        from .model_names import model_info
        key=str(self.w.project.resolve(path));info=model_info(key)
        return f"{self.model_aliases.get(key,info['name'])} [{info['algorithm']}]"+('（试算）' if info['smoke'] else '')

    def refresh_model_choices(self,*_):
        paths=self.model_paths()
        for choice,edit in ((self.w.well_model_choice,self.w.bundle_path),(self.w.seismic_model_choice,self.w.seismic_bundle)):
            current=str(self.w.project.resolve(edit.text().strip())) if edit.text().strip() else ''
            entries=[('请选择项目模型','')]+[(self.model_label(path),path) for path in paths]
            if entries != [(choice.itemText(i),choice.itemData(i)) for i in range(choice.count())]:
                choice.blockSignals(True);choice.clear()
                for label,path in entries:
                    choice.addItem(label,path);choice.setItemData(choice.count()-1,path,Qt.ItemDataRole.ToolTipRole)
                choice.blockSignals(False)
            choice.blockSignals(True);choice.setCurrentIndex(max(0,choice.findData(current)));choice.blockSignals(False)

    def select_model(self,path,page):
        edit=self.w.bundle_path if page==3 else self.w.seismic_bundle
        edit.setText(str(self.w.project.resolve(path)));self.show_page(page)

    def rename_model(self,path):
        from .model_names import model_info
        key=str(self.w.project.resolve(path));old=self.model_aliases.get(key,model_info(key)['name'])
        name,ok=QInputDialog.getText(self.w,'重命名模型','模型显示名称（不改变模型目录）',text=old)
        if not ok:return
        name=name.strip()
        if not name or len(name)>100 or any(ord(c)<32 for c in name):
            QMessageBox.warning(self.w,'名称无效','请输入 1～100 个字符的名称，不含换行等控制字符。');return
        if any(name.casefold()==self.model_aliases.get(other,model_info(other)['name']).casefold() for other in self.model_paths() if other!=key):
            QMessageBox.warning(self.w,'名称重复','项目中已有同名模型，请使用其他名称。');return
        previous=dict(self.model_aliases);self.model_aliases[key]=name
        try:self.persist_workspace()
        except OSError as exc:
            self.model_aliases=previous;QMessageBox.warning(self.w,'名称未保存',str(exc));return
        self.signature=None;self.refresh_tree();self.w.status.setText('模型已重命名：'+name)

    def open_traditional_for_well(self,well):
        self.show_page(7)
        if well:self.w.traditional.analysis_well.setCurrentText(well)

    def open_pressure_tools(self,path):
        self.show_page(6);self.w.slices.result_dir.setText(str(self.w.project.resolve(path)));self.w.slices.load_result()

    def plot_curve_objects(self,records,overlay=False):
        if not records:return
        names={record['well'] for record in records}
        if len(names)!=1:
            QMessageBox.information(self.w,'选择同一口井','请先选择同一口井下的多条曲线；跨井对比后续单独提供。');return
        name=next(iter(names));well=next((well for well in self.w.library.wells if well['name']==name),None)
        if well is None:return
        curves={curve['name']:curve for curve in well['curves']};selected=list(dict.fromkeys(record['curve'] for record in records))
        if any(key not in curves for key in selected):
            QMessageBox.information(self.w,'曲线已更新','部分曲线已移除，请重新选择。');return
        self.show_page(0);self.curve_viewer.show_curves(name,[curves[key] for key in selected],overlay=overlay)

    def interpolate_horizon_object(self,record):
        self.show_page(6)
        if 'layer' in record:self.w.slices.open_horizon_interpolation(record['layer'])
        elif 'horizon' in record:
            config=dict(self.w.catalog.horizons[record['horizon']]);config['path']=str(self.w.project.resolve(config['path']))
            self.w.slices.open_horizon_interpolation(config=config)

    def show_horizon_interpolation_report(self,path):
        try:
            report=json.loads(self.w.project.resolve(path).read_text(encoding='utf-8'))
            unit='线号网格距离' if report['coordinates']=='lines' else '地震 XY 坐标单位'
            lines=[f'层位：{report["names"][0]}',f'方法：线性插值；最大跨度：{report["max_span"]:g}（{unit}）',f'原始点：{report["original_points"]}',f'本次新增插值点：{report["new_interpolated_points"]}',f'当前插值点合计：{report["interpolated_points"]}',f'原始点凸包外未填：{report["outside_convex_hull"]}',f'超过跨度限制未填：{report["span_rejected"]}',f'剩余无层位时间的道：{report["remaining_unmapped_points"]}','统计范围：整个当前压力成果的已有地震道；缺道不参与插值。','原始点不变，不插值压力；本版不自动识别断层。',f'详细报告：{self.w.project.resolve(path)}']
            QMessageBox.information(self.w,'层位空间插值报告','\n'.join(lines))
        except Exception as exc:QMessageBox.warning(self.w,'无法读取插值报告',str(exc))

    def curve_properties(self,record):
        well=next((well for well in self.w.library.wells if well['name']==record['well']),None)
        if well is None:return
        ref=next((ref for ref in well['curves'] if ref['name']==record['curve']),None)
        if ref is None:return
        lines=[f'井名：{well["name"]}',f'曲线：{ref["name"]}',f'源列：{ref["column"]}',f'源单位：{ref.get("unit") or "未填写"}',f'深度列：{ref["depth_column"]}（{ref["depth_unit"]}）',f'记录数：{ref.get("rows", "—")}',f'深度范围：{ref.get("depth_min", "—")} ～ {ref.get("depth_max", "—")} m',f'来源：{self.w.project.resolve(ref["path"])}','需要修改字段或单位时，可右键重新导入该曲线。']
        QMessageBox.information(self.w,'井曲线属性','\n'.join(lines))

    def remove_reference(self,record):
        w=self.w
        if record.get('extra_volume'):
            self.extra_volumes=[item for item in self.extra_volumes if item['path']!=record['path']]
        elif 'batch' in record:
            panel={'训练井文件':w.train_files,'预测井文件':w.well_prediction.log_files,'实测压力点':w.well_prediction.point_files}[record['batch']]
            for i,path in enumerate(panel.all_files()):
                if path==record['path']:panel.list.setCurrentRow(i);panel.remove();break
        elif 'asset' in record:
            asset=w.catalog.assets[record['asset']]
            if record.get('well') and len(asset.get('wells',[]))>1:
                asset.setdefault('excluded_wells',[]).append(record['well']);asset['wells']=[name for name in asset['wells'] if name!=record['well']]
            else:w.catalog.assets.pop(record['asset'])
            w.catalog.refresh()
        elif 'horizon' in record:
            w.catalog.horizons.pop(record['horizon']);w.catalog.refresh()
        elif 'layer' in record:
            w.slices.horizon_entries.pop(record['layer'],None)
            if w.slices.horizon_entries:w.slices.rebuild_horizon_controls();w.slices.redraw()
            else:w.slices.clear_horizon()
        elif 'well' in record:
            if 'curve' in record:
                for well in w.library.wells:
                    if well['name']==record['well']:well['curves']=[c for c in well['curves'] if c['name']!=record['curve']]
            else:
                w.library.wells=[well for well in w.library.wells if well['name']!=record['well']]
                w.catalog.assets=[a for a in w.catalog.assets if a.get('well_name')!=record['well']]
                for asset in w.catalog.assets:
                    if record['well'] in asset.get('wells',[]):
                        asset.setdefault('excluded_wells',[]).append(record['well']);asset['wells'].remove(record['well'])
            w.library.refresh()
        self.refresh_tree()

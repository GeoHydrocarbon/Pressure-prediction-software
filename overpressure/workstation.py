"""Desktop workstation composition. Computation stays in the existing panels."""
from pathlib import Path
import json
from PySide6.QtCore import Qt, QTimer, QUrl, QSize,QItemSelectionModel
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QTabWidget,QDockWidget,QTreeWidget,QTreeWidgetItem,QStackedWidget,QScrollArea,
    QProgressBar,QToolBar,QToolButton,QStyle,QSizePolicy,QMenu,QMessageBox,QBoxLayout,QFormLayout,QDialog,QDialogButtonBox,QAbstractItemView,QFileDialog)


class DocumentTabs(QTabWidget):
    TITLES=['井曲线','模型训练','时深关系','井预测与验证','成果图表','地震预测','地震切片']
    def __init__(self):
        super().__init__();self.setObjectName('documents');self.setDocumentMode(True)
        self.setUsesScrollButtons(True);self.setMinimumSize(0,0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
    def addWidget(self,widget):
        return self.addTab(widget,self.TITLES[self.count()])


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
        if tab is not None:self.w.data_tabs.setCurrentIndex(tab)

    def open_time_depth(self,method=None):
        if method is not None:self.w.td_method.setCurrentText(method)
        self.time_depth_dialog.show();self.time_depth_dialog.raise_();self.time_depth_dialog.activateWindow()

    def load_time_depth(self):
        path,_=QFileDialog.getOpenFileName(self.w,'加载时深关系表',str(self.w.project.path.parent),'时深关系表 (*.xlsx *.csv)')
        if not path:return
        self.w.td_file.setText(str(Path(path).resolve()));self.open_time_depth('table')
        self.w.status.setText('已选择时深表；请核对单位和单/双程时间，并保存检查。')
        self.signature=None;self.refresh_tree()

    def import_batch(self,panel,page):
        self.show_page(page);panel.choose_files();self.refresh_tree()

    def import_asset(self,kind):
        self.w.library.import_spatial(kind);self.refresh_tree()

    def build_menus(self):
        w=self.w;bar=w.menuBar()
        project=bar.addMenu('项目(&P)');data=bar.addMenu('数据(&D)');model=bar.addMenu('模型(&M)')
        predict=bar.addMenu('预测(&R)');view=bar.addMenu('视图(&V)')
        opened=self.action(project,'打开项目…',w.open_project,'Ctrl+O',QStyle.StandardPixmap.SP_DirOpenIcon,True)
        saved=self.action(project,'保存项目设置',w.save_settings,'Ctrl+S',QStyle.StandardPixmap.SP_DialogSaveButton,True)
        outputs=self.action(project,'成果目录',w.open_outputs,icon=QStyle.StandardPixmap.SP_DirIcon)
        project.addSeparator();self.action(project,'退出',w.close,'Alt+F4')
        imports=data.addMenu('导入数据')
        self.action(imports,'井与测井曲线…',w.library.import_files,guard=True)
        self.action(imports,'实测压力点…',lambda:w.library.import_files(measured=True),guard=True)
        for title,kind in [('井口坐标…','coordinates'),('井轨迹…','trajectory'),('井震对比曲线…','curves')]:
            self.action(imports,title,lambda k=kind:self.import_asset(k),guard=True)
        imports.addSeparator()
        self.action(imports,'地震输入与单位…',lambda:self.show_page(0,1))
        self.action(imports,'层位文件…',w.catalog.add_horizon,guard=True)
        data.addSeparator()
        self.action(data,'层位文件管理…',lambda:self.show_page(0,2))
        self.action(data,'检查井数据与地震对应关系',lambda:w.start_task('audit',[]),guard=True)
        self.action(model,'检查所选训练井与曲线',lambda:w.start_task('audit',['--wells-only']),guard=True)
        self.action(model,'开始当前模式训练 / 验证',w.training_button.click,guard=True)
        self.action(model,'最终模型重训设置…',lambda:(self.show_page(1),w.advanced_toggle.setChecked(True)))
        self.action(predict,'开始井预测与验证',w.well_prediction.start_button.click,guard=True)
        predict.addSeparator()
        self.action(predict,'按设置范围预测地震',w.seismic_range_button.click,guard=True)
        self.action(predict,'预测全部地震',w.seismic_full_button.click,guard=True)
        # Keep keyboard navigation without repeating document tabs in the menus.
        self.action(w,'地震切片',lambda:self.show_page(6),'Ctrl+6')
        self.action(w,'成果图表',lambda:self.show_page(4),'Ctrl+5')
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
        self.signature=None;self.refresh_tree()

    def register_result(self,event):
        artifact=event.get('artifact') or ('final_model' if self.w.task_command in ('train','refit',None) else self.w.task_command)
        if artifact in ('audit','seismic_index','seismic_view'):return
        record={'path':event['path'],'artifact':artifact,'smoke':event.get('smoke',False)}
        if record not in self.results:self.results.append(record)
        self.signature=None;self.refresh_tree()

    def entries(self):
        w=self.w
        groups={key:[] for key in ('井','地震输入','时深关系','层位','模型','预测与验证成果')}
        table=w.td_method.currentText()=='table';unit=w.td_unit.currentText();depth=w.td_depth_unit.currentText();convention='双程' if w.td_convention.currentText()=='twt' else '单程'
        path=w.td_file.text().strip() if table else ''
        label=('表：'+(Path(path).name if path else '未选择文件')) if table else f'公式（{unit} → {depth}）'
        details=(str(w.project.resolve(path)) if path else '未选择时深表') if table else '公式系数（常数、一次、二次、三次）：'+w.coefficients.text()
        details+=f'\n时间：{unit}，{convention}；输出深度：{depth}'
        groups['时深关系'].append((label,dict(page=2,time_depth=True,path=path,details=details)))
        for well in w.library.wells:
            children=[]
            for curve in well['curves']:
                label=curve['name']+(' · 实测点' if curve.get('role')=='measurement' else '')
                children.append((label,dict(page=0,well=well['name'],curve=curve['name'],path=curve['path'])))
            for index,asset in enumerate(w.catalog.assets):
                if well['name'] not in asset.get('wells',[asset.get('well_name','')]):continue
                label={'curves':'剖面对比曲线','coordinates':'井口坐标','trajectory':'井轨迹'}[asset['kind']]
                children.append((label+' · '+Path(asset['path']).name,dict(page=0,well=well['name'],path=asset['path'],asset=index)))
            groups['井'].append((well['name'],dict(page=0,well=well['name'],children=children)))
        for name,edit in [('密度',w.den_path),('波阻抗',w.ip_path)]:
            if edit.text().strip():groups['地震输入'].append((name+' · '+Path(edit.text()).name,dict(page=0,tab=1,path=edit.text().strip())))
        loaded_sources={str(Path(entry['source']).resolve()) for entry in w.slices.horizon_entries.values()}
        for i,cfg in enumerate(w.catalog.horizons):
            if str(w.project.resolve(cfg['path'])) not in loaded_sources:groups['层位'].append((cfg.get('display_name',Path(cfg['path']).name),dict(page=0,tab=2,path=cfg['path'],horizon=i)))
        for key,cfg in w.slices.horizon_entries.items():groups['层位'].append((cfg['label'],dict(page=6,path=cfg['source'],layer=key)))
        for path in dict.fromkeys([w.bundle_path.text().strip(),w.seismic_bundle.text().strip()]):
            if path:groups['模型'].append((Path(path).name,dict(page=3,path=path,model=True)))
        paths={}
        if w.pressure_result.text().strip():paths[w.pressure_result.text().strip()]='seismic_volume'
        for result in self.results:
            if result['artifact']=='final_model':
                record=dict(page=3,path=result['path'],model=True)
                if not any(item[1]['path']==result['path'] for item in groups['模型']):
                    groups['模型'].append((Path(result['path']).name+('（试算）' if result['smoke'] else ''),record))
                continue
            if result['artifact'] not in ('seismic_index','seismic_view'):paths[result['path']]=result['artifact']
        for path,kind in paths.items():
            if kind=='final_model':continue
            groups['预测与验证成果'].append((Path(path).name,dict(page=6 if kind in ('seismic_volume','seismic_trial') else 4,path=path,result=kind)))
        return groups

    def refresh_tree(self):
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
        record=item.data(0,Qt.ItemDataRole.UserRole) or {};self.show_page(record.get('page',0),record.get('tab'))
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
            self.w.status.setText('已选择预测模型：'+path)
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
            menu.addAction('查看 / 修改参数…',lambda:self.open_time_depth())
        if 'curve' in record:
            curves=[dict(selected.data(0,Qt.ItemDataRole.UserRole)) for selected in self.tree.selectedItems() if 'curve' in (selected.data(0,Qt.ItemDataRole.UserRole) or {})]
            if len(curves)>1:
                action=menu.addAction(f'绘制所选曲线（{len(curves)} 条）',lambda records=curves:self.plot_curve_objects(records))
                action.setEnabled(len({entry['well'] for entry in curves})==1)
                action.setToolTip('第一版支持同一口井的多条曲线，共用深度轴。')
                menu.addSeparator()
        if record.get('group')=='井' or well:
            menu.addAction('导入测井曲线…' if well else '批量导入井与测井曲线…',lambda:self.w.library.import_files(well=well))
            menu.addAction('导入实测压力点…',lambda:self.w.library.import_files(well=well,measured=True))
            menu.addAction('导入井口坐标…',lambda:self.w.library.import_spatial('coordinates',well))
            menu.addAction('导入井轨迹…',lambda:self.w.library.import_spatial('trajectory',well))
            menu.addSeparator();menu.addAction('选择训练井与曲线',lambda:self.show_page(1))
            menu.addAction('批量井预测与验证',lambda:self.show_page(3));menu.addSeparator()
            menu.addAction('井震空间匹配设置…',self.well_matching_settings)
        if record.get('group')=='地震输入':menu.addAction('地震输入与单位…',lambda:self.show_page(0,1))
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
            menu.addAction('查看训练图表',lambda:self.open_model_charts(record['path']))
        if 'curve' in record:menu.addAction('曲线属性',lambda:self.curve_properties(record))
        if any(key in record for key in ('batch','asset','horizon','layer','well')):
            menu.addAction('移除引用（保留原始文件）',lambda:self.remove_reference(record))

    def open_model_charts(self,path):
        try:self.w.charts.load_directory(self.w.project.resolve(path));self.show_page(4)
        except Exception as exc:QMessageBox.warning(self.w,'无法查看训练图表',str(exc))

    def plot_curve_objects(self,records):
        if not records:return
        names={record['well'] for record in records}
        if len(names)!=1:
            QMessageBox.information(self.w,'选择同一口井','请先选择同一口井下的多条曲线；跨井对比后续单独提供。');return
        name=next(iter(names));well=next((well for well in self.w.library.wells if well['name']==name),None)
        if well is None:return
        curves={curve['name']:curve for curve in well['curves']};selected=list(dict.fromkeys(record['curve'] for record in records))
        if any(key not in curves for key in selected):
            QMessageBox.information(self.w,'曲线已更新','部分曲线已移除，请重新选择。');return
        self.show_page(0);self.curve_viewer.show_curves(name,[curves[key] for key in selected])

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
        if 'batch' in record:
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

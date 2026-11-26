"""A map-centred MDI workspace with nonmodal business-module dialogs."""
from dataclasses import replace
import math
from pathlib import Path

from PySide6.QtCore import Qt,Signal
from PySide6.QtWidgets import (QMdiArea, QMdiSubWindow, QDialog, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QComboBox, QDoubleSpinBox,
                              QCheckBox, QLabel, QFileDialog, QMessageBox, QFormLayout,
                              QDialogButtonBox, QSpinBox, QScrollArea, QSizePolicy)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .charts import FigureStyle, seismic_figure, export_figure
from .map_view import MapView
from .ui_style import WindowBoundsGuard
from .flow_layout import FlowLayout


class SectionPlot(QWidget):
    """A separately zoomable/exportable snapshot, updated when its map line moves."""
    overlayChanged=Signal(dict)
    def __init__(self):
        super().__init__()
        self.section = None
        self.meta = {}
        self.style = FigureStyle(grid=False)
        self.overlay_settings=dict(distance=25.,styles={},legend=True)
        layout = QVBoxLayout(self)
        row = FlowLayout()
        self.palette = QComboBox()
        self.palette.addItems(['seismic', 'viridis', 'RdBu_r', 'gray', 'turbo'])
        row.addWidget(QLabel('色标'))
        row.addWidget(self.palette)
        self.low = QDoubleSpinBox()
        self.high = QDoubleSpinBox()
        for spin in (self.low, self.high):
            spin.setRange(-1e15, 1e15)
            spin.setDecimals(4)
            spin.setKeyboardTracking(False)
            spin.setMaximumWidth(105)
            row.addWidget(spin)
        self.auto = QCheckBox('自动范围')
        self.auto.setChecked(True)
        self.auto.setToolTip('使用当前剖面 1%～99% 分位数设置色标；取消勾选后可手动输入上下限。')
        row.addWidget(self.auto)
        settings = QPushButton('绘图设置…')
        settings.clicked.connect(self.plot_settings)
        row.addWidget(settings)
        self.overlay_button=QPushButton('叠加显示…');self.overlay_button.clicked.connect(self.overlay_dialog);row.addWidget(self.overlay_button)
        export = QPushButton('导出图件…')
        export.clicked.connect(self.export)
        row.addWidget(export)
        layout.addLayout(row)
        self.canvas = FigureCanvasQTAgg(Figure(layout='constrained'))
        self.canvas.setMinimumSize(0,0)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.palette.currentTextChanged.connect(self.redraw)
        self.low.valueChanged.connect(self.redraw)
        self.high.valueChanged.connect(self.redraw)
        self.auto.toggled.connect(self.redraw)

    def update_section(self, section, metadata):
        first = self.section is None
        self.section, self.meta = section, metadata
        if first:
            self.palette.setCurrentText('viridis' if metadata.get('unit') == 'MPa' or metadata.get('positive_only') else 'seismic')
        if self.auto.isChecked():
            self.low.blockSignals(True)
            self.high.blockSignals(True)
            self.low.setValue(metadata['limits'][0])
            self.high.setValue(metadata['limits'][1])
            self.low.blockSignals(False)
            self.high.blockSignals(False)
        self.note.setText(f"{Path(metadata['source']).name} · 时间向下 · 时间抽稀 {metadata['time_stride']} · 空白列 {metadata['missing_columns']}")
        self.note.setToolTip('来源：'+metadata['source']+'\n任意线取最近实际道；无覆盖或无效值处留空，不进行地震数值的空间插值。')
        if metadata.get('overlay_warnings'):self.note.setText(self.note.text()+'\n'+'\n'.join(metadata['overlay_warnings']))
        self.redraw()

    def overlay_dialog(self):
        from .section_overlay_dialog import SectionOverlayDialog
        SectionOverlayDialog(self).exec()

    def set_overlay_settings(self,settings):
        self.overlay_settings=settings;self.redraw();self.overlayChanged.emit(settings)

    def redraw(self, *_):
        self.low.setEnabled(not self.auto.isChecked())
        self.high.setEnabled(not self.auto.isChecked())
        if self.section is None:
            return
        limits = self.meta['limits'] if self.auto.isChecked() else [self.low.value(), self.high.value()]
        if limits[1] < limits[0]:
            self.note.setText('色标上限必须大于或等于下限。')
            return
        figure = seismic_figure(self.section, limits, cmap=self.palette.currentText(), style=self.style)
        figure.axes[-1].set_ylabel(self.meta.get('unit') or 'Amplitude')
        from .section_overlays import render_overlays
        render_overlays(figure.axes[0],self.meta.get('well_overlays',[]),self.overlay_settings,[self.section.y[0],self.section.y[-1]])
        old = self.canvas.figure
        old.clear()
        self.canvas.figure = figure
        figure.set_canvas(self.canvas)
        self.toolbar.update()
        self.canvas.draw_idle()

    def export(self):
        if self.section is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, '导出当前剖面', 'seismic_section.png',
                                                'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if filename:
            try:
                export_figure(self.canvas.figure, filename, self.style)
            except Exception as exc:
                QMessageBox.warning(self, '导出失败', str(exc))

    def plot_settings(self):
        from .plot_settings import TypographyControls, link_dimensions
        dialog = QDialog(self)
        dialog.setWindowTitle('剖面绘图设置')
        dialog.resize(480, 650)
        layout = QVBoxLayout(dialog)
        body = QWidget()
        form = QFormLayout(body)
        typography = TypographyControls(self.style)
        form.addRow(typography)
        width, height = QDoubleSpinBox(), QDoubleSpinBox()
        for widget, value in ((width, self.style.width_cm), (height, self.style.height_cm)):
            widget.setRange(5, 50)
            widget.setValue(value)
        dpi = QSpinBox()
        dpi.setRange(72, 600)
        dpi.setValue(self.style.dpi)
        form.addRow('导出宽（cm）', width)
        form.addRow('导出高（cm）', height)
        form.addRow('DPI', dpi)
        link_dimensions(form, width, height, dpi)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.screen_guard = WindowBoundsGuard(dialog)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.style = replace(self.style, **typography.collect(), width_cm=width.value(), height_cm=height.value(), dpi=dpi.value()).validate()
                self.redraw()
            except Exception as exc:
                QMessageBox.warning(self, '绘图设置需要修改', str(exc))


class PlotSubWindow(QMdiSubWindow):
    """Retain both the document and its content when a user closes a plot."""
    def closeEvent(self, event):
        # QMdiSubWindow's default close also closes its child widget. Showing
        # only the container afterwards leaves the retained canvas hidden.
        event.ignore()
        self.hide()


class WorkspaceViews:
    def __init__(self, shell):
        self.shell = shell
        self.owner = shell.w
        self.area = QMdiArea()
        self.area.setObjectName('mapWorkspace')
        self.area.setViewMode(QMdiArea.ViewMode.SubWindowView)
        self.area.setActivationOrder(QMdiArea.WindowOrder.ActivationHistoryOrder)
        self.documents = {}
        self.module_dialogs = {}
        self.sections = {}
        self.traditional_documents = {}
        self.rock_physics_documents = {}
        # Existing page controls are retained, but their ownership belongs to a
        # module dialog or a plot document, rather than the central tab bar.
        router = self.owner.takeCentralWidget()
        router.setParent(self.owner)
        router.hide()
        router.blockSignals(True)
        for index in range(router.count()):
            if index == 2:
                continue
            widget = router.widget(index)
            title = router.tabText(index)
            router.removeTab(index)
            router.insertTab(index, QWidget(), title)
            if index in (1, 3, 5, 7, 8):
                dialog = QDialog(self.owner)
                dialog.setModal(False)
                dialog.setWindowTitle(title)
                dialog.resize(1400 if index==8 else 850, 850 if index==8 else 760)
                layout = QVBoxLayout(dialog)
                layout.setContentsMargins(8, 8, 8, 8)
                layout.addWidget(widget)
                widget.show()
                close = QPushButton('关闭窗口')
                close.clicked.connect(dialog.hide)
                layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
                dialog.screen_guard = WindowBoundsGuard(dialog)
                self.module_dialogs[index] = dialog
            else:
                sub = self.add_document(title, widget)
                self.documents[index] = sub
        router.blockSignals(False)
        self.map = MapView(self.owner)
        self.map.sectionReady.connect(self.section_ready)
        self.map.sectionRemoved.connect(self.remove_section)
        self.map.opened.connect(self.map_opened)
        self.map_window = self.add_document('项目底图 · Map', self.map)
        self.area.subWindowActivated.connect(self.activated)
        self.owner.setCentralWidget(self.area)
        self.show_map()

    def add_document(self, title, widget):
        sub = PlotSubWindow()
        sub.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        sub.setWidget(widget)
        widget.show()
        sub.setWindowTitle(title)
        sub.resize(850, 620)
        self.area.addSubWindow(sub)
        return sub

    def show_map(self):
        self.map_window.show()
        self.area.setActiveSubWindow(self.map_window)
        if not any(sub.isVisible() and sub is not self.map_window for sub in self.area.subWindowList()):
            self.map_window.showMaximized()

    def show_page(self, index):
        if index in self.module_dialogs:
            dialog = self.module_dialogs[index]
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        elif index in self.documents:
            sub = self.documents[index]
            sub.show()
            self.area.setActiveSubWindow(sub)
            sub.showMaximized()

    def activated(self, sub):
        index = next((key for key, item in self.documents.items() if item is sub), None)
        if index is not None:
            self.owner.pages.blockSignals(True)
            self.owner.pages.setCurrentIndex(index)
            self.owner.pages.blockSignals(False)
            self.shell.page_changed(index)
        else:
            self.shell._switching = True
            self.shell.parameter_dock.hide()
            self.shell._switching = False

    def section_ready(self, ident, section, metadata):
        if ident not in self.sections:
            panel = SectionPlot()
            from copy import deepcopy
            record=self.map.paths.get(ident,{})
            panel.overlay_settings=deepcopy(record.get('overlay_settings',panel.overlay_settings))
            if 'overlay_settings' not in record and metadata.get('coordinate_mode')=='lines':panel.overlay_settings['distance']=.5
            panel.overlayChanged.connect(lambda settings,key=ident:self.overlay_changed(key,settings))
            sub = self.add_document(section.title, panel)
            self.sections[ident] = sub
            sub.show()
            self.tile()
        sub = self.sections[ident]
        if not sub.isVisible():
            sub.show()
            self.tile()
        sub.widget().update_section(section, metadata)
        sub.setWindowTitle(section.title)
        # Updating a dragged line must leave the map active so a drag can continue.

    def overlay_changed(self,ident,settings):
        record=self.map.paths.get(ident)
        if record is None:return
        from .section_overlays import selected_curves
        from copy import deepcopy
        panel=self.sections[ident].widget()
        curves=selected_curves(settings,panel.meta.get('well_overlays',[]))
        previous=record.get('overlay_curves',[])
        record['overlay_settings']=deepcopy(settings);record['overlay_curves']=curves
        if curves!=previous:self.map.request_section(ident,record)

    def open_traditional(self,path):
        from .traditional_views import TraditionalResultView
        key=str(Path(path).resolve())
        if key not in self.traditional_documents:
            panel=TraditionalResultView(key)
            panel.pointsSelected.connect(self.owner.traditional.accept_points)
            title='正常压实趋势' if panel.is_trend else 'Eaton 压力预测'
            self.traditional_documents[key]=self.add_document(title+' · '+Path(key).name,panel)
        sub=self.traditional_documents[key];sub.show();self.area.setActiveSubWindow(sub);sub.showMaximized()
        return sub.widget()

    def open_rock_physics(self,path,result=None):
        from .rock_physics_view import RockPhysicsResultView
        key=str(Path(path).resolve())
        if key not in self.rock_physics_documents:
            if result is None:
                import json
                result=json.loads((Path(key)/'result.json').read_text(encoding='utf-8'))
            if result.get('kind')=='well_log':
                from .well_rock_physics_view import WellRockPhysicsResultView
                panel=WellRockPhysicsResultView(key,result,self.owner)
            else:panel=RockPhysicsResultView(key,result)
            self.rock_physics_documents[key]=self.add_document('岩石物理 · '+Path(key).name,panel)
        sub=self.rock_physics_documents[key];sub.show();self.area.setActiveSubWindow(sub);sub.showMaximized()
        return sub.widget()

    def update_traditional_trend(self,path,trend):
        sub=self.traditional_documents.get(str(path))
        if sub is not None:
            sub.widget().trend=trend;sub.widget().redraw()

    def tile(self):
        visible=[sub for sub in self.area.subWindowList() if sub.isVisible()]
        for sub in visible:sub.showNormal()
        others=[sub for sub in visible if sub is not self.map_window]
        if self.map_window not in visible or not others:
            self.area.tileSubWindows();return
        rect=self.area.viewport().rect();split=round(rect.width()*.52)
        self.map_window.setGeometry(0,0,split,rect.height())
        columns=1 if len(others)<=2 else math.ceil(math.sqrt(len(others)))
        rows=math.ceil(len(others)/columns)
        for number,sub in enumerate(others):
            col,row=number%columns,number//columns
            left=split+(rect.width()-split)*col//columns
            right=split+(rect.width()-split)*(col+1)//columns
            top=rect.height()*row//rows;bottom=rect.height()*(row+1)//rows
            sub.setGeometry(left,top,right-left,bottom-top)

    def remove_section(self,ident):
        sub=self.sections.pop(ident,None)
        if sub is not None:
            self.area.removeSubWindow(sub);sub.deleteLater()

    def map_opened(self, metadata):
        self.map_window.setWindowTitle(metadata['label'] + ' · Map')
        self.owner.status.setText('底图已载入，可在地图上拉剖面。')
        self.shell.restore_map_state(metadata)

    def reset_project(self):
        self.map.reset_project()
        self.owner.rock_physics.stop()
        for sub in self.rock_physics_documents.values():
            self.area.removeSubWindow(sub);sub.deleteLater()
        self.rock_physics_documents.clear()
        for sub in self.traditional_documents.values():
            self.area.removeSubWindow(sub);sub.deleteLater()
        self.traditional_documents.clear()
        for sub in self.sections.values():
            self.area.removeSubWindow(sub)
            sub.deleteLater()
        self.sections.clear()
        for sub in self.documents.values():
            sub.hide()
        for dialog in self.module_dialogs.values():
            dialog.hide()
        self.owner.slices.shutdown()
        self.owner.slices.reset_display()
        self.owner.slices.view = None
        self.owner.slices.remote = False
        self.owner.slices.result_dir.clear()
        self.owner.charts.history = self.owner.charts.history.iloc[:0]
        self.owner.charts.pressures = self.owner.charts.pressures.iloc[:0]
        self.owner.charts.folds = self.owner.charts.folds.iloc[:0]
        self.owner.charts.reference_points = None
        self.owner.charts.directory.clear()
        self.owner.charts.redraw()
        self.show_map()

    def shutdown(self):
        self.map.shutdown()
        for dialog in self.module_dialogs.values():
            dialog.hide()

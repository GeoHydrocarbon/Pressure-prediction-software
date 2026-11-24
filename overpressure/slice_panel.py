from pathlib import Path
import codecs
import json
from dataclasses import replace
from uuid import uuid4

import numpy as np
from PySide6.QtCore import Qt, QProcess, QProcessEnvironment,QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QSlider,
    QDoubleSpinBox,QSpinBox, QFileDialog, QMessageBox, QCheckBox, QColorDialog, QDialog, QFormLayout, QDialogButtonBox,QScrollArea,QSplitter,QSizePolicy,QTabWidget,QListWidget,QListWidgetItem)
from .horizon_layers import add_layers
from .ui_style import Card,polish_layouts,AspectPreview,bound_plot_dialog
from .charts import seismic_figure,FigureStyle,apply_typography
from .plot_frame import PlotResizeControls
from .plot_settings import TypographyControls,link_dimensions
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from .seismic_view import SeismicView
from .seismic_view import Slice
from .volume_reader import VolumeIndex
from .runtime import worker_command, user_data_root


class SliceToolbar(NavigationToolbar2QT):
    def save_figure(self,*_):self.parentWidget().export()


class SlicePanel(QWidget):
    def __init__(self, start_task):
        super().__init__()
        self.start_task = start_task; self.view = None
        self.remote = False; self.reader_process = None; self.reader_ready = False; self.reader_busy = False
        self.pending_request = None; self.latest_id = None; self.stopping = False
        self.pending_import = None; self.import_id = None; self.horizon_path = None; self.horizon_summary = None
        self.horizon_entries={};self.last_horizon_lines=[]
        self.horizon_import_queue=[];self.horizon_import_ids=set()
        self.horizon_config_provider=None;self.project_horizon_attempts=set();self.horizon_load_error=''
        self.interpolation_ids=set();self.interpolation_registered=None;self.pending_interpolation_source=None;self.last_horizon_point_types=None
        self.pending_wells = None; self.wells_id = None; self.well_path = None; self.last_well = None
        self.well_config_provider = None
        self.last_section = None; self.last_limits = None; self.last_overlay = None; self.last_overlay_name = ""
        self.axis_ranges = {}; self.color_limits = None; self.cmap = "viridis"; self.overlay_color = "#e35050"
        self.image_width_cm=24.;self.image_height_cm=15.;self.image_dpi=300;self.preview_mode='fit';self.image_ratio_locked=False
        self.plot_style=FigureStyle(grid=False)
        root=QVBoxLayout(self)
        heading=QLabel('地震切片');heading.setObjectName('heading');root.addWidget(heading)
        self.content_body=QWidget();self.page_content_layout=QVBoxLayout(self.content_body);self.page_content_layout.setContentsMargins(0,0,0,0);root.addWidget(self.content_body,1)
        controls_body=QWidget();layout=QVBoxLayout(controls_body)
        layout.setContentsMargins(0,0,0,0)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(controls_body)
        self.settings_scroll=scroll
        scroll.setMinimumHeight(220)
        self.page_content_layout.addWidget(scroll)
        scroll.setMinimumHeight(230);scroll.setMaximumHeight(250)
        self.control_tabs=QTabWidget();layout.addWidget(self.control_tabs)
        volume_card=Card();self.control_tabs.addTab(volume_card,'切片设置');volume_layout=QVBoxLayout(volume_card)
        volume_layout.addWidget(QLabel('压力成果'));self.result_dir=QLineEdit();volume_layout.addWidget(self.result_dir)
        paths = QHBoxLayout();paths.setProperty('keepHorizontal',True)
        choose = QPushButton('选择成果…'); choose.clicked.connect(self.choose_result); paths.addWidget(choose)
        self.load_button = QPushButton('载入切片'); self.load_button.clicked.connect(self.load_result); paths.addWidget(self.load_button)
        volume_layout.addLayout(paths)
        fields=QFormLayout();self.mode=QComboBox();self.mode.addItems(['Inline 剖面','Crossline 剖面','水平时间切片','沿层位压力切片']);fields.addRow('切片方向',self.mode)
        self.position_group=QWidget();position_layout=QHBoxLayout(self.position_group);position_layout.setContentsMargins(0,0,0,0);position_layout.setProperty('keepHorizontal',True)
        self.position=QComboBox();position_layout.addWidget(self.position)
        self.remote_position=QDoubleSpinBox();self.remote_position.setDecimals(0);self.remote_position.setKeyboardTracking(False);self.remote_position.hide();position_layout.addWidget(self.remote_position)
        self.position_label=QLabel('线号 / 时间');fields.addRow(self.position_label,self.position_group);volume_layout.addLayout(fields)
        plot_actions=QHBoxLayout();plot_actions.setProperty('keepHorizontal',True)
        self.export_button=QPushButton('导出图件…');self.export_button.clicked.connect(self.export);plot_actions.addWidget(self.export_button)
        self.settings_button=QPushButton('绘图设置…');self.settings_button.setToolTip('设置坐标范围、色标、字体、图片比例及导出尺寸');self.settings_button.clicked.connect(self.display_settings);plot_actions.addWidget(self.settings_button);volume_layout.addLayout(plot_actions)
        self.along_controls=QWidget();along_layout=QVBoxLayout(self.along_controls);along_layout.setContentsMargins(0,0,0,0)
        along_fields=QFormLayout();self.along_label=QLabel('取值层位')
        self.horizon_combo=QComboBox();self.horizon_combo.setObjectName('along_horizon_selector');self.horizon_combo.setPlaceholderText('请导入或载入项目层位');along_fields.addRow(self.along_label,self.horizon_combo)
        self.sampling=QComboBox();self.sampling.addItems(['时间线性插值','最近时间采样']);along_fields.addRow('时间取值',self.sampling);along_layout.addLayout(along_fields)
        self.interpolation_marks=QCheckBox('标记插值点（空心圆）');self.interpolation_marks.setChecked(True);self.interpolation_marks.toggled.connect(self.refresh_display);along_layout.addWidget(self.interpolation_marks)
        self.along_import_button=QPushButton('导入层位…');self.along_import_button.clicked.connect(self.import_horizon);along_layout.addWidget(self.along_import_button)
        self.along_note=QLabel('按层位时间提取每道压力，显示沿层平面图。');self.along_note.setObjectName('hint');self.along_note.setWordWrap(True);along_layout.addWidget(self.along_note)
        volume_layout.addWidget(self.along_controls);self.along_controls.hide()
        layer_card=Card('层位显示');self.control_tabs.addTab(layer_card,'层位叠加');layer_layout=QVBoxLayout(layer_card)
        horizon_row = QHBoxLayout()
        self.horizon_button = QPushButton("导入层位（线号或 XY）"); self.horizon_button.clicked.connect(self.import_horizon); horizon_row.addWidget(self.horizon_button)
        self.overlay_enabled = QCheckBox("剖面叠加层位"); self.overlay_enabled.setChecked(True); horizon_row.addWidget(self.overlay_enabled)
        self.sampling.setToolTip("仅用于沿层位压力切片的时间取值方式"); self.sampling.setEnabled(False)
        self.horizon_list=QListWidget();self.horizon_list.setMinimumHeight(85);self.horizon_list.setMaximumHeight(130)
        self.horizon_list.setToolTip('勾选的层位同时叠加到 Inline / Crossline 剖面；选中一项后可设置该层的颜色和线宽。')
        self.sampling_note=QLabel();self.sampling_note.setWordWrap(True);self.sampling_note.setObjectName('hint')
        layer_actions=QHBoxLayout()
        color = QPushButton("层位颜色");self.layer_color_button=color; color.clicked.connect(self.choose_overlay_color); layer_actions.addWidget(color)
        layer_actions.addWidget(QLabel('所选层线宽 (pt)'))
        self.layer_width=QDoubleSpinBox();self.layer_width.setRange(.2,12);self.layer_width.setDecimals(1);self.layer_width.setValue(1.6);layer_actions.addWidget(self.layer_width)
        remove=QPushButton('移除所选层位');remove.clicked.connect(self.remove_selected_horizon);layer_actions.addWidget(remove)
        report = QPushButton("导出匹配报告"); report.clicked.connect(self.export_horizon_report); layer_actions.addWidget(report)
        clear = QPushButton("清除全部层位"); clear.clicked.connect(self.clear_horizon); layer_actions.addWidget(clear)
        layer_actions.addStretch();layer_layout.addLayout(horizon_row)
        layer_layout.addWidget(QLabel('剖面显示层位（勾选多层；选中一项后设置颜色和线宽）'))
        layer_layout.addWidget(self.horizon_list);layer_layout.addLayout(layer_actions);layer_layout.addWidget(self.sampling_note)
        self.horizon_list.itemChanged.connect(self.layer_checked)
        self.horizon_list.currentItemChanged.connect(self.layer_selected)
        self.layer_width.valueChanged.connect(self.change_layer_width)
        well_card=Card();self.control_tabs.addTab(well_card,'井曲线对比');well_layout=QVBoxLayout(well_card)
        self.well_load_button=QPushButton('载入项目井数据');self.well_load_button.clicked.connect(self.import_wells);well_layout.addWidget(self.well_load_button)
        well_fields=QFormLayout();self.well_combo=QComboBox();well_fields.addRow('井',self.well_combo);self.curve_combo=QComboBox();well_fields.addRow('曲线',self.curve_combo)
        self.well_width=QDoubleSpinBox();self.well_width.setRange(.1,1000);self.well_width.setValue(3);self.well_width.setSuffix(' 线号宽度');well_fields.addRow('显示宽度',self.well_width);well_layout.addLayout(well_fields)
        self.well_enabled=QCheckBox('叠加井曲线');self.well_enabled.setChecked(True);well_layout.addWidget(self.well_enabled)
        well_actions=QHBoxLayout();well_actions.setProperty('keepHorizontal',True);well_layout.addLayout(well_actions)
        self.well_locate_button=QPushButton('定位剖面');self.well_locate_button.setToolTip('跳到所选井旁的地震剖面');self.well_locate_button.clicked.connect(self.jump_well);well_actions.addWidget(self.well_locate_button)
        self.well_compare_button=QPushButton('对比与导出…');self.well_compare_button.setToolTip('井曲线压力与地震预测压力对比及导出');self.well_compare_button.clicked.connect(self.compare_well);well_actions.addWidget(self.well_compare_button)
        self.well_combo.currentTextChanged.connect(self.redraw); self.curve_combo.currentTextChanged.connect(self.redraw)
        self.well_enabled.toggled.connect(self.refresh_display); self.well_width.valueChanged.connect(self.refresh_display)
        plot_body=Card();plot_layout=QVBoxLayout(plot_body);self.page_content_layout.addWidget(plot_body)
        plot_body.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self.slider = QSlider(Qt.Orientation.Horizontal); plot_layout.addWidget(self.slider)
        self.context = QLabel("载入已完成的预测成果。程序建立轻量几何索引，仅按需读取当前切片；灰色为无效区或缺失道，时间域不等于深度域。")
        self.context.setWordWrap(True);self.context.setObjectName('hint');plot_layout.addWidget(self.context)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(8, 5), layout="constrained")); self.toolbar = SliceToolbar(self.canvas, self)
        self.preview=QScrollArea();self.preview.setWidgetResizable(True);self.preview.setMinimumHeight(100)
        self.aspect_preview=AspectPreview(self.canvas);self.preview.setWidget(self.aspect_preview)
        self.preview.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview.setMinimumSize(0,0);self.preview.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        self.aspect_preview.set_plot_size(self.image_width_cm,self.image_height_cm)
        self.aspect_preview.set_preview_mode(self.preview_mode)
        self.preview_controls=PlotResizeControls(self.aspect_preview,'slice',self)
        self.preview_controls.lock.toggled.connect(self.set_preview_lock)
        plot_layout.addWidget(self.toolbar);plot_layout.addWidget(self.preview_controls);plot_layout.addWidget(self.preview,1)
        self.mode.currentIndexChanged.connect(self.update_positions)
        self.position.currentIndexChanged.connect(self.position_changed)
        self.slider.valueChanged.connect(self.slider_changed)
        self.remote_position.valueChanged.connect(self.remote_changed)
        self.horizon_combo.currentTextChanged.connect(self.redraw)
        self.horizon_combo.currentIndexChanged.connect(self.update_sampling_controls)
        self.sampling.currentIndexChanged.connect(self.redraw)
        self.overlay_enabled.toggled.connect(self.refresh_display)
        self.control_tabs.currentChanged.connect(self.update_settings_height)
        self.layer_selected()
        self.update_sampling_controls()

    def set_preview_lock(self,checked):
        self.image_ratio_locked=bool(checked);self.aspect_preview.lock_ratio=bool(checked)

    def update_settings_height(self,index):
        if index==1:
            height=self.control_tabs.widget(1).minimumSizeHint().height()+self.control_tabs.tabBar().sizeHint().height()+24
            self.settings_scroll.setMaximumHeight(16777215);self.settings_scroll.setMinimumHeight(max(230,height))
        else:self.settings_scroll.setMinimumHeight(230);self.settings_scroll.setMaximumHeight(250)

    def choose_result(self):
        path = QFileDialog.getExistingDirectory(self, "选择地震预测成果目录")
        if path: self.result_dir.setText(path)

    def load_result(self):
        if not self.result_dir.text().strip():
            QMessageBox.information(self, "选择成果", "请先选择已完成的地震成果目录。"); return
        self.start_task("prepare-seismic-view", ["--result", self.result_dir.text().strip()])

    def load_cache(self, path):
        self.shutdown(); self.remote = False; self.position.show(); self.remote_position.hide()
        self.reset_display()
        self.view = SeismicView.from_cache(path)
        self.result_dir.setText(str(Path(path).resolve().parent))
        self.context.setText(f"时间域压力（MPa）；{len(self.view.pressure)} 道。灰色为无效区或未覆盖的线号交点，无插值。各方向使用统一色标。")
        if self.view.purpose == "synthetic_fixture": self.context.setText("合成数据软件验证样例，不代表真实预测。" + self.context.text())
        self.update_positions()

    def load_index(self, path):
        self.shutdown(); self.stopping = False
        self.reset_display()
        self.view = VolumeIndex.load(path); self.remote = True
        self.canvas.figure.clear(); self.canvas.draw_idle(); self.export_button.setEnabled(False)
        self.position.hide(); self.remote_position.show(); self.result_dir.setText(str(self.view.path.parent))
        self.base_context = f"时间域压力（MPa），{len(self.view.inline)} 道；灰色为无效/缺失，无插值，统一色标。"
        if self.view.purpose == "synthetic_fixture": self.base_context = "合成验收，非真实预测。" + self.base_context
        self.context.setText(self.base_context)
        proc = QProcess(self); self.reader_process = proc
        env = QProcessEnvironment.systemEnvironment(); env.insert("PYTHONIOENCODING", "utf-8"); proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(user_data_root()))
        self.reader_buffer = ""; self.reader_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.reader_ready = False; self.reader_busy = False; self.pending_request = None
        proc.readyReadStandardOutput.connect(self.read_reader)
        proc.readyReadStandardError.connect(lambda: self.context.setText(bytes(proc.readAllStandardError()).decode("utf-8", errors="replace")))
        proc.errorOccurred.connect(lambda error: self.context.setText("切片读取进程错误：" + proc.errorString()))
        proc.finished.connect(lambda code, status, p=proc: self.reader_finished(p, code))
        proc.start(*worker_command("overpressure.slice_worker", ["--index", str(self.view.path)]))
        self.update_positions()

    def update_positions(self, *_):
        self.update_sampling_controls()
        if self.view is None: return
        mode = self.mode.currentIndex()
        self.slider.setVisible(mode < 3)
        self.position.setVisible(not self.remote and mode < 3); self.remote_position.setVisible(self.remote and mode < 3)
        if mode == 3: self.ensure_project_horizons();self.redraw(); return
        if self.remote:
            self.export_button.setEnabled(False)
            self.positions = np.unique(self.view.inline) if mode == 0 else np.unique(self.view.crossline) if mode == 1 else None
            self.remote_position.blockSignals(True); self.slider.blockSignals(True)
            self.remote_position.setDecimals(0 if mode < 2 else 6)
            self.remote_position.setSuffix("" if mode < 2 else " ms")
            if mode < 2:
                self.remote_position.setRange(float(self.positions[0]), float(self.positions[-1])); self.remote_position.setSingleStep(1)
                self.remote_position.setValue(float(self.positions[0])); self.slider.setRange(0, len(self.positions) - 1)
            else:
                self.remote_position.setRange(self.view.t0_ms, self.view.t0_ms + (self.view.nt - 1) * self.view.interval_ms)
                self.remote_position.setSingleStep(self.view.interval_ms); self.remote_position.setValue(self.view.t0_ms); self.slider.setRange(0, self.view.nt - 1)
            self.slider.setValue(0); self.slider.blockSignals(False); self.remote_position.blockSignals(False)
            self.redraw(); return
        values = np.unique(self.view.inline) if mode == 0 else np.unique(self.view.crossline) if mode == 1 else self.view.times_ms
        self.position.blockSignals(True); self.position.clear()
        self.position.addItems([str(int(v)) if mode < 2 else f"{v:g} ms" for v in values]); self.position.setCurrentIndex(0); self.position.blockSignals(False)
        self.slider.blockSignals(True); self.slider.setRange(0, len(values) - 1); self.slider.setValue(0); self.slider.blockSignals(False)
        self.redraw()

    def position_changed(self, index):
        if index < 0: return
        self.slider.blockSignals(True); self.slider.setValue(index); self.slider.blockSignals(False)
        self.redraw()

    def slider_changed(self, index):
        if not self.remote: self.position.setCurrentIndex(index); return
        value = self.positions[index] if self.mode.currentIndex() < 2 else self.view.t0_ms + index * self.view.interval_ms
        self.remote_position.setValue(float(value))

    def remote_changed(self, value):
        if not self.remote: return
        self.export_button.setEnabled(False)
        if self.mode.currentIndex() < 2:
            index = int(np.searchsorted(self.positions, value))
            if index >= len(self.positions) or self.positions[index] != int(value):
                self.context.setText("该线号不在当前成果中，请选择已有线号。"); return
        else:
            index = int(round((value - self.view.t0_ms) / self.view.interval_ms))
            if abs(value - (self.view.t0_ms + index * self.view.interval_ms)) > 1e-5:
                self.context.setText("该时间不在采样网格上，请按采样间隔选择。"); return
        self.slider.blockSignals(True); self.slider.setValue(index); self.slider.blockSignals(False); self.redraw()

    def redraw(self, *_):
        if self.view is None: return
        mode = self.mode.currentIndex()
        if mode == 3 and (not self.remote or not self.horizon_path or not self.horizon_combo.currentText()):
            message=self.horizon_load_error or ('正在载入项目层位，完成后自动显示沿层压力图…' if self.horizon_import_ids else '请在右侧“切片设置”中导入层位，再选择需要沿层取值的层位。' if self.remote else '沿层位取值需要先载入 SEG-Y 压力成果。')
            self.context.setText(message);self.latest_id=uuid4().hex;self.pending_request=None
            self.export_button.setEnabled(False); self.settings_button.setEnabled(False); self.canvas.figure.clear()
            self.canvas.figure.text(.5,.5,message,ha='center',va='center',wrap=True,fontfamily=['Microsoft YaHei','DejaVu Sans'],fontsize=10)
            self.canvas.draw_idle(); self.last_section = None; return
        if self.remote:
            self.export_button.setEnabled(False)
            self.settings_button.setEnabled(False)
            value = int(self.remote_position.value()) if mode < 2 else int(round((self.remote_position.value() - self.view.t0_ms) / self.view.interval_ms))
            request = {"id": uuid4().hex, "mode": ("inline", "crossline", "time", "horizon")[mode], "value": value}
            if self.horizon_path and self.horizon_combo.currentText():
                entry=self.horizon_entries.get(self.horizon_combo.currentData())
                request.update(horizon_path=entry['path'] if entry else self.horizon_path,horizon_name=entry['name'] if entry else self.horizon_combo.currentText(),sampling="nearest" if self.sampling.currentIndex() else "linear")
            request['overlay_layers']=[{k:item[k] for k in ('id','path','name')} for item in self.horizon_entries.values() if item['checked']]
            if self.well_path and mode<2:
                request.update(well_path=self.well_path,well_name=self.well_combo.currentText(),well_curve=self.curve_combo.currentText())
            self.latest_id = request["id"]; self.pending_request = request; self.send_pending(); return
        if self.position.currentIndex() < 0: return
        section = self.view.section(("inline", "crossline", "time")[mode], int(self.position.currentText()) if mode < 2 else self.position.currentIndex())
        self.render(section, self.view.limits)

    def render(self, section, limits, overlay=None, overlay_name=""):
        self.last_section, self.last_limits = section, limits
        self.last_overlay, self.last_overlay_name = overlay, overlay_name
        xlim, ylim = self.axis_ranges.get(self.mode.currentIndex(), (None, None))
        layers=None
        if self.horizon_entries:
            layers=[dict(self.horizon_entries[line['id']],time=line['time']) for line in self.last_horizon_lines if line['id'] in self.horizon_entries and self.horizon_entries[line['id']]['checked']]
        figure = seismic_figure(section, self.color_limits or limits, figure=self.canvas.figure, cmap=self.cmap, xlim=xlim, ylim=ylim,
                                overlay=overlay if self.overlay_enabled.isChecked() else None, overlay_color=self.overlay_color, overlay_name=overlay_name,
                                well=self.last_well if self.well_enabled.isChecked() else None, well_label=self.well_combo.currentText()+' '+self.curve_combo.currentText(), well_width=self.well_width.value(),style=self.plot_style,
                                overlays=layers if self.overlay_enabled.isChecked() else [])
        if self.mode.currentIndex()==3 and not np.isfinite(section.values).any():
            figure.axes[0].text(.5,.5,'该层位上没有有效压力值\n请查看图上方的匹配与时间范围提示',transform=figure.axes[0].transAxes,ha='center',va='center',fontfamily=[self.plot_style.font_family,'Microsoft YaHei','DejaVu Sans'],fontsize=self.plot_style.font_size,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        if self.mode.currentIndex()==3 and self.interpolation_marks.isChecked() and self.last_horizon_point_types is not None and self.last_horizon_point_types.shape==section.values.shape:
            row,col=np.nonzero((self.last_horizon_point_types==2)&np.isfinite(section.values))
            if len(row):
                step=max(1,int(np.ceil(len(row)/20000)));figure.axes[0].scatter(section.x[col[::step]],section.y[row[::step]],s=9,facecolors='none',edgecolors='#d97520',linewidths=.5,label='插值层位点'+('（标记抽稀）' if step>1 else ''),rasterized=True)
                figure.axes[0].legend();apply_typography(figure,self.plot_style)
        self.aspect_preview.set_plot_size(self.image_width_cm,self.image_height_cm)
        self.aspect_preview.set_preview_mode(self.preview_mode)
        self.toolbar.update(); self.canvas.draw_idle()
        fresh = not self.remote or (not self.reader_busy and self.pending_request is None)
        self.export_button.setEnabled(fresh); self.settings_button.setEnabled(fresh)

    def send_pending(self):
        if not self.reader_process or not self.reader_ready or self.reader_busy: return
        if self.pending_import is not None: request = self.pending_import; self.pending_import = None
        elif self.horizon_import_queue:request=self.horizon_import_queue.pop(0)
        elif self.pending_wells is not None: request = self.pending_wells; self.pending_wells = None
        elif self.pending_request is not None: request = self.pending_request; self.pending_request = None
        else: return
        self.reader_busy = True
        self.reader_process.write((json.dumps(request) + "\n").encode("utf-8"))
        self.context.setText(self.base_context + ('\n正在进行层位空间插值，保留原始点并检查跨度限制……' if request.get('command')=='interpolate_horizon' else "\n正在读取并匹配层位数据……" if request.get("command") == "import_horizon" else "\n正在按需读取切片……"))

    def read_reader(self):
        if self.reader_process is None: return
        self.reader_buffer += self.reader_decoder.decode(bytes(self.reader_process.readAllStandardOutput()))
        while "\n" in self.reader_buffer:
            line, self.reader_buffer = self.reader_buffer.split("\n", 1)
            try: event = json.loads(line)
            except ValueError: continue
            if self.stopping: continue
            if event["event"] == "ready": self.reader_ready = True; self.send_pending(); continue
            if event["event"] in ("slice", "error", "horizon_loaded", "wells_loaded",'horizon_interpolated'):
                self.reader_busy = False
                if event["event"] == "error":
                    self.interpolation_ids.discard(event.get('id'))
                    if event.get('id') in self.horizon_import_ids or event.get('id')==self.import_id:self.horizon_load_error='层位载入失败：'+event['message']
                    self.horizon_import_ids.discard(event.get('id'));self.context.setText("切片读取失败：" + event["message"])
                elif event['event'] == 'wells_loaded' and event.get('id') == self.wells_id:
                    self.well_path = event['path']
                    for box,items in [(self.well_combo,event['names']),(self.curve_combo,event['curves'])]:
                        box.blockSignals(True); box.clear(); box.addItems(items); box.blockSignals(False)
                    self.context.setText(f"井数据加载完成：{event['rows']} 点；定位 {event['located_rows']} 点；压力配对 {event['pressure_pairs']} 点。点击‘跳到井旁剖面’。")
                    self.redraw()
                elif event["event"] in ('horizon_loaded','horizon_interpolated') and (event.get("id") == self.import_id or event.get('id') in self.horizon_import_ids):
                    self.horizon_import_ids.discard(event.get('id'))
                    self.interpolation_ids.discard(event.get('id'))
                    self.horizon_load_error=''
                    self.horizon_path = event["path"]; self.horizon_summary = event
                    try:self.horizon_entries=add_layers(self.horizon_entries,event);self.rebuild_horizon_controls()
                    except ValueError as exc:self.context.setText(str(exc));self.send_pending();continue
                    if self.interpolation_registered:self.interpolation_registered(event['settings'])
                    if event['event']=='horizon_interpolated':
                        self.interpolation_ids.discard(event.get('id'))
                        key=next(key for key,entry in self.horizon_entries.items() if entry['path']==event['path'] and entry['name']==event['names'][0])
                        self.horizon_combo.setCurrentIndex(self.horizon_combo.findData(key));self.mode.setCurrentIndex(3);self.control_tabs.setCurrentIndex(0)
                    elif self.pending_interpolation_source==str(Path(event['source']).resolve()):
                        self.pending_interpolation_source=None
                        key=next(key for key,entry in self.horizon_entries.items() if entry['source']==str(Path(event['source']).resolve()))
                        QTimer.singleShot(0,lambda selected=key:self.open_horizon_interpolation(selected))
                    self.context.setText(self.base_context + self.horizon_message()); self.redraw()
                elif event["id"] == self.latest_id:
                    try:
                        self.last_stats = event.get("stats", {})
                        with np.load(event["path"], allow_pickle=False) as data:
                            self.last_horizon_lines=[dict(id=str(key),time=times.copy()) for key,times in zip(data['horizon_ids'],data['horizon_times'])] if 'horizon_ids' in data else []
                            self.last_horizon_point_types=data['horizon_point_types'].copy() if 'horizon_point_types' in data and data['horizon_point_types'].size else None
                            self.last_well = {k:data['well_'+k] for k in ('center','time','curve','seismic')} if 'well_center' in data else None
                            section = Slice(data["values"], data["x"], data["y"], str(data["xlabel"].item()), str(data["ylabel"].item()), str(data["title"].item()), bool(data["invert_y"].item()))
                            self.render(section, tuple(data["limits"].tolist()), data["horizon_time"] if "horizon_time" in data else None,
                                        str(data["horizon_name"].item()) if "horizon_name" in data else "")
                            sx, sy = int(data["x_stride"].item()), int(data["y_stride"].item())
                        self.context.setText(self.base_context + self.horizon_message() + self.along_diagnostics() + (f"\n显示抽稀：横向每 {sx} 点、纵向每 {sy} 点；仅影响显示，完整 SEG-Y 不变。" if sx > 1 or sy > 1 else "\n当前切片未抽稀。"))
                    except Exception as exc: self.context.setText("切片显示失败：" + str(exc))
                self.send_pending()

    def reader_finished(self, proc, code):
        if self.reader_process is proc:
            self.reader_ready = False; self.reader_busy = False; self.reader_process = None
            if not self.stopping and code: self.context.setText(self.context.text() + " 切片读取进程已停止，请重新载入成果。")
        proc.deleteLater()

    def shutdown(self):
        self.stopping = True; self.pending_request = None; self.pending_import = None; self.import_id = None
        self.pending_wells = None; self.wells_id = None
        proc = self.reader_process
        if proc is not None and proc.state() != QProcess.ProcessState.NotRunning:
            proc.closeWriteChannel()
            if not proc.waitForFinished(1500): proc.kill(); proc.waitForFinished(1000)
        self.reader_process = None; self.reader_ready = False; self.reader_busy = False

    def reset_display(self):
        self.axis_ranges = {}; self.last_section = None; self.last_overlay = None
        self.well_path = None; self.last_well = None; self.well_combo.clear(); self.curve_combo.clear()
        self.clear_horizon(redraw=False)

    def clear_horizon(self, *_ , redraw=True):
        self.horizon_path = None; self.horizon_summary = None; self.pending_import = None; self.import_id = None
        self.horizon_entries={};self.last_horizon_lines=[]
        self.project_horizon_attempts=set()
        self.horizon_load_error=''
        self.interpolation_ids=set();self.pending_interpolation_source=None;self.last_horizon_point_types=None
        self.horizon_import_queue=[];self.horizon_import_ids=set();self.last_overlay=None;self.last_overlay_name=''
        self.latest_id = uuid4().hex
        self.horizon_combo.blockSignals(True); self.horizon_combo.clear(); self.horizon_combo.blockSignals(False)
        self.horizon_list.blockSignals(True);self.horizon_list.clear();self.horizon_list.blockSignals(False);self.update_sampling_controls()
        if redraw: self.redraw()

    def horizon_message(self):
        item = self.horizon_summary
        if not item: return ""
        message=f"\n已加载 {len(self.horizon_entries)} 个层位，勾选显示 {sum(v['checked'] for v in self.horizon_entries.values())} 层；最近导入 {item['records']} 条记录，映射值 {item['mapped_trace_values']} 个；缺失与未匹配留空，详见匹配报告。"
        if 'new_interpolated_points' in item:message+=f'\n最近一次空间插值新增 {item["new_interpolated_points"]} 点，原始点保持不变；右键派生层位可查看插值报告。'
        return message

    def rebuild_horizon_controls(self):
        selected=self.horizon_combo.currentData();current=self.horizon_list.currentItem();current_id=current.data(Qt.ItemDataRole.UserRole) if current else None
        self.horizon_combo.blockSignals(True);self.horizon_list.blockSignals(True);self.horizon_combo.clear();self.horizon_list.clear()
        for key,entry in self.horizon_entries.items():
            self.horizon_combo.addItem(entry['label'],key)
            item=QListWidgetItem(entry['label']);item.setData(Qt.ItemDataRole.UserRole,key);item.setToolTip(entry['source'])
            item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable);item.setCheckState(Qt.CheckState.Checked if entry['checked'] else Qt.CheckState.Unchecked);self.horizon_list.addItem(item)
            if key==current_id:self.horizon_list.setCurrentItem(item)
        found=self.horizon_combo.findData(selected)
        if self.horizon_combo.count():self.horizon_combo.setCurrentIndex(found if found>=0 else 0)
        if self.horizon_list.currentRow()<0 and self.horizon_list.count():self.horizon_list.setCurrentRow(0)
        self.horizon_combo.blockSignals(False);self.horizon_list.blockSignals(False);self.layer_selected();self.update_sampling_controls()

    def layer_selected(self,*_):
        item=self.horizon_list.currentItem();entry=self.horizon_entries.get(item.data(Qt.ItemDataRole.UserRole)) if item else None
        self.layer_width.blockSignals(True);self.layer_width.setValue(entry['width'] if entry else 1.6);self.layer_width.setEnabled(entry is not None);self.layer_width.blockSignals(False)
        self.layer_color_button.setEnabled(entry is not None)

    def layer_checked(self,item):
        self.horizon_entries[item.data(Qt.ItemDataRole.UserRole)]['checked']=item.checkState()==Qt.CheckState.Checked
        if self.mode.currentIndex()<2:self.redraw()
        else:self.refresh_display()

    def change_layer_width(self,value):
        item=self.horizon_list.currentItem()
        if item:self.horizon_entries[item.data(Qt.ItemDataRole.UserRole)]['width']=value;self.refresh_display()

    def remove_selected_horizon(self):
        item=self.horizon_list.currentItem()
        if item:
            self.horizon_entries.pop(item.data(Qt.ItemDataRole.UserRole),None)
            if not self.horizon_entries:self.clear_horizon();return
            self.rebuild_horizon_controls();self.redraw()

    def update_sampling_controls(self,*_):
        along=self.mode.currentIndex()==3;ready=self.remote and self.horizon_combo.currentIndex()>=0
        self.position_group.setVisible(not along);self.position_label.setVisible(not along)
        self.along_controls.setVisible(along);self.horizon_combo.setEnabled(along and ready)
        entry=self.horizon_entries.get(self.horizon_combo.currentData(),{});self.interpolation_marks.setVisible(along and entry.get('derived',False))
        self.sampling.setEnabled(along and ready)
        self.sampling_note.setText('沿层取值：按层位时间在相邻压力采样间线性插值，或选择最近采样点。' if along and ready else '时间取值仅用于“沿层位压力切片”；请先载入成果并选择层位。' if along else '时间线性插值仅用于“沿层位压力切片”，不用于 Inline / Crossline 剖面上的层位线。')

    def ensure_project_horizons(self):
        if not self.remote or not self.reader_process or not self.horizon_config_provider:return
        try:
            known={str(Path(entry['source']).resolve()) for entry in self.horizon_entries.values()}
            for config in self.horizon_config_provider():
                signature=json.dumps(config,sort_keys=True,ensure_ascii=False)
                if str(Path(config['path']).resolve()) in known or signature in self.project_horizon_attempts:continue
                self.project_horizon_attempts.add(signature);self.import_horizon_config(config)
        except Exception as exc:
            self.horizon_load_error='项目层位载入失败：'+str(exc);self.context.setText(self.horizon_load_error)

    def along_diagnostics(self):
        if self.mode.currentIndex()!=3:return ''
        stats=self.last_stats;matched=stats.get('horizon_points',0);inside=stats.get('horizon_in_range_points',0);valid=stats.get('valid_pressure_points',0)
        message=f'\n沿层位：{self.horizon_combo.currentText()}；已匹配 {matched} 道，时间在道范围内 {inside} 道，本次显示有效压力 {valid} 点。'
        if self.horizon_entries.get(self.horizon_combo.currentData(),{}).get('derived'):message+=f'\n层位来源：原始点 {stats.get("original_horizon_points",0)} 个，空间插值点 {stats.get("interpolated_horizon_points",0)} 个；空心圆标记有效压力处的插值点。'
        if not valid:
            message+='\n'+('没有匹配到地震道，请核对层位坐标、线号与匹配距离。' if not matched else '层位时间不在地震道范围内，请核对时间单位和单/双程时间类型。' if not inside else '层位时间处于压力无效区，或插值相邻压力点无效；请核对压力体有效区。')
        return message

    def import_horizon(self):
        if not self.remote or not self.reader_process:
            QMessageBox.information(self, "先载入地震成果", "请先选择 SEG-Y 成果目录并点击“载入切片”。"); return
        from .horizon_dialog import HorizonImportDialog
        dialog = HorizonImportDialog(self, self.view.time_convention)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.import_horizon_config(dialog.config())

    def import_horizon_config(self, config):
        if not self.remote or not self.reader_process:
            QMessageBox.information(self,'先载入压力成果','请先在地震数据中载入压力成果，再加载层位。'); return
        self.horizon_load_error='';self.import_id = uuid4().hex
        self.horizon_import_ids.add(self.import_id)
        self.horizon_import_queue.append({'id':self.import_id,'command':'import_horizon','config':config});self.send_pending()

    def open_horizon_interpolation(self,key=None,config=None):
        if not self.remote or not self.reader_process:
            QMessageBox.information(self,'先载入压力成果','请先载入 SEG-Y 压力成果，插值仅针对已有地震道进行。');return
        if self.interpolation_ids:
            QMessageBox.information(self,'正在插值','请等待当前层位插值结束。');return
        if config is not None:
            source=str(Path(config['path']).resolve());matches={k:v for k,v in self.horizon_entries.items() if v['source']==source}
            if not matches:self.pending_interpolation_source=source;self.import_horizon_config(config);return
            key=next(iter(matches))
        if key not in self.horizon_entries:return
        source=self.horizon_entries[key]['source'];entries={k:v for k,v in self.horizon_entries.items() if v['source']==source}
        from .horizon_interpolation_dialog import HorizonInterpolationDialog
        dialog=HorizonInterpolationDialog(self,entries,key)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            try:self.interpolate_horizon(dialog.parameters())
            except ValueError as exc:QMessageBox.warning(self,'无法开始插值',str(exc))

    def interpolate_horizon(self,parameters):
        if not self.remote or not self.reader_process:raise ValueError('请先载入 SEG-Y 压力成果')
        if self.interpolation_ids:raise ValueError('请等待当前层位插值结束')
        if len(self.horizon_entries)>=64:raise ValueError('层位数量已达上限，请先移除不需要的层位')
        request_id=uuid4().hex;self.interpolation_ids.add(request_id);self.horizon_import_ids.add(request_id)
        self.horizon_import_queue.append(dict(parameters,id=request_id,command='interpolate_horizon'));self.send_pending()

    def import_wells(self):
        if not self.remote or not self.reader_process:
            QMessageBox.information(self,'先载入压力成果','请先载入 SEG-Y 压力成果。'); return
        try:
            config = self.well_config_provider() if self.well_config_provider else None
            if not config or not config['assets']: raise ValueError('请先从左侧井对象导入曲线、井口坐标或轨迹。')
            self.wells_id=uuid4().hex; self.pending_wells={'id':self.wells_id,'command':'import_wells','config':config}; self.send_pending()
        except Exception as exc: QMessageBox.warning(self,'井数据加载失败',str(exc))

    def mapped_well(self):
        if not self.well_path: raise ValueError('请先加载项目井数据。')
        from .well_spatial import WellMap
        return WellMap.load(self.well_path,self.view)

    def jump_well(self):
        try:
            mapping=self.mapped_well(); rows=mapping.trace_row[(mapping.wells==self.well_combo.currentText()) & (mapping.trace_row>=0)]
            if not len(rows):raise ValueError('所选井没有定位成功的井点。')
            mode=self.mode.currentIndex()
            if mode>1:self.mode.setCurrentIndex(0); mode=0
            lines=self.view.inline[rows] if mode==0 else self.view.crossline[rows]
            values,counts=np.unique(lines,return_counts=True); self.remote_position.setValue(float(values[np.argmax(counts)]))
        except Exception as exc:QMessageBox.warning(self,'定位井剖面',str(exc))

    def compare_well(self):
        try:
            from .well_compare import WellCompareDialog
            dialog=WellCompareDialog(self,self.mapped_well(),self.well_combo.currentText()); dialog.exec()
        except Exception as exc:QMessageBox.warning(self,'井震对比',str(exc))

    def refresh_display(self, *_):
        if self.last_section is not None:
            self.render(self.last_section, self.last_limits, self.last_overlay, self.last_overlay_name)

    def choose_overlay_color(self):
        from PySide6.QtGui import QColor
        item=self.horizon_list.currentItem();entry=self.horizon_entries.get(item.data(Qt.ItemDataRole.UserRole)) if item else None
        color = QColorDialog.getColor(QColor(entry['color'] if entry else self.overlay_color), self, "选择所选层位线颜色")
        if color.isValid():
            if entry:entry['color']=color.name()
            else:self.overlay_color=color.name()
            self.refresh_display()

    def export_horizon_report(self):
        if not self.horizon_path:
            QMessageBox.information(self, "尚无层位", "请先导入层位数据。"); return
        path, _ = QFileDialog.getSaveFileName(self, "导出层位匹配明细", "horizon_matches.csv", "CSV 表格 (*.csv)")
        if path:
            try:
                import shutil
                item=self.horizon_list.currentItem();entry=self.horizon_entries.get(item.data(Qt.ItemDataRole.UserRole)) if item else None
                source = Path(entry['path'] if entry else self.horizon_path)
                if Path(path).resolve() != source.with_suffix(".csv").resolve(): shutil.copyfile(source.with_suffix(".csv"), path)
                target = Path(path).with_suffix(".settings.json")
                if target.resolve() != source.with_suffix(".json").resolve(): shutil.copyfile(source.with_suffix(".json"), target)
            except Exception as exc: QMessageBox.warning(self, "报告导出失败", str(exc))

    def display_settings(self):
        if self.last_section is None:
            QMessageBox.information(self, "先载入切片", "请先载入并显示一张切片。"); return
        dialog = self.create_display_dialog(); dialog.exec()

    def create_display_dialog(self):
        dialog = QDialog(self); dialog.setWindowTitle("切片绘图设置 · 坐标 / 色标 / 图片比例")
        dialog.resize(640,600)
        outer=QVBoxLayout(dialog);tabs=QTabWidget();outer.addWidget(tabs)
        coordinates=QWidget();form=QFormLayout(coordinates);tabs.addTab(coordinates,'坐标与色标')
        section = self.last_section
        automatic = QCheckBox("自动坐标范围"); automatic.setChecked(self.mode.currentIndex() not in self.axis_ranges); form.addRow(automatic)
        limits = self.axis_ranges.get(self.mode.currentIndex(), ((float(section.x.min()), float(section.x.max())), (float(section.y.min()), float(section.y.max()))))
        fields = []
        for axis, label in ((0, section.xlabel), (1, section.ylabel)):
            for end, title in ((0, "最小值"), (1, "最大值")):
                edit = QDoubleSpinBox();edit.setObjectName(f'slice_axis_{axis}_{end}'); edit.setDecimals(6); edit.setRange(-1e12, 1e12); edit.setValue(limits[axis][end]); form.addRow(label + " " + title, edit); fields.append(edit)
        auto_color = QCheckBox("自动压力色标范围"); auto_color.setChecked(self.color_limits is None); form.addRow(auto_color)
        low = QDoubleSpinBox(); high = QDoubleSpinBox()
        low.setObjectName('slice_color_min');high.setObjectName('slice_color_max')
        for edit, value, label in ((low, (self.color_limits or self.last_limits)[0], "色标最小压力 (MPa)"), (high, (self.color_limits or self.last_limits)[1], "色标最大压力 (MPa)")):
            edit.setDecimals(6); edit.setRange(-1e12, 1e12); edit.setValue(value); form.addRow(label, edit)
        cmap = QComboBox();cmap.setObjectName('slice_cmap'); cmap.addItems(["viridis", "cividis", "turbo", "RdYlBu_r", "RdBu_r", "seismic", "gray"]); cmap.setCurrentText(self.cmap); form.addRow("配色", cmap)
        typography=TypographyControls(self.plot_style,default_style=FigureStyle(grid=False));tabs.addTab(typography,'字体与标题')
        dimensions=QWidget();size_form=QFormLayout(dimensions);tabs.addTab(dimensions,'图片尺寸与预览')
        width=QDoubleSpinBox();height=QDoubleSpinBox()
        for edit,name,value in [(width,'slice_width_cm',self.image_width_cm),(height,'slice_height_cm',self.image_height_cm)]:
            edit.setObjectName(name);edit.setRange(5,100);edit.setDecimals(2);edit.setValue(value)
        size_form.addRow('图片宽度 (cm)',width);size_form.addRow('图片高度 (cm)',height)
        dpi=QSpinBox();dpi.setObjectName('slice_dpi');dpi.setRange(72,600);dpi.setValue(self.image_dpi);size_form.addRow('导出分辨率 (DPI)',dpi)
        ratio,lock=link_dimensions(size_form,width,height,dpi,prefix='slice',locked=self.image_ratio_locked)
        preview_mode=QLabel('完整显示，可拖动图框调整预览');preview_mode.setObjectName('slice_preview_mode');size_form.addRow('屏幕预览',preview_mode)
        hint = QLabel('拖动图框边缘或四角可调整预览大小及纵横比，始终限制在绘图区内。“适应窗口”保留当前比例，“恢复设置比例”恢复设置中的宽高比；导出仍使用指定图宽、图高及 DPI。坐标范围按切片方向保存；色标及图片比例适用于各切片方向和沿层位图，不改变预测数据。恢复自动范围后点击“应用”。')
        hint.setWordWrap(True); size_form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("应用")
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText("恢复自动范围")
        outer.addWidget(buttons)
        def toggles():
            for edit in fields: edit.setEnabled(not automatic.isChecked())
            low.setEnabled(not auto_color.isChecked()); high.setEnabled(not auto_color.isChecked())
        automatic.toggled.connect(toggles); auto_color.toggled.connect(toggles); toggles()
        def apply():
            try:style=replace(self.plot_style,**typography.collect()).validate()
            except ValueError as exc:QMessageBox.warning(dialog,'绘图设置需要修改',str(exc));return
            xlim = (fields[0].value(), fields[1].value()); ylim = (fields[2].value(), fields[3].value())
            if not automatic.isChecked() and (xlim[0] >= xlim[1] or ylim[0] >= ylim[1]):
                QMessageBox.warning(dialog, "范围需要修改", "每个坐标轴的最小值必须小于最大值。"); return
            if not auto_color.isChecked() and low.value() >= high.value():
                QMessageBox.warning(dialog, "色标需要修改", "色标最小压力必须小于最大压力。"); return
            if automatic.isChecked(): self.axis_ranges.pop(self.mode.currentIndex(), None)
            else: self.axis_ranges[self.mode.currentIndex()] = (xlim, ylim)
            self.color_limits = None if auto_color.isChecked() else (low.value(), high.value()); self.cmap = cmap.currentText()
            self.image_width_cm=width.value();self.image_height_cm=height.value();self.image_dpi=dpi.value()
            self.plot_style=style
            self.preview_mode='fit';self.image_ratio_locked=lock.isChecked()
            self.preview_controls.lock.setChecked(self.image_ratio_locked);self.set_preview_lock(self.image_ratio_locked);self.refresh_display()
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(apply)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(lambda: (automatic.setChecked(True), auto_color.setChecked(True)))
        buttons.rejected.connect(dialog.reject)
        polish_layouts(dialog)
        return bound_plot_dialog(dialog)

    def export(self):
        if self.view is None: return
        path, _ = QFileDialog.getSaveFileName(self, "导出当前切片", "pressure_slice.png", "PNG 图片 (*.png);;PDF 图件 (*.pdf);;SVG 图件 (*.svg)")
        if path:
            try: self.save_slice_figure(path)
            except Exception as exc: QMessageBox.warning(self, "导出失败", str(exc))

    def save_slice_figure(self,path):
        figure=self.canvas.figure;size=figure.get_size_inches().copy()
        try:
            figure.set_size_inches(self.image_width_cm/2.54,self.image_height_cm/2.54,forward=False)
            figure.savefig(path,dpi=self.image_dpi,facecolor='white')
        finally:
            figure.set_size_inches(size,forward=False);self.canvas.draw_idle()

"""Interactive project basemap; heavy SEG-Y access runs in a separate worker."""
import codecs
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                              QPushButton, QDoubleSpinBox, QCheckBox, QSizePolicy)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .runtime import worker_command, user_data_root
from .seismic_view import Slice
from .flow_layout import FlowLayout
from .charts import FigureStyle, apply_typography


class MapView(QWidget):
    sectionReady = Signal(str, object, dict)
    sectionRemoved = Signal(str)
    opened = Signal(dict)

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.process = None
        self.metadata = None
        self.geometry = {}
        self.wells = []
        self.paths = {}
        self.active_line = None
        self.draft = []
        self.drag = None
        self.ready = False
        self.busy = False
        self.pending = {}
        self.latest = {}
        self.buffer = ''
        self.map_values = None
        self.map_time = None
        self.desired_config = None
        self.open_config = None
        self.generation = 0
        self.cache_path = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.dataset = QComboBox()
        self.dataset.setMinimumContentsLength(12)
        self.dataset.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        row.addWidget(QLabel('显示数据'))
        row.addWidget(self.dataset, 1)
        import_button = QPushButton('导入 SEG-Y…')
        import_button.clicked.connect(lambda: owner.shell.import_seismic())
        row.addWidget(import_button)
        reload_button = QPushButton('重新载入')
        reload_button.clicked.connect(self.reload)
        row.addWidget(reload_button)
        layout.addLayout(row)
        row = FlowLayout()
        self.coordinates = QComboBox()
        self.coordinates.addItem('线号底图', 'lines')
        self.coordinates.addItem('XY 底图', 'xy')
        row.addWidget(self.coordinates)
        self.tool = QComboBox()
        for title, mode in [('选择 / 拖动剖面', 'select'), ('Inline 剖面', 'inline'),
                            ('Crossline 剖面', 'crossline'), ('任意直线剖面', 'arbitrary'),
                            ('任意折线剖面', 'polyline')]:
            self.tool.addItem(title, mode)
        row.addWidget(self.tool)
        finish = QPushButton('完成折线')
        finish.clicked.connect(self.finish_path)
        row.addWidget(finish)
        remove = QPushButton('移除选中剖面线')
        remove.clicked.connect(self.remove_line)
        row.addWidget(remove)
        layout.addLayout(row)
        row = FlowLayout()
        self.attribute = QCheckBox('显示时间切片')
        row.addWidget(self.attribute)
        self.time = QDoubleSpinBox()
        self.time.setRange(-1e9, 1e9)
        self.time.setDecimals(3)
        self.time.setSuffix(' ms')
        self.time.setKeyboardTracking(False)
        self.time.setMaximumWidth(140)
        row.addWidget(self.time)
        self.equal_scale=QCheckBox('等比例坐标')
        self.equal_scale.setToolTip('勾选后保持横纵坐标尺度一致；取消勾选时适应窗口。剖面取道始终使用原始坐标。')
        self.equal_scale.toggled.connect(lambda:self.draw_map(preserve=True))
        row.addWidget(self.equal_scale)
        self.line_label = QLabel()
        row.addWidget(self.line_label)
        layout.addLayout(row)
        self.figure = Figure(figsize=(9, 6), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(0,0)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel('导入地震后显示工区底图；选择剖面工具，用鼠标在底图上拉剖面。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(180)
        self.redraw_timer.timeout.connect(self.request_active_section)
        self.dataset.currentIndexChanged.connect(self.load_selected)
        self.coordinates.currentIndexChanged.connect(self.coordinate_changed)
        self.tool.currentIndexChanged.connect(self.tool_changed)
        self.attribute.toggled.connect(self.request_map)
        self.time.valueChanged.connect(self.request_map)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('motion_notify_event', self.move)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('key_press_event', self.key_press)
        self.draw_map()

    def set_datasets(self, datasets, selected=None):
        current = selected or self.dataset.currentData()
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for config in datasets:
            self.dataset.addItem(config['label'], config)
        index = next((i for i, item in enumerate(datasets) if item['path'] == (current or {}).get('path')), 0)
        if datasets:
            self.dataset.setCurrentIndex(index)
        self.dataset.blockSignals(False)
        self.load_selected()

    def load_selected(self, *_):
        config = self.dataset.currentData()
        if config == self.desired_config:
            return
        self.desired_config = config
        self.shutdown()
        self.metadata = None
        self.geometry = {}
        self.map_values = None
        self.wells = []
        self.paths = {}
        self.active_line = None
        self.draft = []
        self.drag = None
        self.draw_map()
        if config is None:
            self.status.setText('导入地震后显示工区底图；可从左侧对象或上方按钮导入。')
            return
        self.open_config = config
        self.status.setText('正在读取地震道头和采样信息……')
        proc = QProcess(self)
        self.process = proc
        generation = self.generation
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONIOENCODING', 'utf-8')
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(user_data_root()))
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.cache_path = user_data_root() / 'outputs' / 'browser_cache' / uuid4().hex
        proc.readyReadStandardOutput.connect(lambda: self.read_output(proc, generation))
        proc.readyReadStandardError.connect(lambda: self.read_error(proc, generation))
        proc.finished.connect(lambda code: self.finished(proc, generation, code))
        proc.errorOccurred.connect(lambda error: self.failed_start(proc, generation, error))
        proc.start(*worker_command('overpressure.browser_worker', ['--cache', str(self.cache_path)]))

    def reload(self):
        self.desired_config = None
        self.load_selected()

    def enqueue(self, key, request):
        request = dict(request, id=uuid4().hex)
        self.latest[key] = request['id']
        self.pending[key] = request
        self.send_next()

    def send_next(self):
        if not self.process or not self.ready or self.busy or not self.pending:
            return
        key = next(iter(self.pending))
        request = self.pending.pop(key)
        self.busy = True
        self.process.write((json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8'))

    def read_error(self, proc, generation):
        message = bytes(proc.readAllStandardError()).decode('utf-8', errors='replace')
        if generation == self.generation and message.strip():
            self.owner.log.appendPlainText(message.strip())

    def read_output(self, proc, generation):
        raw = bytes(proc.readAllStandardOutput())
        if generation != self.generation:
            return
        self.buffer += self.decoder.decode(raw)
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            try:
                event = json.loads(line)
                self.handle_event(event)
            except Exception as exc:
                self.status.setText('地震浏览失败：' + str(exc))
                self.busy = False
        self.send_next()

    def handle_event(self, event):
        kind = event.get('event')
        if kind == 'ready':
            self.ready = True
            self.enqueue('open', dict(command='open', config=self.open_config))
            return
        if kind not in ('opened', 'map', 'section', 'error'):
            return
        self.busy = False
        key = next((key for key, ident in self.latest.items() if ident == event.get('id')), None)
        if key is None:
            return
        if kind == 'error':
            self.status.setText('地震浏览失败：' + event['message'])
            return
        if kind == 'opened':
            self.metadata = event['metadata']
            self.wells = self.metadata.get('wells', [])
            with np.load(event['path'], allow_pickle=False) as data:
                self.geometry = {name: data[name].copy() for name in data.files}
            self.coordinates.blockSignals(True)
            self.coordinates.model().item(1).setEnabled(self.metadata['xy_available'])
            self.coordinates.setCurrentIndex(1 if self.metadata['xy_available'] else 0)
            self.coordinates.blockSignals(False)
            self.time.blockSignals(True)
            self.time.setRange(*self.metadata['time_range'])
            self.time.setSingleStep(self.metadata['interval_ms'])
            self.time.setValue(float(np.mean(self.metadata['time_range'])))
            self.time.blockSignals(False)
            self.opened.emit(self.metadata)
            self.draw_map()
            self.status.setText(f"{self.metadata['label']} · {self.metadata['traces']:,} 道；底图显示 {self.metadata['map_points']:,} 个道位置。选择剖面工具开始查看。")
            if self.metadata.get('warnings'):
                self.owner.log.appendPlainText('\n'.join(self.metadata['warnings']))
            if self.attribute.isChecked():
                self.request_map()
        elif kind == 'map':
            with np.load(event['path'], allow_pickle=False) as data:
                self.map_values = data['values'].copy()
            self.map_time = event['time_ms']
            self.draw_map(preserve=True)
            self.status.setText(f"时间切片 {self.map_time:g} ms；显示点按道序抽取，未做空间插值。")
        elif kind == 'section':
            with np.load(event['path'], allow_pickle=False) as data:
                section = Slice(data['values'].copy(), data['x'].copy(), data['y'].copy(),
                                str(data['xlabel'].item()), str(data['ylabel'].item()), str(data['title'].item()), True)
            view_id = event['view_id']
            meta = event['metadata']
            if view_id in self.paths:
                self.paths[view_id].update(line=meta['line'], **{k: meta[k] for k in ('position',) if k in meta})
            self.sectionReady.emit(view_id, section, meta)
            self.draw_map(preserve=True)
            self.status.setText('剖面已更新；切换到“选择 / 拖动剖面”可移动剖面线或端点。')

    def request_map(self, *_):
        if not self.metadata:
            return
        if self.attribute.isChecked():
            self.status.setText('正在按需读取时间切片……')
            self.enqueue('map', dict(command='map', time_ms=self.time.value()))
        else:
            self.draw_map(preserve=True)

    def coordinate_changed(self, *_):
        self.paths = {}
        self.active_line = None
        self.draft = []
        self.drag = None
        self.draw_map()

    def tool_changed(self, *_):
        self.draft = []
        self.drag = None
        if self.toolbar.mode:
            if str(self.toolbar.mode) == 'pan/zoom':
                self.toolbar.pan()
            else:
                self.toolbar.zoom()
        self.status.setText('直线：拖动或点击起终点；折线：连续单击，双击 / 右键结束；选择模式：拖动端点或移动线。Esc 取消绘制。')
        self.draw_map(preserve=True)

    def map_points(self):
        if not self.geometry:
            return np.empty((0, 2))
        if self.coordinates.currentData() == 'xy':
            return np.column_stack((self.geometry['x'], self.geometry['y']))
        return np.column_stack((self.geometry['crossline'], self.geometry['inline']))

    def set_wells(self, wells):
        self.wells = wells
        self.draw_map(preserve=True)

    def draw_map(self, preserve=False):
        previous = (self.ax.get_xlim(), self.ax.get_ylim()) if preserve and hasattr(self, 'ax') and self.geometry else None
        self.figure.clear()
        self.ax = self.figure.subplots()
        points = self.map_points()
        xy = self.coordinates.currentData() == 'xy'
        if len(points):
            if self.attribute.isChecked() and self.map_values is not None:
                finite = self.map_values[np.isfinite(self.map_values)]
                low, high = np.quantile(finite, [.01, .99]) if len(finite) else (0, 1)
                if low == high:
                    high = low + max(1., abs(low) * .01)
                image = self.ax.scatter(points[:, 0], points[:, 1], c=self.map_values, s=5, marker='s',
                                        cmap='viridis', vmin=low, vmax=high, linewidths=0, rasterized=True)
                self.figure.colorbar(image, ax=self.ax, label=self.metadata.get('unit') or 'Amplitude')
            else:
                self.ax.scatter(points[:, 0], points[:, 1], s=2, c='#9aaebd', linewidths=0, rasterized=True)
        for well in self.wells:
            if xy:
                position = (well['x'], well['y'])
            elif 'inline' in well:
                position = (well['crossline'], well['inline'])
            else:
                continue
            self.ax.plot(*position, marker='o', color='#243b53', markersize=5)
            self.ax.annotate(well['name'], position, xytext=(5, 5), textcoords='offset points', fontsize=9)
        for ident, record in self.paths.items():
            line = np.asarray(record.get('line', record.get('vertices', [])))
            if not len(line):
                continue
            selected = ident == self.active_line
            self.ax.plot(line[:, 0], line[:, 1], color='#cb641d' if selected else '#297ca2',
                         linewidth=1.8 if selected else 1., label=record['label'])
            if selected:
                vertices = np.asarray(record.get('vertices') or line[[0, -1]].tolist())
                self.ax.plot(vertices[:, 0], vertices[:, 1], 'o', color='#cb641d', markersize=5)
            self.ax.annotate(record['label'], line[0], fontsize=8, xytext=(4, 4), textcoords='offset points')
        if self.draft:
            points = np.asarray(self.draft)
            self.ax.plot(points[:, 0], points[:, 1], 'o--', color='#cb641d', linewidth=1.)
        self.ax.set(xlabel='X (seismic coordinates)' if xy else 'Crossline',
                    ylabel='Y (seismic coordinates)' if xy else 'Inline',
                    title=self.metadata['label'] + (' · time slice' if self.attribute.isChecked() and self.map_values is not None else ' · Map') if self.metadata else 'Project Map')
        self.ax.set_aspect('equal' if self.equal_scale.isChecked() else 'auto', adjustable='box')
        self.ax.ticklabel_format(useOffset=False, style='plain')
        self.ax.grid(True, alpha=.18)
        if not self.geometry:
            self.ax.text(.5, .5, 'Import SEG-Y to display the project map', ha='center', va='center', transform=self.ax.transAxes)
        if previous:
            self.ax.set_xlim(*previous[0])
            self.ax.set_ylim(*previous[1])
        apply_typography(self.figure,FigureStyle(font_size=9,title_size=12,grid=True))
        self.canvas.draw_idle()

    def nearest_line(self, mode, point):
        if self.coordinates.currentData() == 'lines':
            return int(round(point[1] if mode == 'inline' else point[0]))
        points = self.map_points()
        nearest = int(np.argmin(np.sum((points - point) ** 2, axis=1)))
        return int(self.geometry[mode][nearest])

    def create_line(self, mode, vertices=None, position=None):
        if not self.metadata:
            return
        ident = uuid4().hex
        number=1
        labels={record['label'] for record in self.paths.values()}
        while f'S{number}' in labels:number+=1
        label = f'S{number}'
        self.paths[ident] = dict(mode=mode, vertices=vertices or [], position=position, label=label,
                                 coordinate_mode=self.coordinates.currentData())
        self.active_line = ident
        self.line_label.setText('当前剖面：' + label)
        self.request_active_section()

    def request_active_section(self):
        record = self.paths.get(self.active_line)
        if not record or not self.metadata:
            return
        self.status.setText('正在按需读取剖面……')
        self.enqueue(self.active_line, dict(command='section', view_id=self.active_line,
                                           **{k: record[k] for k in ('mode', 'vertices', 'position', 'coordinate_mode')}))

    def valid_event(self, event):
        return self.metadata is not None and event.inaxes is self.ax and event.xdata is not None and event.ydata is not None and not self.toolbar.mode

    def press(self, event):
        if not self.valid_event(event):
            return
        if event.button == 3:
            self.finish_path()
            return
        if event.button != 1:
            return
        point = [float(event.xdata), float(event.ydata)]
        mode = self.tool.currentData()
        if mode in ('inline', 'crossline'):
            self.create_line(mode, position=self.nearest_line(mode, point))
            self.drag = ('fixed', self.active_line, 0)
        elif mode == 'arbitrary':
            if len(self.draft) == 1:
                self.draft.append(point)
                self.finish_path()
            else:
                self.draft = [point]
                self.drag = ('new', None, 0)
        elif mode == 'polyline':
            if not self.draft or np.linalg.norm(np.asarray(point) - self.draft[-1]) > 1e-8:
                self.draft.append(point)
            if event.dblclick:
                self.finish_path()
        else:
            for ident, record in reversed(list(self.paths.items())):
                vertices = np.asarray(record.get('vertices') or record.get('line', []))
                if not len(vertices):
                    continue
                screen = self.ax.transData.transform(vertices)
                index = int(np.argmin(np.linalg.norm(screen - [event.x, event.y], axis=1)))
                if np.linalg.norm(screen[index] - [event.x, event.y]) < 12:
                    self.active_line = ident
                    self.drag = ('fixed' if record['mode'] in ('inline', 'crossline') else 'vertex', ident, index)
                    self.line_label.setText('当前剖面：' + record['label'])
                    break
                line=np.asarray(record.get('line') or record.get('vertices',[]))
                if len(line)<2:continue
                pixels=self.ax.transData.transform(line);a,b=pixels[:-1],pixels[1:]
                delta=b-a;denominator=np.sum(delta*delta,axis=1)
                weight=np.clip(np.sum((np.array([event.x,event.y])-a)*delta,axis=1)/np.maximum(denominator,1e-12),0,1)
                distance=np.linalg.norm(a+weight[:,None]*delta-[event.x,event.y],axis=1)
                if distance.min()<10:
                    self.active_line=ident
                    self.drag=('fixed',ident,0) if record['mode'] in ('inline','crossline') else ('translate',ident,point)
                    self.line_label.setText('当前剖面：'+record['label'])
                    break
        self.draw_map(preserve=True)

    def move(self, event):
        if not self.drag or not self.valid_event(event):
            return
        kind, ident, index = self.drag
        point = [float(event.xdata), float(event.ydata)]
        if kind == 'new':
            self.draft = [self.draft[0], point]
        elif kind == 'fixed':
            record = self.paths[ident]
            record['position'] = self.nearest_line(record['mode'], point)
            self.redraw_timer.start()
        elif kind == 'translate':
            record=self.paths[ident];delta=np.asarray(point)-index
            record['vertices']=(np.asarray(record['vertices'])+delta).tolist()
            record['line']=record['vertices'];self.drag=('translate',ident,point)
            self.redraw_timer.start()
        else:
            record = self.paths[ident]
            record['vertices'][index] = point
            record['line'] = record['vertices']
            self.redraw_timer.start()
        self.draw_map(preserve=True)

    def release(self, event):
        if not self.drag:
            return
        kind = self.drag[0]
        self.drag = None
        if kind == 'new' and len(self.draft) == 2:
            self.finish_path()
        elif kind != 'new':
            self.redraw_timer.stop()
            self.request_active_section()

    def finish_path(self):
        if len(self.draft) < 2:
            return
        mode = self.tool.currentData()
        if mode not in ('arbitrary', 'polyline'):
            return
        vertices = [point[:] for point in self.draft]
        self.draft = []
        self.create_line(mode, vertices=vertices)
        self.draw_map(preserve=True)

    def key_press(self, event):
        if event.key == 'escape':
            self.draft = []
            self.drag = None
            self.draw_map(preserve=True)

    def remove_line(self):
        if self.active_line:
            self.sectionRemoved.emit(self.active_line)
            self.paths.pop(self.active_line, None)
            self.pending.pop(self.active_line, None)
            self.latest.pop(self.active_line, None)
        self.active_line = None
        self.line_label.clear()
        self.draw_map(preserve=True)

    def failed_start(self, proc, generation, error):
        if generation == self.generation and error == QProcess.ProcessError.FailedToStart:
            self.status.setText('地震浏览器无法启动：' + proc.errorString())

    def finished(self, proc, generation, code):
        if generation == self.generation:
            self.process = None
            self.ready = False
            self.busy = False
            if code:
                self.status.setText('地震浏览器已停止，请重新载入。')
        proc.deleteLater()

    def shutdown(self):
        self.generation += 1
        self.redraw_timer.stop()
        proc, self.process = self.process, None
        self.ready = False
        self.busy = False
        self.pending = {}
        self.latest = {}
        self.buffer = ''
        if proc is not None:
            if proc.state() != QProcess.ProcessState.NotRunning:
                proc.closeWriteChannel()
                if not proc.waitForFinished(1000):
                    proc.kill()
                    proc.waitForFinished(1000)

    def reset_project(self):
        self.shutdown()
        self.desired_config = None
        self.open_config = None
        self.metadata = None
        self.geometry = {}
        self.wells = []
        self.paths = {}
        self.active_line = None
        self.draft = []
        self.drag = None
        self.map_values = None
        self.line_label.clear()
        self.coordinates.blockSignals(True)
        self.coordinates.setCurrentIndex(0)
        self.coordinates.blockSignals(False)
        self.attribute.blockSignals(True)
        self.attribute.setChecked(False)
        self.attribute.blockSignals(False)
        self.equal_scale.blockSignals(True)
        self.equal_scale.setChecked(False)
        self.equal_scale.blockSignals(False)
        self.dataset.blockSignals(True)
        self.dataset.clear()
        self.dataset.blockSignals(False)
        self.draw_map()

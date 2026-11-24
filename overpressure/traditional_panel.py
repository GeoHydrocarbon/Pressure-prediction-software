"""One traditional-method module: calibrate a trend, adopt it, then predict wells."""
from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QLabel,
                              QComboBox, QListWidget, QListWidgetItem, QTabWidget,
                              QDoubleSpinBox, QPushButton, QMessageBox)

from .traditional import defaults, two_point_trend, validate_prediction
from .ui_style import Card


class TraditionalPanel(QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.loading = True
        self.applied_trends = {}
        self.trend_report = None
        self.trend_path = None
        layout = QVBoxLayout(self)
        hint = QLabel('从井库选择数据，先分析正常压实趋势，再明确采用参数并执行 Eaton 预测。所有图件在中央工作区打开。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        trend_page = QWidget()
        trend_layout = QVBoxLayout(trend_page)
        form = QFormLayout()
        self.analysis_well = QComboBox()
        self.analysis_curve = QComboBox()
        form.addRow('分析井', self.analysis_well)
        form.addRow('声波时差曲线', self.analysis_curve)
        self.analysis_min = self.number(0, 100000, 3)
        self.analysis_max = self.number(0, 100000, 3)
        form.addRow('分析起始深度（m）', self.analysis_min)
        form.addRow('分析结束深度（m）', self.analysis_max)
        trend_layout.addLayout(form)
        group = Card('两点定义正常压实趋势 · 坐标为 深度（m）与 LN(AC [us/ft])')
        form = QFormLayout(group)
        self.point_edits = []
        for index in (1, 2):
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            depth = self.number(0, 100000, 3)
            lnac = self.number(-100, 100, 6)
            line.addWidget(QLabel('深度'))
            line.addWidget(depth)
            line.addWidget(QLabel('LN(AC)'))
            line.addWidget(lnac)
            form.addRow(f'第 {index} 点', row)
            self.point_edits.append((depth, lnac))
        self.equation = QLabel()
        self.equation.setWordWrap(True)
        form.addRow(self.equation)
        trend_layout.addWidget(group)
        self.preview_button = owner.button(trend_layout, '生成正常压实趋势图', lambda: self.start('traditional-trend'))
        row = QHBoxLayout()
        self.apply_target = QComboBox()
        self.apply_target.addItems(['当前分析井', '勾选的预测井（共用该趋势）'])
        row.addWidget(self.apply_target)
        self.adopt_button = QPushButton('采用此趋势用于 Eaton')
        self.adopt_button.clicked.connect(self.adopt)
        row.addWidget(self.adopt_button)
        owner.actions.append(self.adopt_button)
        trend_layout.addLayout(row)
        self.trend_note = QLabel('生成趋势图后可修改两点，或在中央图上取两点。修改后需再次采用，才会改变预测参数。')
        self.trend_note.setWordWrap(True)
        trend_layout.addWidget(self.trend_note)
        trend_layout.addStretch()
        self.tabs.addTab(trend_page, '1  正常压实趋势')
        pressure_page = QWidget()
        pressure_layout = QVBoxLayout(pressure_page)
        self.wells = QListWidget()
        self.wells.setMinimumHeight(110)
        self.wells.setMaximumHeight(180)
        pressure_layout.addWidget(QLabel('待预测井 · 悬停查看已采用的趋势参数'))
        pressure_layout.addWidget(self.wells)
        selection = QHBoxLayout()
        for title, checked in [('全选', True), ('清空', False)]:
            button = QPushButton(title)
            button.clicked.connect(lambda _, value=checked: self.select_all(value))
            selection.addWidget(button)
        pressure_layout.addLayout(selection)
        form = QFormLayout()
        self.ac_curve = QComboBox()
        form.addRow('待预测井共同的 AC 曲线', self.ac_curve)
        self.fields = {}
        for key, title, low, high, decimals in [
            ('eaton_exponent', 'Eaton 指数', .01, 20, 4),
            ('rock_density', '平均岩石密度（g/cc）', .01, 10, 4),
            ('fluid_density', '流体密度（g/cc）', .01, 10, 4),
            ('gravity', '重力加速度（m/s²）', .01, 100, 4),
            ('depth_min', '预测起始深度（m）', 0, 100000, 3),
            ('depth_max', '预测结束深度（m）', 0, 100000, 3),
            ('coefficient_min', '压力系数标记下限', -100, 100, 4),
            ('coefficient_max', '压力系数标记上限', -100, 100, 4),
            ('max_gap_m', '实测点最大匹配间隔（m；0 自动）', 0, 100000, 3)]:
            widget = self.number(low, high, decimals)
            self.fields[key] = widget
            form.addRow(title, widget)
        self.lithology = QComboBox()
        self.lithology.addItem('全部有效 AC 点（请按适用地层选择深度范围）', 'all')
        self.lithology.addItem('仅泥岩（读取 AC 源文件的岩性列）', 'mudstone')
        form.addRow('岩性范围', self.lithology)
        from PySide6.QtWidgets import QLineEdit
        self.lithology_column = QLineEdit('岩性')
        form.addRow('岩性列名', self.lithology_column)
        self.reference_curve = QComboBox()
        self.reference_curve.addItem('不使用压力曲线，仅用已导入实测点', '')
        form.addRow('参考压力曲线（可选）', self.reference_curve)
        pressure_layout.addLayout(form)
        note = QLabel('上覆压力采用常密度近似；压力系数 = Eaton 压力 / 静水压力。区间外结果保留并标记，不截断。实测点优先于参考压力曲线，不插补压力标签。')
        note.setWordWrap(True)
        pressure_layout.addWidget(note)
        self.adoption_summary = QLabel()
        self.adoption_summary.setWordWrap(True)
        pressure_layout.addWidget(self.adoption_summary)
        self.predict_button = owner.button(pressure_layout, '开始 Eaton 井压力预测', lambda: self.start('traditional-predict'))
        pressure_layout.addStretch()
        self.tabs.addTab(pressure_page, '2  Eaton 压力预测')
        self.analysis_well.currentTextChanged.connect(self.analysis_well_changed)
        for widget in (self.analysis_curve,):widget.currentTextChanged.connect(self.update_trend)
        for widget in (self.analysis_min, self.analysis_max, *[w for pair in self.point_edits for w in pair]):
            widget.valueChanged.connect(self.update_trend)
        self.wells.itemChanged.connect(self.selection_changed)
        self.lithology.currentIndexChanged.connect(lambda: self.lithology_column.setEnabled(self.lithology.currentData() == 'mudstone'))
        self.loading = False

    @staticmethod
    def number(low, high, decimals):
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(decimals)
        widget.setKeyboardTracking(False)
        return widget

    def selected_wells(self):
        return [self.wells.item(i).text() for i in range(self.wells.count()) if self.wells.item(i).checkState() == Qt.CheckState.Checked]

    def points(self):
        return [[depth.value(), lnac.value()] for depth, lnac in self.point_edits]

    def scope(self):
        return (self.analysis_well.currentText(), self.analysis_curve.currentText(), self.analysis_min.value(), self.analysis_max.value())

    @staticmethod
    def report_scope(report):
        return (report['well'], report['ac_curve'], report['analysis_min'], report['analysis_max'])

    def ac_names(self, well):
        record = next((item for item in self.owner.library.wells if item['name'] == well), {})
        return [ref['name'] for ref in record.get('curves', []) if ref.get('unit') in ('us/ft', 'us/m') and ref.get('role') not in ('measurement', 'derived')]

    def refresh_library(self, *_):
        if self.loading:return
        self.loading = True
        well = self.analysis_well.currentText()
        selected = set(self.selected_wells())
        self.analysis_well.clear()
        self.analysis_well.addItems([item['name'] for item in self.owner.library.wells])
        self.analysis_well.setCurrentText(well)
        self.wells.clear()
        for record in self.owner.library.wells:
            item = QListWidgetItem(record['name'])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if record['name'] in selected else Qt.CheckState.Unchecked)
            trend = self.applied_trends.get(record['name'])
            item.setToolTip(f"已采用：a={trend['a']:.8g}，b={trend['b']:.8g}；来源：{trend.get('source_well', record['name'])}" if trend else '尚未采用正常压实趋势')
            self.wells.addItem(item)
        self.loading = False
        self.analysis_well_changed()
        self.selection_changed()

    def analysis_well_changed(self, *_):
        if self.loading:return
        current = self.analysis_curve.currentText()
        self.analysis_curve.blockSignals(True)
        self.analysis_curve.clear()
        self.analysis_curve.addItems(self.ac_names(self.analysis_well.currentText()))
        self.analysis_curve.setCurrentText(current or 'AC')
        self.analysis_curve.blockSignals(False)
        self.update_trend()

    def selection_changed(self, *_):
        if self.loading:return
        selected = self.selected_wells()
        current = self.ac_curve.currentText()
        candidates = [self.ac_names(name) for name in selected]
        common = [name for name in candidates[0] if all(name in group for group in candidates[1:])] if candidates else ['AC']
        self.ac_curve.clear()
        self.ac_curve.addItems(common)
        self.ac_curve.setCurrentText(current or 'AC')
        reference = self.reference_curve.currentData()
        names = list(dict.fromkeys(ref['name'] for well in self.owner.library.wells if well['name'] in selected
                                  for ref in well['curves'] if ref.get('unit') in ('MPa', 'kPa', 'psi') and ref.get('role') not in ('measurement', 'derived')))
        self.reference_curve.clear()
        self.reference_curve.addItem('不使用压力曲线，仅用已导入实测点', '')
        for name in names:self.reference_curve.addItem(name, name)
        index = self.reference_curve.findData(reference if reference is not None else 'Pp')
        self.reference_curve.setCurrentIndex(max(0, index))
        ready = sum(name in self.applied_trends for name in selected)
        self.adoption_summary.setText(f'已勾选 {len(selected)} 口井，其中 {ready} 口已采用趋势。')

    def select_all(self, checked):
        self.loading = True
        for i in range(self.wells.count()):self.wells.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.loading = False
        self.selection_changed()

    def update_trend(self, *_):
        if self.loading:return
        try:
            trend = two_point_trend(self.points())
            self.equation.setText(f"a = {trend['a']:.9g}；b = {trend['b']:.9g}\n正常 AC = exp(a × 深度 + b)" + ('\n当前趋势随深度增大而增大，请核对正常压实段。' if trend['a'] >= 0 else ''))
            ready = self.trend_report is not None and self.report_scope(self.trend_report) == self.scope()
            self.adopt_button.setEnabled(ready and self.owner.process is None)
            if ready and hasattr(self.owner, 'shell'):
                self.owner.shell.workspace.update_traditional_trend(self.trend_path, trend)
        except ValueError as exc:
            self.equation.setText(str(exc))
            self.adopt_button.setEnabled(False)

    def populate(self, settings):
        cfg = defaults(settings)
        self.trend_report = None
        self.trend_path = None
        self.applied_trends = deepcopy(cfg['applied_trends'])
        self.refresh_library()
        self.loading = True
        self.analysis_well.setCurrentText(cfg['analysis_well'])
        self.loading = False
        self.analysis_well_changed()
        self.loading = True
        self.analysis_curve.setCurrentText(cfg['analysis_curve'])
        self.analysis_min.setValue(cfg['analysis_min'])
        self.analysis_max.setValue(cfg['analysis_max'])
        for widgets, point in zip(self.point_edits, cfg['points']):
            for widget, value in zip(widgets, point):widget.setValue(value)
        for key, widget in self.fields.items():widget.setValue(cfg[key])
        self.lithology.setCurrentIndex(self.lithology.findData(cfg['lithology_filter']))
        self.lithology_column.setText(cfg['lithology_column'])
        for i in range(self.wells.count()):
            self.wells.item(i).setCheckState(Qt.CheckState.Checked if self.wells.item(i).text() in cfg['wells'] else Qt.CheckState.Unchecked)
        self.loading = False
        self.selection_changed()
        self.ac_curve.setCurrentText(cfg['ac_curve'])
        self.reference_curve.setCurrentIndex(max(0, self.reference_curve.findData(cfg['reference_curve'])))
        self.lithology_column.setEnabled(self.lithology.currentData()=='mudstone')
        self.update_trend()

    def collect(self):
        return dict(analysis_well=self.analysis_well.currentText(), analysis_curve=self.analysis_curve.currentText(), ac_curve=self.ac_curve.currentText(),
                    analysis_min=self.analysis_min.value(), analysis_max=self.analysis_max.value(), points=self.points(),
                    wells=self.selected_wells(), applied_trends=deepcopy(self.applied_trends),
                    lithology_filter=self.lithology.currentData(), lithology_column=self.lithology_column.text().strip(),
                    reference_curve=self.reference_curve.currentData() or '', **{key: widget.value() for key, widget in self.fields.items()})

    def start(self, command):
        try:
            if not self.analysis_well.currentText() and command == 'traditional-trend':raise ValueError('请先在左侧井库导入井与 AC 曲线')
            if command == 'traditional-trend':
                two_point_trend(self.points())
                if not self.analysis_curve.currentText():raise ValueError('所选井缺少源单位为 us/ft 或 us/m 的声波时差曲线')
                if self.analysis_min.value()>=self.analysis_max.value():raise ValueError('分析起始深度必须小于结束深度')
            else:
                validate_prediction(self.collect())
                if not self.ac_curve.currentText():raise ValueError('所选井没有共同的 AC 曲线，请核对曲线名称和单位')
            self.owner.start_task(command, [])
        except ValueError as exc:QMessageBox.warning(self, '传统压力预测设置需要修改', str(exc))

    def accept_points(self, points, report):
        if self.report_scope(report) != self.scope():
            QMessageBox.information(self, '选择当前分析井', '请先在传统压力预测模块选择与图件相同的井、曲线和分析范围。')
            return
        self.trend_report=report
        self.trend_path=Path(report['run_path'])
        self.loading = True
        for widgets, point in zip(self.point_edits, points):
            for widget, value in zip(widgets, point):widget.setValue(value)
        self.loading = False
        self.update_trend()

    def adopt(self):
        if self.trend_report is None or self.scope() != self.report_scope(self.trend_report):
            QMessageBox.warning(self, '先生成趋势图', '井、曲线或分析范围已改变，请重新生成趋势图。')
            return
        targets = self.selected_wells() if self.apply_target.currentIndex() else [self.analysis_well.currentText()]
        if not targets:
            QMessageBox.warning(self, '选择应用井', '请先在 Eaton 页勾选要采用此趋势的井。')
            return
        trend = two_point_trend(self.points())
        for name in targets:
            self.applied_trends[name] = dict(trend, confirmed=True, source_well=self.analysis_well.currentText(),
                                             source_curve=self.analysis_curve.currentText(), source_report=str(self.trend_path),
                                             analysis_min=self.analysis_min.value(), analysis_max=self.analysis_max.value(),
                                             source_data_sha256=self.trend_report['data_sha256'])
        selected = set(self.selected_wells()) | set(targets)
        self.refresh_library()
        for i in range(self.wells.count()):
            if self.wells.item(i).text() in selected:self.wells.item(i).setCheckState(Qt.CheckState.Checked)
        if self.owner.save_settings('traditional'):
            self.trend_note.setText(f'已将当前趋势应用到 {len(targets)} 口井。可切换到 Eaton 页开始预测。')

    def load_result(self, path, artifact):
        path = Path(path).resolve()
        report = json.loads((path / 'traditional.json').read_text(encoding='utf-8'))
        if report.get('status')!='completed':raise ValueError('传统方法任务尚未完成')
        self.owner.shell.workspace.open_traditional(path)
        if artifact == 'traditional_trend':
            self.trend_report, self.trend_path = report, path
        else:
            self.register_curves(path, report)
        self.update_trend()

    def register_curves(self, path, report):
        from .well_library import merge_wells
        from .data import sha256_file
        definitions = [('Eaton压力', 'Pp_eaton', 'MPa'), ('Eaton压力系数', 'pressure_coefficient', '无量纲'),
                       ('正常压实AC', 'AC_normal', 'us/ft'), ('静水压力', 'P_h', 'MPa'), ('上覆压力', 'P_o', 'MPa')]
        incoming = []
        for entry in report['per_well_files']:
            source = path / entry['path']
            if sha256_file(source) != entry['sha256']:raise ValueError('Eaton 成果文件校验失败，未登记派生曲线')
            base = dict(path=str(source), well_column='well', well_value=entry['well'], depth_column='Depth', depth_unit='m',
                        rows=entry['rows'], depth_min=entry['depth_min'], depth_max=entry['depth_max'], role='derived', method='eaton', result=str(path))
            existing = next((well['curves'] for well in self.owner.library.wells if well['name']==entry['well']), [])
            protected = {ref['name'] for ref in existing if ref.get('role')!='derived' or ref.get('method')!='eaton'}
            curves = []
            for name, column, unit in definitions:
                candidate = name
                suffix = 1
                while candidate in protected:
                    candidate = f'Eaton·{name}' if suffix==1 else f'Eaton·{name} ({suffix})'
                    suffix += 1
                curves.append(dict(base, name=candidate, column=column, unit=unit))
            incoming.append(dict(name=entry['well'], curves=curves))
        self.owner.library.wells = merge_wells(self.owner.library.wells, incoming)
        self.owner.library.refresh()

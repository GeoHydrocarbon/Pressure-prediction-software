from pathlib import Path
import json

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox,
    QLabel, QFileDialog, QMessageBox, QScrollArea, QStackedWidget, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QDialog, QFormLayout, QFontComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QDialogButtonBox,QTabWidget,QSizePolicy)
from .plot_frame import PlotResizeControls
from .plot_settings import TypographyControls,link_dimensions
from .charts import FigureStyle, training_figure, pressure_figure, folds_figure, scatter_figure, export_figure
from .evaluation import evaluation_table, MAPE_RULE
from .well_points import ROLE_LABELS, ROLE_ENGLISH
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from .ui_style import Card,AspectPreview,polish_layouts,bound_plot_dialog


class PlotToolbar(NavigationToolbar2QT):
    def save_figure(self, *args):
        self.parentWidget().export()


class ChartPanel(QWidget):
    def __init__(self, settings_path=None):
        super().__init__()
        from .runtime import user_data_root
        self.settings_path = Path(settings_path) if settings_path else user_data_root() / "outputs" / "chart_style.json"
        self.style = FigureStyle.load(self.settings_path)
        self.history = pd.DataFrame(); self.pressures = pd.DataFrame(); self.folds = pd.DataFrame()
        self.reference_points = None; self.point_settings = {}; self.well_roles = {}
        self.source_label = "Evaluation data"; self.source_file = None
        self.preview_mode='fit';self.image_ratio_locked=False
        layout = QVBoxLayout(self)
        heading=QLabel('成果图表');heading.setObjectName('heading');layout.addWidget(heading)
        card=Card();self.settings_card=card;layout.addWidget(card);options=QVBoxLayout(card)
        row = QHBoxLayout(); self.directory = QLineEdit(); row.addWidget(self.directory, 1)
        select = QPushButton("选择成果目录"); select.clicked.connect(self.choose_directory); row.addWidget(select)
        options.addLayout(row)
        controls = QHBoxLayout(); self.mode = QComboBox()
        self.mode.addItems(["训练曲线", "压力深度曲线", "留一井误差", "预测 vs 真实散点图", "评价指标表"])
        self.mode.addItem("实测点匹配明细")
        self.well = QComboBox(); controls.addWidget(self.mode); controls.addWidget(self.well)
        self.style_button = QPushButton("绘图设置"); self.style_button.clicked.connect(self.show_plot_settings); controls.addWidget(self.style_button)
        self.export_button = QPushButton("导出图片"); self.export_button.clicked.connect(self.export); controls.addWidget(self.export_button)
        self.table_export = QPushButton("导出指标表"); self.table_export.clicked.connect(self.export_metrics); controls.addWidget(self.table_export)
        controls.addStretch(); options.addLayout(controls)
        self.context = QLabel("选择模型、井预测或留一井验证的成果目录。已有结果可直接生成散点图和指标表，无需重新训练。")
        self.context.setWordWrap(True);self.context.setObjectName('hint');options.addWidget(self.context)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(8, 5), layout="constrained"))
        self.toolbar = PlotToolbar(self.canvas, self); layout.addWidget(self.toolbar)
        self.views = QStackedWidget(); layout.addWidget(self.views, 1)
        self.preview = QScrollArea(); self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aspect_preview=AspectPreview(self.canvas);self.preview.setWidget(self.aspect_preview);self.preview.setWidgetResizable(True)
        self.preview.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for widget in (self.preview,self.views):
            widget.setMinimumSize(0,0);widget.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        self.preview_controls=PlotResizeControls(self.aspect_preview,'chart',self)
        layout.insertWidget(layout.indexOf(self.views),self.preview_controls)
        self.preview_controls.lock.toggled.connect(self.set_preview_lock)
        self.aspect_preview.set_preview_mode('fit')
        self.views.addWidget(self.preview)
        self.table = QTableWidget(); self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide(); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.views.addWidget(self.table)
        self.note = QLabel("MAPE 按百分比显示；真实值为零的点仅从 MAPE 中排除，并单列数量；接近零时 MAPE 可能偏大。R² 无法定义时显示 —，允许负值。整体指标按有效采样点汇总。")
        self.note.setWordWrap(True); layout.addWidget(self.note); self.note.hide()
        self.mode.currentIndexChanged.connect(self.redraw); self.well.currentTextChanged.connect(self.redraw)
        self.redraw()

    def set_preview_lock(self,checked):
        self.image_ratio_locked=bool(checked);self.aspect_preview.lock_ratio=bool(checked)

    def choose_directory(self):
        path = QFileDialog.getExistingDirectory(self, "选择成果目录")
        if path:
            try: self.load_directory(path)
            except Exception as exc: QMessageBox.warning(self, "无法读取成果", str(exc))

    def load_directory(self, path):
        path = Path(path)
        history = pd.read_csv(path / "history.csv") if (path / "history.csv").is_file() else pd.DataFrame()
        folds = pd.read_csv(path / "folds.csv") if (path / "folds.csv").is_file() else pd.DataFrame()
        pressure_file = next((path / name for name in ("predictions.csv", "oof_predictions.csv", "validation_predictions.csv") if (path / name).is_file()), None)
        pressures = pd.read_csv(pressure_file, encoding="utf-8-sig") if pressure_file else pd.DataFrame()
        if history.empty and folds.empty and pressures.empty: raise ValueError("目录中没有可显示的训练或井预测成果")
        self.history, self.folds, self.pressures = history, folds, pressures
        self.reference_points = pd.read_csv(path / "reference_points.csv", encoding="utf-8-sig") if (path / "reference_points.csv").is_file() else None
        self.point_settings = {}; self.well_roles = {}
        self.source_file = pressure_file; self.directory.setText(str(path))
        if self.reference_points is not None: self.source_file = path / "reference_points.csv"
        self.well.blockSignals(True); self.well.clear(); self.well.addItem("全部井（汇总）", None)
        if not pressures.empty and "well" in pressures:
            names = pressures.well.dropna().astype(str).tolist()
            if self.reference_points is not None: names += self.reference_points.well.dropna().astype(str).tolist()
            for well in dict.fromkeys(names): self.well.addItem(well, well)
        self.well.blockSignals(False)
        self.source_label = "Well prediction · supplied reference labels"
        info = "井预测：与当前文件提供的参考标签配对，不自动等同于独立盲井评价。"
        if pressure_file and pressure_file.name == "validation_predictions.csv":
            self.source_label = "Internal validation"; info = "内部验证集：用于早停及选参，不能当作独立盲井精度。"
        elif pressure_file and pressure_file.name == "oof_predictions.csv":
            self.source_label = "Development wells · leave-one-well-out"; info = "开发井留一井验证结果，不含最终盲井。"
        for filename in ("manifest.json", "summary.json", "metrics.json"):
            if not (path / filename).is_file(): continue
            meta = json.loads((path / filename).read_text(encoding="utf-8"))
            if meta.get("workflow") == "sparse_well_validation":
                self.point_settings = meta.get("settings", {}); self.well_roles = meta.get("well_roles", {})
                self.source_label = "Measured pressure points · matched depths"
                info = f"实测点 {meta['measurement_rows']}，已匹配 {meta['matched_rows']}，未匹配 {meta['unmatched_rows']}；评价性质见逐井表。测点很少时指标仅作有限参考；没有有效实测点时只预测。"
            if filename == "manifest.json" and pressure_file and pressure_file.name == "validation_predictions.csv":
                strategy = meta.get("split_strategy", {})
                if strategy.get("method") == "random_points":
                    ratio = strategy.get("train_ratio", .8)
                    self.source_label = f"Internal validation · random {ratio:.0%} / {1-ratio:.0%} split"
            if meta.get("blind_only"):
                self.source_label = "Independent blind-well evaluation"; info = "独立盲井评价：只展示所选模型的保留盲井配对结果。"
            if meta.get("purpose") == "smoke_test":
                info = "软件流程试算：曲线与误差仅用于检查链路，不代表模型精度。"; self.source_label += " · SMOKE TEST"
            elif meta.get("purpose") == "final_refit": info = "进阶最终重训：当前仅显示训练误差，未重新产生内部验证点。"
            if meta.get("status", "completed") != "completed": info += " 此任务尚未完成，仅供诊断。"
        self.context.setText(info + " 压力单位 MPa；散点与指标使用完全相同的有效配对点。")
        self.mode.blockSignals(True); self.mode.setCurrentIndex(2 if not folds.empty else 0 if not history.empty else 1); self.mode.blockSignals(False)
        self.redraw()

    def live_history(self, records):
        if len(records) <= 1:
            self.pressures = pd.DataFrame(); self.folds = pd.DataFrame(); self.source_file = None
            self.reference_points = None; self.point_settings = {}; self.well_roles = {}
            self.directory.clear(); self.well.blockSignals(True); self.well.clear(); self.well.blockSignals(False)
        self.history = pd.DataFrame(records)
        self.context.setText("实时训练损失；内部验证集用于早停与选参，最终盲井不参与。" if "validation_data" in self.history else "固定轮数最终重训：不使用验证集或盲井，仅显示训练损失。")
        self.mode.blockSignals(True); self.mode.setCurrentIndex(0); self.mode.blockSignals(False); self.redraw()

    def selected_well(self): return self.well.currentData()

    def evaluation_frame(self):
        return self.reference_points if self.reference_points is not None else self.pressures

    def current_metrics(self):
        frame = self.evaluation_frame(); well = self.selected_well()
        if well is not None and "well" in frame: frame = frame.loc[frame.well.astype(str) == str(well)]
        result = evaluation_table(frame)
        if self.reference_points is not None:
            known = set(result.loc[result.scope == "well", "well"])
            from .evaluation import metrics
            for name in self.pressures.well.dropna().astype(str).unique():
                if name not in known and (well is None or str(well) == name): result.loc[len(result)] = {"scope": "well", "well": name, **metrics([], [])}
            result["evaluation_role"] = result.well.map(self.well_roles).map(ROLE_LABELS).fillna("汇总 / 见逐井独立性")
        if well is not None: result = result.loc[result.scope == "well"].copy()
        result.insert(0, "dataset", self.source_label)
        return result

    def populate_table(self):
        result = self.current_metrics()
        fields = [("well", "井 / 汇总"), ("count", "有效点数"), ("rmse", "RMSE (MPa)"), ("mae", "MAE (MPa)"),
                  ("mape", "MAPE (%)"), ("r2", "R²"), ("bias", "Bias (MPa)"),
                  ("mape_count", "MAPE 点数"), ("mape_excluded_zero_count", "MAPE 排除\n零值数")]
        if "evaluation_role" in result: fields.insert(1, ("evaluation_role", "评价性质"))
        self.table.setSortingEnabled(False); self.table.clear(); self.table.setRowCount(len(result)); self.table.setColumnCount(len(fields))
        self.table.setHorizontalHeaderLabels([label for _, label in fields])
        for i, (_, row) in enumerate(result.iterrows()):
            for j, (field, _) in enumerate(fields):
                value = row[field]; item = QTableWidgetItem()
                if field in ("well", "evaluation_role"): item.setText(str(value))
                elif pd.isna(value): item.setText("—")
                else: item.setData(Qt.ItemDataRole.DisplayRole, int(value) if "count" in field else round(float(value), 4))
                if j: item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip(MAPE_RULE if field.startswith("mape") else self.source_label)
                self.table.setItem(i, j, item)
        self.table.setSortingEnabled(True); self.table.resizeRowsToContents()

    def populate_matches(self):
        frame = self.reference_points
        if frame is None: frame = pd.DataFrame()
        if self.selected_well() is not None and "well" in frame: frame = frame.loc[frame.well.astype(str) == str(self.selected_well())]
        fields = [("well", "井"), ("source_row", "原表行号"), ("Depth", "测点深度 (m)"), ("Pp", "实测压力 (MPa)"),
                  ("Pp_pred", "对应预测 (MPa)"), ("error_MPa", "预测 - 实测 (MPa)"), ("match_reason", "匹配状态 / 原因")]
        self.table.setSortingEnabled(False); self.table.clear(); self.table.setRowCount(len(frame)); self.table.setColumnCount(len(fields))
        self.table.setHorizontalHeaderLabels([label for _, label in fields])
        for i, (_, row) in enumerate(frame.iterrows()):
            for j, (field, _) in enumerate(fields):
                value = row.get(field); item = QTableWidgetItem("—" if pd.isna(value) else str(round(value, 4)) if isinstance(value, float) else str(value))
                self.table.setItem(i, j, item)
        self.table.setSortingEnabled(True); self.table.resizeRowsToContents()

    def redraw(self, *_):
        mode = self.mode.currentIndex(); self.well.setEnabled(mode in (1, 3, 4, 5))
        if mode == 1 and self.selected_well() is None and self.well.count() > 1:
            self.well.blockSignals(True); self.well.setCurrentIndex(1); self.well.blockSignals(False)
        is_table = mode in (4, 5); self.views.setCurrentIndex(int(is_table)); self.note.setVisible(is_table)
        self.preview_controls.setVisible(not is_table)
        self.toolbar.setVisible(not is_table); self.style_button.setEnabled(not is_table); self.export_button.setEnabled(not is_table)
        self.table_export.setEnabled(not self.pressures.empty)
        self.table_export.setText("导出匹配明细" if mode == 5 else "导出指标表")
        if is_table:
            self.populate_matches() if mode == 5 else self.populate_table()
            return
        if mode == 0 and not self.history.empty: figure = training_figure(self.history, self.style)
        elif mode == 1 and not self.pressures.empty and self.selected_well() is not None and (self.pressures.well.astype(str) == str(self.selected_well())).any():
            figure = pressure_figure(self.pressures, self.selected_well(), self.style, observations=self.reference_points, max_gap_m=self.point_settings.get("max_gap_m", 0), source_label=ROLE_ENGLISH.get(self.well_roles.get(str(self.selected_well()))))
        elif mode == 2 and not self.folds.empty: figure = folds_figure(self.folds, self.style)
        elif mode == 3 and not self.pressures.empty:
            label = self.source_label + (" · " + ROLE_ENGLISH.get(self.well_roles.get(str(self.selected_well())), "") if self.selected_well() is not None and self.well_roles else "")
            figure = scatter_figure(self.evaluation_frame(), self.selected_well(), self.style, label)
        else:
            figure = Figure(figsize=(8, 5), layout="constrained")
            figure.subplots().text(.5, .5, "No data for this view", ha="center", va="center", transform=figure.axes[0].transAxes)
        self.canvas.figure.clear(); self.canvas.figure = figure; figure.set_canvas(self.canvas)
        self.aspect_preview.set_plot_size(self.style.width_cm,self.style.height_cm)
        self.toolbar.update(); self.canvas.draw_idle()

    def apply_style(self, style, persist=True):
        style.validate()
        if persist: style.save(self.settings_path)
        self.style = style; self.redraw()

    def create_style_dialog(self):
        dialog = QDialog(self); dialog.setWindowTitle("绘图设置 · 预览与导出共用");dialog.resize(640,600)
        layout = QVBoxLayout(dialog);tabs=QTabWidget();layout.addWidget(tabs)
        typography=TypographyControls(self.style);tabs.addTab(typography,'字体与标题')
        dimensions=QWidget();form=QFormLayout(dimensions);tabs.addTab(dimensions,'图片尺寸与预览')
        inputs = {}
        for key, label, low, high, integer in (("width_cm", "图宽 (cm)", 5, 50, False),
                ("height_cm", "图高 (cm)", 5, 50, False), ("dpi", "导出分辨率 (DPI)", 72, 600, True),
                ("point_size", "散点大小 (pt²)", 1, 100, False), ("alpha", "散点不透明度", .05, 1, False)):
            edit = QSpinBox() if integer else QDoubleSpinBox(); edit.setRange(low, high)
            if not integer: edit.setDecimals(2)
            edit.setObjectName('chart_'+key);inputs[key] = edit; form.addRow(label, edit);edit.setValue(getattr(self.style,key))
        ratio,lock=link_dimensions(form,inputs['width_cm'],inputs['height_cm'],inputs['dpi'],prefix='chart',locked=self.image_ratio_locked)
        preview_mode=QLabel('完整显示，可拖动图框调整预览');preview_mode.setObjectName('chart_preview_mode');form.addRow('屏幕预览',preview_mode)
        def fill(style):
            lock.setChecked(False);typography.fill(style)
            for key, edit in inputs.items(): edit.setValue(getattr(style, key))
        explanation = QLabel('拖动图框边缘或四角可调整预览大小及纵横比，图像始终完整显示在绘图区内。“适应窗口”保留当前比例，“恢复设置比例”恢复本页指定的宽高比；导出仍使用指定图宽、图高及 DPI。SVG/PDF 支持矢量文字与线条，密集散点按所选 DPI 栅格化。')
        explanation.setWordWrap(True); layout.addWidget(explanation)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText('应用')
        buttons.button(QDialogButtonBox.StandardButton.Close).setText('关闭')
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText('恢复默认设置')
        def apply():
            try:
                style=FigureStyle(**typography.collect(),**{key:edit.value() for key,edit in inputs.items()}).validate()
                style.save(self.settings_path)
                self.preview_mode='fit';self.image_ratio_locked=lock.isChecked()
                self.preview_controls.lock.setChecked(self.image_ratio_locked);self.set_preview_lock(self.image_ratio_locked)
                self.aspect_preview.set_preview_mode(self.preview_mode);self.apply_style(style,persist=False)
            except Exception as exc: QMessageBox.warning(dialog, "绘图设置需要修改", str(exc))
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(apply)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(lambda: fill(FigureStyle()))
        buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        polish_layouts(dialog)
        return bound_plot_dialog(dialog)

    def show_plot_settings(self): self.create_style_dialog().exec()

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出当前图表", "pressure_chart.png", "PNG 图片 (*.png);;SVG 矢量图 (*.svg);;PDF 图件 (*.pdf)")
        if path:
            try: export_figure(self.canvas.figure, path, self.style)
            except Exception as exc: QMessageBox.warning(self, "导出失败", str(exc))

    def save_metrics(self, path):
        result = self.current_metrics(); result.to_csv(path, index=False, encoding="utf-8-sig")
        notes = {"dataset": self.source_label, "source": str(self.source_file) if self.source_file else None,
                 "well_roles": self.well_roles, "matching_settings": self.point_settings,
                 "mape_definition": MAPE_RULE, "r2_definition": "1 - sum((prediction-reference)^2)/sum((reference-mean(reference))^2); undefined if reference is constant or fewer than two pairs",
                 "overall": "Pooled finite valid pairs, weighted by point count; not a macro average of wells", "units": {"rmse": "MPa", "mae": "MPa", "bias": "MPa", "mape": "%"}}
        Path(path).with_suffix(".notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    def export_metrics(self):
        if self.mode.currentIndex() == 5:
            path, _ = QFileDialog.getSaveFileName(self, "导出实测点匹配明细", "reference_points.csv", "CSV 表格 (*.csv)")
            if path and self.reference_points is not None:
                try:
                    frame = self.reference_points
                    if self.selected_well() is not None: frame = frame.loc[frame.well.astype(str) == str(self.selected_well())]
                    frame.to_csv(path, index=False, encoding="utf-8-sig")
                except Exception as exc: QMessageBox.warning(self, "导出失败", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出当前范围的评价指标", "evaluation_metrics.csv", "CSV 表格 (*.csv)")
        if path:
            try: self.save_metrics(path)
            except Exception as exc: QMessageBox.warning(self, "导出失败", str(exc))

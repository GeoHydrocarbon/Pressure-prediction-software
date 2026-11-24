"""Central, exportable plot documents for normal compaction and Eaton results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
                              QCheckBox, QFileDialog, QMessageBox, QStackedWidget,
                              QTableWidget, QTableWidgetItem, QSizePolicy, QAbstractItemView)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from .charts import FigureStyle, export_figure, scatter_figure
from .traditional_figures import trend_figure, eaton_figure
from .data import sha256_file
from .flow_layout import FlowLayout


class TraditionalResultView(QWidget):
    pointsSelected = Signal(object, dict)

    def __init__(self, path):
        super().__init__()
        self.path = Path(path).resolve()
        self.report = json.loads((self.path / 'traditional.json').read_text(encoding='utf-8'))
        if self.report.get('status') != 'completed' or self.report.get('schema_version') != 1:
            raise ValueError('仅能打开已完成的传统压力预测成果')
        self.report['run_path']=str(self.path)
        self.is_trend = self.report['artifact'] == 'traditional_trend'
        name = 'trend_data.csv' if self.is_trend else 'predictions.csv'
        if sha256_file(self.path / name) != self.report['data_sha256']:
            raise ValueError('传统方法成果数据校验失败')
        self.frame = pd.read_csv(self.path / name, encoding='utf-8-sig', dtype={'well':'string'})
        self.observations = None
        if not self.is_trend:
            if sha256_file(self.path / 'reference_points.csv') != self.report['reference_sha256']:
                raise ValueError('实测压力匹配结果校验失败')
            self.observations = pd.read_csv(self.path / 'reference_points.csv', encoding='utf-8-sig', dtype={'well':'string'})
        self.trend = self.report.get('trend')
        self.style = FigureStyle(width_cm=28., height_cm=20.)
        self.pending_points = []
        layout = QVBoxLayout(self)
        row = FlowLayout()
        self.well = QComboBox()
        self.well.addItems(self.frame.well.dropna().unique().astype(str).tolist())
        row.addWidget(QLabel('井'))
        row.addWidget(self.well)
        self.mode = QComboBox()
        self.mode.addItems(['正常压实趋势'] if self.is_trend else ['压力与压力系数', '预测 vs 参考压力', '评价指标'])
        row.addWidget(self.mode)
        self.pick = QCheckBox('在图上取两点')
        self.pick.setVisible(self.is_trend)
        self.pick.setToolTip('横坐标为深度（m），纵坐标为 LN(AC [us/ft])。取完两点后仍需在模块内明确采用。')
        row.addWidget(self.pick)
        for title, callback in [('绘图设置…', self.plot_settings), ('导出图件…', self.export_plot),
                                ('导出数据…', self.export_data), ('成果目录', self.open_folder)]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.canvas = FigureCanvasQTAgg(Figure(layout='constrained'))
        self.canvas.setMinimumSize(0,0)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        self.toolbar = NavigationToolbar2QT(self.canvas,self)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        layout.addWidget(self.toolbar)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.canvas)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stack.addWidget(self.table)
        layout.addWidget(self.stack,1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.well.currentTextChanged.connect(self.redraw)
        self.mode.currentIndexChanged.connect(self.redraw)
        self.pick.toggled.connect(self.pick_toggled)
        self.canvas.mpl_connect('button_press_event',self.pick_point)
        self.redraw()

    def pick_toggled(self,on):
        self.pending_points=[]
        if on and self.toolbar.mode:
            if str(self.toolbar.mode)=='pan/zoom':self.toolbar.pan()
            else:self.toolbar.zoom()
        if on:self.note.setText('依次单击两个点定义趋势；然后在“传统压力预测”模块内采用此趋势。')

    def pick_point(self,event):
        if not self.pick.isChecked() or not self.is_trend or event.button!=1 or not event.inaxes or self.toolbar.mode:
            return
        if event.xdata is None or event.ydata is None:return
        self.pending_points.append([float(event.xdata),float(event.ydata)])
        if len(self.pending_points)==2:
            self.pointsSelected.emit(self.pending_points[:],self.report)
            self.pick.setChecked(False)
        else:self.note.setText('已选第一个点，请选择第二个点。')

    def update_trend(self,trend):
        if self.is_trend:
            self.trend=trend
            self.redraw()

    def selected(self):
        name=self.well.currentText()
        frame=self.frame.loc[self.frame.well==name]
        points=self.observations.loc[self.observations.well==name] if self.observations is not None else None
        return name,frame,points

    def redraw(self,*_):
        name,frame,points=self.selected()
        is_table=not self.is_trend and self.mode.currentIndex()==2
        self.stack.setCurrentIndex(int(is_table))
        self.toolbar.setVisible(not is_table)
        if is_table:
            from .evaluation import evaluation_table
            table=evaluation_table(points)
            labels={'well':'井 / 汇总','count':'有效点数','mae':'MAE (MPa)','rmse':'RMSE (MPa)','bias':'Bias (MPa)','r2':'R²','mape':'MAPE (%)'}
            fields=['well','count','rmse','mae','mape','r2','bias']
            self.table.setRowCount(len(table));self.table.setColumnCount(len(fields))
            self.table.setHorizontalHeaderLabels([labels[key] for key in fields])
            for row,(_,record) in enumerate(table.iterrows()):
                for col,key in enumerate(fields):
                    value=record[key];text='—' if pd.isna(value) else f'{value:.5g}' if isinstance(value,(float,np.floating)) else str(value)
                    self.table.setItem(row,col,QTableWidgetItem(text))
            self.table.resizeColumnsToContents()
            self.note.setText('仅使用成功匹配的参考压力点。参数可能已用这些井标定，此处不自动认定为独立验证。')
            return
        if self.is_trend:
            figure=trend_figure(frame,self.trend,name,self.style)
            self.note.setText(f"LN(AC) = {self.trend['a']:.9g} × 深度 + {self.trend['b']:.9g}；两点标定，需在模块内采用后才用于预测。")
        elif self.mode.currentIndex()==1:
            figure=scatter_figure(points,style=self.style,source_label='Eaton · supplied pressure references; calibration may use these wells')
            self.note.setText('仅对成功匹配的参考压力点计算指标；没有参考压力时不生成精度结论。')
        else:
            figure=eaton_figure(frame,points,name,self.style)
            flagged=int(frame.quality_flags.fillna('').ne('').sum())
            self.note.setText(f"有效预测 {int(frame.prediction_valid.sum()):,} 点；标记 {flagged:,} 点。原始计算结果未按压力系数范围裁剪；压力系数 = 预测压力 / 静水压力。")
        old=self.canvas.figure;old.clear();self.canvas.figure=figure;figure.set_canvas(self.canvas)
        self.toolbar.update();self.canvas.draw_idle()

    def plot_settings(self):
        from .plot_settings import edit_figure_style
        style=edit_figure_style(self,self.style)
        if style is not None:self.style=style;self.redraw()

    def export_plot(self):
        if self.stack.currentIndex()==1:return
        filename,_=QFileDialog.getSaveFileName(self,'导出当前图件','traditional_pressure.png','PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if filename:
            try:export_figure(self.canvas.figure,filename,self.style)
            except Exception as exc:QMessageBox.warning(self,'导出失败',str(exc))

    def export_data(self):
        filename,_=QFileDialog.getSaveFileName(self,'导出当前井完整数据','traditional_pressure.csv','CSV (*.csv)')
        if filename:
            try:
                if self.stack.currentIndex()==1:
                    from .evaluation import evaluation_table
                    data=evaluation_table(self.selected()[2])
                else:data=self.selected()[1]
                data.to_csv(filename,index=False,encoding='utf-8-sig')
            except Exception as exc:QMessageBox.warning(self,'导出失败',str(exc))

    def open_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path)))

"""Comparison of two predictions, deliberately labelled as differences."""
import numpy as np
from PySide6.QtWidgets import QDialog,QVBoxLayout,QLabel,QPushButton,QFileDialog,QTableWidget,QTableWidgetItem,QHBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from .charts import finish_style
from .ui_style import polish_layouts


class WellCompareDialog(QDialog):
    def __init__(self,parent,mapping,well):
        super().__init__(parent); self.setWindowTitle('井压力与地震压力差异 — '+well); self.resize(1000,740)
        self.frame=mapping.table(well); layout=QVBoxLayout(self)
        valid=self.frame[['well_pressure_MPa','seismic_pressure_MPa']].notna().all(axis=1)
        paired=self.frame.loc[valid]; delta=paired.seismic_minus_well_MPa.to_numpy()
        text=f'井：{well}；配对点 {len(delta)} / {len(self.frame)}。'
        if len(delta):text+=f' 平均差 {delta.mean():.3f} MPa；平均绝对差 {np.abs(delta).mean():.3f} MPa；均方根差 {np.sqrt(np.mean(delta**2)):.3f} MPa。'
        label=QLabel(text+'\n差值＝地震压力－井压力。这是两种结果之间的差异；只有井压力来自实测时，才具有实测验证含义。'); label.setWordWrap(True); layout.addWidget(label)
        figure=Figure(figsize=(9,4),layout='constrained'); ax,dx=figure.subplots(1,2,sharey=True)
        ax.plot(self.frame.well_pressure_MPa,self.frame.depth_m,color='#d92344',label='Well pressure')
        ax.plot(self.frame.seismic_pressure_MPa,self.frame.depth_m,color='#008baa',label='Seismic at well')
        ax.set(xlabel='Pressure (MPa)',ylabel='Depth (m)'); ax.invert_yaxis(); ax.legend()
        dx.plot(self.frame.seismic_minus_well_MPa,self.frame.depth_m,color='#6955aa'); dx.axvline(0,color='#999999',linewidth=.7); dx.set_xlabel('Seismic - well (MPa)')
        finish_style(figure); self.canvas=FigureCanvasQTAgg(figure); layout.addWidget(NavigationToolbar2QT(self.canvas,self)); layout.addWidget(self.canvas,1)
        table=QTableWidget(min(1000,len(self.frame)),len(self.frame.columns)); table.setHorizontalHeaderLabels(['井名','深度 (m)','时间 (ms)','X','Y','地震道索引','井压力 (MPa)','地震压力 (MPa)','地震－井 (MPa)'])
        for r,row in enumerate(self.frame.head(1000).itertuples(index=False,name=None)):
            for c,v in enumerate(row):table.setItem(r,c,QTableWidgetItem(f'{v:.4f}' if isinstance(v,float) else str(v)))
        table.resizeColumnsToContents(); table.setMaximumHeight(180); layout.addWidget(table)
        layout.addWidget(QLabel('表格预览前 1000 行；CSV 导出包含全部点及未配对空值。'))
        row=QHBoxLayout(); layout.addLayout(row)
        for label,callback in [('导出全部对比 CSV',self.export),('导出对比图',self.export_plot)]:
            b=QPushButton(label); b.clicked.connect(callback); row.addWidget(b)
        polish_layouts(self)

    def export(self):
        path,_=QFileDialog.getSaveFileName(self,'导出对比数据','well_seismic_comparison.csv','CSV (*.csv)')
        if path:self.frame.to_csv(path,index=False,encoding='utf-8-sig')

    def export_plot(self):
        path,_=QFileDialog.getSaveFileName(self,'导出对比图','well_seismic_comparison.png','PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if path:self.canvas.figure.savefig(path,dpi=300)

"""Central MDI result document for scalar, vector and matrix rock models."""
import csv
import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton,
    QTableWidget,QTableWidgetItem,QTabWidget,QDoubleSpinBox,QLineEdit,QFileDialog,QMessageBox,
    QAbstractItemView,QSizePolicy)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib.colors import is_color_like
from matplotlib.figure import Figure

from .charts import FigureStyle,apply_typography,export_figure
from .flow_layout import FlowLayout
from .plot_settings import edit_figure_style


class RockPhysicsResultView(QWidget):
    @staticmethod
    def display_name(name):
        return name[6:]+'（输入）' if isinstance(name,str) and name.startswith('input.') else name

    def __init__(self,path,result=None):
        super().__init__();self.path=Path(path).resolve();self.style=FigureStyle();self.figure=None
        if result is None:
            payload=json.loads((self.path/'result.json').read_text(encoding='utf-8'))
            result=payload.get('result',payload)
        if result.get('status','completed')!='completed':raise ValueError('岩石物理成果尚未完成')
        self.result=result;self.setMinimumSize(0,0)
        layout=QVBoxLayout(self);self.title=QLabel(result.get('title',result.get('model_id','岩石物理结果')));self.title.setWordWrap(True);layout.addWidget(self.title)
        row=FlowLayout();self.mode=QComboBox()
        for title,value in [('曲线','line'),('散点','scatter'),('矩阵热图','matrix')]:self.mode.addItem(title,value)
        self.x=QComboBox();self.y=QComboBox();self.matrix=QComboBox()
        for label,widget in [('图形',self.mode),('横轴',self.x),('纵轴',self.y),('矩阵',self.matrix)]:row.addWidget(QLabel(label));row.addWidget(widget)
        layout.addLayout(row)
        row=FlowLayout();self.color=QLineEdit('#236889');self.color.setMaximumWidth(100);self.width=QDoubleSpinBox();self.width.setRange(.1,12);self.width.setValue(1.6)
        self.point_size=QDoubleSpinBox();self.point_size.setRange(1,30);self.point_size.setValue(4)
        for label,widget in [('颜色',self.color),('线宽',self.width),('点大小（pt）',self.point_size)]:row.addWidget(QLabel(label));row.addWidget(widget)
        settings=QPushButton('绘图设置…');settings.clicked.connect(self.plot_settings);row.addWidget(settings)
        export=QPushButton('导出图件…');export.clicked.connect(self.choose_export_figure);row.addWidget(export)
        data_export=QPushButton('导出数据…');data_export.clicked.connect(self.choose_export_data);row.addWidget(data_export);layout.addLayout(row)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        plot=QWidget();plot_layout=QVBoxLayout(plot);plot_layout.setContentsMargins(0,0,0,0)
        self.canvas=FigureCanvasQTAgg(Figure(layout='constrained'));self.canvas.setMinimumSize(0,0);self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        self.toolbar=NavigationToolbar2QT(self.canvas,self);self.toolbar.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        plot_layout.addWidget(self.toolbar);plot_layout.addWidget(self.canvas,1);self.tabs.addTab(plot,'图形')
        self.table=QTableWidget();self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);self.tabs.addTab(self.table,'结果表')
        self.parameters_table=QTableWidget(0,2);self.parameters_table.setHorizontalHeaderLabels(['输入参数','数值']);self.parameters_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);self.tabs.addTab(self.parameters_table,'本次参数')
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.columns=[];self.rows=[];self.numeric={};self.matrices={};self.populate()
        for combo in (self.mode,self.x,self.y,self.matrix):combo.currentIndexChanged.connect(self.redraw)
        self.color.editingFinished.connect(self.redraw);self.width.valueChanged.connect(self.redraw);self.point_size.valueChanged.connect(self.redraw)
        self.redraw()

    def populate(self):
        table=self.result.get('table',{});self.columns=list(table.get('columns',[]));self.rows=list(table.get('rows',[]))
        if not self.columns:
            # Older or externally supplied results may have only scalar outputs.
            for output in self.result.get('outputs',[]):
                values=np.asarray(output.get('values'))
                if values.ndim==0:self.columns.append(output['name'])
            if self.columns:self.rows=[[next(o['values'] for o in self.result['outputs'] if o['name']==name) for name in self.columns]]
        if len(self.columns)>256 or len(self.rows)>100000:raise ValueError('成果表过大，不能直接显示')
        self.table.setColumnCount(len(self.columns));self.table.setHorizontalHeaderLabels([self.display_name(name) for name in self.columns]);self.table.setRowCount(min(20000,len(self.rows)))
        for row,values in enumerate(self.rows[:20000]):
            for column,value in enumerate(values[:len(self.columns)]):
                item=QTableWidgetItem('' if value is None else f'{value:.9g}' if isinstance(value,(float,int)) else str(value));self.table.setItem(row,column,item)
        self.x.addItem('样点序号',None)
        for index,name in enumerate(self.columns):
            try:values=np.array([row[index] if index<len(row) else None for row in self.rows],dtype=float)
            except (TypeError,ValueError):continue
            self.numeric[name]=values;self.x.addItem(self.display_name(name),name);self.y.addItem(self.display_name(name),name)
        inputs=[name for name in self.numeric if name.startswith('input.')]
        if inputs:self.x.setCurrentIndex(self.x.findData(inputs[0]))
        elif len(self.rows)>1 and self.numeric:
            # An explicit index is preferable to guessing that an output is an input axis.
            self.x.setCurrentIndex(0)
        output_names=[output['name'] for output in self.result.get('outputs',[])]
        outputs=[name for output in output_names for name in self.numeric if name==output or name.startswith(output+' [')]
        if not outputs:outputs=[name for name in self.numeric if name not in inputs and name not in ('index','i1','i2','row','column')]
        if outputs:self.y.setCurrentIndex(self.y.findData(outputs[0]))
        for output in self.result.get('outputs',[]):
            try:values=np.asarray(output.get('values'),dtype=float)
            except (ValueError,TypeError):continue
            if values.ndim==2:
                self.matrices[output['name']]=(values,output.get('unit',''));self.matrix.addItem(output['name'],output['name'])
        self.mode.model().item(2).setEnabled(bool(self.matrices))
        vectors=any(np.asarray(output.get('values')).ndim==1 and np.asarray(output.get('values')).size>1 for output in self.result.get('outputs',[]))
        if self.matrices and not vectors:self.mode.setCurrentIndex(2)
        for row,(name,value) in enumerate(self.result.get('parameters',{}).items()):
            self.parameters_table.insertRow(row);self.parameters_table.setItem(row,0,QTableWidgetItem(name))
            self.parameters_table.setItem(row,1,QTableWidgetItem(json.dumps(value,ensure_ascii=False)))
        self.parameters_table.resizeColumnToContents(0)

    def redraw(self,*_):
        color=self.color.text().strip()
        if not is_color_like(color):self.status.setText('颜色无效，请输入颜色名称或 #RRGGBB');return
        matrix_mode=self.mode.currentData()=='matrix';self.x.setEnabled(not matrix_mode);self.y.setEnabled(not matrix_mode);self.matrix.setEnabled(matrix_mode)
        self.color.setEnabled(not matrix_mode);self.width.setEnabled(not matrix_mode);self.point_size.setEnabled(not matrix_mode)
        figure=Figure(figsize=(self.style.width_cm/2.54,self.style.height_cm/2.54),layout='constrained');ax=figure.subplots();ax.set_title(self.result.get('title',self.result.get('model_id','岩石物理结果')))
        if matrix_mode and self.matrix.currentData() in self.matrices:
            values,unit=self.matrices[self.matrix.currentData()];image=ax.imshow(np.ma.masked_invalid(values),cmap='viridis',aspect='auto')
            ax.set(xlabel='列号（从 1 开始）',ylabel='行号（从 1 开始）');ax.set_xticks(np.arange(values.shape[1]),labels=np.arange(1,values.shape[1]+1));ax.set_yticks(np.arange(values.shape[0]),labels=np.arange(1,values.shape[0]+1));figure.colorbar(image,ax=ax,label=unit)
        elif len(self.rows)<=1:
            ax.set_axis_off();summary=[]
            for output in self.result.get('outputs',[]):
                array=np.asarray(output.get('values'))
                if array.size==1:
                    value=array.reshape(-1)[0].item();label=f'{value:.9g}' if isinstance(value,(float,int)) else str(value)
                    summary.append(output['name']+' = '+label+(' '+output['unit'] if output.get('unit') else ''))
            ax.text(.04,.9,'\n'.join(summary[:16]) or '请在结果表中查看数值或选择矩阵热图。',va='top',transform=ax.transAxes,fontsize=12)
        else:
            name=self.y.currentData();values=self.numeric.get(name);xname=self.x.currentData()
            xx=self.numeric.get(xname) if xname else np.arange(1,len(self.rows)+1,dtype=float)
            if values is not None and xx is not None and len(values)==len(xx):
                valid=np.isfinite(xx)&np.isfinite(values)
                if self.mode.currentData()=='scatter':ax.scatter(xx[valid],values[valid],c=color,s=self.point_size.value()**2,alpha=.8)
                else:ax.plot(np.where(valid,xx,np.nan),np.where(valid,values,np.nan),color=color,linewidth=self.width.value())
                ax.set(xlabel=self.display_name(xname) if xname else '样点序号（从 1 开始）',ylabel=self.display_name(name))
            else:ax.text(.5,.5,'请选择可绘制的横轴和纵轴',ha='center',transform=ax.transAxes)
            ax.grid(self.style.grid,alpha=.2)
        apply_typography(figure,self.style,title_axes=[ax]);old=self.canvas.figure;old.clear();self.canvas.figure=figure;figure.set_canvas(self.canvas);self.figure=figure;self.toolbar.update();self.canvas.draw_idle()
        warnings=self.result.get('warnings',[])
        summary=f'{self.result.get("model_id", "")} · {len(self.rows):,} 条结果；表格预览最多 20,000 行，导出保留完整数据。'
        self.status.setText(summary+('\n'+'\n'.join(map(str,warnings)) if warnings else ''))

    def plot_settings(self):
        style=edit_figure_style(self,self.style)
        if style is not None:self.style=style;self.redraw()

    def export_data(self,path):
        path=Path(path)
        if path.suffix.lower()=='.json':path.write_text(json.dumps(self.result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        else:
            with path.open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(self.columns);writer.writerows(self.rows)

    def choose_export_data(self):
        path,_=QFileDialog.getSaveFileName(self,'导出岩石物理数据',str(self.path/'rock_physics.csv'),'CSV (*.csv);;JSON (*.json)')
        if not path:return
        try:self.export_data(path)
        except Exception as exc:QMessageBox.warning(self,'导出失败',str(exc))

    def export_plot(self,path):
        if self.figure is not None:export_figure(self.figure,path,self.style)

    def choose_export_figure(self):
        path,_=QFileDialog.getSaveFileName(self,'导出岩石物理图件',str(self.path/'rock_physics.png'),'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if not path:return
        try:self.export_plot(path)
        except Exception as exc:QMessageBox.warning(self,'导出失败',str(exc))

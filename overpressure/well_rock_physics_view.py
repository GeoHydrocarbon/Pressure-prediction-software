"""Well-log comparison document, sharing the existing mixed-track viewer."""
from copy import deepcopy
import json
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QTabWidget,QComboBox,
    QPlainTextEdit,QTableWidget,QTableWidgetItem,QDialog,QDialogButtonBox,QFileDialog,QMessageBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib.figure import Figure

from .well_curve_viewer import CurveViewer
from .rock_physics_view import RockPhysicsResultView
from .rock_physics.well_model import OUTPUT_UNITS
from .charts import FigureStyle,export_figure,apply_typography
from .flow_layout import FlowLayout


class WellRockPhysicsResultView(QWidget):
    def __init__(self,path,result=None,owner=None):
        super().__init__();self.path=Path(path).resolve();self.owner=owner
        self.result=result or json.loads((self.path/'result.json').read_text(encoding='utf-8'))
        self.style=FigureStyle();self.frame=self.result['table'];self.columns=self.frame['columns'];self.rows=self.frame['rows'];self.numeric={}
        for i,name in enumerate(self.columns):
            try:self.numeric[name]=np.array([r[i] for r in self.rows],float)
            except (ValueError,TypeError):pass
        layout=QVBoxLayout(self);layout.addWidget(QLabel(self.result['title']))
        summary=self.result['summary'];layout.addWidget(QLabel(f"建模 {summary['modeled_rows']} / {summary['rows']} 个样本；无效 {summary['invalid_rows']} 个。原始曲线保留。"))
        row=FlowLayout()
        for title,slot in [('重新编辑 / 使用标定参数',self.edit_scheme),('保存成果曲线到井',self.choose_attach),('恢复默认对比图',self.reset_tracks),('保存当前图件设置',self.save_plot_state),('导出当前图件…',self.export_plot)]:
            button=QPushButton(title);button.clicked.connect(slot);row.addWidget(button)
        layout.addLayout(row);self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        self.viewer=CurveViewer(owner);self.tabs.addTab(self.viewer,'测井综合对比');self.make_tracks()
        self.cross=QWidget();box=QVBoxLayout(self.cross);row=QHBoxLayout();self.x=QComboBox();self.y=QComboBox();self.color=QComboBox();self.color.addItem('不着色','')
        for name in self.numeric:self.x.addItem(name);self.y.addItem(name);self.color.addItem(name,name)
        for label,widget in [('横轴',self.x),('纵轴',self.y),('着色',self.color)]:row.addWidget(QLabel(label));row.addWidget(widget)
        box.addLayout(row);self.cross_canvas=FigureCanvasQTAgg(Figure(layout='constrained'));box.addWidget(NavigationToolbar2QT(self.cross_canvas,self));box.addWidget(self.cross_canvas,1)
        self.tabs.addTab(self.cross,'交会图');self.x.setCurrentText('vp');self.y.setCurrentText('VP_RP');self.color.setCurrentIndex(self.color.findData('Depth'))
        for combo in (self.x,self.y,self.color):combo.currentIndexChanged.connect(self.draw_cross)
        self.draw_cross()
        self.scan_page=QWidget();box=QVBoxLayout(self.scan_page);self.scan_canvas=FigureCanvasQTAgg(Figure(layout='constrained'));box.addWidget(NavigationToolbar2QT(self.scan_canvas,self));box.addWidget(self.scan_canvas,1)
        self.scan_table=QTableWidget();box.addWidget(self.scan_table);self.tabs.addTab(self.scan_page,'敏感性 / 标定');self.draw_scan()
        self.data=RockPhysicsResultView(self.path,self.result);self.tabs.addTab(self.data,'结果表与数据导出')
        report=QPlainTextEdit();report.setReadOnly(True)
        labels=dict(all='全建模区间（含标定样本）',before='初始模型，全区间',calibration='实际标定样本',validation='独立验证区间')
        lines=[]
        for group,values in self.result['metrics'].items():
            lines.append(labels[group])
            if not values:lines.append('  无有效实测参考样本')
            for key,m in values.items():
                unit='g/cm³' if key=='rho' else 'm/s';r2='—' if m['r2'] is None else f"{m['r2']:.5g}"
                lines.append(f"  {key}: N={m['count']}, RMSE={m['rmse']:.6g} {unit}, Bias={m['bias']:.6g} {unit}, R²={r2}")
        lines.extend(['','标定信息',json.dumps({k:v for k,v in self.result['calibration'].items() if k not in ('scan','sample_depths')},ensure_ascii=False,indent=2),'','处理记录与适用条件',*self.result['warnings'],'','方案与来源',json.dumps(self.result['parameters'],ensure_ascii=False,indent=2)])
        report.setPlainText('\n'.join(lines));self.tabs.addTab(report,'误差统计与记录')
        self.status=QLabel('');self.status.setWordWrap(True);layout.addWidget(self.status);self.restore_plot_state()

    def make_tracks(self):
        datasets=[];tracks={};colors={};track=0;depth=self.numeric['Depth'];source=str(self.path/'results.csv')
        def add(column,title,unit,ident,color,linestyle='-'):
            values=self.numeric.get(column)
            if values is None or not np.isfinite(values).any():return
            datasets.append(dict(x=values,z=depth,unit=unit,well=self.result['well'],name=title,source=source,column=column,points=False))
            tracks[title]=ident;colors[title]=(color,linestyle)
        for column,title in [('phi','孔隙度'),('sw','含水饱和度')]:
            track+=1;add(column,title,'fraction',track,'#238b65')
        for obs,model,label,unit in [('vp','VP_RP','Vp','m/s'),('vs','VS_RP','Vs','m/s'),('rho','DEN_RP','密度','g/cm3')]:
            track+=1;add(obs,'实测 '+label,unit,track,'#263746');add(model,'模型 '+label,unit,track,'#cf513c')
        for column,label,unit in [('vp_error','Vp 误差','m/s'),('vs_error','Vs 误差','m/s'),('rho_error','密度误差','g/cm3')]:
            track+=1;add(column,label,unit,track,'#7d56a2')
        self.viewer.render(datasets)
        rows=self.viewer.setting_rows()
        for row in rows:
            row['track']=tracks[row['name']];row['color']=colors[row['name']][0];row['linestyle']=colors[row['name']][1]
        self.viewer.apply_plot_settings(rows);self.viewer.caption.setText(self.result['title']+' · 实测与模型同道；可自由调整分道、线点样式和图例。')

    def reset_tracks(self):
        self.viewer.datasets=[];self.viewer.curve_styles={};self.viewer.legend_settings={};self.make_tracks()

    def draw_cross(self):
        figure=self.cross_canvas.figure;figure.clear();ax=figure.add_subplot(111)
        x=self.numeric.get(self.x.currentText());y=self.numeric.get(self.y.currentText())
        if x is None or y is None:return
        valid=np.isfinite(x)&np.isfinite(y);color=self.numeric.get(self.color.currentData())
        if color is not None:
            valid&=np.isfinite(color);scatter=ax.scatter(x[valid],y[valid],c=color[valid],s=12,alpha=.7);figure.colorbar(scatter,ax=ax,label=self.color.currentData())
        else:ax.scatter(x[valid],y[valid],s=12,alpha=.7)
        if (self.x.currentText(),self.y.currentText()) in [('vp','VP_RP'),('vs','VS_RP'),('rho','DEN_RP')] and valid.any():
            lo=min(x[valid].min(),y[valid].min());hi=max(x[valid].max(),y[valid].max());ax.plot([lo,hi],[lo,hi],'--',c='gray',label='1:1');ax.legend()
        ax.set(xlabel=self.x.currentText(),ylabel=self.y.currentText());ax.grid(alpha=.2)
        apply_typography(figure,self.style);self.cross_canvas.draw_idle()

    def draw_scan(self):
        c=self.result['calibration'];history=c.get('scan',[]);figure=self.scan_canvas.figure;figure.clear();ax=figure.add_subplot(111)
        if not history:
            ax.text(.5,.5,'本次为正演，未执行参数标定',ha='center',transform=ax.transAxes)
            apply_typography(figure,self.style);self.scan_canvas.draw_idle();return
        keys=list(history[0]['parameters']);columns=keys+['score'];self.scan_table.setColumnCount(len(columns));self.scan_table.setHorizontalHeaderLabels(columns);self.scan_table.setRowCount(len(history));self.scan_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row,h in enumerate(history):
            for col,value in enumerate([h['parameters'][k] for k in keys]+[h['score']]):self.scan_table.setItem(row,col,QTableWidgetItem(f'{value:.8g}'))
        if c['mode']=='scan' and len(keys)==2:
            points=ax.scatter([h['parameters'][keys[0]] for h in history],[h['parameters'][keys[1]] for h in history],c=[h['score'] for h in history],marker='s',s=150)
            ax.set(xlabel=keys[0],ylabel=keys[1]);figure.colorbar(points,ax=ax,label='Relative error score')
        elif c['mode']=='scan':
            ax.plot([h['parameters'][keys[0]] for h in history],[h['score'] for h in history],'-o');ax.set(xlabel=keys[0],ylabel='Relative error score')
        else:ax.plot(np.arange(1,len(history)+1),[h['score'] for h in history]);ax.set(xlabel='Evaluation',ylabel='Relative error score')
        ax.grid(alpha=.2);apply_typography(figure,self.style);self.scan_canvas.draw_idle()

    def edit_scheme(self):
        if self.owner is None:return
        panel=self.owner.rock_physics
        if panel.process is not None:self.status.setText('请等待当前岩石物理任务完成或取消后再编辑。');return
        panel.workflow.set_scheme(deepcopy(self.result['parameters']['adopted_scheme']));panel.workflow_tabs.setCurrentIndex(0);self.owner.shell.show_page(8)

    def attach(self,selection):
        if self.owner is None:raise ValueError('请从项目窗口保存成果曲线')
        from .rock_physics.well_results import attach_curves
        before=deepcopy(self.owner.library.wells)
        after=attach_curves(before,self.path,self.result,selection)
        self.owner.library.wells=after
        try:self.owner.shell.persist_workspace()
        except Exception:
            self.owner.library.wells=before;raise
        self.owner.library.refresh();self.status.setText(f'已保存 {len(selection)} 条成果曲线到井 {self.result["well"]}。')

    def choose_attach(self):
        dialog=QDialog(self);dialog.setWindowTitle('选择并命名成果曲线');box=QVBoxLayout(dialog);table=QTableWidget(len(OUTPUT_UNITS),2);table.setHorizontalHeaderLabels(['保存曲线','井库名称（不能覆盖已有曲线）'])
        suffix=re.sub(r'[^\w\u4e00-\u9fff-]+','_',self.result['scheme_name'])
        existing={c['name'] for w in getattr(getattr(self.owner,'library',None),'wells',[]) if w['name']==self.result['well'] for c in w['curves']}
        for row,column in enumerate(OUTPUT_UNITS):
            item=QTableWidgetItem(column);item.setFlags((item.flags()|Qt.ItemFlag.ItemIsUserCheckable)&~Qt.ItemFlag.ItemIsEditable);item.setCheckState(Qt.CheckState.Checked if column in ('VP_RP','VS_RP','DEN_RP') else Qt.CheckState.Unchecked);table.setItem(row,0,item)
            name=column+'_'+suffix;base=name;counter=2
            while name in existing:name=base+'_'+str(counter);counter+=1
            table.setItem(row,1,QTableWidgetItem(name))
        box.addWidget(table);buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);box.addWidget(buttons);dialog.resize(620,400)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            try:self.attach({table.item(r,0).text():table.item(r,1).text() for r in range(table.rowCount()) if table.item(r,0).checkState()==Qt.CheckState.Checked})
            except Exception as exc:QMessageBox.warning(self,'保存失败',str(exc))

    def save_plot_state(self):
        try:
            state=dict(rows=self.viewer.setting_rows(),legends=[dict(names=[i[1] for i in key],settings=value) for key,value in self.viewer.legend_settings.items()],
                cross=dict(x=self.x.currentText(),y=self.y.currentText(),color=self.color.currentData()))
            (self.path/'display.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');self.status.setText('图件设置已保存，下次打开此成果时恢复。')
        except Exception as exc:self.status.setText('保存图件设置失败：'+str(exc))

    def restore_plot_state(self):
        path=self.path/'display.json'
        if not path.is_file():return
        try:
            state=json.loads(path.read_text(encoding='utf-8'));names={d['name'] for d in self.viewer.datasets}
            rows=[r for r in state['rows'] if r['name'] in names]
            kept={r['name'] for r in rows};self.viewer.datasets=[d for d in self.viewer.datasets if d['name'] in kept]
            if rows:self.viewer.apply_plot_settings(rows)
            else:self.viewer.render([])
            for legend in state.get('legends',[]):
                key=tuple(sorted(self.viewer.identity(d) for d in self.viewer.datasets if d['name'] in legend['names']));self.viewer.legend_settings[key]=legend['settings']
            self.viewer.draw_tracks();cross=state.get('cross',{});self.x.setCurrentText(cross.get('x','vp'));self.y.setCurrentText(cross.get('y','VP_RP'));self.color.setCurrentIndex(max(0,self.color.findData(cross.get('color','Depth'))))
        except Exception as exc:self.status.setText('图件设置未恢复：'+str(exc))

    def export_plot(self):
        path,_=QFileDialog.getSaveFileName(self,'导出当前图件',str(self.path/'well_rock_physics.png'),'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if not path:return
        figure=self.cross_canvas.figure if self.tabs.currentWidget() is self.cross else self.scan_canvas.figure if self.tabs.currentWidget() is self.scan_page else self.viewer.canvas.figure
        try:export_figure(figure,path,self.style)
        except Exception as exc:QMessageBox.warning(self,'导出失败',str(exc))

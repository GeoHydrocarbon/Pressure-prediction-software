"""Interactive same-well log tracks with bounded frame and track resizing."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt,QThread,Signal,QPointF
from PySide6.QtGui import QPainter,QPen,QColor
from PySide6.QtWidgets import (QWidget,QFrame,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,QCheckBox,
    QPushButton,QListWidget,QListWidgetItem,QAbstractItemView,QSplitter,QSizePolicy,QComboBox,QDialog)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from .plot_frame import ResizeFrame,ResizablePlot
from .data import read_table
from .config import UNITS,CANONICAL_UNITS,unit_factor
from .charts import apply_typography,FigureStyle,PALETTE
from .well_points import gap_limit
from .flow_layout import FlowLayout


class CurveReader(QThread):
    loaded=Signal(object);failed=Signal(str)
    def __init__(self,well,refs,parent):
        super().__init__(parent);self.well=well;self.refs=[refs] if isinstance(refs,dict) else refs
    def run(self):
        try:
            tables={};result=[]
            for ref in self.refs:
                if self.isInterruptionRequested():return
                path=ref['path']
                if path not in tables:
                    dtype={r['well_column']:'string' for r in self.refs if r['path']==path and r.get('well_column')}
                    tables[path]=read_table(Path(path),dtype=dtype or None)
                raw=tables[path]
                if ref.get('well_column'):raw=raw.loc[raw[ref['well_column']].astype('string').str.strip().eq(ref.get('well_value',self.well)).fillna(False)]
                if raw.empty:raise ValueError(f'{self.well} / {ref["name"]} 没有可读取的数据')
                z=pd.to_numeric(raw[ref['depth_column']],errors='coerce').to_numpy(dtype=float,na_value=np.nan)*unit_factor('Depth',ref['depth_unit'])
                x=pd.to_numeric(raw[ref['column']],errors='coerce').to_numpy(dtype=float,na_value=np.nan)
                # Derived/custom names still carry physical units (e.g. Eaton
                # MPa versus a measured kPa curve); normalize before overlay.
                physical = ref['name'] if ref['name'] in UNITS else next((key for key,units in UNITS.items() if ref.get('unit') in units), None)
                if physical:x=x*unit_factor(physical,ref['unit']);unit=CANONICAL_UNITS[physical]
                else:unit=ref.get('unit','')
                order=np.argsort(z,kind='stable');z=z[order];x=x[order]
                x=np.where(np.isfinite(x)&np.isfinite(z),x,np.nan)
                predicted=(ref.get('role') in ('derived','prediction') or ref.get('method')=='eaton' or
                           ref['column'] in ('Pp_pred','Pp_eaton'))
                result.append(dict(x=x,z=z,unit=unit,well=self.well,name=ref['name'],source=path,column=ref['column'],
                                   points=ref.get('role')=='measurement' or (ref['name']=='Pp' and not predicted)))
            if not self.isInterruptionRequested():self.loaded.emit(result)
        except Exception as exc:self.failed.emit(str(exc))


class TrackCanvas(FigureCanvasQTAgg):
    def __init__(self,figure):super().__init__(figure);self.dividers=lambda:[]
    def paintEvent(self,event):
        super().paintEvent(event)
        painter=QPainter(self);painter.setPen(QPen(QColor('#b6c6d3'),1,Qt.PenStyle.DotLine))
        for x,top,bottom in self.dividers():
            painter.drawLine(QPointF(x,top),QPointF(x,bottom));middle=(top+bottom)/2
            painter.drawLine(QPointF(x-2,middle-5),QPointF(x-2,middle+5));painter.drawLine(QPointF(x+2,middle-5),QPointF(x+2,middle+5))
        painter.end()


class CurveViewer(QWidget):
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.worker=None;self.generation=0;self.pending=None;self.closing=False
        self.datasets=[];self.weights=[];self.axes=[];self.base_axes=[];self.track_ids=[]
        self.curve_styles={};self.assignments={};self.populating=False;self.track_drag=None
        self.legend_settings={};self.legends={};self.picked_legend=None;self.legend_release_cid=None
        layout=QVBoxLayout(self);self.caption=QLabel('双击查看单曲线；Ctrl / Shift 多选同井曲线后，右键绘制。');self.caption.setWordWrap(True);layout.addWidget(self.caption)
        row=FlowLayout();self.mode=QComboBox();self.mode.addItems(['分道显示','同框叠加','自定义分道']);row.addWidget(self.mode)
        self.mode.setToolTip('叠加时相同单位共用横轴；不同单位使用独立横轴。曲线保持原始深度采样，实测压力以点显示。')
        self.settings_button=QPushButton('分道与曲线样式…');row.addWidget(self.settings_button)
        self.settings_button.clicked.connect(self.plot_settings)
        self.legend_button=QPushButton('图例设置…');row.addWidget(self.legend_button)
        self.legend_button.clicked.connect(self.edit_legends)
        self.lock=QCheckBox('锁定比例');row.addWidget(self.lock)
        self.fit_button=QPushButton('适应窗口');row.addWidget(self.fit_button);self.reset_button=QPushButton('恢复布局');row.addWidget(self.reset_button)
        self.list_toggle=QCheckBox('曲线列表');self.list_toggle.setChecked(True);row.addWidget(self.list_toggle)
        self.size_label=QLabel();row.addWidget(self.size_label);layout.addLayout(row)
        self.canvas=TrackCanvas(Figure(figsize=(8,6),layout='constrained'));self.toolbar=NavigationToolbar2QT(self.canvas,self);layout.addWidget(self.toolbar)
        self.canvas.setToolTip('使用工具栏缩放或平移；拖动测井道之间的虚线可调整道宽。')
        self.body=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(self.body,1)
        self.track_panel=QWidget();track_layout=QVBoxLayout(self.track_panel);track_layout.setContentsMargins(0,0,0,0)
        label=QLabel('曲线（拖动排序）');label.setWordWrap(True);track_layout.addWidget(label)
        self.track_list=QListWidget();self.track_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.track_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove);self.track_list.setDefaultDropAction(Qt.DropAction.MoveAction);track_layout.addWidget(self.track_list)
        self.remove_button=QPushButton('移除所选');self.remove_button.clicked.connect(self.remove_tracks);track_layout.addWidget(self.remove_button)
        self.track_panel.setMinimumWidth(110);self.track_panel.setMaximumWidth(185);self.body.addWidget(self.track_panel)
        self.plot_box=ResizablePlot(self.canvas);self.body.addWidget(self.plot_box);self.body.setStretchFactor(1,1);self.body.setSizes([150,900]);self.body.setChildrenCollapsible(False)
        self.canvas.dividers=self.dividers
        self.lock.toggled.connect(lambda value:setattr(self.plot_box,'lock_ratio',value))
        self.fit_button.clicked.connect(self.plot_box.fit);self.reset_button.clicked.connect(self.restore_layout)
        self.reset_button.setToolTip('恢复图框比例和等宽测井道，保留当前曲线顺序。')
        self.list_toggle.toggled.connect(self.update_track_panel);self.plot_box.sizeChanged.connect(lambda w,h:self.size_label.setText(f'预览 {w} × {h}'))
        self.size_label.setToolTip('拖动图框调整预览宽高；导出保留当前图片比例。')
        self.track_list.model().rowsMoved.connect(self.reorder_tracks)
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.canvas.mpl_connect('button_press_event',self.press_divider);self.canvas.mpl_connect('motion_notify_event',self.move_divider);self.canvas.mpl_connect('button_release_event',self.release_divider)
        self.canvas.mpl_connect('pick_event',self.pick_legend)
        self.update_track_panel()
    def show_curve(self,well,ref):self.show_curves(well,[ref])
    def show_curves(self,well,refs,overlay=False):
        if not refs:return
        refs=deepcopy(refs)
        for ref in refs:ref['path']=str(self.owner.project.resolve(ref['path']))
        self.mode.blockSignals(True);self.mode.setCurrentIndex(int(overlay));self.mode.blockSignals(False)
        self.generation+=1;self.closing=False;request=(well,refs,self.generation)
        self.caption.setText(f'正在读取 {well} / {len(refs)} 条曲线…')
        if self.worker is not None:
            self.pending=request;self.worker.requestInterruption();return
        self.start_read(request)
    def start_read(self,request):
        well,refs,generation=request;worker=CurveReader(well,refs,self);self.worker=worker
        worker.loaded.connect(lambda data:self.render(data) if generation==self.generation and not self.closing else None)
        worker.failed.connect(lambda message:self.caption.setText('读取失败：'+message) if generation==self.generation and not self.closing else None)
        worker.finished.connect(lambda:self.done(worker));worker.start()
    def done(self,worker):
        if self.worker is worker:self.worker=None
        worker.deleteLater()
        if self.pending is not None and not self.closing:
            request=self.pending;self.pending=None;self.start_read(request)
    def render(self,data):
        previous={self.identity(d):self.curve_styles[d['name']] for d in self.datasets if d['name'] in self.curve_styles}
        self.datasets=[data] if isinstance(data,dict) else list(data)
        self.assignments={d['name']:i+1 for i,d in enumerate(self.datasets)}
        self.curve_styles={d['name']:deepcopy(previous.get(self.identity(d),self.default_style(d,i))) for i,d in enumerate(self.datasets)}
        self.track_ids=[]
        self.populate_tracks();self.draw_tracks();self.plot_box.restore(self.default_ratio())
    @staticmethod
    def identity(data):return (data['well'],data['name'],data['source'],data.get('column',''))
    @staticmethod
    def default_style(data,index):
        color=PALETTE[index%len(PALETTE)]
        return dict(kind='points' if data.get('points') else 'line',color=color,linestyle='-',linewidth=1.,
                    marker_color=color,marker='o',markersize=4.2,legend_label=data['name'])
    def groups(self):
        groups={}
        for i,data in enumerate(self.datasets):
            ident=1 if self.mode.currentIndex()==1 else self.assignments[data['name']] if self.mode.currentIndex()==2 else i+1
            groups.setdefault(ident,[]).append(data)
        return sorted(groups.items())
    def default_ratio(self):return max(4/3,len(self.groups())*.32)
    def setting_rows(self):
        tracks={d['name']:i+1 for i,(_,group) in enumerate(self.groups()) for d in group}
        return [dict(name=d['name'],track=tracks[d['name']],**self.curve_styles[d['name']]) for d in self.datasets]
    def plot_settings(self):
        if not self.datasets:return
        from .curve_settings import CurveSettingsDialog
        dialog=CurveSettingsDialog(self,self.setting_rows())
        if dialog.exec()==QDialog.DialogCode.Accepted:self.apply_plot_settings(dialog.values())
    def apply_plot_settings(self,rows):
        order=sorted({int(row['track']) for row in rows})
        self.assignments={row['name']:order.index(int(row['track']))+1 for row in rows}
        self.curve_styles={row['name']:{key:value for key,value in row.items() if key not in ('name','track')} for row in rows}
        self.mode.blockSignals(True);self.mode.setCurrentIndex(2);self.mode.blockSignals(False)
        self.populate_tracks();self.draw_tracks();self.plot_box.restore(self.default_ratio())
    def legend_key(self,datasets):return tuple(sorted(self.identity(data) for data in datasets))
    def legend_rows(self):
        from .legend_settings import legend_defaults
        return [dict(key=self.legend_key(group),title=f"第 {index+1} 道 · "+' / '.join(d['name'] for d in group),
                     count=len(group),settings=deepcopy(self.legend_settings.get(self.legend_key(group),legend_defaults(len(group)))))
                for index,(_,group) in enumerate(self.groups())]
    def edit_legends(self):
        if not self.datasets:return
        from .legend_settings import LegendSettingsDialog
        labels={d['name']:self.curve_styles[d['name']].get('legend_label',d['name']) for d in self.datasets}
        dialog=LegendSettingsDialog(self,self.legend_rows(),labels)
        if dialog.exec()==QDialog.DialogCode.Accepted:self.apply_legend_settings(*dialog.values())
    def apply_legend_settings(self,tracks,labels):
        for track in tracks:self.legend_settings[track['key']]=deepcopy(track['settings'])
        for name,label in labels.items():self.curve_styles[name]['legend_label']=label or name
        self.draw_tracks()
    def pick_legend(self,event):
        self.picked_legend=next((key for key,(legend,base,_) in self.legends.items() if event.artist is legend),None)
    def remember_legend_position(self,event):
        key=self.picked_legend;self.picked_legend=None
        if key not in self.legends:return
        legend,base,options=self.legends[key]
        if not options['draggable']:return
        # This callback is connected after the draggable legends finalize the
        # gesture. Store axes-relative coordinates so resize/export can reuse it.
        xy=base.transAxes.inverted().transform(legend.get_window_extent().p0)
        if np.isfinite(xy).all():
            options.update(position='custom',custom_xy=xy.tolist())
            self.legend_settings[key]=deepcopy(options)
            legend.set_bbox_to_anchor(None);legend.set_loc(tuple(xy))
            self.canvas.draw_idle()
    def add_legend(self,base,unit_axes,datasets,handles):
        from .legend_settings import legend_defaults
        key=self.legend_key(datasets)
        options=deepcopy(self.legend_settings.get(key,legend_defaults(len(datasets))))
        if not options['visible']:return
        position=options['position'];kwargs={}
        if position in ('above','below'):
            top=position=='above'
            offset=(34+36*(len(unit_axes)-1)) if top else -40
            transform=base.transAxes+ScaledTranslation(0,offset/72,self.canvas.figure.dpi_scale_trans)
            kwargs=dict(loc='lower center' if top else 'upper center',bbox_to_anchor=(.5,1 if top else 0),bbox_transform=transform)
        elif position=='custom':kwargs=dict(loc=tuple(options['custom_xy']))
        else:kwargs=dict(loc=position)
        # Put the legend above all twinned axes rather than beneath their data.
        host=list(unit_axes.values())[-1]
        legend=host.legend(handles=handles,labels=[h.get_label() for h in handles],
                           ncols=options['columns'],frameon=True,framealpha=options['alpha'],**kwargs)
        legend.get_frame().set_linewidth(.8 if options['frame'] else 0.)
        legend.set_draggable(options['draggable'],use_blit=False,update='loc')
        self.legends[key]=(legend,base,options)
    def change_mode(self,*_):
        self.populate_tracks();self.draw_tracks();self.plot_box.restore(self.default_ratio())
    def populate_tracks(self):
        self.populating=True;self.track_list.clear()
        for data in self.datasets:
            text=f"第 {self.assignments[data['name']]} 道 · {data['name']}" if self.mode.currentIndex()==2 else data['name']
            item=QListWidgetItem(text);item.setData(Qt.ItemDataRole.UserRole,data['name']);item.setToolTip(data['name']+' ('+data['unit']+')\n'+data['source']);self.track_list.addItem(item)
        self.populating=False;self.update_track_panel()
    def update_track_panel(self,*_):
        self.list_toggle.setEnabled(len(self.datasets)>1);self.track_panel.setVisible(len(self.datasets)>1 and self.list_toggle.isChecked())
    def draw_tracks(self):
        for legend,_,_ in self.legends.values():legend.set_draggable(False)
        if self.legend_release_cid is not None:self.canvas.mpl_disconnect(self.legend_release_cid)
        self.legends={};self.picked_legend=None
        self.track_drag=None;self.canvas.figure.clear();self.axes=[];self.base_axes=[]
        if not self.datasets:
            self.caption.setText('当前没有曲线；可从左侧井对象重新选择。');self.canvas.draw_idle();return
        tracks=self.groups();ids=[key for key,_ in tracks]
        if ids!=self.track_ids:self.weights=[1.]*len(tracks)
        self.track_ids=ids
        bases=self.canvas.figure.subplots(1,len(tracks),sharey=True,squeeze=False,gridspec_kw=dict(width_ratios=self.weights))[0]
        self.base_axes=list(bases)
        finite_depth=[]
        for index,(base,(ident,datasets)) in enumerate(zip(bases,tracks)):
            unit_axes={};handles=[]
            for data in datasets:
                key=data['unit'] or ('unknown',data['name'])
                if key not in unit_axes:
                    ax=base if not unit_axes else base.twiny()
                    if unit_axes:
                        ax.spines['top'].set_position(('outward',36*(len(unit_axes)-1)))
                        ax.patch.set_visible(False)
                    ax.set_xlabel(data['unit'] or data['name']+'（未指定单位）')
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=3));unit_axes[key]=ax;self.axes.append(ax)
                ax=unit_axes[key];style=self.curve_styles[data['name']]
                x,z=np.asarray(data['x'],dtype=float),np.asarray(data['z'],dtype=float)
                if style['kind']!='points':
                    gaps=np.flatnonzero(np.diff(z)>gap_limit(z))+1
                    ax.plot(np.insert(x,gaps,np.nan),np.insert(z,gaps,np.nan),color=style['color'],
                            linewidth=style['linewidth'],linestyle=style['linestyle'],label=data['name'])
                if style['kind']!='line':
                    valid=np.isfinite(x)&np.isfinite(z)
                    ax.scatter(x[valid],z[valid],color=style['marker_color'],s=style['markersize']**2,
                               marker=style['marker'],label=data['name'] if style['kind']=='points' else '_nolegend_',zorder=3)
                handles.append(Line2D([],[],label=style.get('legend_label',data['name']),color=style['color'],
                    linestyle=style['linestyle'] if style['kind']!='points' else 'None',linewidth=style['linewidth'],
                    marker=style['marker'] if style['kind']!='line' else 'None',markersize=style['markersize'],
                    markerfacecolor=style['marker_color'],markeredgecolor=style['marker_color']))
                depth=z[np.isfinite(z)]
                if len(depth):finite_depth.extend([float(depth.min()),float(depth.max())])
            base.set_title(datasets[0]['name'] if len(datasets)==1 else f'第 {index+1} 道')
            base.grid(True,alpha=.18)
            for ax in list(unit_axes.values())[1:]:ax.grid(False)
            self.add_legend(base,unit_axes,datasets,handles)
            if index==0:base.set_ylabel('Depth (m)')
        if finite_depth:
            low,high=min(finite_depth),max(finite_depth)
            if low==high:low-=.5;high+=.5
            bases[0].set_ylim(high,low)
        style=FigureStyle(font_size=10,title_size=11)
        apply_typography(self.canvas.figure,style)
        for legend,_,options in self.legends.values():
            for text in legend.get_texts():text.set_fontsize(options['fontsize'])
        self.legend_release_cid=self.canvas.mpl_connect('button_release_event',self.remember_legend_position)
        self.canvas.figure.suptitle(self.datasets[0]['well'],fontsize=12,fontfamily=[style.font_family,'Microsoft YaHei','DejaVu Sans'])
        description=self.datasets[0]['name'] if len(self.datasets)==1 else f'{len(self.datasets)} 条测井曲线'
        self.caption.setText(f'{self.datasets[0]["well"]} · {description} · {self.mode.currentText()} · 共用深度轴（m）')
        self.caption.setToolTip('各曲线按自身采样绘制，不重采样或插值；图框边缘调图片尺寸，虚线调道宽。')
        self.toolbar.update();self.canvas.draw_idle()
    def reorder_tracks(self,*_):
        if self.populating:return
        lookup={d['name']:d for d in self.datasets}
        ordered=[lookup[self.track_list.item(i).data(Qt.ItemDataRole.UserRole)] for i in range(self.track_list.count())]
        old_weights={d['name']:w for d,w in zip(self.datasets,self.weights)}
        group_weights=dict(zip(self.track_ids,self.weights))
        self.datasets=ordered
        if self.mode.currentIndex()==0:self.weights=[old_weights[d['name']] for d in ordered]
        elif self.mode.currentIndex()==2:
            groups=list(dict.fromkeys(self.assignments[d['name']] for d in ordered))
            self.assignments={d['name']:groups.index(self.assignments[d['name']])+1 for d in ordered}
            self.weights=[group_weights[key] for key in groups];self.track_ids=list(range(1,len(groups)+1))
            self.populate_tracks()
        self.draw_tracks()
    def remove_tracks(self):
        names={item.data(Qt.ItemDataRole.UserRole) for item in self.track_list.selectedItems()}
        self.datasets=[d for d in self.datasets if d['name'] not in names]
        self.populate_tracks();self.draw_tracks()
    def restore_layout(self):
        self.weights=[1.]*len(self.groups());self.draw_tracks();self.plot_box.restore(self.default_ratio())
    def dividers(self):
        if len(self.base_axes)<2:return []
        dpr=self.canvas.device_pixel_ratio;result=[]
        for a,b in zip(self.base_axes[:-1],self.base_axes[1:]):
            x=(a.bbox.x1+b.bbox.x0)/2/dpr;top=self.canvas.height()-min(a.bbox.y1,b.bbox.y1)/dpr;bottom=self.canvas.height()-max(a.bbox.y0,b.bbox.y0)/dpr;result.append((x,top,bottom))
        return result
    def event_position(self,event):
        if event.x is None or event.y is None:return None
        return event.x/self.canvas.device_pixel_ratio,self.canvas.height()-event.y/self.canvas.device_pixel_ratio
    def press_divider(self,event):
        if event.button!=1 or self.toolbar.mode or self.picked_legend is not None:return
        pos=self.event_position(event)
        if pos is None:return
        for i,(x,top,bottom) in enumerate(self.dividers()):
            if abs(pos[0]-x)<=7 and top<=pos[1]<=bottom:
                pixels=(self.base_axes[i].bbox.width+self.base_axes[i+1].bbox.width)/self.canvas.device_pixel_ratio
                self.track_drag=(i,pos[0],list(self.weights),max(1,pixels));return
    def move_divider(self,event):
        if self.toolbar.mode:return
        pos=self.event_position(event)
        if pos is None:return
        if self.track_drag:
            i,x,weights,pixels=self.track_drag;total=weights[i]+weights[i+1];left=np.clip(weights[i]+(pos[0]-x)/pixels*total,.15*total,.85*total)
            self.weights=list(weights);self.weights[i]=float(left);self.weights[i+1]=float(total-left)
            self.base_axes[0].get_subplotspec().get_gridspec().set_width_ratios(self.weights);self.canvas.setCursor(Qt.CursorShape.SplitHCursor);self.canvas.draw_idle();return
        hit=any(abs(pos[0]-x)<=7 and top<=pos[1]<=bottom for x,top,bottom in self.dividers())
        self.canvas.setCursor(Qt.CursorShape.SplitHCursor if hit else Qt.CursorShape.ArrowCursor)
    def release_divider(self,event):self.track_drag=None;self.canvas.unsetCursor()
    def stop(self):
        self.closing=True;self.generation+=1;self.pending=None
        if self.worker is not None:self.worker.requestInterruption();self.worker.wait();self.worker=None
    def reset(self):
        self.generation+=1;self.pending=None;self.datasets=[];self.weights=[];self.legend_settings={};self.populate_tracks();self.draw_tracks()
        if self.worker is not None:self.worker.requestInterruption()

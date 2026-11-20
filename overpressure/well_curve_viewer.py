"""Interactive same-well log tracks with bounded frame and track resizing."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt,QThread,Signal,QPointF
from PySide6.QtGui import QPainter,QPen,QColor
from PySide6.QtWidgets import (QWidget,QFrame,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,QCheckBox,
    QPushButton,QListWidget,QListWidgetItem,QAbstractItemView,QSplitter,QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from .data import read_table
from .config import UNITS,CANONICAL_UNITS,unit_factor
from .charts import apply_typography,FigureStyle,PALETTE


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
                if ref['name'] in UNITS:x=x*unit_factor(ref['name'],ref['unit']);unit=CANONICAL_UNITS[ref['name']]
                else:unit=ref.get('unit','')
                order=np.argsort(z,kind='stable');z=z[order];x=x[order]
                x=np.where(np.isfinite(x)&np.isfinite(z),x,np.nan)
                result.append(dict(x=x,z=z,unit=unit,well=self.well,name=ref['name'],source=path))
            if not self.isInterruptionRequested():self.loaded.emit(result)
        except Exception as exc:self.failed.emit(str(exc))


class ResizeFrame(QFrame):
    BORDER=8
    def __init__(self,canvas,host):
        super().__init__(host);self.host=host;self.drag=None;self.setMouseTracking(True)
        layout=QVBoxLayout(self);layout.setContentsMargins(self.BORDER,self.BORDER,self.BORDER,self.BORDER);layout.addWidget(canvas)
        self.setToolTip('拖动图框边缘或角点调整图片大小；勾选“锁定比例”可保持纵横比。')
    def edges(self,pos):
        return (-1 if pos.x()<self.BORDER else 1 if pos.x()>=self.width()-self.BORDER else 0,
                -1 if pos.y()<self.BORDER else 1 if pos.y()>=self.height()-self.BORDER else 0)
    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:
            edge=self.edges(event.position())
            if any(edge):self.drag=(event.globalPosition(),self.width(),self.height(),edge);event.accept();return
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self.drag:
            start,w,h,(ex,ey)=self.drag;delta=event.globalPosition()-start
            dw=2*delta.x()*ex if ex else 0;dh=2*delta.y()*ey if ey else 0
            width=max(1,w+dw-2*self.BORDER);height=max(1,h+dh-2*self.BORDER)
            if self.host.lock_ratio:
                if ex and (not ey or abs(dw/max(w,1))>=abs(dh/max(h,1))):height=width/self.host.ratio
                else:width=height*self.host.ratio
            self.host.set_image_size(width,height);event.accept();return
        ex,ey=self.edges(event.position())
        self.setCursor(Qt.CursorShape.SizeFDiagCursor if ex==ey and ex else Qt.CursorShape.SizeBDiagCursor if ex and ey else Qt.CursorShape.SizeHorCursor if ex else Qt.CursorShape.SizeVerCursor if ey else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        self.drag=None;self.unsetCursor();super().mouseReleaseEvent(event)
    def leaveEvent(self,event):
        if self.drag is None:self.unsetCursor()
        super().leaveEvent(event)
    def paintEvent(self,event):
        super().paintEvent(event);painter=QPainter(self);painter.setPen(QPen(QColor('#90a9bc'),1));painter.drawRect(self.rect().adjusted(1,1,-2,-2))
        painter.setBrush(QColor('#90a9bc'))
        for x,y in [(3,3),(self.width()//2,3),(self.width()-4,3),(3,self.height()//2),(self.width()-4,self.height()//2),(3,self.height()-4),(self.width()//2,self.height()-4),(self.width()-4,self.height()-4)]:painter.drawRect(x-2,y-2,4,4)
        painter.end()


class ResizablePlot(QWidget):
    sizeChanged=Signal(int,int)
    def __init__(self,canvas):
        super().__init__();self.ratio=4/3;self.scale=1.;self.lock_ratio=False;self.canvas=canvas;self.adjusting=False
        self.setMinimumSize(0,0);self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        layout=QGridLayout(self);layout.setContentsMargins(0,0,0,0)
        self.frame=ResizeFrame(canvas,self);layout.addWidget(self.frame,0,0,Qt.AlignmentFlag.AlignCenter)
        canvas.setMinimumSize(0,0);self.refresh_size()
    def maximum_image_size(self):
        aw=max(1,self.width()-2*ResizeFrame.BORDER);ah=max(1,self.height()-2*ResizeFrame.BORDER)
        return (ah*self.ratio,ah) if aw/ah>self.ratio else (aw,aw/self.ratio)
    def set_image_size(self,width,height):
        aw=max(1,self.width()-16);ah=max(1,self.height()-16)
        if self.lock_ratio:
            mw,mh=self.maximum_image_size();minimum=min(1.,max(min(160,aw)/mw,min(120,ah)/mh))
            self.scale=min(1.,max(minimum,min(width/mw,height/mh)))
        else:
            width=min(aw,max(min(160,aw),width));height=min(ah,max(min(120,ah),height));self.ratio=width/height
            mw,mh=self.maximum_image_size();self.scale=min(1.,width/mw,height/mh)
        self.refresh_size()
    def fit(self):self.scale=1.;self.refresh_size()
    def restore(self,ratio):self.ratio=ratio;self.scale=1.;self.refresh_size()
    def refresh_size(self):
        if self.adjusting:return
        self.adjusting=True
        try:
            w,h=self.maximum_image_size();w=max(1,round(w*self.scale));h=max(1,round(h*self.scale))
            self.frame.setFixedSize(min(max(1,self.width()),w+16),min(max(1,self.height()),h+16))
            self.sizeChanged.emit(w,h)
        finally:self.adjusting=False
    def resizeEvent(self,event):super().resizeEvent(event);self.refresh_size()


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
        self.datasets=[];self.weights=[];self.axes=[];self.populating=False;self.track_drag=None
        layout=QVBoxLayout(self);self.caption=QLabel('双击查看单曲线；Ctrl / Shift 多选同井曲线后，右键绘制。');self.caption.setWordWrap(True);layout.addWidget(self.caption)
        row=QHBoxLayout();self.lock=QCheckBox('锁定比例');row.addWidget(self.lock)
        self.fit_button=QPushButton('适应窗口');row.addWidget(self.fit_button);self.reset_button=QPushButton('恢复布局');row.addWidget(self.reset_button)
        self.list_toggle=QCheckBox('曲线列表');self.list_toggle.setChecked(True);row.addWidget(self.list_toggle);row.addStretch()
        self.size_label=QLabel();row.addWidget(self.size_label);layout.addLayout(row)
        self.canvas=TrackCanvas(Figure(figsize=(8,6),layout='constrained'));self.toolbar=NavigationToolbar2QT(self.canvas,self);layout.addWidget(self.toolbar)
        self.canvas.setToolTip('使用工具栏缩放或平移；拖动测井道之间的虚线可调整道宽。')
        self.body=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(self.body,1)
        self.track_panel=QWidget();track_layout=QVBoxLayout(self.track_panel);track_layout.setContentsMargins(0,0,0,0)
        label=QLabel('测井道（拖动排序）');label.setWordWrap(True);track_layout.addWidget(label)
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
        self.canvas.mpl_connect('button_press_event',self.press_divider);self.canvas.mpl_connect('motion_notify_event',self.move_divider);self.canvas.mpl_connect('button_release_event',self.release_divider)
        self.update_track_panel()
    def show_curve(self,well,ref):self.show_curves(well,[ref])
    def show_curves(self,well,refs):
        if not refs:return
        refs=deepcopy(refs)
        for ref in refs:ref['path']=str(self.owner.project.resolve(ref['path']))
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
        self.datasets=[data] if isinstance(data,dict) else list(data);self.weights=[1.]*len(self.datasets)
        self.populate_tracks();self.draw_tracks();self.plot_box.restore(self.default_ratio())
    def default_ratio(self):return max(4/3,len(self.datasets)*.32)
    def populate_tracks(self):
        self.populating=True;self.track_list.clear()
        for data in self.datasets:
            item=QListWidgetItem(data['name']);item.setData(Qt.ItemDataRole.UserRole,data['name']);item.setToolTip(data['name']+' ('+data['unit']+')\n'+data['source']);self.track_list.addItem(item)
        self.populating=False;self.update_track_panel()
    def update_track_panel(self,*_):
        self.list_toggle.setEnabled(len(self.datasets)>1);self.track_panel.setVisible(len(self.datasets)>1 and self.list_toggle.isChecked())
    def draw_tracks(self):
        self.track_drag=None;self.canvas.figure.clear();self.axes=[]
        if not self.datasets:
            self.caption.setText('当前没有曲线；可从左侧井对象重新选择。');self.canvas.draw_idle();return
        axes=self.canvas.figure.subplots(1,len(self.datasets),sharey=True,squeeze=False,gridspec_kw=dict(width_ratios=self.weights))[0];self.axes=list(axes)
        finite_depth=[]
        for i,(ax,data) in enumerate(zip(axes,self.datasets)):
            ax.plot(data['x'],data['z'],color=PALETTE[i%len(PALETTE)],linewidth=1.)
            ax.set_title(data['name']);ax.set_xlabel(data['unit'] or '未指定单位');ax.grid(True,alpha=.18);ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            if i==0:ax.set_ylabel('Depth (m)')
            depth=data['z'][np.isfinite(data['z'])]
            if len(depth):finite_depth.extend([float(depth.min()),float(depth.max())])
        if finite_depth:
            low,high=min(finite_depth),max(finite_depth)
            if low==high:low-=.5;high+=.5
            axes[0].set_ylim(high,low)
        style=FigureStyle(font_size=10,title_size=11);apply_typography(self.canvas.figure,style)
        self.canvas.figure.suptitle(self.datasets[0]['well'],fontsize=12,fontfamily=[style.font_family,'Microsoft YaHei','DejaVu Sans'])
        description=self.datasets[0]['name'] if len(self.datasets)==1 else f'{len(self.datasets)} 条测井曲线'
        self.caption.setText(f'{self.datasets[0]["well"]} · {description} · 共用深度轴（m）')
        self.caption.setToolTip('各曲线按自身采样绘制，不重采样或插值；图框边缘调图片尺寸，虚线调道宽。')
        self.toolbar.update();self.canvas.draw_idle()
    def reorder_tracks(self,*_):
        if self.populating:return
        lookup={d['name']:(d,w) for d,w in zip(self.datasets,self.weights)}
        ordered=[lookup[self.track_list.item(i).data(Qt.ItemDataRole.UserRole)] for i in range(self.track_list.count())]
        self.datasets=[d for d,w in ordered];self.weights=[w for d,w in ordered];self.draw_tracks()
    def remove_tracks(self):
        names={item.data(Qt.ItemDataRole.UserRole) for item in self.track_list.selectedItems()}
        retained=[(d,w) for d,w in zip(self.datasets,self.weights) if d['name'] not in names]
        self.datasets=[d for d,w in retained];self.weights=[w for d,w in retained];self.populate_tracks();self.draw_tracks()
    def restore_layout(self):
        self.weights=[1.]*len(self.datasets);self.draw_tracks();self.plot_box.restore(self.default_ratio())
    def dividers(self):
        if len(self.axes)<2:return []
        dpr=self.canvas.device_pixel_ratio;result=[]
        for a,b in zip(self.axes[:-1],self.axes[1:]):
            x=(a.bbox.x1+b.bbox.x0)/2/dpr;top=self.canvas.height()-min(a.bbox.y1,b.bbox.y1)/dpr;bottom=self.canvas.height()-max(a.bbox.y0,b.bbox.y0)/dpr;result.append((x,top,bottom))
        return result
    def event_position(self,event):
        if event.x is None or event.y is None:return None
        return event.x/self.canvas.device_pixel_ratio,self.canvas.height()-event.y/self.canvas.device_pixel_ratio
    def press_divider(self,event):
        if event.button!=1 or self.toolbar.mode:return
        pos=self.event_position(event)
        if pos is None:return
        for i,(x,top,bottom) in enumerate(self.dividers()):
            if abs(pos[0]-x)<=7 and top<=pos[1]<=bottom:
                pixels=(self.axes[i].bbox.width+self.axes[i+1].bbox.width)/self.canvas.device_pixel_ratio
                self.track_drag=(i,pos[0],list(self.weights),max(1,pixels));return
    def move_divider(self,event):
        if self.toolbar.mode:return
        pos=self.event_position(event)
        if pos is None:return
        if self.track_drag:
            i,x,weights,pixels=self.track_drag;total=weights[i]+weights[i+1];left=np.clip(weights[i]+(pos[0]-x)/pixels*total,.15*total,.85*total)
            self.weights=list(weights);self.weights[i]=float(left);self.weights[i+1]=float(total-left)
            self.axes[0].get_subplotspec().get_gridspec().set_width_ratios(self.weights);self.canvas.setCursor(Qt.CursorShape.SplitHCursor);self.canvas.draw_idle();return
        hit=any(abs(pos[0]-x)<=7 and top<=pos[1]<=bottom for x,top,bottom in self.dividers())
        self.canvas.setCursor(Qt.CursorShape.SplitHCursor if hit else Qt.CursorShape.ArrowCursor)
    def release_divider(self,event):self.track_drag=None;self.canvas.unsetCursor()
    def stop(self):
        self.closing=True;self.generation+=1;self.pending=None
        if self.worker is not None:self.worker.requestInterruption();self.worker.wait();self.worker=None
    def reset(self):
        self.generation+=1;self.pending=None;self.datasets=[];self.weights=[];self.populate_tracks();self.draw_tracks()
        if self.worker is not None:self.worker.requestInterruption()

"""Shared bounded frames for well curves, result charts and seismic slices."""
from PySide6.QtCore import Qt,Signal
from PySide6.QtGui import QPainter,QPen,QColor
from PySide6.QtWidgets import QWidget,QFrame,QVBoxLayout,QHBoxLayout,QGridLayout,QLayout,QSizePolicy,QCheckBox,QPushButton,QLabel


class ResizeFrame(QFrame):
    BORDER=8
    def __init__(self,canvas,host):
        super().__init__(host);self.host=host;self.drag=None;self.setMouseTracking(True)
        layout=QVBoxLayout(self);layout.setContentsMargins(self.BORDER,self.BORDER,self.BORDER,self.BORDER);layout.setProperty("preserveMargins",True);layout.addWidget(canvas)
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
        layout=QGridLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
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
            self.frame.layout().activate();self.layout().activate()
            self.sizeChanged.emit(w,h)
        finally:self.adjusting=False
    def resizeEvent(self,event):super().resizeEvent(event);self.refresh_size()


class PlotResizeControls(QWidget):
    """Compact screen-preview controls independent of export dimensions."""
    def __init__(self,preview,prefix,parent=None):
        super().__init__(parent);self.preview=preview
        self.setObjectName(prefix+'_resize_controls')
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        row=QHBoxLayout(self);row.setContentsMargins(0,0,0,0)
        self.lock=QCheckBox('锁定纵横比');self.lock.setObjectName(prefix+'_preview_ratio_lock');row.addWidget(self.lock)
        self.fit_button=QPushButton('适应窗口');self.fit_button.setObjectName(prefix+'_fit_preview');row.addWidget(self.fit_button)
        self.reset_button=QPushButton('恢复设置比例');self.reset_button.setObjectName(prefix+'_reset_preview');row.addWidget(self.reset_button)
        row.addStretch()
        self.size_label=QLabel();self.size_label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed);row.addWidget(self.size_label)
        self.setToolTip('拖动图框边缘或四角改变预览大小和比例；导出仍使用绘图设置里的宽、高和 DPI。')
        self.lock.toggled.connect(lambda checked:setattr(preview,'lock_ratio',checked))
        self.fit_button.clicked.connect(preview.fit)
        self.reset_button.clicked.connect(preview.restore_configured)
        preview.sizeChanged.connect(lambda w,h:self.size_label.setText(f'预览 {w} × {h}'))


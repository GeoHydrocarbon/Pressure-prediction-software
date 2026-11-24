"""Shared layout and appearance primitives. Contains no data or task logic."""
from PySide6.QtCore import Qt,Signal,QEvent,QObject,QTimer
from shiboken6 import isValid
from .plot_frame import ResizablePlot
from PySide6.QtWidgets import (QFrame, QLabel, QWidget, QVBoxLayout, QHBoxLayout,QGridLayout,QFormLayout,
    QLayout, QLineEdit, QComboBox, QAbstractSpinBox, QSizePolicy, QPushButton,QAbstractScrollArea,QScrollArea,QMainWindow,QStackedWidget,QToolBar,QSplitter,QApplication,QTabWidget)

PAGE_PADDING = 12
CARD_PADDING = 12
SPACING = 8
FIELD_SPACING = 8


class Card(QFrame):
    def __init__(self, title='', parent=None):
        super().__init__(parent)
        self.setObjectName('card'); self.caption = title; self.caption_widget = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

    def prepare(self):
        layout = self.layout()
        if layout is None: return
        layout.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        if self.caption and self.caption_widget is None:
            label = QLabel(self.caption); label.setObjectName('cardTitle'); label.setWordWrap(True)
            policy=QSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Maximum);policy.setHeightForWidth(True);label.setSizePolicy(policy)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
            self.caption_widget = label
            if isinstance(layout, QFormLayout): layout.insertRow(0, label)
            elif isinstance(layout, QVBoxLayout): layout.insertWidget(0, label)


def polish_layouts(root):
    """Normalize existing nested layouts without reconnecting controls or signals."""
    for layout in root.findChildren(QLayout):
        if not isValid(layout) or layout.property("preserveMargins"): continue
        parent=layout.parent()
        if not isValid(layout):continue
        if type(layout) not in (QVBoxLayout,QHBoxLayout,QGridLayout,QFormLayout):continue
        if isinstance(parent,(QAbstractScrollArea,QMainWindow,QStackedWidget,QToolBar,QSplitter,QComboBox,QAbstractSpinBox)):continue
        layout.setSpacing(SPACING)
        if isinstance(layout, QFormLayout):
            layout.setHorizontalSpacing(SPACING); layout.setVerticalSpacing(FIELD_SPACING)
            layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if isinstance(parent,QWidget) and parent.layout() is layout:
            if not isinstance(parent,Card):
                margins=layout.contentsMargins()
                if any((margins.left(),margins.top(),margins.right(),margins.bottom())):
                    layout.setContentsMargins(PAGE_PADDING,PAGE_PADDING,PAGE_PADDING,PAGE_PADDING)
        else: layout.setContentsMargins(0,0,0,0)
    for card in root.findChildren(Card): card.prepare()
    for edit in root.findChildren(QLineEdit):
        edit.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        edit.setMinimumWidth(80)
    for combo in root.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(7)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
    for spin in root.findChildren(QAbstractSpinBox):
        spin.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
    for label in root.findChildren(QLabel):
        if label.objectName() not in ('title','cardTitle','heading','statusMessage','projectPath'):
            label.setWordWrap(True)
            policy=label.sizePolicy();policy.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
            policy.setVerticalPolicy(QSizePolicy.Policy.Maximum);policy.setHeightForWidth(True);label.setSizePolicy(policy)
    for button in root.findChildren(QPushButton):
        button.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Fixed)


class AspectPreview(ResizablePlot):
    """Mouse-resizable preview with independent, fixed export dimensions."""
    heightRequired=Signal(int)
    def __init__(self,canvas):
        super().__init__(canvas)
        self.width_inches=18/2.54;self.preview_mode='fit';self._configured_size=None

    def set_plot_size(self,width_cm,height_cm):
        size=(float(width_cm),float(height_cm))
        changed=size!=self._configured_size
        self._configured_size=size;self.width_inches=size[0]/2.54
        if changed:self.restore(size[0]/size[1])
        else:self.refresh_size()

    def set_preview_mode(self,mode):
        # Legacy modes cannot enlarge the viewport or reset mouse adjustments.
        self.preview_mode='fit';self.setMinimumSize(0,0);self.refresh_size()

    def restore_configured(self):
        width,height=self._configured_size or (18.,13.)
        self.restore(width/height)

    def refresh_size(self):
        super().refresh_size()
        canvas=self.canvas;figure=canvas.figure;dpr=canvas.device_pixel_ratio
        dpi=max(1,min(160,max(1,canvas.width())/getattr(self,'width_inches',18/2.54)))*dpr
        figure.set_dpi(dpi)
        figure.set_size_inches(max(1,canvas.width())*dpr/dpi,max(1,canvas.height())*dpr/dpi,forward=False)
        canvas.draw_idle();self.heightRequired.emit(0)


class PageScroll(QScrollArea):
    """Scroll the whole results page, including wheel events over Matplotlib."""
    def __init__(self,content,canvas=None):
        super().__init__();self.setWidgetResizable(True);self.setWidget(content)
        # The scroll viewport, rather than its potentially very tall content,
        # participates in the parent layout's size negotiation.
        self.setMinimumSize(0,0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        if canvas is not None:canvas.installEventFilter(self)

    def eventFilter(self,watched,event):
        if event.type()==QEvent.Type.Wheel and not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            bar=self.verticalScrollBar()
            delta=event.pixelDelta().y() or event.angleDelta().y()
            if delta and bar.maximum()>0:
                bar.setValue(bar.value()-delta);event.accept();return True
        return super().eventFilter(watched,event)


class WindowBoundsGuard(QObject):
    """Keep the decorated window within the current screen's usable work area."""
    def __init__(self,window):
        super().__init__(window);self.window=window;self.available_rect=None;self.connected=False
        self.pending=False;self.adjusting=False;self.screen=None
        window.installEventFilter(self)

    def schedule(self,*_):
        if not self.pending and not self.adjusting:
            self.pending=True;QTimer.singleShot(0,self.fit)

    def eventFilter(self,watched,event):
        if event.type() in (QEvent.Type.Show,QEvent.Type.Resize,QEvent.Type.Move,QEvent.Type.LayoutRequest,QEvent.Type.ScreenChangeInternal):self.schedule()
        return False

    def fit(self):
        self.pending=False;window=self.window
        if not isValid(window) or not window.isVisible() or self.adjusting:return
        handle=window.windowHandle()
        if handle is not None and not self.connected:
            handle.screenChanged.connect(self.schedule);self.connected=True
        screen=window.screen() or QApplication.primaryScreen()
        if screen is None:return
        if self.screen is not screen:
            if self.screen is not None:
                try:self.screen.availableGeometryChanged.disconnect(self.schedule)
                except (RuntimeError,TypeError):pass
            self.screen=screen;screen.availableGeometryChanged.connect(self.schedule)
        available=self.available_rect() if self.available_rect is not None else screen.availableGeometry()
        if available.width()<1 or available.height()<1:return
        frame=window.frameGeometry();client=window.geometry()
        extra_width=max(0,frame.width()-client.width());extra_height=max(0,frame.height()-client.height())
        maximum_width=max(1,available.width()-extra_width-16);maximum_height=max(1,available.height()-extra_height-16)
        self.adjusting=True
        try:
            window.setMaximumSize(maximum_width,maximum_height)
            if window.isMaximized() or window.isFullScreen():return
            window.resize(min(window.width(),maximum_width),min(window.height(),maximum_height))
            frame=window.frameGeometry()
            x=max(available.left()+8,min(frame.left(),available.right()-frame.width()+1-8))
            y=max(available.top()+8,min(frame.top(),available.bottom()-frame.height()+1-8))
            if (x,y)!=(frame.left(),frame.top()):window.move(x,y)
        finally:self.adjusting=False


def bound_plot_dialog(dialog):
    """Scroll settings content while keeping the dialog's action buttons visible."""
    layout=dialog.layout()
    for index in range(layout.count()):
        widget=layout.itemAt(index).widget()
        if isinstance(widget,QTabWidget):
            layout.takeAt(index)
            scroll=PageScroll(widget);layout.insertWidget(index,scroll,1)
            break
    dialog.screen_guard=WindowBoundsGuard(dialog)
    return dialog


QSS = '''
QMainWindow, QDialog { background: #f2f5f9; color: #26384b; }
QWidget { color: #26384b; }
QWidget#pageBody { background: #f2f5f9; }
QLabel { background: transparent; }
QLabel#title { font-size: 24px; font-weight: 700; color: #193852; }
QLabel#heading { font-size: 20px; font-weight: 700; color: #193852; }
QLabel#cardTitle { font-size: 14px; font-weight: 700; color: #24465e; padding-bottom: 4px; }
QLabel#subtitle, QLabel#hint { color: #64788b; }
QFrame#card { background: #ffffff; border: 1px solid #dfe7ef; border-radius: 10px; }
QFrame#notice { background: #edf5fa; border: 1px solid #d5e7f1; border-radius: 8px; padding: 10px; }
QPushButton { background: #ffffff; border: 1px solid #ccd9e5; border-radius: 7px; padding: 7px 12px; min-height: 20px; }
QPushButton:hover { background: #edf5fc; border-color: #8cb3d0; }
QPushButton:pressed { background: #deedf8; }
QPushButton:focus { border: 1px solid #4484b1; }
QPushButton[role="primary"] { background: #28648a; border-color: #28648a; color: white; font-weight: 600; }
QPushButton[role="primary"]:hover { background: #32769f; }
QPushButton:disabled { color: #9ba9b7; background: #f4f6f8; border-color: #e1e7ec; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #fbfcfe; border: 1px solid #d3dee8; border-radius: 6px; padding: 6px 8px; min-height: 20px; selection-background-color: #28648a; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #4e8cb6; background: #ffffff; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background: #f0f3f6; color: #96a4b0; }
QComboBox::drop-down { width: 24px; border: none; }
QComboBox QAbstractItemView { background: white; selection-background-color: #dbeaf5; selection-color: #193852; padding: 4px; }
QListWidget { background: #ffffff; border: 1px solid #dfe7ef; border-radius: 8px; outline: none; }
QListWidget::item { padding: 9px 10px; border-radius: 5px; }
QListWidget::item:selected { background: #e2eef8; color: #1e567d; }
QListWidget#navigation { background: #e8eff6; border: none; padding: 6px; }
QListWidget#navigation::item { padding: 12px 10px; margin: 2px 0; }
QListWidget#navigation::item:selected { background: #28648a; color: white; }
QTabWidget::pane { border: 1px solid #dfe7ef; border-radius: 8px; background: #f2f5f9; }
QTabBar::tab { padding: 10px 22px; background: #e9eff5; color: #60758a; border: none; margin-right: 4px; border-top-left-radius: 7px; border-top-right-radius: 7px; }
QTabBar::tab:selected { background: #ffffff; color: #245f86; font-weight: 600; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #eef2f6; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: #becdda; border-radius: 4px; min-height: 28px; }
QScrollBar:horizontal { background: #eef2f6; height: 9px; margin: 0; }
QScrollBar::handle:horizontal { background: #becdda; border-radius: 4px; min-width: 28px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QSplitter::handle { background: #f2f5f9; }
QSplitter::handle:hover { background: #c8dce9; }
QProgressBar { border: 1px solid #dce5ed; border-radius: 5px; background: #e8eef4; min-height: 12px; text-align: center; font-size: 11px; }
QProgressBar::chunk { background: #6c9fbd; border-radius: 4px; }
QPlainTextEdit, QTableWidget { background: white; border: 1px solid #dfe7ef; border-radius: 8px; alternate-background-color: #f3f7fb; gridline-color: #e3ebf2; }
QHeaderView::section { background: #eaf1f7; color: #315269; padding: 8px; border: none; border-bottom: 1px solid #d6e1eb; font-weight: 600; }
QTableWidget::item { padding: 6px; }
QCheckBox { spacing: 7px; }
QToolBar { border: none; background: transparent; spacing: 6px; }
QToolButton { border: none; border-radius: 5px; padding: 4px; }
QToolButton:hover { background: #e5eff7; }
QSlider::groove:horizontal { height: 5px; background: #dde6ee; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #377aa4; border-radius: 7px; }
QToolTip { background: #243d52; color: white; border: none; padding: 6px; }
/* Compact workstation chrome; scientific plots retain their own typography. */
QMainWindow { background: #e9edf1; }
QMenuBar { background: #f7f8fa; border-bottom: 1px solid #cbd3dc; padding: 2px 6px; }
QMenuBar::item { padding: 5px 12px; background: transparent; }
QMenuBar::item:selected { background: #dce9f4; }
QMenu { background: #ffffff; border: 1px solid #bbc8d4; padding: 4px; }
QMenu::item { padding: 7px 25px; }
QMenu::item:selected { background: #dce9f4; }
QMenu::item:disabled { color: #97a2ae; }
QToolBar#mainTools { background: #f1f3f6; border-bottom: 1px solid #cbd3dc; padding: 4px; spacing: 3px; }
QToolBar#mainTools QToolButton { padding: 5px 7px; border-radius: 3px; }
QDockWidget { font-weight: 600; }
QDockWidget::title { background: #e5eaf0; padding: 6px 8px; border-bottom: 1px solid #cbd3dc; text-align: left; }
QTreeWidget { background: #ffffff; border: 1px solid #d0d8e1; outline: none; font-weight: 400; }
QTreeWidget::item { padding: 5px 2px; }
QTreeWidget::item:selected { background: #dcebf7; color: #174d73; }
QTabWidget::pane { border-radius: 0; border: 1px solid #ced7e0; }
QTabBar::tab { padding: 7px 12px; margin-right: 1px; border-top-left-radius: 2px; border-top-right-radius: 2px; }
QTabBar::tab:selected { border-top: 2px solid #317cad; }
QFrame#card { border-radius: 3px; }
QPushButton { border-radius: 3px; padding: 5px 9px; min-height: 18px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { border-radius: 3px; padding: 4px 6px; min-height: 18px; }
QLabel#heading { font-size: 16px; }
QLabel#cardTitle { font-size: 13px; padding-bottom: 2px; }
QListWidget, QPlainTextEdit, QTableWidget { border-radius: 2px; }
QStatusBar { background: #eef2f6; border-top: 1px solid #cbd3dc; font-size: 11px; }
QStatusBar::item { border: none; }
'''

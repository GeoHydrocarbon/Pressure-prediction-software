"""Shared font/title controls and image ratio linking for both plotting panels."""
from PySide6.QtWidgets import QWidget,QFormLayout,QFontComboBox,QDoubleSpinBox,QCheckBox,QLineEdit,QLabel,QPushButton
from .charts import FigureStyle


class TypographyControls(QWidget):
    def __init__(self,style,parent=None,default_style=None):
        super().__init__(parent);form=QFormLayout(self)
        self.font=QFontComboBox();self.font.setEditable(True);form.addRow('字体（中文建议 Microsoft YaHei）',self.font)
        self.sizes={}
        for key,label,maximum in [('axis_label_size','坐标轴标签字号 (pt)',36),('tick_size','坐标轴刻度字号 (pt)',36),('legend_size','图例字号 (pt)',36),('title_size','标题字号 (pt)',48)]:
            spin=QDoubleSpinBox();spin.setRange(5,maximum);spin.setDecimals(1);spin.setObjectName('plot_'+key);self.sizes[key]=spin;form.addRow(label,spin)
        self.show_title=QCheckBox('显示标题');self.show_title.setObjectName('plot_show_title');form.addRow(self.show_title)
        self.title=QLineEdit();self.title.setObjectName('plot_title');self.title.setPlaceholderText('留空使用当前图件默认标题');form.addRow('自定义标题',self.title)
        self.grid=QCheckBox('显示浅色网格');self.grid.setObjectName('plot_grid');form.addRow(self.grid)
        reset=QPushButton('恢复字体与标题默认值');reset.clicked.connect(lambda:self.fill(default_style or FigureStyle()));form.addRow(reset)
        self.fill(style)

    def fill(self,style):
        self.font.setCurrentText(style.font_family)
        for key,spin in self.sizes.items():spin.setValue(getattr(style,key) or style.font_size)
        self.show_title.setChecked(style.show_title);self.title.setText(style.title);self.grid.setChecked(style.grid)

    def collect(self):
        return dict(font_family=self.font.currentText().strip(),font_size=self.sizes['axis_label_size'].value(),
                    show_title=self.show_title.isChecked(),title=self.title.text(),grid=self.grid.isChecked(),
                    **{key:spin.value() for key,spin in self.sizes.items()})


def link_dimensions(form,width,height,dpi,*,prefix='plot',locked=False):
    ratio=QDoubleSpinBox();ratio.setObjectName(prefix+'_ratio');ratio.setRange(.05,20);ratio.setDecimals(4);ratio.setValue(width.value()/height.value())
    form.addRow('纵横比例（宽 ÷ 高）',ratio)
    lock=QCheckBox('锁定纵横比例（修改宽或高时联动）');lock.setObjectName(prefix+'_ratio_lock');lock.setChecked(locked);form.addRow(lock)
    pixels=QLabel();form.addRow('导出像素',pixels)
    changing={'active':False}
    def update(source):
        if changing['active']:return
        changing['active']=True
        try:
            if source=='ratio':height.setValue(width.value()/ratio.value())
            elif lock.isChecked():
                if source=='width':height.setValue(width.value()/ratio.value())
                elif source=='height':width.setValue(height.value()*ratio.value())
            ratio.setValue(width.value()/height.value())
            pixels.setText(f'约 {round(width.value()/2.54*dpi.value())} × {round(height.value()/2.54*dpi.value())} px')
        finally:changing['active']=False
    width.valueChanged.connect(lambda:update('width'));height.valueChanged.connect(lambda:update('height'));ratio.valueChanged.connect(lambda:update('ratio'));dpi.valueChanged.connect(lambda:update('dpi'));update('initial')
    return ratio,lock

"""WellModuleUI controls and interactions."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QPlainTextEdit

class WellModuleUI:
    def build_results(self):
        page = self.page('井预测与验证','选择模型与待预测井，设置实测点匹配后开始预测；数据从左侧井对象导入。',show_heading=False)
        page.setAlignment(Qt.AlignmentFlag.AlignTop)
        form = QFormLayout()
        self.bundle_path = self.path_field(form,'模型目录',directory=True)
        from ..well_panel import WellPredictionPanel
        self.well_prediction = WellPredictionPanel(self,model_form=form); page.addWidget(self.well_prediction)
        self.bundle_path.textChanged.connect(lambda: self.well_prediction.independent.setChecked(False))
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(240); page.addWidget(self.log, 1)

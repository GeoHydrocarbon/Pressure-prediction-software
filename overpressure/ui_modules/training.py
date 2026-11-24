"""TrainingModuleUI controls and interactions."""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLineEdit, QCheckBox, QSpinBox, QLabel
from ..ui_style import Card
from ..algorithms.registry import ALGORITHM_SPECS

class TrainingModuleUI:
    def build_training(self):
        page = self.page("模型训练", "勾选的井全部参与所选训练模式。随机划分默认 80% 训练 / 20% 验证，训练比例可自定义，剩余部分用于验证；也可选择留一井验证。")
        page.addWidget(self.library.training_card)
        self.training_data_check=self.button(page,'检查所选训练井与曲线',lambda:self.start_task('audit',['--wells-only']))
        group = Card("训练方法与参数 · MLP");self.training_algorithm_card=group; form = QFormLayout(group)
        form.addRow('训练模式',self.split_method);form.addRow('随机模式训练比例（%，默认 80）',self.train_percent)
        self.algorithm = self.combo([])
        for spec in ALGORITHM_SPECS.values():
            self.algorithm.addItem(spec.title if spec.available else spec.title+'（待接入）', spec.name)
            item=self.algorithm.model().item(self.algorithm.count()-1)
            item.setEnabled(spec.available)
            item.setToolTip('已实现；支持井预测及三维地震预测' if spec.available else '本阶段仅预留算法入口，尚不能训练')
        form.addRow("算法", self.algorithm)
        self.hidden = QLineEdit(); form.addRow("隐藏层宽度", self.hidden)
        self.dropout = self.number(0, .95, 2); form.addRow("Dropout", self.dropout)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 100000); form.addRow("最多训练轮数", self.epochs)
        self.batch = QSpinBox(); self.batch.setRange(1, 65536); form.addRow("批量大小", self.batch)
        self.lr = self.number(.0000001, 1, 7); form.addRow("学习率", self.lr)
        self.patience = QSpinBox(); self.patience.setRange(1, 10000); form.addRow("早停耐心轮数", self.patience)
        self.seed = QLineEdit(); form.addRow("随机种子（划分与训练）", self.seed)
        page.addWidget(group)
        physical = Card("可选物理约束");self.training_physics_card=physical; physform = QFormLayout(physical)
        self.physical_enabled = QCheckBox("训练中加入 Eaton 软物理损失")
        physform.addRow(self.physical_enabled)
        self.physics_edits = {}
        for key, caption, low, high, decimals in (
            ("weight", "物理损失权重", 0, .9999, 4), ("eaton_exponent", "Eaton 指数", .01, 20, 4),
            ("fluid_density", "流体密度（g/cc）", .01, 10, 4), ("rho_rock", "岩石密度（g/cc）", .01, 10, 4),
            ("gravity", "重力加速度（m/s²）", .01, 100, 4), ("nac_a", "正常压实趋势 a", -100, 100, 9),
            ("nac_b", "正常压实趋势 b", -100, 100, 6)):
            edit = self.number(low, high, decimals); self.physics_edits[key] = edit; physform.addRow(caption, edit)
        self.physical_enabled.toggled.connect(lambda on: [edit.setEnabled(on) for edit in self.physics_edits.values()])
        page.addWidget(physical)
        self.button(page, "检查当前训练模式（小样本）", lambda: self.start_training_mode(True))
        self.training_button=self.button(page, "开始随机划分训练", lambda:self.start_training_mode(False))
        self.split_method.currentIndexChanged.connect(self.update_training_mode_controls)
        self.train_percent.valueChanged.connect(self.update_training_mode_controls)
        self.update_training_mode_controls()
        self.advanced_toggle = QCheckBox("留一井完成后：使用全部参与井训练最终模型")
        page.addWidget(self.advanced_toggle)
        self.advanced_training = QWidget(); advanced = QVBoxLayout(self.advanced_training)
        hint = QLabel("留一井验证输出各井验证结果和汇总指标；预测新井或地震体时，可在这里根据验证记录训练最终模型。随机划分模式无需再次重训。"); hint.setWordWrap(True); advanced.addWidget(hint)
        refit_form = QFormLayout(); self.refit_cv = self.path_field(refit_form, "留一井验证成果目录", directory=True); advanced.addLayout(refit_form)
        self.button(advanced, "验证全部参与井重训流程（小样本）", lambda: self.start_task("refit", ["--cv", self.refit_cv.text(), "--smoke"]))
        self.button(advanced, "使用全部参与井重训最终模型", lambda: self.start_task("refit", ["--cv", self.refit_cv.text()]))
        page.addWidget(self.advanced_training); self.advanced_training.hide()
        self.advanced_toggle.toggled.connect(self.advanced_training.setVisible)
        page.addStretch()

    def start_training_mode(self,smoke=False):
        self.start_task('cross-validate' if self.split_method.currentIndex() else 'train',['--smoke'] if smoke else [])

    def update_training_mode_controls(self,*_):
        leave_one=bool(self.split_method.currentIndex());self.train_percent.setEnabled(not leave_one)
        ratio=self.train_percent.value()
        self.training_button.setText('开始留一井验证' if leave_one else f'开始随机划分训练（{ratio:g}% / {100-ratio:g}%）')

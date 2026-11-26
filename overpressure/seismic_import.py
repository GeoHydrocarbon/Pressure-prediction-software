"""Per-file SEG-Y header configuration and asynchronous, read-only preflight."""
import json
from pathlib import Path
from PySide6.QtCore import QProcess,QProcessEnvironment,QTimer
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QFormLayout,QLabel,QLineEdit,QComboBox,
    QSpinBox,QPushButton,QDialogButtonBox,QPlainTextEdit,QScrollArea,QWidget)
from .runtime import worker_command,user_data_root
from .ui_style import WindowBoundsGuard

STANDARD=dict(inline=189,crossline=193,x=181,y=185,coordinate_scalar=71,coordinate_units=89)
LEGACY=dict(inline=21,crossline=1,x=73,y=77,coordinate_scalar=71,coordinate_units=89)


class SeismicImportDialog(QDialog):
    def __init__(self,parent,path,initial):
        super().__init__(parent)
        self.setWindowTitle('SEG-Y 导入 · 道头设置与预检');self.resize(690,740)
        self.path=Path(path).resolve();self.process=None;self.checked=None;self.checked_config=None;self.closing=False;self.preview_complete=False
        layout=QVBoxLayout(self);body=QWidget();form=QFormLayout(body)
        file_label=QLabel(str(self.path));file_label.setWordWrap(True);form.addRow('文件',file_label)
        hint=QLabel('道头位置使用从 1 开始的字节编号，不是实际线号。先设置位置并预检，通过后才能导入。\n预检仅读取道头和采样信息，不读取地震幅值体。')
        hint.setWordWrap(True);form.addRow(hint)
        self.label=QLineEdit(initial.get('label',self.path.stem));self.unit=QLineEdit(initial.get('unit','Amplitude'))
        form.addRow('数据名称',self.label);form.addRow('幅值单位（仅标注）',self.unit)
        self.convention=QComboBox();self.convention.addItems(['twt','owt']);self.convention.setCurrentText(initial.get('time_convention','twt'))
        form.addRow('时间类型（双程 / 单程）',self.convention)
        self.presets=QComboBox();self.presets.addItem('当前设置',dict(initial.get('headers',STANDARD)))
        self.presets.addItem('常见标准：Inline 189 / Crossline 193',STANDARD)
        self.presets.addItem('常见旧格式：Inline 21 / Crossline 1，XY 73 / 77',LEGACY)
        form.addRow('道头预设（仍需核对预检结果）',self.presets)
        self.headers={}
        for key,title in [('inline','Inline（4 字节）'),('crossline','Crossline（4 字节）'),
                          ('x','X 坐标（4 字节）'),('y','Y 坐标（4 字节）'),
                          ('coordinate_scalar','坐标缩放（2 字节）'),('coordinate_units','坐标单位（2 字节）')]:
            edit=QSpinBox();edit.setRange(1,239 if key in ('coordinate_scalar','coordinate_units') else 237)
            edit.setValue(initial.get('headers',STANDARD).get(key,STANDARD[key]));self.headers[key]=edit
            form.addRow(title+' 起始字节',edit);edit.valueChanged.connect(self.invalidate)
        self.preview_button=QPushButton('读取道头并预检');form.addRow(self.preview_button)
        self.summary=QPlainTextEdit();self.summary.setReadOnly(True);self.summary.setMinimumHeight(160)
        form.addRow(self.summary)
        note=QLabel('预检通过仅表示当前道头与采样可供 Map／剖面浏览，不代表幅值质量或井震标定通过。')
        note.setWordWrap(True);form.addRow(note)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(body);layout.addWidget(scroll)
        self.buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.ok_button=self.buttons.button(QDialogButtonBox.StandardButton.Ok);self.ok_button.setText('确认导入');self.ok_button.setEnabled(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject);layout.addWidget(self.buttons)
        self.preview_button.clicked.connect(self.preview);self.presets.currentIndexChanged.connect(self.use_preset)
        self.convention.currentIndexChanged.connect(self.invalidate)
        self.screen_guard=WindowBoundsGuard(self)
        QTimer.singleShot(0,self.preview)

    def config(self):
        return dict(path=str(self.path),headers={key:edit.value() for key,edit in self.headers.items()},
                    time_convention=self.convention.currentText())

    def values(self):
        return dict(self.config(),label=self.label.text().strip() or self.path.stem,unit=self.unit.text().strip() or 'Amplitude')

    def invalidate(self,*_):
        self.checked=None;self.checked_config=None;self.preview_complete=False;self.ok_button.setEnabled(False)
        self.summary.setPlainText('道头或时间类型已改变，请重新预检。')

    def use_preset(self,*_):
        for key,value in self.presets.currentData().items():self.headers[key].setValue(value)
        self.invalidate()

    def set_busy(self,on):
        self.preview_button.setEnabled(not on);self.presets.setEnabled(not on);self.convention.setEnabled(not on)
        for edit in self.headers.values():edit.setEnabled(not on)

    def preview(self):
        if self.process is not None or self.closing:return
        self.invalidate();self.summary.setPlainText('正在读取道头及采样信息……')
        config=self.config();proc=QProcess(self);self.process=proc;self.set_busy(True)
        env=QProcessEnvironment.systemEnvironment();env.insert('PYTHONIOENCODING','utf-8');proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(user_data_root()))
        proc.started.connect(lambda:(proc.write(json.dumps(dict(config,operation='preview'),ensure_ascii=False).encode('utf-8')),proc.closeWriteChannel()))
        proc.finished.connect(lambda *_:self.finished(proc,config))
        proc.errorOccurred.connect(lambda error:self.finished(proc,config) if error==QProcess.ProcessError.FailedToStart else None)
        try:proc.start(*worker_command('overpressure.seismic_geometry'))
        except Exception as exc:
            self.process=None;proc.deleteLater();self.preview_complete=True;self.set_busy(False);self.summary.setPlainText('预检失败：'+str(exc))

    def finished(self,proc,config):
        if self.process is not proc:return
        output=bytes(proc.readAllStandardOutput()).decode('utf-8',errors='replace')
        errors=bytes(proc.readAllStandardError()).decode('utf-8',errors='replace')
        self.process=None;self.set_busy(False)
        try:
            result=json.loads(output)
            lines=[]
            if 'traces' in result:
                lines.extend([f"地震道数：{result['traces']:,}；每道采样点：{result['samples']:,}",
                    f"Inline：{result['inline'][0]} ～ {result['inline'][1]}；Crossline：{result['crossline'][0]} ～ {result['crossline'][1]}",
                    f"重复线号位置：{result['duplicate_pairs']:,}"])
            if result.get('passed'):
                lines.extend([f"采样间隔：{result['interval_ms']:g} ms；时间：{result['time_range'][0]:g} ～ {result['time_range'][1]:g} ms",
                    'XY 坐标：可用' if result['xy_available'] else 'XY 坐标：不可用，将使用线号底图',
                    '预检通过。请核对线号范围和道头设置后确认导入。'])
                self.checked=result;self.checked_config=config;self.ok_button.setEnabled(True)
            else:lines.append('预检未通过：'+result.get('error','未知错误')+'\n请修改道头设置，或选择相符的预设，再重新预检。')
            self.summary.setPlainText('\n'.join(lines))
        except Exception as exc:self.summary.setPlainText('预检失败：'+(errors.strip() or str(exc) or proc.errorString()))
        finally:self.preview_complete=True;proc.deleteLater()

    def stop(self):
        self.closing=True
        proc,self.process=self.process,None
        if proc is not None:
            if proc.state()!=QProcess.ProcessState.NotRunning:proc.kill();proc.waitForFinished(1000)
            proc.deleteLater()

    def accept(self):
        if self.checked is None or self.checked_config!=self.config():return
        try:
            stat=self.path.stat()
            if self.checked['stamp']!=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns):
                self.invalidate();self.summary.setPlainText('文件已发生变化，请重新预检。');return
        except OSError as exc:self.invalidate();self.summary.setPlainText(str(exc));return
        self.stop();super().accept()

    def reject(self):self.stop();super().reject()
    def closeEvent(self,event):self.stop();super().closeEvent(event)

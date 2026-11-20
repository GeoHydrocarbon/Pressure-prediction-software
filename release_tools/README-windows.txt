超压预测软件 Windows x64 CPU 版

1. 完整解压 ZIP 到本地文件夹。不要直接在压缩包内运行。
2. 双击 PressurePrediction.exe。保留 PressureWorker.exe 与 _internal 文件夹。
3. 不需要安装 Python、Conda 或 CUDA。CPU 版训练速度取决于电脑性能。
4. 首次运行创建空白项目；从左侧项目对象右键导入井、地震、层位和时深关系。
5. 项目与默认成果保存在 %LOCALAPPDATA%\PressurePrediction，也可另存项目。
6. 时深系数和物理参数仅为格式示例，必须按项目实际情况设置后再用于预测。
7. 本包不含实际井数据、SEG-Y、预训练模型和历史成果。先导入数据并训练，或加载兼容模型。
8. 现阶段训练算法为逐点 MLP；RF 与 XGBoost 尚未实现。

本版本未经数字签名。遇到启动失败时保留错误信息，并向维护者报告。
项目：https://github.com/GeoHydrocarbon/Pressure-prediction-software

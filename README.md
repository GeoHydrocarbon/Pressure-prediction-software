# 超压预测软件 · Pressure Prediction

面向井资料训练、井压力预测与验证、三维 SEG-Y 压力预测的 Windows 桌面软件。

## 同事下载安装

打开本仓库的 [Releases](https://github.com/GeoHydrocarbon/Pressure-prediction-software/releases)，下载 `PressurePrediction-Windows-x64-CPU.zip`，完整解压并双击 `PressurePrediction.exe`。

运行包自带 Python 和计算依赖，无需安装 Conda 或 CUDA。CPU 训练速度取决于电脑性能。运行包包含 `PressureWorker.exe` 和 `_internal`，请保持目录完整。项目与默认成果保存在 `%LOCALAPPDATA%\PressurePrediction`，也可以另存项目。

尚无发行包时，请查看 Actions 的构建状态；只有通过打包后启动和计算检查的版本才会发布。

## 主要功能

- 按井管理测井曲线、井坐标、轨迹和时深关系，右键导入，支持多曲线显示。
- 用户选择训练曲线；逐点 MLP，可选 Eaton 物理损失及其参数。
- 随机样本划分（训练比例默认 80%，可修改）或留一井验证。
- 多井批量预测、独立稀疏压力点对比；RMSE、MAE、MAPE、R² 和散点图。
- 自定义 inline/crossline 范围的 SEG-Y 压力预测，支持单位转换和有效样本掩码。
- inline、crossline、时间与沿层位切片，多层位显示和空间插值、井曲线叠加及图件导出。

当前算法为逐点 MLP；RF 和 XGBoost 尚未实现。本仓库不包含 LSTM 训练流程。

## 首次使用

1. 从左侧项目对象右键导入井的曲线及相关资料。
2. 在模型训练中选择井、输入曲线、压力目标、划分方式和参数，执行检查后训练。
3. 在井预测与验证中选择模型及待预测井，按需导入独立实测压力点。
4. 进行地震预测前，加载密度、阻抗 SEG-Y，核对属性单位、道头、时间单位和单/双程时间，设置项目实际的时深关系。
5. 预测完成后查看成果图表或地震切片，按需要导入层位和叠加井曲线。

空白项目中的时深系数、物理参数仅用于示例格式，必须按实际项目设置。没有预训练模型；不应将未校准的示例设置用于实际解释。详细使用手册将在稳定版本另行更新。

## 从源码运行

建议使用独立 Python 3.12 环境，在 Windows x64 安装依赖后运行：

```powershell
python -m pip install -r requirements.txt
python -m overpressure.gui
```

源码默认使用同一 Python 执行后台任务；已有本地 `pytorch_env` 时仍支持原开发环境。可通过 `PRESSURE_WORKER_PYTHON` 指定后台解释器。

## 维护与 Windows 构建

GitHub Actions 的 `Windows release` 流程先运行合成数据测试，再使用 PyInstaller 构建 GUI 和计算组件，并在移除开发环境路径后验证打包后的训练和 GUI 启动。

在 Actions 手动运行并勾选 `publish`，或推送版本标签 `v*`，将通过检查的 ZIP 和 SHA256 校验文件发布到 Releases。普通源码推送只运行测试；Windows 打包在手动运行或推送版本标签时执行。版本号来自 `overpressure/__init__.py`。

本地构建使用 Windows x64 / Python 3.12：

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r release_tools/requirements-windows.txt
python -m PyInstaller --noconfirm --clean release_tools/windows.spec
python release_tools/smoke_frozen.py dist/PressurePrediction
```

发行包未做数字签名。源码、运行包都不包含实际井资料、地震体、历史训练模型和项目成果。

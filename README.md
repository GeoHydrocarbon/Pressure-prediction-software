# 超压预测软件 · Pressure Prediction

面向井资料训练、井压力预测与验证、三维 SEG-Y 压力预测的 Windows 桌面软件。

当前版本：**0.15.0**。包含 SEG-Y 道头设置与预检、传统 Eaton 压力预测、正常压实趋势标定、混合测井道、逐曲线线／点样式和逐道图例设置。

本版新增 RF / XGBoost、TXT 井空间数据、MD / 方位角 / 井斜角轨迹、逐井应用时深关系及 Map / 剖面井曲线叠加。岩石物理工作台采用折叠设置与实时正演预览，支持矿物曲线、常见矿物预设、参数标定、完整成果保存及曲线回写。自动预览最多 240 个深度样点，不自动保存；正式成果需执行完整计算。详细操作见 [测井岩石物理建模](docs/测井岩石物理建模.md)。

岩石物理库来源：[GeoHydrocarbon/shale_oil_rp/rockphypy-main](https://github.com/GeoHydrocarbon/shale_oil_rp/tree/main/rockphypy-main)。库源代码及 LGPL-3.0 许可证、来源与修改说明保存在 `overpressure/vendor/rockphypy/`。

## 同事下载安装

打开本仓库的 [Releases](https://github.com/GeoHydrocarbon/Pressure-prediction-software/releases)，下载对应版本的 Windows x64 CPU ZIP 运行包，完整解压并双击 `PressurePrediction.exe`。

运行包自带 Python 和计算依赖，无需安装 Conda 或 CUDA。CPU 训练速度取决于电脑性能。运行包包含 `PressureWorker.exe` 和 `_internal`，请保持目录完整。项目与默认成果保存在 `%LOCALAPPDATA%\PressurePrediction`，也可以另存项目。

尚无发行包时，请查看 Actions 的构建状态；只有通过打包后启动和计算检查的版本才会发布。

## 主要功能

- 按井管理测井曲线、井坐标、轨迹和时深关系，右键导入，支持多曲线显示。
- Map 中央绘图工作区与独立模块弹窗，可从原始 SEG-Y 显示底图并交互拉取剖面。
- SEG-Y 导入前设置线号与坐标的道头起始字节，提供常见预设和只读预检；通过后按文件保存设置，再打开 Map。
- 传统压力预测：两点定义正常压实趋势，明确采用后执行单井／多井 Eaton 预测，无需训练模型；压力系数为 Pp / Ph。
- 测井图可混合独立道与叠加道；每条曲线独立选择线、点或线＋点，设置颜色、线型、点形状和大小。
- 逐道设置图例位置、字号、透明度、边框、列数和名称，支持拖动与导出；修复图窗关闭后无法重新显示的问题。
- 用户选择训练曲线；逐点 MLP，可选 Eaton 物理损失及其参数。
- 随机样本划分（训练比例默认 80%，可修改）或留一井验证。
- 多井批量预测、独立稀疏压力点对比；RMSE、MAE、MAPE、R² 和散点图。
- 自定义 inline/crossline 范围的 SEG-Y 压力预测，支持单位转换和有效样本掩码。
- inline、crossline、时间与沿层位切片，多层位显示和空间插值、井曲线叠加及图件导出。

训练算法包括逐点 MLP、RF 和 XGBoost；RF 与 XGBoost 本版不加物理约束。本仓库不包含 LSTM 训练流程。

## 首次使用

1. 从左侧项目对象右键导入井的曲线及相关资料。
2. 在模型训练中选择井、输入曲线、压力目标、划分方式和参数，执行检查后训练。
3. 在井预测与验证中选择模型及待预测井，按需导入独立实测压力点。
4. 进行地震预测前，加载密度、阻抗 SEG-Y，核对属性单位、道头、时间单位和单/双程时间，设置项目实际的时深关系。
5. 预测完成后查看成果图表或地震切片，按需要导入层位和叠加井曲线。

传统 Eaton 流程：导入深度与 AC → 打开“传统压力预测” → 生成正常压实趋势图并取两点 → 明确采用趋势 → Eaton 井压力预测。上覆压力使用平均岩石密度近似；原始计算结果保留，异常值只标记。曲线叠加时，在同一口井下 Ctrl / Shift 多选曲线后右键绘制，用“分道与曲线样式…”指定道号与线／点样式；图例遮挡时打开“图例设置…”，选择“图框外上方”或直接拖动。

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

在 Actions 手动运行并勾选 `publish`、推送版本标签 `v*`，或向 main 推送提交标题以 `Release ` 开头的发布提交，将通过检查的 ZIP 和 SHA256 校验文件发布到 Releases。普通源码推送只运行测试。发布前需更新 `overpressure/__init__.py` 中的版本号；已有同名 Release 时会拒绝覆盖。

本地构建使用 Windows x64 / Python 3.12：

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r release_tools/requirements-windows.txt
python -m PyInstaller --noconfirm --clean release_tools/windows.spec
python release_tools/smoke_frozen.py dist/PressurePrediction
```

发行包未做数字签名。源码、运行包都不包含实际井资料、地震体、历史训练模型和项目成果。

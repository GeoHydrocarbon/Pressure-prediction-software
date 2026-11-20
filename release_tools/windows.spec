from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = [(str(root / 'project.example.yaml'), '.')]
for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'PySide6', 'shiboken6', 'matplotlib', 'segyio', 'openpyxl', 'PyYAML'):
    datas += copy_metadata(name)
a = Analysis(
    [str(root / 'release_tools' / 'entry.py')],
    pathex=[str(root)],
    binaries=[], datas=datas,
    hiddenimports=collect_submodules('overpressure') + ['scipy.spatial._qhull', 'sklearn.utils._typedefs', 'openpyxl', 'matplotlib.backends.backend_qtagg'],
    hookspath=[], runtime_hooks=[],
    excludes=['tensorflow', 'IPython', 'jupyter', 'notebook', 'pytest', 'PyQt5', 'PyQt6', 'tkinter'],
    noarchive=False,
)
pyz = PYZ(a.pure)
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name='PressurePrediction', console=False, debug=False, strip=False, upx=False)
worker = EXE(pyz, a.scripts, [], exclude_binaries=True, name='PressureWorker', console=True, debug=False, strip=False, upx=False)
coll = COLLECT(gui, worker, a.binaries, a.datas, strip=False, upx=False, name='PressurePrediction')

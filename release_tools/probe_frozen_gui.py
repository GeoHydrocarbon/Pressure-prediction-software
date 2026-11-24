"""Run only the frozen GUI probe with isolated user data and persistent evidence."""
import json,os,subprocess,sys
from pathlib import Path
app=Path(sys.argv[1]).resolve();folder=Path(sys.argv[2]).resolve();folder.mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
for key in ('PYTHONPATH','PYTHONHOME','PRESSURE_WORKER_PYTHON','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'):env.pop(key,None)
env['PATH']=os.pathsep.join((str(Path(env['WINDIR'])/'System32'),env['WINDIR']))
env['LOCALAPPDATA']=str(folder/'userdata')
report=folder/'gui.json'
result=subprocess.run([str(app/'PressurePrediction.exe'),'--gui-smoke',str(report)],env=env,cwd=folder,timeout=60)
print('returncode',result.returncode)
if report.exists():print(report.read_text(encoding='utf-8'))
else:print('No GUI report')
raise SystemExit(result.returncode)

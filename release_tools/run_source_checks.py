"""Run every source test file in a fresh process to isolate Qt event loops."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1];failures=[]
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',QT_QPA_PLATFORM='offscreen')
    for file in sorted((root/'tests').glob('test_*.py')):
        print('\nSource test file: '+file.name,flush=True)
        try:
            result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',file.name,'-v'],cwd=root,env=env,timeout=300)
            if result.returncode:failures.append(file.name)
        except subprocess.TimeoutExpired:failures.append(file.name+' (timeout)')
    if failures:print('FAILED test files: '+', '.join(failures),flush=True);return 1
    print('All isolated source test files passed.',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())

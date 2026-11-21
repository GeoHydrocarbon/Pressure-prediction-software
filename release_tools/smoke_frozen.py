"""Exercise packaged workers using entirely synthetic wells and SEG-Y fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import segyio
import yaml


def main():
    app_dir = Path(sys.argv[1]).resolve()
    worker = app_dir / 'PressureWorker.exe'
    gui = app_dir / 'PressurePrediction.exe'
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='pressure_release_') as temp:
        folder = Path(temp) / '中文 项目'
        folder.mkdir()
        cfg = yaml.safe_load((root / 'project.example.yaml').read_text(encoding='utf-8'))
        cfg['data']['training_selection'] = 'checked_files'
        cfg['training'].update(epochs=2, patience=2, batch_size=32)
        cfg['algorithm']['hidden_sizes'] = [8]
        files = []
        for i in range(2):
            depth = np.linspace(100, 200, 80)
            path = folder / f'Synthetic_{i}.csv'
            pd.DataFrame({'井':f'Synthetic_{i}', 'Depth':depth, 'AC':90 + depth*.01+i,
                          'DEN':2.2+depth*.001, 'Ip':6000+depth*3, 'Pp':depth*.012+i*.1}).to_csv(path,index=False)
            files.append(str(path))
        cfg['data']['well_files'] = files
        cfg['data']['well_imports'] = files
        cfg['output_dir'] = str(folder / 'outputs')
        project = folder / 'project.yaml'
        project.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
        env = dict(os.environ)
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PRESSURE_WORKER_PYTHON', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            env.pop(key, None)
        # No development interpreter or site-packages can satisfy bundled imports.
        env['PATH'] = os.pathsep.join((str(Path(os.environ['WINDIR'])/'System32'), os.environ['WINDIR']))
        env['LOCALAPPDATA'] = str(folder/'userdata')
        def run(module, args, payload=None):
            result = subprocess.run([str(worker),'--pressure-worker',module,*args], input=payload,
                                    text=True, encoding='utf-8', errors='replace', capture_output=True,
                                    env=env,cwd=folder,timeout=180)
            print(result.stdout)
            if result.returncode: raise RuntimeError(result.stderr + result.stdout)
            return result.stdout
        run('overpressure',['train','--project',str(project)])
        bundles = list((folder/'outputs').rglob('weights.pt'))
        if len(bundles)!=1: raise RuntimeError('Packaged training did not produce weights.pt')
        spec = segyio.spec(); spec.sorting=2; spec.format=5
        spec.samples=[0.,10.,20.]; spec.ilines=[10,11]; spec.xlines=[20,21]
        seismic = folder/'synthetic.sgy'
        with segyio.create(str(seismic), spec) as f:
            f.bin[segyio.BinField.Interval] = 10000
            for i in range(4):
                f.header[i] = {189:10+i//2,193:20+i%2,109:0,115:3,117:10000}
                f.trace[i] = np.full(3,2300,dtype='float32')
        geometry = json.loads(run('overpressure.seismic_geometry', [], json.dumps({'files':{'density':str(seismic)},'headers':{'inline':189,'crossline':193}})))
        assert geometry['files']['density']['inline']==[10,11]
        assert geometry['files']['density']['crossline']==[20,21]
        impedance=folder/'impedance.sgy'
        import shutil
        shutil.copyfile(seismic,impedance)
        with segyio.open(str(impedance),'r+',ignore_geometry=True) as f:
            for i in range(4): f.trace[i]=np.full(3,7_000_000,dtype='float32')
        cfg['seismic'].update(density_file=str(seismic),impedance_file=str(impedance))
        cfg['time_depth']['coefficients']=[100.,1000.]
        project.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        run('overpressure',['predict-seismic','--project',str(project),'--bundle',str(bundles[0].parent),'--full'])
        results=list((folder/'outputs').rglob('result.json'))
        seismic_results=[p for p in results if (p.parent/'pressure.sgy').exists()]
        assert len(seismic_results)==1
        result=json.loads(seismic_results[0].read_text(encoding='utf-8'))
        assert result['predicted_points']==12 and result['round_trip_verified']
        run('overpressure',['prepare-seismic-view','--project',str(project),'--result',str(seismic_results[0].parent)])
        index=seismic_results[0].parent/'volume_index.npz'
        assert index.is_file()
        requests=[{'id':'a'*32,'mode':'inline','value':10},{'command':'close'}]
        messages=run('overpressure.slice_worker',['--index',str(index)],'\n'.join(map(json.dumps,requests))+'\n')
        events=[json.loads(line) for line in messages.splitlines() if line.startswith('{')]
        assert any(event.get('event')=='ready' for event in events)
        assert any(event.get('event')=='slice' for event in events), messages
        assert not any(event.get('event')=='error' for event in events), messages
        report = folder/'gui.json'
        result = subprocess.run([str(gui),'--gui-smoke',str(report)],env=env,cwd=folder,timeout=60)
        if result.returncode or not report.exists(): raise RuntimeError('Packaged GUI did not start')
        assert json.loads(report.read_text(encoding='utf-8'))['gui']=='passed'
        print('Packaged training, SEG-Y inference, geometry, slice reader and GUI smoke tests passed')


if __name__ == '__main__':
    main()

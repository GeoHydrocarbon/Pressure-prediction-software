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
    sys.path.insert(0,str(root))
    with tempfile.TemporaryDirectory(prefix='pressure_release_') as temp:
        folder = Path(temp) / '中文 项目'
        folder.mkdir()
        cfg = yaml.safe_load((app_dir / '_internal' / 'project.example.yaml').read_text(encoding='utf-8'))
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
        for algorithm in ('rf','xgboost'):
            cfg['algorithm']['name']=algorithm
            project.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
            run('overpressure',['train','--project',str(project),'--smoke'])
            manifests=[json.loads(p.read_text(encoding='utf-8')) for p in (folder/'outputs/models').rglob('manifest.json')]
            assert any(m.get('algorithm')==algorithm and m.get('status')=='completed' for m in manifests),algorithm
        cfg['algorithm']['name']='mlp'
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
        browser_requests=[dict(command='open',id='open',config=dict(path=str(seismic),headers=cfg['seismic']['header_fields'],unit='Amplitude')),
                          dict(command='section',id='section',view_id='test',mode='inline',position=10),dict(command='close')]
        messages=run('overpressure.browser_worker',['--cache',str(folder/'browser_cache')],
                     '\n'.join(map(json.dumps,browser_requests))+'\n')
        browser_events=[json.loads(line) for line in messages.splitlines() if line.startswith('{')]
        assert any(event.get('event')=='section' for event in browser_events),messages
        assert not any(event.get('event')=='error' for event in browser_events),messages
        from overpressure.well_library import inspect_curve_file
        cfg['project_data']['wells']=inspect_curve_file(files[0],well_column='井')
        from overpressure.traditional import defaults,two_point_trend
        trend=two_point_trend([[100.,4.5],[200.,4.4]])
        cfg['traditional_prediction']=dict(defaults({}),analysis_well='Synthetic_0',analysis_min=100.,analysis_max=200.,
            wells=['Synthetic_0'],applied_trends={'Synthetic_0':dict(trend,confirmed=True)})
        project.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
        run('overpressure',['traditional-trend','--project',str(project)])
        run('overpressure',['traditional-predict','--project',str(project)])
        reports=list((folder/'outputs'/'traditional').rglob('traditional.json'))
        assert len(reports)==2
        for path in reports:assert json.loads(path.read_text(encoding='utf-8'))['status']=='completed'
        from overpressure.rock_physics.well_model import default_scheme
        rp=default_scheme();rp.update(well='Synthetic_0',depth_min=100.,depth_max=200.)
        rp['bindings']['phi']=dict(curve='',constant=.12,unit='fraction')
        well=cfg['project_data']['wells'][0]
        for operation in ('well_live','well_log'):
            before=set((folder/'outputs').rglob('result.json'))
            text=run('overpressure.rock_physics.worker',[],json.dumps(dict(operation=operation,scheme=rp,well=well,output_dir=str(folder/'outputs/rp'))))
            event=[json.loads(line) for line in text.splitlines() if line.startswith('{')][-1]
            assert event['event']==('live_preview' if operation=='well_live' else 'completed'),event
            assert event['result']['summary']['modeled_rows']>0
            if operation=='well_live':assert set((folder/'outputs').rglob('result.json'))==before
        report = folder/'gui.json'
        result = subprocess.run([str(gui),'--gui-smoke',str(report),str(seismic)],env=env,cwd=folder,timeout=90)
        if result.returncode or not report.exists():
            detail=report.read_text(encoding='utf-8') if report.exists() else 'No report'
            raise RuntimeError('Packaged GUI did not pass: '+detail)
        gui_report=json.loads(report.read_text(encoding='utf-8'))
        assert gui_report['gui']=='passed',gui_report
        for field in ('blank_project','traditional_module','mixed_tracks','curve_styles','legend_settings','reopen_plot','rock_physics_live','rock_physics_save'):
            assert gui_report[field],gui_report
        assert gui_report['segy_import']['passed'] and gui_report['segy_import']['section'],gui_report
        if len(sys.argv)>2:
            reports_dir=Path(sys.argv[2]).resolve();reports_dir.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(report,reports_dir/'packaged_gui.json')
            shutil.copyfile(report.with_suffix('.png'),reports_dir/'packaged_gui.png')
            (reports_dir/'packaged_checks.json').write_text(json.dumps(dict(status='passed',gui=gui_report,
                checks=['CPU training','SEG-Y inference','geometry','slice reader','Map section','traditional trend',
                        'Eaton prediction','blank GUI','mixed tracks','curve styles','legend settings','reopen plot','SEG-Y import settings and preflight',
                        'RF training','XGBoost training','RockPhyPy resources','rock physics live worker','rock physics saved worker','live GUI','saved GUI']),
                ensure_ascii=False,indent=2),encoding='utf-8')
        print('Packaged CPU training, SEG-Y, Map section, Eaton, blank GUI, mixed tracks and legend smoke tests passed')


if __name__ == '__main__':
    main()

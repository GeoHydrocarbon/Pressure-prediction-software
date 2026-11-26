import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from overpressure.config import Project
from overpressure.model import ModelBundle
from overpressure.modules.training import train, cross_validate, refit
from overpressure.algorithms.trees import save_forest
from overpressure.data import sha256_file
from overpressure.jobs import Cancelled

ROOT=Path(__file__).resolve().parents[1]


class TreeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)
        cfg=yaml.safe_load((ROOT/'project.yaml').read_text(encoding='utf-8'))
        cfg['physics']['enabled']=False
        cfg['data'].pop('well_files',None);cfg['data'].pop('well_imports',None)
        cfg['data'].update(training_selection='named_wells',well_file='wells.csv',development_wells=['A','V'],validation_wells=['V'],
                           blind_wells=['B'],split_method='well_holdout',features=['Depth','AC','DEN','Ip'],target_curve='Pp')
        cfg['data']['columns']={k:k for k in ('well','Depth','AC','DEN','Ip','Pp')}
        cfg['algorithm'].update(name='rf',rf=dict(n_estimators=6,n_jobs=1),xgboost=dict(n_estimators=8,early_stopping_rounds=2,n_jobs=1))
        cfg['output_dir']='outputs'
        self.project=Project(self.folder/'project.yaml',cfg)
        rng=np.random.default_rng(42);n=90;depth=np.tile(np.linspace(100,200,30),3)
        self.frame=pd.DataFrame(dict(well=np.repeat(['A','V','B'],30),Depth=depth,AC=rng.uniform(70,110,n),DEN=rng.uniform(2,2.6,n),Ip=rng.uniform(6000,9000,n),Pp=.1*depth+rng.normal(0,.1,n)))
        self.frame.to_csv(self.folder/'wells.csv',index=False)

    def quiet(self,fn,*args,**kwargs):
        with redirect_stdout(io.StringIO()):return fn(*args,**kwargs)

    def test_both_train_reload_predict_wells_and_keep_blind_excluded(self):
        from overpressure.modules.wells import predict_wells
        for name in ('rf','xgboost'):
            with self.subTest(name=name):
                self.project.settings['algorithm']['name']=name
                run=self.quiet(train,self.project);bundle=ModelBundle(run)
                self.assertEqual(set(bundle.metadata['split']),{'train','validation'})
                self.assertEqual(bundle.metadata['preprocessing']['fit_wells'],['A'])
                x=self.frame[['Depth','AC','DEN','Ip']].to_numpy(copy=True);x[1,1]=np.nan
                pred=bundle.predict(x,batch_size=7)
                self.assertTrue(np.isnan(pred[1]));self.assertTrue(np.isfinite(pred[np.arange(len(pred))!=1]).all())
                np.testing.assert_array_equal(pred,ModelBundle(run).predict(x,batch_size=8192))
                out=self.quiet(predict_wells,self.project,run,blind_only=True)
                points=pd.read_csv(out/'predictions.csv');self.assertEqual(set(points.well),{'B'})
                self.assertTrue(np.isfinite(points.Pp_pred).all())
                weights=run/('forest.npz' if name=='rf' else 'model.json');weights.write_bytes(weights.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'校验'):ModelBundle(run)

    def test_rf_serialization_matches_sklearn(self):
        x=self.frame[['Depth','AC','DEN','Ip']].to_numpy();y=self.frame.Pp.to_numpy()
        forest=RandomForestRegressor(n_estimators=5,max_depth=5,random_state=6).fit(x,y)
        save_forest(forest,self.folder/'forest.npz')
        meta=dict(schema_version=1,status='completed',algorithm='rf',features=['Depth','AC','DEN','Ip'],weights_sha256=sha256_file(self.folder/'forest.npz'))
        (self.folder/'manifest.json').write_text(json.dumps(meta),encoding='utf-8')
        np.testing.assert_allclose(ModelBundle(self.folder).predict(x,batch_size=3),forest.predict(x),rtol=1e-6)

    def test_random_split_and_smoke_models(self):
        from overpressure.modules.wells import predict_wells
        self.project.settings['data'].update(split_method='random_points',train_ratio=.7,validation_wells=[])
        for name in ('rf','xgboost'):
            self.project.settings['algorithm']['name']=name
            run=self.quiet(train,self.project,smoke=True);bundle=ModelBundle(run)
            self.assertEqual(bundle.metadata['split_strategy']['method'],'random_points')
            assignments=pd.read_csv(run/'split_assignments.csv')
            self.assertFalse(assignments.source_row.duplicated().any())
            self.assertNotIn('B',set(assignments.well))
            with self.assertRaisesRegex(ValueError,'试算'):self.quiet(predict_wells,self.project,run)

    def test_xgboost_early_stop_saves_only_best_rounds(self):
        cfg=self.project.settings;cfg['algorithm']['name']='xgboost'
        self.frame.loc[self.frame.well=='A','Pp']=10.
        self.frame.loc[self.frame.well=='V','Pp']=100.
        self.frame.to_csv(self.folder/'wells.csv',index=False)
        run=self.quiet(train,self.project);bundle=ModelBundle(run)
        self.assertEqual(bundle.metadata['best_epoch'],1)
        self.assertEqual(bundle.metadata['epochs_completed'],3)
        self.assertEqual(bundle._backend.model.num_boosted_rounds(),1)

    def test_leave_one_well_and_refit_for_each_backend(self):
        self.project.settings['data']['split_method']='leave_one_well_out'
        for name in ('rf','xgboost'):
            with self.subTest(name=name):
                self.project.settings['algorithm']['name']=name
                cv=self.quiet(cross_validate,self.project);summary=json.loads((cv/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual({r['validation_well'] for r in summary['folds']},{'A','V'})
                out=self.quiet(refit,self.project,cv);bundle=ModelBundle(out)
                self.assertEqual(set(bundle.metadata['split']),{'train'})
                self.assertEqual(set(bundle.metadata['preprocessing']['fit_wells']),{'A','V'})
                self.assertNotIn('internal_validation_metrics',bundle.metadata)
                self.assertEqual(bundle.metadata['best_epoch'],summary['suggested_refit_epochs'])

    def test_cancel_and_no_physics(self):
        stop=self.folder/'cancel';stop.touch()
        for name in ('rf','xgboost'):
            self.project.settings['algorithm']['name']=name
            with self.assertRaises(Cancelled):self.quiet(train,self.project,cancel_file=stop)
        runs=list((self.folder/'outputs'/'models').iterdir())
        for run in runs:
            self.assertEqual(json.loads((run/'manifest.json').read_text(encoding='utf-8'))['status'],'cancelled')
            with self.assertRaises(ValueError):ModelBundle(run)

    def test_both_predict_real_segy_pipeline(self):
        import segyio
        from overpressure.seismic import predict_seismic_volume
        cfg=self.project.settings;cfg['time_depth']=dict(method='polynomial',coefficients=[100.,1000.],time_unit='s',time_convention='twt',depth_unit='m')
        cfg['seismic'].update(density_file='den.sgy',impedance_file='ip.sgy',density_unit='g/cc',impedance_unit='g/cc*m/s')
        cfg['seismic']['header_fields'].update(inline=21,crossline=1,x=73,y=77)
        for filename,value in [('den.sgy',2.4),('ip.sgy',8000.)]:
            spec=segyio.spec();spec.format=5;spec.samples=[0,1,2];spec.tracecount=2
            with segyio.create(str(self.folder/filename),spec) as f:
                f.bin[segyio.BinField.Interval]=1000
                for i in range(2):
                    f.header[i]={1:20+i,21:10,73:i,77:1,109:0,115:3,117:1000}
                    f.trace[i]=np.array([value,0,value],np.float32)
        for name in ('rf','xgboost'):
            cfg['algorithm']['name']=name;run=self.quiet(train,self.project)
            out=self.quiet(predict_seismic_volume,self.project,run,block_traces=1)
            with segyio.open(str(out/'pressure.sgy'),'r',ignore_geometry=True) as f:
                self.assertEqual(f.header[1][1],21)
                self.assertTrue(np.isfinite(f.trace[0][[0,2]]).all());self.assertTrue(np.isnan(f.trace[0][1]))
            self.assertTrue(json.loads((out/'result.json').read_text(encoding='utf-8'))['round_trip_verified'])

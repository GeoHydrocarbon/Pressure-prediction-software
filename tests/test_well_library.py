"""End-to-end contracts for well-centric data and variable input features."""
from copy import deepcopy
from contextlib import redirect_stdout
import io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from overpressure.config import Project,validate_settings
from overpressure.well_library import inspect_curve_file,merge_wells,read_library_frame,read_library_measurements
from overpressure.data import prepare_data
ROOT=Path(__file__).resolve().parents[1]

class WellLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        cfg=yaml.safe_load((ROOT/'project.yaml').read_text(encoding='utf-8'))
        cfg['data'].update(training_selection='well_library',features=['GR','DEN'],feature_units={'GR':'API'},target_curve='Pp',development_wells=['A','B'],validation_wells=[],blind_wells=[],split_method='random_points',train_ratio=.7)
        cfg['physics']['enabled']=False;cfg['algorithm'].update(hidden_sizes=[4],dropout=0);cfg['training'].update(epochs=1,batch_size=8)
        wells=[]
        for name in ('A','B','C'):
            path=self.folder/(name+'.csv');z=np.arange(20.)+100
            pd.DataFrame(dict(Depth=z,GR=np.arange(20.)+10,DEN=2.4,Pp=10+z/20)).to_csv(path,index=False)
            incoming=inspect_curve_file(path,definitions=[dict(column=k,name=k,unit=u) for k,u in [('GR','API'),('DEN','g/cc'),('Pp','MPa')]])
            wells=merge_wells(wells,incoming)
        cfg['project_data']['wells']=wells;self.project=Project(self.folder/'project.yaml',cfg)
    def test_selected_wells_features_depth_alignment_and_units(self):
        validate_settings(self.project.settings);prepared=prepare_data(self.project)
        self.assertEqual(prepared.scaler.mean_.shape,(2,));self.assertEqual(set(prepared.audit.frame.well),{'A','B'})
        self.assertEqual(len(prepared.splits['train']),28);self.assertEqual(len(prepared.splits['validation']),12)
        self.project.settings['data']['feature_units']['GR']='other'
        with self.assertRaisesRegex(ValueError,'单位'):prepare_data(self.project)
    def test_reimport_curve_merges_by_well_and_keeps_original_depth_grid(self):
        path=self.folder/'extra.csv';pd.DataFrame(dict(well=['A','A'],Depth=[100.,102.],NPHI=[.2,.3])).to_csv(path,index=False)
        incoming=inspect_curve_file(path,well_column='well',definitions=[dict(name='NPHI',column='NPHI',unit='v/v')])
        before=self.project.settings['project_data']['wells'];original=before[0]['curves'][0]
        merged=merge_wells(before,incoming);self.assertEqual(len(merged),3);self.assertEqual(merged[0]['curves'][0],original)
        self.project.settings['project_data']['wells']=merged;cfg=self.project.settings['data'];cfg.update(features=['NPHI'],feature_units={'NPHI':'v/v'},development_wells=['A'])
        frame,_,_=read_library_frame(self.project,['A'],['NPHI']);self.assertEqual(frame.Depth.tolist(),[100,102])
        with self.assertRaisesRegex(ValueError,'缺少曲线'):read_library_frame(self.project,['B'],['NPHI'])
    def test_no_implicit_interpolation_for_different_grids(self):
        path=self.folder/'shifted.csv';pd.DataFrame(dict(Depth=np.arange(20.)+100.5,DEN=2400)).to_csv(path,index=False)
        incoming=inspect_curve_file(path,forced_well='A',definitions=[dict(name='DEN',column='DEN',unit='kg/m3')])
        self.project.settings['project_data']['wells']=merge_wells(self.project.settings['project_data']['wells'],incoming)
        with self.assertRaisesRegex(ValueError,'有效'):prepare_data(self.project)
    def test_sparse_measurements_remain_separate_and_keep_all_points(self):
        path=self.folder/'points.csv';pd.DataFrame(dict(well=['C','C'],Depth=[101.5,104.5],Pp=[16000,18000])).to_csv(path,index=False)
        incoming=inspect_curve_file(path,well_column='well',definitions=[dict(name='实测压力',column='Pp',unit='kPa',role='measurement')])
        self.project.settings['project_data']['wells']=merge_wells(self.project.settings['project_data']['wells'],incoming)
        points=read_library_measurements(self.project,['C']);np.testing.assert_allclose(points.Pp,[16,18])
        self.assertEqual(len(self.project.settings['project_data']['wells'][2]['curves']),5)
    def test_dynamic_train_reload_predict_and_reject_unsupported_seismic(self):
        from overpressure.training import train
        from overpressure.model import ModelBundle
        from overpressure.well_workflow import predict_new_wells
        from overpressure.predict_volume import predict
        with redirect_stdout(io.StringIO()):run=train(self.project)
        bundle=ModelBundle(run,device='cpu');self.assertEqual(bundle.features,('GR','DEN'));self.assertEqual(bundle.feature_units['GR'],'API')
        self.assertTrue(np.isfinite(bundle.predict([[12.,2.4]])).all())
        with self.assertRaises(ValueError):bundle.predict([[12.,2.4,7.]])
        self.project.settings['data']['features']=['Depth'] # Current training selection cannot alter the model inputs.
        self.project.settings['well_prediction']=dict(use_library=True,library_wells=['C'],independence_confirmed=False,max_gap_m=0)
        with redirect_stdout(io.StringIO()):out=predict_new_wells(self.project,run)
        result=pd.read_csv(out/'predictions.csv');self.assertEqual(len(result),20);self.assertTrue(np.isfinite(result.Pp_pred).all())
        with self.assertRaisesRegex(ValueError,'GR'):predict(self.project,run,full=True)

    def test_dynamic_leave_one_and_refit_preserve_features(self):
        from overpressure.cross_validation import cross_validate
        from overpressure.refit import refit
        from overpressure.model import ModelBundle
        self.project.settings['data']['split_method']='leave_one_well_out'
        with redirect_stdout(io.StringIO()):
            cv=cross_validate(self.project);model=refit(self.project,cv)
        self.assertEqual(ModelBundle(model).features,('GR','DEN'))
        self.project.settings['data']['features']=['DEN']
        with self.assertRaisesRegex(ValueError,'曲线'):refit(self.project,cv)

    def test_seismic_subset_uses_model_order(self):
        import torch,segyio
        from overpressure.model import PointMLP
        from overpressure.data import sha256_file
        from overpressure.predict_volume import predict
        from test_volume_pipeline import VolumePipelineTests
        fixture=VolumePipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        model=PointMLP([2],0,input_size=2)
        for parameter in model.parameters():parameter.data.zero_()
        # x[0] is density and x[1] depth; swapping them changes the output.
        model.network[0].weight.data.copy_(torch.tensor([[1.,0.],[0.,1.]]))
        model.network[-1].weight.data.copy_(torch.tensor([[10.,.01]]))
        torch.save(model.state_dict(),fixture.bundle/'weights.pt')
        meta=json.loads((fixture.bundle/'manifest.json').read_text(encoding='utf-8'))
        meta.update(features=['DEN','Depth'],architecture={'hidden_sizes':[2],'dropout':0},weights_sha256=sha256_file(fixture.bundle/'weights.pt'),preprocessing=dict(mean=[0,0],scale=[1,1],minimum=[0,0],maximum=[100,10000]))
        (fixture.bundle/'manifest.json').write_text(json.dumps(meta),encoding='utf-8')
        with redirect_stdout(io.StringIO()):out=predict(fixture.project,fixture.bundle,trace_count=1)
        with segyio.open(str(out/'pressure.sgy'),ignore_geometry=True) as f:
            values=np.array(f.trace[0]);np.testing.assert_allclose(values[[0,2]], [25.,25.02],atol=1e-4)

if __name__=='__main__':unittest.main()

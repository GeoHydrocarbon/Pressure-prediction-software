"""Scientific invariants, depth/QC semantics and calibration separation."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from overpressure.rock_physics.well_model import (default_scheme,prepare_data,forward,run_workflow,
    align,convert,validate_scheme,calibrate)
from overpressure.rock_physics.storage import save_result
from overpressure.rock_physics.well_results import attach_curves


class WellRockPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.raw=pd.DataFrame(dict(Depth=np.arange(100.,110.),POR=np.linspace(5.,20.,10),SW=np.full(10,60.),VP=np.full(10,4200.),VS=np.full(10,2500.),DEN=np.full(10,2.4)))
        self.path=self.root/'logs.csv';self.raw.to_csv(self.path,index=False)
        self.well=dict(name='A',curves=[dict(name=c,column=c,path=str(self.path),depth_column='Depth',depth_unit='m',well_column='',well_value='A',unit='') for c in self.raw])
        self.s=default_scheme();self.s.update(well='A',depth_min=100.,depth_max=109.,dry_model='cripor')
        for key,column,unit in [('phi','POR','%'),('sw','SW','%'),('vp','VP','m/s'),('vs','VS','m/s'),('rho','DEN','g/cm3')]:self.s['bindings'][key]=dict(curve=column,constant=None,unit=unit)

    def test_units_and_no_extrapolation_or_missing_gap_bridge(self):
        np.testing.assert_allclose(convert([100,200],'vp','us/ft'),[3048,1524])
        np.testing.assert_allclose(convert([1000,2000],'rho','kg/m3'),[1,2])
        np.testing.assert_allclose(convert([10,30],'phi','%'),[.1,.3])
        values=align(np.array([1.,2.,3.,5.]),np.array([10.,np.nan,30.,50.]),np.array([.5,1.5,2.,2.5,4.,5.,6.]),True,2.)
        self.assertTrue(np.isnan(values[:4]).all());self.assertEqual(values[4],40);self.assertEqual(values[5],50);self.assertTrue(np.isnan(values[6]))
        self.assertTrue(np.isnan(align(np.array([1.,3.]),np.array([10.,30.]),np.array([2.]))[0]))

    def test_independent_critical_porosity_gassmann_equations(self):
        data,_=prepare_data(self.s,self.well);r=forward(data,self.s)
        f=np.array([.7,.3]);k=np.array([37.,21.]);g=np.array([44.,7.])
        k0=((f*k).sum()+1/(f/k).sum())/2;g0=((f*g).sum()+1/(f/g).sum())/2
        phi=r.phi.to_numpy();kd=k0*(1-phi/.4);gd=g0*(1-phi/.4);kf=r.K_fluid.to_numpy()
        ks=kd+(1-kd/k0)**2/(phi/kf+(1-phi)/k0-kd/k0**2)
        rho=(1-phi)*2.635+phi*r.rho_fluid
        np.testing.assert_allclose(r.K_RP,ks,rtol=1e-12);np.testing.assert_allclose(r.G_RP,gd,rtol=1e-12)
        np.testing.assert_allclose(r.VP_RP,np.sqrt((ks+4*gd/3)/rho)*1000,rtol=1e-12)
        np.testing.assert_allclose(r.AC_RP,304800/r.VP_RP)

    def test_zero_porosity_and_dual_dem_endmembers(self):
        data,_=prepare_data(self.s,self.well);data.loc[:,'phi']=0
        r=forward(data,self.s);np.testing.assert_allclose(r.K_RP,r.K_matrix);np.testing.assert_allclose(r.DEN_RP,data.matrix_density)
        data.loc[:,'phi']=.1
        for sf,alpha in [(0,.15),(1,.01)]:
            s=deepcopy(self.s);s['dry_model']='dual_dem';s['parameters']['soft_fraction']=sf
            dual=forward(data,s);s['dry_model']='single_dem';s['parameters']['alpha_hard']=alpha;single=forward(data,s)
            np.testing.assert_allclose(dual.VP_RP,single.VP_RP,rtol=1e-10)

    def test_weight_fraction_and_toc_mass_accounting(self):
        s=deepcopy(self.s);s['mineral_basis']='weight';s['kerogen']=True;s['bindings']['toc']['constant']=.03
        data,_=prepare_data(s,self.well)
        expected=np.array([.7*.96/2.65,.3*.96/2.6,.04/1.25]);expected/=expected.sum()
        np.testing.assert_allclose(data[['volume_0','volume_1','volume_2']].iloc[0],expected)
        np.testing.assert_allclose(data.matrix_density,np.dot(expected,[2.65,2.6,1.25]))

    def test_invalid_rows_are_retained_and_optional_observations_missing(self):
        self.raw.loc[2,'POR']=-999.25;self.raw.loc[3,'SW']=120;self.raw.to_csv(self.path,index=False)
        s=deepcopy(self.s);s['bindings']['vs']=dict(curve='',constant=None,unit='m/s')
        result=run_workflow(s,self.well);self.assertEqual(result['summary']['rows'],10);self.assertEqual(result['summary']['invalid_rows'],2)
        columns=result['table']['columns'];rows=result['table']['rows']
        self.assertIsNone(rows[2][columns.index('VP_RP')]);self.assertTrue(rows[2][columns.index('qc_reason')]);self.assertNotIn('vs',result['metrics']['all'])
        json.dumps(result,allow_nan=False);self.assertEqual(result['parameters']['scheme']['bindings']['phi']['unit'],'%')

    def test_scan_recovers_known_parameter_and_ignores_validation_targets(self):
        truth=deepcopy(self.s);truth['parameters']['critical_porosity']=.5;data,_=prepare_data(truth,self.well);pred=forward(data,truth)
        data['vp']=pred.VP_RP;data['vs']=pred.VS_RP;data['rho']=pred.DEN_RP
        s=deepcopy(self.s);s['calibration'].update(mode='scan',parameters=[dict(name='critical_porosity',min=.3,max=.6,steps=4)],depth_min=100,depth_max=105,validation_min=106,validation_max=109,max_samples=4)
        fitted,report=calibrate(data,s);self.assertAlmostEqual(fitted['parameters']['critical_porosity'],.5)
        self.assertLess(report['score'],1e-20);self.assertEqual(report['sample_count'],4)
        data.loc[data.Depth>=106,['vp','vs','rho']]=100000
        other,_=calibrate(data,s);self.assertEqual(fitted['parameters'],other['parameters'])
        s['calibration']['mode']='optimize';s['calibration']['max_samples']=10
        fitted,report=calibrate(data,s);self.assertAlmostEqual(fitted['parameters']['critical_porosity'],.5,places=3)
        self.assertTrue(report['converged'])

    def test_validation_overlap_and_layer_override_are_rejected(self):
        s=deepcopy(self.s);s['calibration'].update(mode='scan',parameters=[dict(name='critical_porosity',min=.3,max=.6,steps=4)],depth_min=100,depth_max=105,validation_min=105,validation_max=109)
        data,_=prepare_data(s,self.well)
        with self.assertRaisesRegex(ValueError,'重叠'):calibrate(data,s)
        s['calibration']['validation_min']=106;s['layers']=[{'from':100,'to':105,'parameters':{'critical_porosity':.4}}]
        with self.assertRaisesRegex(ValueError,'分层覆盖'):calibrate(data,s)

    def test_two_parameter_dem_scan_and_uninformative_density_rejected(self):
        s=deepcopy(self.s);s['dry_model']='dual_dem';s['parameters']['soft_fraction']=.2
        data,_=prepare_data(s,self.well);truth=forward(data,s);data['vp']=truth.VP_RP;data['vs']=truth.VS_RP;data['rho']=truth.DEN_RP
        s['parameters']['soft_fraction']=.15
        s['calibration'].update(mode='scan',parameters=[dict(name='soft_fraction',min=.1,max=.3,steps=3),dict(name='alpha_soft',min=.005,max=.01,steps=2)],depth_min=100,depth_max=105,max_samples=4)
        fitted,report=calibrate(data,s)
        self.assertEqual(len(report['scan']),6);self.assertAlmostEqual(fitted['parameters']['soft_fraction'],.2)
        self.assertAlmostEqual(fitted['parameters']['alpha_soft'],.01)
        s['calibration']['weights']=dict(vp=0,vs=0,rho=1)
        with self.assertRaisesRegex(ValueError,'缺乏敏感性'):calibrate(data,s)

    def test_layers_have_explicit_boundaries_and_no_overlap(self):
        s=deepcopy(self.s);s['layers']=[{'from':100,'to':105,'parameters':{'critical_porosity':.3}},{'from':105,'to':109,'parameters':{'critical_porosity':.5}}]
        data,_=prepare_data(s,self.well);r=forward(data,s);self.assertAlmostEqual(r.K_dry.iloc[5]/r.K_matrix.iloc[5],1-r.phi.iloc[5]/.5)
        self.assertAlmostEqual(r.K_dry.iloc[-1]/r.K_matrix.iloc[-1],1-r.phi.iloc[-1]/.5)
        s['layers'][1]['from']=104
        with self.assertRaisesRegex(ValueError,'重叠'):validate_scheme(s)

    def test_save_attach_is_selective_and_preserves_sources(self):
        original=self.path.read_bytes();result=run_workflow(self.s,self.well);path=save_result(result,self.root/'results');before=deepcopy(self.well)
        wells=attach_curves([self.well],path,result,{'VP_RP':'VP_model'})
        self.assertEqual(self.well,before);self.assertEqual(self.path.read_bytes(),original)
        ref=wells[0]['curves'][-1];self.assertEqual(ref['role'],'derived');self.assertEqual(ref['column'],'VP_RP')
        with self.assertRaisesRegex(ValueError,'重复'):attach_curves(wells,path,result,{'VP_RP':'VP_model'})
        self.assertEqual(json.loads((path/'parameters.json').read_text(encoding='utf-8'))['parameters']['adopted_scheme']['name'],self.s['name'])

    def test_worker_preview_and_model_protocol(self):
        root=Path(__file__).resolve().parents[1]
        for operation in ('well_preview','well_log','well_live'):
            request=dict(operation=operation,scheme=self.s,well=self.well,output_dir=str(self.root/'worker'))
            completed=subprocess.run([sys.executable,'-m','overpressure.rock_physics.worker'],input=json.dumps(request),capture_output=True,text=True,encoding='utf-8',cwd=root,timeout=30)
            self.assertEqual(completed.returncode,0,completed.stderr);events=[json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual(events[-1]['event'],dict(well_preview='preview',well_log='completed',well_live='live_preview')[operation])
            if operation=='well_log':self.assertTrue((Path(events[-1]['path'])/'result.json').is_file())
            if operation=='well_live':self.assertNotIn('path',events[-1]);self.assertIsNotNone(events[-1]['result']['preview'])

    def test_live_preview_is_bounded_forward_only_and_keeps_hidden_gaps(self):
        from overpressure.rock_physics.well_model import preview_sample
        frame=pd.DataFrame(dict(Depth=np.arange(1000.),model_valid=np.ones(1000,bool),vp=np.ones(1000),vs=np.ones(1000),rho=np.ones(1000)))
        frame.loc[501,'model_valid']=False;frame.loc[600,'vp']=np.nan
        sample,meta=preview_sample(frame,8)
        self.assertEqual(len(sample),8);self.assertTrue(meta['breaks']['model']);self.assertTrue(meta['breaks']['vp'])
        frame['model_valid']=False;frame.loc[501,'model_valid']=True
        sample,meta=preview_sample(frame,8);self.assertTrue(sample.model_valid.any())
        s=deepcopy(self.s);s['calibration']['mode']='optimize'
        result=run_workflow(s,self.well,preview_limit=8)
        self.assertEqual(result['calibration']['mode'],'none');self.assertEqual(len(result['table']['rows']),8)
        self.assertEqual(s['calibration']['mode'],'optimize')
        with self.assertRaisesRegex(ValueError,'实时预览'):save_result(result,self.root/'not_saved')
        with self.assertRaisesRegex(ValueError,'实时预览'):attach_curves([self.well],self.root,result,{'VP_RP':'bad'})
        self.assertFalse((self.root/'not_saved').exists())


if __name__=='__main__':unittest.main()

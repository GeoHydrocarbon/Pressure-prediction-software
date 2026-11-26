"""Numerical invariants independent of upstream snapshots of known bugs."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from overpressure.rock_physics import catalog,calculate,save_result


class RockPhysicsTests(unittest.TestCase):
    def values(self,result):return [np.asarray(output['values']) for output in result['outputs']]

    def test_catalog_is_lazy_unique_and_documents_units(self):
        models=catalog();self.assertEqual(len({m['id'] for m in models}),len(models));self.assertGreaterEqual(len(models),110)
        self.assertEqual(len({m['category'] for m in models}),10)
        velocity=next(m for m in models if m['id']=='utils.V')
        self.assertEqual(next(p for p in velocity['parameters'] if p['name']=='rho')['unit'],'g/cc')
        process=subprocess.run([sys.executable,'-c',"from overpressure.rock_physics import catalog; catalog(); import sys; assert not {'torch','scipy','matplotlib','kde_diffusion','rockphypy'} & set(sys.modules)"],capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertNotIn('EM.DEM',{m['id'] for m in models})

    def test_all_explicit_presets_produce_finite_reproducible_results(self):
        examples=[m for m in catalog() if m['validated_example']]
        self.assertEqual(len(examples),44)
        for model in examples:
            with self.subTest(model=model['id']):
                result=calculate(model['id'],model['example']);self.assertEqual(result['status'],'completed')
                self.assertTrue(result['outputs']);json.dumps(result,allow_nan=False)
                for value in self.values(result):self.assertTrue(np.isfinite(value).all())
                repeated=calculate(model['id'],result['parameters'])
                for a,b in zip(self.values(result),self.values(repeated)):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

    def test_velocities_and_moduli_roundtrip_in_declared_units(self):
        vp,vs=self.values(calculate('utils.V',dict(K=37.,G=44.,rho=2.65)))
        np.testing.assert_allclose([vp,vs],[6008.379892351815,4074.7728261714988],rtol=1e-12)
        k,g=self.values(calculate('utils.M_from_V',dict(den=2.65,vp=float(vp),vs=float(vs))))
        np.testing.assert_allclose([k,g],[37,44],rtol=1e-12)

    def test_vrh_analytical_bounds_and_optional_defaults(self):
        result=calculate('EM.VRH',dict(volumes=[.7,.3],M=[37.,20.9]))
        v,r,h=self.values(result);self.assertAlmostEqual(float(v),.7*37+.3*20.9)
        self.assertAlmostEqual(float(r),1/(.7/37+.3/20.9));self.assertAlmostEqual(float(h),(float(v)+float(r))/2)
        hs=calculate('EM.HS',dict(f=.5,K1=37.,K2=20.9,G1=44.,G2=6.85))
        self.assertEqual(hs['parameters']['bound'],'upper')

    def test_empirical_and_permeability_declared_units(self):
        vp,vs=self.values(calculate('Empirical.han',dict(phi=.1,C=.2)))
        np.testing.assert_allclose([vp,vs],[4335.,2651.],rtol=1e-12)
        result=calculate('Permeability.Kozeny_Carman',dict(phi=.2,d=1e-4))
        self.assertEqual(result['outputs'][0]['unit'],'m²')
        self.assertAlmostEqual(result['outputs'][0]['values'],1e-8/180*.2**3/(1-.2)**2)

    def test_gassmann_manual_formula_and_same_fluid_identity(self):
        result=calculate('Fluid.Gassmann',dict(K_dry=15.,G_dry=12.,K_mat=37.,Kf=2.2,phi=.2))
        k,g=self.values(result);expected=15+(1-15/37)**2/(.2/2.2+.8/37-15/37**2)
        self.assertAlmostEqual(float(k),expected);self.assertEqual(float(g),12.)
        k2=self.values(calculate('Fluid.Gassmann_sub',dict(phi=.2,K0=37.,Ksat1=20.,Kfl1=2.2,Kfl2=2.2)))[0]
        self.assertAlmostEqual(float(k2),20.)
        vp,vs=self.values(calculate('Fluid.Gassmann_vels',dict(Vp1=3500.,Vs1=2000.,rho1=2.25,rhofl1=1.,Kfl1=2.2,rhofl2=1.,Kfl2=2.2,K0=37.,phi=.2)))
        np.testing.assert_allclose([vp,vs],[3500.,2000.],rtol=1e-12)

    def test_dem_exact_endpoint_and_no_porosity_host(self):
        for phi in (0.,.105,.2):
            with self.subTest(phi=phi):
                k,g,p=self.values(calculate('EM.Berryman_DEM',dict(Km=37.,Gm=44.,Ki=0.,Gi=0.,alpha=.1,phi=phi)))
                self.assertEqual(float(p[-1]),phi);self.assertFalse((p>phi).any())
                self.assertEqual(k[0],37.);self.assertEqual(g[0],44.)
                self.assertTrue((np.diff(k)<=1e-8).all());self.assertTrue((np.diff(g)<=1e-8).all())

    def test_zoeppritz_matches_independent_boundary_equations(self):
        vp1,vs1,rho1,vp2,vs2,rho2=3000.,1600.,2200.,3500.,1900.,2400.
        for degrees in (0.,10.,20.,30.):
            theta1=np.deg2rad(degrees);p=np.sin(theta1)/vp1;theta2=np.arcsin(p*vp2);phi1=np.arcsin(p*vs1);phi2=np.arcsin(p*vs2)
            matrix=np.array([
                [-np.sin(theta1),-np.cos(phi1),np.sin(theta2),np.cos(phi2)],
                [np.cos(theta1),-np.sin(phi1),np.cos(theta2),-np.sin(phi2)],
                [2*rho1*vs1*np.sin(phi1)*np.cos(theta1),rho1*vs1*(1-2*np.sin(phi1)**2),2*rho2*vs2*np.sin(phi2)*np.cos(theta2),rho2*vs2*(1-2*np.sin(phi2)**2)],
                [-rho1*vp1*(1-2*np.sin(phi1)**2),rho1*vs1*np.sin(2*phi1),rho2*vp2*(1-2*np.sin(phi2)**2),-rho2*vs2*np.sin(2*phi2)]])
            rhs=np.array([np.sin(theta1),np.cos(theta1),2*rho1*vs1*np.sin(phi1)*np.cos(theta1),rho1*vp1*(1-2*np.sin(phi1)**2)])
            expected=np.linalg.solve(matrix,rhs)[:2]
            actual=self.values(calculate('AVO.zoeppritz',dict(vp1=vp1,vs1=vs1,rho1=rho1,vp2=vp2,vs2=vs2,rho2=rho2,theta=degrees)))
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_matrix_results_and_invalid_physical_inputs(self):
        c=self.values(calculate('utils.write_iso',dict(K=37.,G=44.)))[0]
        rotated=self.values(calculate('Anisotropy.Bond_trans',dict(C=c.tolist(),theta=37.,axis=3)))[0]
        np.testing.assert_allclose(c,rotated,atol=1e-12)
        anisotropic=c.copy();anisotropic[0,0]+=10;anisotropic[1,1]+=5
        rotated=self.values(calculate('Anisotropy.Bond_trans',dict(C=anisotropic.tolist(),theta=23.,axis=1)))[0]
        restored=self.values(calculate('Anisotropy.Bond_trans',dict(C=rotated.tolist(),theta=-23.,axis=1)))[0]
        np.testing.assert_allclose(anisotropic,restored,atol=1e-12)
        failures=[('utils.V',dict(K=True,G=44.,rho=2.65)),('utils.V',dict(K='37',G=44.,rho=2.65)),
                  ('utils.V',dict(K=-1.,G=44.,rho=2.65)),('utils.V',dict(K=37.,G=44.,rho=0.)),
                  ('EM.VRH',dict(volumes=[70,30],M=[37.,20.9])),('EM.VRH',dict(volumes=[.5,.5],M=[37.])),
                  ('EM.cripor',dict(K0=37.,G0=44.,phi=.6,phic=.4)),
                  ('Fluid.Gassmann',dict(K_dry=50.,G_dry=12.,K_mat=37.,Kf=2.2,phi=.2)),
                  ('utils.V',dict(K=float('nan'),G=44.,rho=2.65))]
        failures.append(('utils.V',dict(K=[True,37.],G=44.,rho=2.65)))
        failures.append(('Anisotropy.Bond_trans',dict(C=c.tolist(),theta=30.,axis=2)))
        for model,parameters in failures:
            with self.subTest(model=model,parameters=parameters),self.assertRaises(ValueError):calculate(model,parameters)
        with self.assertRaisesRegex(ValueError,'未知'):calculate('__import__',{})
        with self.assertRaisesRegex(ValueError,'未知参数'):calculate('utils.V',dict(K=37.,G=44.,rho=2.65,unexpected=1))

    def test_worker_and_saved_results_are_complete_and_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            request=dict(model_id='EM.cripor',parameters=next(m['example'] for m in catalog() if m['id']=='EM.cripor'),output_dir=temporary)
            process=subprocess.run([sys.executable,'-m','overpressure.rock_physics.worker'],input=json.dumps(request),capture_output=True,text=True,encoding='utf-8',timeout=40)
            self.assertEqual(process.returncode,0,process.stderr)
            event=json.loads(process.stdout);self.assertEqual(event['event'],'completed');path=Path(event['path'])
            self.assertTrue(path.is_relative_to(Path(temporary).resolve()))
            saved=json.loads((path/'result.json').read_text(encoding='utf-8'));self.assertEqual(saved,event['result'])
            with (path/'results.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.reader(stream));self.assertEqual(rows[0],saved['table']['columns']);self.assertEqual(len(rows)-1,len(saved['table']['rows']))
            request['model_id']='invalid';process=subprocess.run([sys.executable,'-m','overpressure.rock_physics.worker'],input=json.dumps(request),capture_output=True,text=True,encoding='utf-8',timeout=40)
            self.assertNotEqual(process.returncode,0);self.assertEqual(json.loads(process.stdout)['event'],'error')
            self.assertEqual(len(list(Path(temporary).iterdir())),1)

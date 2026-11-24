"""Analytical Eaton values, raw-row retention, unit contracts, and complete artifacts."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from overpressure.config import Project
from overpressure.jobs import Cancelled
from overpressure.traditional import (defaults, two_point_trend, calculate_eaton,
                                     analyze_trend, predict_eaton, WellInputs, validate_prediction)
from overpressure.well_library import inspect_curve_file


class TraditionalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        wells = []
        for name in ('A', 'B'):
            path = self.folder / (name + '.csv')
            pd.DataFrame(dict(Depth=[1000.,1001.,1002.,1010.],AC=[100.,101.,np.nan,105.],Pp=[20.,21.,np.nan,24.],
                              岩性=['泥岩','砂岩','泥岩','砂质泥岩'])).to_csv(path,index=False)
            wells.extend(inspect_curve_file(path, definitions=[dict(column='AC',name='AC',unit='us/ft'),dict(column='Pp',name='Pp',unit='MPa')]))
        cfg=defaults({})
        cfg.update(analysis_well='A',analysis_min=900.,analysis_max=1100.,points=[[900.,np.log(50.)],[1100.,np.log(50.)]],wells=['A','B'])
        trend=dict(two_point_trend(cfg['points']),confirmed=True,source_well='A')
        cfg['applied_trends']={name:deepcopy(trend) for name in ('A','B')}
        self.project=Project(self.folder/'project.yaml',dict(output_dir='outputs',project_data=dict(wells=wells),traditional_prediction=cfg))
        self.cfg=cfg

    def test_two_point_equation_and_invalid_calibration(self):
        result=two_point_trend([[1000,4.5],[3000,3.8]])
        self.assertAlmostEqual(result['a'],-.00035)
        self.assertAlmostEqual(result['b'],4.85)
        for points in ([[1,4],[1,5]],[[1,np.inf],[2,3]],[[-1,4],[2,3]]):
            with self.assertRaises(ValueError):two_point_trend(points)

    def test_eaton_matches_hand_calculation_and_does_not_clip(self):
        self.cfg['eaton_exponent']=2.
        self.cfg['coefficient_min']=.8;self.cfg['coefficient_max']=1.2
        frame=pd.DataFrame(dict(well=['A']*4,Depth=[1000.,1000.,0.,1000.],AC=[100.,10.,100.,0.],lithology=['泥岩']*4))
        result=calculate_eaton(frame,two_point_trend(self.cfg['points']),self.cfg)
        expected=24.01-(24.01-10.094)*.25
        self.assertAlmostEqual(result.Pp_eaton.iloc[0],expected)
        self.assertAlmostEqual(result.pressure_coefficient.iloc[0],expected/10.094)
        self.assertIn('coefficient_outside_range',result.quality_flags.iloc[0])
        self.assertLess(result.Pp_eaton.iloc[1],0)
        self.assertIn('negative_pressure',result.quality_flags.iloc[1])
        self.assertEqual(result.Pp_eaton.iloc[2],0)
        self.assertTrue(np.isnan(result.pressure_coefficient.iloc[2]))
        self.assertTrue(np.isnan(result.Pp_eaton.iloc[3]))
        self.assertEqual(len(result),4)

    def test_explicit_rock_and_depth_selection_keep_excluded_rows(self):
        inputs=WellInputs(self.project);frame=inputs.ac('A','AC')
        cfg=deepcopy(self.cfg);cfg.update(lithology_filter='mudstone',depth_max=1005.)
        result=calculate_eaton(frame,two_point_trend(cfg['points']),cfg)
        self.assertEqual(result.calculation_status.tolist(),['calculated','excluded_lithology','invalid_ac_or_depth','outside_depth_range'])
        self.assertEqual(result.prediction_valid.sum(),1)
        frame.attrs['has_lithology']=False
        with self.assertRaisesRegex(ValueError,'岩性列'):calculate_eaton(frame,two_point_trend(cfg['points']),cfg)

    def test_normal_compaction_pressure_is_hydrostatic_without_roundoff_flags(self):
        depth=np.array([1000.,1001.,1002.])
        trend=two_point_trend([[1000,4.5],[3000,3.8]])
        normal=np.exp(trend['a']*depth+trend['b'])
        frame=pd.DataFrame(dict(Depth=depth,AC=normal*(1-1e-15)))
        result=calculate_eaton(frame,trend,self.cfg)
        np.testing.assert_allclose(result.Pp_eaton,result.P_h)
        self.assertTrue(result.quality_flags.eq('').all())

    def test_curve_plots_break_at_missing_samples_and_large_depth_gaps(self):
        from overpressure.traditional_figures import eaton_figure
        frame=WellInputs(self.project).ac('A','AC')
        result=calculate_eaton(frame,two_point_trend(self.cfg['points']),self.cfg)
        figure=eaton_figure(result,None,'A')
        line=figure.axes[1].lines[2]
        self.assertEqual(np.isfinite(line.get_xdata()).sum(),3)
        self.assertTrue(np.isnan(line.get_ydata()).any(), 'Unsampled interval must break the line')

    def test_only_depth_and_ac_are_required_and_units_are_normalized(self):
        path=self.folder/'C.csv'
        pd.DataFrame(dict(Depth=[1000/.3048,1001/.3048],AC=[100/.3048,101/.3048])).to_csv(path,index=False)
        wells=inspect_curve_file(path,depth_unit='ft',definitions=[dict(column='AC',name='AC',unit='us/m')])
        self.project.settings['project_data']['wells']=wells
        frame=WellInputs(self.project).ac('C','AC')
        np.testing.assert_allclose(frame.Depth,[1000,1001]);np.testing.assert_allclose(frame.AC,[100,101])
        cfg=self.project.settings['traditional_prediction'];cfg.update(wells=['C'],analysis_well='C',applied_trends={'C':dict(two_point_trend(cfg['points']),confirmed=True)})
        with redirect_stdout(io.StringIO()):out=predict_eaton(self.project)
        meta=json.loads((out/'traditional.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['matched_points'],0)
        self.assertEqual(meta['valid_points'],2)

    def test_batch_outputs_preserve_rows_original_sources_and_reference_points(self):
        source=self.folder/'A.csv';before=source.read_bytes()
        with redirect_stdout(io.StringIO()):
            trend=analyze_trend(self.project);run=predict_eaton(self.project)
        trend_meta=json.loads((trend/'traditional.json').read_text(encoding='utf-8'))
        self.assertEqual(trend_meta['valid_points'],3)
        self.assertEqual(trend_meta['status'],'completed')
        meta=json.loads((run/'traditional.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['status'],'completed');self.assertEqual(len(meta['per_well_files']),2)
        self.assertEqual(len(pd.read_csv(run/'predictions.csv')),8)
        self.assertEqual(meta['matched_points'],6)
        self.assertEqual(source.read_bytes(),before)
        self.assertTrue((run/'eaton_001.png').stat().st_size>1000)

    def test_separate_sparse_measurements_take_priority_and_do_not_cross_gaps(self):
        path=self.folder/'measured.csv'
        pd.DataFrame(dict(Depth=[1000.5,1005.,999.],Pp=[20000.,21000.,22000.])).to_csv(path,index=False)
        incoming=inspect_curve_file(path,forced_well='A',definitions=[dict(column='Pp',name='实测压力',unit='kPa',role='measurement')])
        from overpressure.well_library import merge_wells
        self.project.settings['project_data']['wells']=merge_wells(self.project.settings['project_data']['wells'],incoming)
        self.cfg['wells']=['A']
        with redirect_stdout(io.StringIO()):run=predict_eaton(self.project)
        points=pd.read_csv(run/'reference_points.csv')
        self.assertEqual(len(points),3)
        self.assertEqual(points.match_status.tolist(),['matched_interpolated','gap','outside_range'])
        self.assertEqual(points.Pp.tolist(),[20.,21.,22.])

    def test_missing_trend_and_invalid_physics_fail_before_outputs(self):
        cfg=deepcopy(self.cfg);cfg['applied_trends']={}
        with self.assertRaisesRegex(ValueError,'采用'):validate_prediction(cfg)
        cfg=deepcopy(self.cfg);cfg['rock_density']=1.
        with self.assertRaisesRegex(ValueError,'岩石密度'):validate_prediction(cfg)
        self.project.settings['traditional_prediction']['applied_trends']={}
        with self.assertRaises(ValueError):predict_eaton(self.project)
        self.assertFalse((self.folder/'outputs').exists())

    def test_cancelled_exports_never_mark_a_run_complete(self):
        with patch('overpressure.traditional_figures.eaton_figure',side_effect=Cancelled('cancelled')):
            with self.assertRaises(Cancelled),redirect_stdout(io.StringIO()):predict_eaton(self.project)
        report=next((self.folder/'outputs'/'traditional'/'predictions').glob('*/traditional.json'))
        self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['status'],'cancelled')


if __name__=='__main__':unittest.main()

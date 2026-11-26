import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from overpressure.seismic_browser import SeismicDataset
from overpressure.well_overlays import project_points,curve_id
from overpressure.section_overlays import render_overlays,default_style,well_key,curve_key
from overpressure.time_depth_library import relation
from overpressure.config import Project


class WellOverlayTests(unittest.TestCase):
    def setUp(self):
        from tests.test_seismic_browser import SeismicBrowserTests
        fixture=SeismicBrowserTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.folder=fixture.path.parent;self.config=fixture.config
        heads=self.folder/'heads.csv';pd.DataFrame(dict(well=['A','B','C'],X=[300525,300550,300525],Y=[4000250,4000250,4000275])).to_csv(heads,index=False)
        track=self.folder/'track.csv';pd.DataFrame(dict(well=['A','A'],MD=[1000,1018],X=[300525,300540],Y=[4000250,4000250])).to_csv(track,index=False)
        logs=self.folder/'logs.csv';pd.DataFrame(dict(well=['A']*10,Depth=np.arange(1000,1020,2),AC=np.arange(10)*2+80,Pp=np.arange(10)+20)).to_csv(logs,index=False)
        self.td=relation(Project(self.folder/'project.yaml',{}),dict(method='polynomial',coefficients=[0.,1000.],time_unit='s',time_convention='twt',depth_unit='m'))
        self.refs=[dict(path=str(logs),name=name,column=name,well_column='well',well_value='A',depth_column='Depth',depth_unit='m',unit=unit,role=role) for name,unit,role in [('AC','us/ft','log'),('Pp','MPa','measurement')]]
        self.ids=[curve_id('A',ref) for ref in self.refs]
        self.config.update(well_assets=[dict(path=str(heads),kind='coordinates',columns=dict(well='well',x='X',y='Y')),dict(path=str(track),kind='trajectory',columns=dict(well='well',depth='MD',x='X',y='Y'))],
            well_library=[dict(name='A',curves=self.refs,time_depth_id=self.td['id']),dict(name='B',curves=[]),dict(name='C',curves=[])],time_depth_relations=[self.td],time_convention='twt')

    def reader(self):
        ds=SeismicDataset(self.config);self.addCleanup(ds.close);return ds

    def test_map_has_trajectory_and_curves_are_loaded_only_when_selected(self):
        ds=self.reader();context=ds.well_context()
        with patch.object(context,'curve_values',wraps=context.curve_values) as read:
            wells,warnings=ds.well_locations();self.assertFalse(warnings)
            self.assertEqual(len(wells),3);self.assertEqual(wells[0]['trajectory_xy'],[[300525.,4000250.],[300540.,4000250.]])
            np.testing.assert_allclose(wells[0]['trajectory_lines'],[[21,10],[21.6,10]],atol=1e-8)
            _,meta=ds.section('inline',position=10,coordinate_mode='xy');read.assert_not_called()
            a=meta['well_overlays'][0];self.assertEqual(len(a['curves']),2);self.assertEqual(a['curve_data'],[])
            np.testing.assert_allclose([a['track']['time'][0],a['track']['time'][-1]],[1000,1018],atol=1e-8)
            _,meta=ds.section('inline',position=10,coordinate_mode='xy',overlay_curves=self.ids);self.assertEqual(read.call_count,2)
            self.assertEqual(len(meta['well_overlays'][0]['curve_data']),2)
            json.dumps(meta,allow_nan=False)

    def test_fallback_vertical_corridor_and_independent_styles_keep_extent(self):
        ds=self.reader();section,meta=ds.section('inline',position=10,coordinate_mode='xy',overlay_curves=self.ids)
        styles={well_key('A'):dict(default_style('well'),color='red',linewidth=3),curve_key(self.ids[0]):dict(default_style('curve'),visible=True,color='green',width=10),curve_key(self.ids[1]):dict(default_style('curve','points'),visible=True,marker='s',markersize=7)}
        figure=Figure();ax=figure.subplots();ax.set_xlim(20,23);ax.set_ylim(1018,1000)
        render_overlays(ax,meta['well_overlays'],dict(distance=1.,styles=styles),[1000,1018])
        self.assertEqual(ax.get_xlim(),(20,23));self.assertEqual(ax.get_ylim(),(1018,1000))
        self.assertEqual({text.get_text() for text in ax.texts},{'A','B'})
        self.assertEqual(ax.lines[0].get_color(),'red');self.assertEqual(ax.lines[0].get_linewidth(),3)
        vertical=next(line for line in ax.lines if len(line.get_xdata())==2)
        np.testing.assert_allclose(vertical.get_xdata(),[22,22],atol=1e-8)
        self.assertEqual(len(ax.collections),1);self.assertEqual(ax.collections[0].get_sizes()[0],49)

    def test_polyline_projection_and_owt_conversion(self):
        x,d=project_points([[5,1],[11,5],[-5,0]],[[0,0],[10,0],[10,10]],[0,10,20])
        np.testing.assert_allclose(x[:2],[5,15]);np.testing.assert_allclose(d[:2],[1,1]);self.assertTrue(np.isnan(x[2]))
        self.td['config'].update(coefficients=[0.,2000.],time_convention='owt')
        ds=self.reader();_,meta=ds.section('polyline',coordinate_mode='xy',vertices=[[300500,4000250],[300550,4000250],[300550,4000300]],overlay_curves=[self.ids[0]])
        a=meta['well_overlays'][0];np.testing.assert_allclose([a['track']['time'][0],a['track']['time'][-1]],[1000,1018],atol=1e-8)
        self.assertAlmostEqual(a['head_x'],25)

    def test_missing_binding_uses_vertical_and_skips_depth_curves(self):
        self.config['well_library'][0].pop('time_depth_id')
        ds=self.reader();_,meta=ds.section('inline',position=10,overlay_curves=[self.ids[0]])
        a=meta['well_overlays'][0];self.assertIsNone(a['track']);self.assertEqual(a['curve_data'],[])
        self.assertIn('缺少已应用',meta['overlay_warnings'][0])

    def test_sparse_trajectory_intersection_selects_whole_projected_well(self):
        path=Path(self.config['well_assets'][1]['path'])
        pd.DataFrame(dict(well=['A','A'],MD=[1000,1018],X=[300525,300525],Y=[4000240,4000280])).to_csv(path,index=False)
        ds=self.reader();_,meta=ds.section('inline',position=10,coordinate_mode='xy')
        well=meta['well_overlays'][0]
        self.assertLess(well['geometry_distance'],1e-8)
        self.assertTrue(any(abs(t-1004.5)<1e-7 for t in well['track']['time']))
        figure=Figure();ax=figure.subplots();ax.set_xlim(20,23);ax.set_ylim(1018,1000)
        render_overlays(ax,[well],dict(distance=0,styles={}),[1000,1018])
        self.assertTrue(np.isfinite(ax.lines[0].get_xdata()).all())

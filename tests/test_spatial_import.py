"""Physical row parsing and independently known directional-survey geometries."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from overpressure.spatial_table import read_spatial_table,preview_source
from overpressure.well_spatial import read_asset,prepare_wells,WellMap
from overpressure.directional_survey import minimum_curvature


class SpatialImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)

    def test_txt_preamble_gbk_blank_rows_and_leading_zero_names(self):
        path=self.folder/'井坐标.txt';path.write_text('说明：测试坐标\n井名\tX\tY\n单位：m\n001\t10\t20\n\n002\t30\t40\n',encoding='gb18030')
        options=dict(header_row=2,data_row=4,delimiter='\t')
        raw=read_spatial_table(path,options)
        self.assertEqual(raw['井名'].tolist(),['001','002']);self.assertEqual(raw.attrs['source_rows'],[4,6])
        self.assertEqual(preview_source(path)[0],(1,'说明：测试坐标'))
        _,frame=read_asset(dict(path=str(path),kind='coordinates',columns=dict(well='井名',x='X',y='Y'),table_import=options))
        self.assertEqual(frame.source_row.tolist(),[4,6]);np.testing.assert_array_equal(frame[['x','y']],[[10,20],[30,40]])

    def test_auto_csv_semicolon_whitespace_and_headerless(self):
        for separator in (',',';','\t','  '):
            path=self.folder/'coord.txt';path.write_text('preamble\n'+separator.join(['well','X','Y'])+'\n'+separator.join(['A','1','2'])+'\n',encoding='utf-8')
            raw=read_spatial_table(path,dict(header_row=2,data_row=3));self.assertEqual(raw.iloc[0].tolist(),['A','1','2'])
        path.write_text('A 1 2\nB 3 4\n',encoding='utf-8')
        raw=read_spatial_table(path,dict(header_row=0,data_row=1));self.assertEqual(list(raw),['Column_1','Column_2','Column_3'])
        self.assertEqual(raw.attrs['source_rows'],[1,2])
        with self.assertRaisesRegex(ValueError,'表头行之后'):read_spatial_table(path,dict(header_row=2,data_row=2))

    def test_excel_and_quoted_multiline_csv_keep_original_rows(self):
        from openpyxl import Workbook
        path=self.folder/'coords.xlsx';book=Workbook();sheet=book.active
        for row in [['说明'],['well','X','Y'],['单位'],['001',1,2],[],['002',3,4]]:sheet.append(row)
        book.save(path);book.close()
        raw=read_spatial_table(path,dict(header_row=2,data_row=4));self.assertEqual(raw.attrs['source_rows'],[4,6])
        self.assertEqual(preview_source(path)[1],(2,'well | X | Y'))
        path=self.folder/'quoted.csv';path.write_text('well,X,Y\n"A\nB",1,2\nC,3,4\n',encoding='utf-8')
        raw=read_spatial_table(path);self.assertEqual(raw.attrs['source_rows'],[2,4])

    def test_minimum_curvature_vertical_tangent_build_and_north_wrap(self):
        np.testing.assert_allclose(minimum_curvature([0,100,200],[0,0,0],[0,0,0]),[[0,0,0],[0,0,100],[0,0,200]],atol=1e-10)
        tangent=minimum_curvature([0,100],[30,30],[90,90]);np.testing.assert_allclose(tangent[-1],[50,0,100*np.sqrt(3)/2],atol=1e-10)
        build=minimum_curvature([0,100],[0,90],[0,0]);np.testing.assert_allclose(build[-1],[0,200/np.pi,200/np.pi],atol=1e-10)
        wrap=minimum_curvature([0,100],[30,30],[359,1]);self.assertAlmostEqual(wrap[-1,0],0,places=10);self.assertGreater(wrap[-1,1],49)
        rad=minimum_curvature([0,100],[0,np.pi/2],[0,0],angle_unit='rad');np.testing.assert_allclose(rad,build)

    def test_invalid_survey_and_explicit_first_segment(self):
        for md,inc,azi in [([0,0],[0,0],[0,0]),([10,0],[0,0],[0,0]),([0,100],[0,np.nan],[0,0]),([0,100],[0,181],[0,0]),([0,100],[0,0],[0,361]),([0,100],[0,180],[0,0])]:
            with self.subTest(md=md,inc=inc,azi=azi),self.assertRaises(ValueError):minimum_curvature(md,inc,azi)
        with self.assertRaisesRegex(ValueError,'首个测点'):minimum_curvature([100,200],[30,30],[90,90])
        got=minimum_curvature([100,200],[30,30],[90,90],initial_segment='first_direction')
        np.testing.assert_allclose(got[:,0],[50,100]);np.testing.assert_allclose(got[:,2],[100*np.sqrt(3)/2,100*np.sqrt(3)])

    def test_multiwell_filter_units_and_actual_mapping_to_seismic(self):
        path=self.folder/'survey.txt';path.write_text('说明\nwell MD AZI INC\nA 0 90 30\nB 0 0 0\nA 328.0839895013123 90 30\nB 328.0839895013123 0 0\n',encoding='utf-8')
        survey=dict(path=str(path),kind='trajectory',trajectory_mode='survey',columns=dict(well='well',depth='MD',azimuth='AZI',inclination='INC'),depth_unit='ft',table_import=dict(header_row=2,data_row=3))
        _,frame=read_asset(survey);np.testing.assert_allclose(frame.x,[0,0,50,0],atol=1e-9)
        self.assertEqual(frame.source_row.tolist(),[3,4,5,6])
        _,only_a=read_asset(dict(survey,well_name='A'));self.assertEqual(only_a.source_row.tolist(),[3,5])
        coords=self.folder/'heads.txt';coords.write_text('well X Y\nA 100 200\nB 400 500\n',encoding='utf-8')
        logs=self.folder/'logs.csv';pd.DataFrame(dict(well=['A','A'],MD=[0,100],time=[1,1.001],value=[1,2])).to_csv(logs,index=False)
        class Reader:
            index=SimpleNamespace(path=self.folder/'index.npz',inline=np.array([1,1,1]),crossline=np.array([1,2,3]),x=np.array([100.,150.,999.]),y=np.array([200.,200.,999.]),coordinate_units=np.array([1,1,1]),time_convention='twt',t0_ms=1000,interval_ms=1,offsets=np.array([0,0,0]),ns=2)
            def values(self,row,samples):
                out=np.array([10.,20.])[samples];self.last_request['sample_values_read']+=len(out);return out
        config=dict(assets=[survey,dict(path=str(coords),kind='coordinates',columns=dict(well='well',x='X',y='Y')),dict(path=str(logs),kind='curves',columns=dict(well='well',depth='MD'),time_column='time',time_unit='s',time_convention='twt')],max_distance=1)
        result=prepare_wells(Reader(),config);mapping=WellMap.load(result['path'],Reader.index)
        np.testing.assert_allclose(mapping.x,[100,150],atol=1e-8);np.testing.assert_allclose(mapping.y,[200,200],atol=1e-8)
        np.testing.assert_array_equal(mapping.trace_row,[0,1]);np.testing.assert_allclose(mapping.seismic_pressure,[10,20])

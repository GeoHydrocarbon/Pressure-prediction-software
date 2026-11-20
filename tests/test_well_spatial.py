import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from overpressure.time_depth import TimeDepth
from overpressure.config import Project
from overpressure.well_spatial import interpolate_path, sample_well_pressure, well_overlay, WellMap, prepare_wells


class WellSpatialTests(unittest.TestCase):
    def test_inverse_roundtrip_and_outside_mask(self):
        cfg=dict(method='polynomial',coefficients=[0,1000],time_unit='s',depth_unit='m',time_convention='twt',valid_time_range=[1,2],extrapolation='mask')
        td=TimeDepth(Project(Path('project.yaml'),{'time_depth':cfg}),time_bounds_seconds=(1,2))
        np.testing.assert_allclose(td.inverse([1000,1500,2000]),[1,1.5,2])
        self.assertTrue(np.isnan(td.inverse([900,2100,np.nan])).all())

    def test_path_no_extrapolation_no_conflicting_duplicate_bridge(self):
        path=pd.DataFrame(dict(depth=[100,200,200,300],x=[0,10,11,20],y=[0,0,0,0]))
        xy=interpolate_path(path,np.array([90,100,150,200,250,300,310]))
        np.testing.assert_allclose(xy[[1,5]],[ [0,0],[20,0] ])
        self.assertTrue(np.isnan(xy[[0,2,3,4,6]]).all())

    def test_sampling_offsets_mask_bounds_and_deduplicated_reads(self):
        class Reader:
            index=SimpleNamespace(t0_ms=1000,interval_ms=2,offsets=np.array([0,1]),ns=4)
            def values(self,row,samples):
                values=np.array([[20.,22.,np.nan,26.],[30.,32.,34.,36.]])[row,samples]
                self.last_request['sample_values_read']+=len(values)
                return values
        reader=Reader()
        got=sample_well_pressure(reader,np.array([0,0,0,0,1,1,-1]),np.array([1000,1001,1003,1010,1002,1003,1000]))
        np.testing.assert_allclose(got[[0,1,4,5]],[20,21,30,31])
        self.assertTrue(np.isnan(got[[2,3,6]]).all())
        self.assertEqual(reader.last_request['sample_values_read'],5)

    def test_overlay_only_intersection_with_current_line(self):
        idx=SimpleNamespace(inline=np.array([10,11]),crossline=np.array([20,21]))
        m=WellMap(np.array(['A','A']),np.array([100.,200.]),np.array([1000.,1100.]),np.zeros(2),np.zeros(2),np.array([0,1]),['Pp'],np.array([[20.,25.]]),np.array(['Pp','Pp']),np.array([20.,25.]),np.array([21.,26.]),'')
        overlay=well_overlay(m,idx,'inline',10,'A','Pp')
        self.assertEqual(overlay['center'][0],20); self.assertTrue(np.isnan(overlay['center'][1]))
        self.assertEqual(well_overlay(m,idx,'time',10,'A','Pp'),{})

    def test_absolute_trajectory_direct_time_pressure_units_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            pd.DataFrame(dict(well=['A']*3,depth=[100,150,200],time=[1,1.001,1.002],PP=[20000,21000,22000])).to_csv(folder/'logs.csv',index=False)
            pd.DataFrame(dict(well=['A']*2,depth=[100,200],X=[0,0],Y=[0,0])).to_csv(folder/'path.csv',index=False)
            class Reader:
                index=SimpleNamespace(path=folder/'index.npz',inline=np.array([1,1]),crossline=np.array([1,2]),x=np.array([0.,25.]),y=np.array([0.,0.]),coordinate_units=np.array([1,1]),time_convention='twt',t0_ms=1000,interval_ms=1,offsets=np.array([0,0]),ns=3)
                def values(self,row,samples):
                    out=np.array([20.,21.,22.])[samples]; self.last_request['sample_values_read']+=len(out); return out
            config=dict(assets=[dict(kind='curves',path=str(folder/'logs.csv'),columns=dict(well='well',depth='depth'),time_column='time',time_unit='s',time_convention='twt',pressure_column='PP',pressure_unit='kPa'),dict(kind='trajectory',path=str(folder/'path.csv'),columns=dict(well='well',depth='depth',x='X',y='Y'))],max_distance=1)
            summary=prepare_wells(Reader(),config); mapping=WellMap.load(summary['path'],Reader.index)
            np.testing.assert_allclose(mapping.well_pressure,[20,21,22]); np.testing.assert_allclose(mapping.seismic_pressure,[20,21,22])
            self.assertEqual(summary['pressure_pairs'],3); self.assertEqual(summary['read_stats']['sample_values_read'],3)
            self.assertEqual(mapping.wells.tolist(),['A']*3)


if __name__=='__main__':unittest.main()

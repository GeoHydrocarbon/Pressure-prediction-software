"""Contracts for bounded horizon-time interpolation and provenance round trips."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,unittest
import numpy as np
from overpressure.horizon_interpolation import interpolate_times,interpolate_horizon
from overpressure.horizons import HorizonMap,geometry_hash,import_horizons

class HorizonInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        x=np.array([0,4,0,4,2,1,6]);y=np.array([0,0,4,4,2,1,6])
        self.index=SimpleNamespace(path=self.folder/'index.npz',inline=y,crossline=x,x=x.astype(float)*100,y=y.astype(float)*100,coordinate_units=np.ones(7),time_convention='twt',t0_ms=0.,interval_ms=1.,offsets=np.zeros(7,dtype=int),ns=2000)
        self.times=np.array([1000,1008,1012,1020,np.nan,np.nan,np.nan])
    def test_linear_plane_originals_preserved_and_no_extrapolation(self):
        result,kinds,report=interpolate_times(self.index,self.times,max_span=6)
        np.testing.assert_array_equal(result[:4],self.times[:4]);np.testing.assert_allclose(result[4:6],[1010,1005]);self.assertTrue(np.isnan(result[6]))
        np.testing.assert_array_equal(kinds,[1,1,1,1,2,2,0]);self.assertEqual(report['new_interpolated_points'],2);self.assertEqual(report['outside_convex_hull'],1)
        self.assertEqual(len(result),len(self.index.inline),'only existing traces may be represented')
    def test_span_is_checked_on_all_edges_and_respects_coordinate_units(self):
        result,kinds,report=interpolate_times(self.index,self.times,max_span=4)
        self.assertTrue(np.isnan(result[4:]).all());self.assertEqual(report['span_rejected'],2)
        result,_,_=interpolate_times(self.index,self.times,max_span=6,coordinates='xy');self.assertTrue(np.isnan(result[4:]).all())
        result,_,_=interpolate_times(self.index,self.times,max_span=600,coordinates='xy');self.assertTrue(np.isfinite(result[4:6]).all())
    def test_previous_interpolated_points_do_not_become_new_seeds(self):
        times=self.times.copy();times[3]=1020.;kinds=np.array([1,1,1,2,0,0,0],dtype=np.uint8)
        self.index.inline[4]=3;self.index.crossline[4]=3
        result,_,report=interpolate_times(self.index,times,max_span=100,point_types=kinds)
        self.assertTrue(np.isnan(result[4]));self.assertEqual(result[3],1020.);self.assertEqual(report['previous_interpolated_points'],1)
    def test_derived_file_reimport_keeps_point_classification(self):
        source=self.folder/'source.npz';np.savez_compressed(source,names=['Original'],times_ms=self.times[None,:],geometry=np.array(geometry_hash(self.index)))
        before=source.read_bytes()
        with patch('overpressure.volume_reader.require_fresh'):
            summary=interpolate_horizon(self.index,source,'Original',new_name='Filled',max_span=6)
        self.assertEqual(source.read_bytes(),before)
        mapped=HorizonMap.load(summary['path'],self.index);self.assertEqual(mapped.names,['Filled']);self.assertEqual((mapped.point_types==2).sum(),2)
        restored=import_horizons(self.index,summary['settings']);again=HorizonMap.load(restored['path'],self.index)
        np.testing.assert_array_equal(again.times_ms,mapped.times_ms);np.testing.assert_array_equal(again.point_types,mapped.point_types)
        self.assertTrue(restored['derived']);self.assertEqual(restored['interpolated_points'],2)
    def test_invalid_parameters_and_collinear_sources_are_rejected(self):
        with self.assertRaises(ValueError):interpolate_times(self.index,self.times,max_span=0)
        self.index.inline[:4]=0
        self.index.crossline[:4]=np.arange(4)
        with self.assertRaisesRegex(ValueError,'二维'):interpolate_times(self.index,self.times,max_span=10)

if __name__=='__main__':unittest.main()

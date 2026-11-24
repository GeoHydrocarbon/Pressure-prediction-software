"""Sparse horizon samples must survive display decimation."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from overpressure.volume_reader import VolumeReader

class AlongHorizonDisplayTests(unittest.TestCase):
    def reader(self):
        reader=VolumeReader.__new__(VolumeReader)
        reader.index=SimpleNamespace(path='unused',inline=np.array([0,100,1000000]),crossline=np.array([100,0,1000000]),t0_ms=0.,interval_ms=1.,offsets=np.zeros(3,dtype=int),ns=5)
        reader.last_request={}
        def values(row,position):
            reader.last_request['sample_values_read']+=1;reader.last_request['traces_read']+=1
            return np.array([20.+row])
        reader.values=values;return reader
    def test_sparse_off_grid_points_are_not_discarded(self):
        reader=self.reader()
        with patch('overpressure.volume_reader.require_fresh'):
            section=reader.horizon_section([2.,2.,np.nan],'Sparse',max_cells=4)
        self.assertEqual(section.values.size,4)
        self.assertEqual(np.isfinite(section.values).sum(),2)
        np.testing.assert_array_equal(np.sort(section.values[np.isfinite(section.values)]),[20.,21.])
        self.assertLess(section.x.max(),1000000)
        self.assertEqual(reader.last_request['valid_pressure_points'],2)
    def test_outside_time_range_is_distinguishable_from_no_matches(self):
        reader=self.reader()
        with patch('overpressure.volume_reader.require_fresh'):
            section=reader.horizon_section([200.,200.,np.nan],'Outside',max_cells=4)
        self.assertTrue(np.isnan(section.values).all())
        self.assertEqual(reader.last_request['horizon_points'],2)
        self.assertEqual(reader.last_request['horizon_in_range_points'],0)
        self.assertEqual(reader.last_request['traces_read'],0)
if __name__=='__main__':unittest.main()

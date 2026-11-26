"""Header preflight must honor the chosen bytes and leave all trace values alone."""
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from overpressure.seismic_geometry import preview_segy

STANDARD=dict(inline=189,crossline=193,x=181,y=185,coordinate_scalar=71,coordinate_units=89)
LEGACY=dict(inline=21,crossline=1,x=73,y=77,coordinate_scalar=71,coordinate_units=89)


def legacy_segy(path):
    binary=bytearray(400)
    for offset,value in ((16,2000),(20,3),(24,5),(54,1)):struct.pack_into('>h',binary,offset,value)
    with Path(path).open('wb') as f:
        f.write(b' '*3200);f.write(binary)
        for il,xl in ((10,20),(11,20),(10,21),(11,21)):
            header=bytearray(240)
            for offset,value in ((20,il),(0,xl),(72,3000000+xl*250),(76,40000000+il*250)):
                struct.pack_into('>i',header,offset,value)
            for offset,value in ((70,-10),(88,1),(108,1000),(114,3),(116,2000)):
                struct.pack_into('>h',header,offset,value)
            f.write(header);f.write(np.array([1,2,3],dtype='>f4').tobytes())


class SeismicImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'legacy.sgy';legacy_segy(self.path)

    def test_wrong_mapping_reports_zero_ranges_and_duplicate_positions(self):
        result=preview_segy(dict(path=str(self.path),headers=STANDARD))
        self.assertFalse(result['passed']);self.assertEqual(result['duplicate_pairs'],3)
        self.assertEqual(result['inline'],[0,0]);self.assertIn('重复',result['error'])

    def test_legacy_mapping_passes_without_reading_amplitudes_or_changing_source(self):
        before=self.path.read_bytes()
        with patch('overpressure.seismic_browser.SeismicDataset.values',side_effect=AssertionError('No amplitude reads')):
            result=preview_segy(dict(path=str(self.path),headers=LEGACY))
        self.assertTrue(result['passed'],result)
        self.assertEqual(result['inline'],[10,11]);self.assertEqual(result['crossline'],[20,21])
        self.assertEqual(result['duplicate_pairs'],0);self.assertTrue(result['xy_available'])
        self.assertEqual(result['interval_ms'],2.);self.assertEqual(result['time_range'],[1000.,1004.])
        self.assertEqual(self.path.read_bytes(),before)

    def test_missing_or_invalid_source_is_an_actionable_failure(self):
        self.path.write_bytes(b'invalid')
        result=preview_segy(dict(path=str(self.path),headers=LEGACY))
        self.assertFalse(result['passed']);self.assertTrue(result['error'])


if __name__=='__main__':unittest.main()

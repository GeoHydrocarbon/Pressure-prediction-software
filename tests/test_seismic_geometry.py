from pathlib import Path
import tempfile
import unittest
import numpy as np
import segyio
from overpressure.seismic_geometry import read_geometry


class SeismicGeometryTests(unittest.TestCase):
    def test_actual_custom_headers_unsorted_extents_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            files={}
            for kind,lines in [('density',[(30,80),(10,100),(20,90)]),('impedance',[(30,80),(10,100),(20,91)])]:
                path=Path(directory)/(kind+'.sgy'); files[kind]=str(path)
                spec=segyio.spec();spec.format=5;spec.tracecount=3;spec.samples=np.arange(4)
                with segyio.create(str(path),spec) as dst:
                    for i,(il,xl) in enumerate(lines):dst.header[i]={189:il,193:xl};dst.trace[i]=np.zeros(4,dtype=np.float32)
            result=read_geometry(dict(files=files,headers=dict(inline=189,crossline=193)))
            self.assertEqual(result['files']['density'],dict(inline=[10,30],crossline=[80,100],traces=3))
            self.assertFalse(result['geometry_matches'])
            files.pop('impedance');result=read_geometry(dict(files=files,headers=dict(inline=189,crossline=193)))
            self.assertNotIn('geometry_matches',result)

    def test_missing_file_is_reported_per_input(self):
        result=read_geometry(dict(files=dict(density='does-not-exist.sgy'),headers=dict(inline=189,crossline=193)))
        self.assertIn('error',result['files']['density'])


if __name__=='__main__':unittest.main()

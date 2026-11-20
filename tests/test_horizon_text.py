from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from overpressure.horizon_io import read_horizon_table
from overpressure.horizons import map_horizon


class HorizonTextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)

    def test_headerless_txt_keeps_first_point_and_real_source_lines(self):
        path=self.folder/'points.txt';path.write_text('# description\n20 10 100 200 1000\n\n! comment\n21 10 110 200 1002\n',encoding='utf-8')
        raw=read_horizon_table(path);self.assertEqual(raw.shape,(2,5));self.assertEqual(raw.iloc[0].tolist(),[20,10,100,200,1000])
        self.assertEqual(raw.attrs['source_rows'].tolist(),[2,5])
        index=SimpleNamespace(inline=np.array([10,10]),crossline=np.array([20,21]),x=None,coordinate_units=None,time_convention='twt',t0_ms=1000,interval_ms=2,offsets=np.zeros(2),ns=3)
        cfg=dict(coordinates='lines',columns=dict(inline='第2列',crossline='第1列',time='第5列',name=''),time_unit='ms',time_convention='twt')
        _,times,table=map_horizon(raw,index,cfg);np.testing.assert_allclose(times,[[1000,1002]])
        self.assertEqual(table.source_row.tolist(),[2,5])

    def test_udf_semicolon_header_skip_and_chinese_gb18030(self):
        path=self.folder/'layer.udf';path.write_text('EXPORT SETTINGS\nUNITS: ms\nX;Y;Time;Horizon\n100;200;1002;层位甲\n110;200;1004;层位甲\n',encoding='gb18030')
        raw=read_horizon_table(path,dict(skip_rows=2,header='present',delimiter='semicolon'))
        self.assertEqual(raw.columns.tolist(),['X','Y','Time','Horizon']);self.assertEqual(raw.Horizon.tolist(),['层位甲']*2)
        self.assertEqual(raw.attrs['source_rows'].tolist(),[4,5])
        preview=read_horizon_table(path,dict(skip_rows=2),nrows=1);self.assertEqual(len(preview),1)

    def test_utf16_and_quoted_whitespace_layer_names(self):
        path=self.folder/'quoted.udf';path.write_text('100 200 1002 "T6 # Upper"\n110 200 1004 "T6 # Upper"\n',encoding='utf-16')
        raw=read_horizon_table(path);self.assertEqual(raw.shape,(2,4));self.assertEqual(raw['第4列'].iloc[0],'T6 # Upper')

    def test_binary_and_limit_are_explicit_errors(self):
        path=self.folder/'binary.udf';path.write_bytes(b'UDF\x00\x01\x02')
        with self.assertRaisesRegex(ValueError,'文本表格'):read_horizon_table(path)
        path=self.folder/'large.txt';path.write_text('1 2 3\n4 5 6\n7 8 9\n',encoding='utf-8')
        with patch('overpressure.horizon_io.MAX_HORIZON_RECORDS',2):
            with self.assertRaisesRegex(ValueError,'300 万'):read_horizon_table(path)

    def test_headerless_numeric_with_invalid_cell_is_not_dropped(self):
        path=self.folder/'invalid.txt';path.write_text('100 200 NULL\n110 200 1002\n',encoding='utf-8')
        self.assertEqual(len(read_horizon_table(path)),2)


if __name__=='__main__':unittest.main()

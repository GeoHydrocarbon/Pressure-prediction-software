import unittest
import numpy as np
from overpressure.horizon_layers import add_layers
from overpressure.charts import seismic_figure
from overpressure.seismic_view import Slice


class HorizonLayerTests(unittest.TestCase):
    def test_multiple_files_and_same_names_are_kept_distinct(self):
        entries=add_layers({},dict(source='a.txt',path='a.npz',names=['A','B']))
        entries=add_layers(entries,dict(source='b.udf',path='b.npz',names=['A']))
        self.assertEqual(len(entries),3);self.assertEqual(len({v['label'] for v in entries.values()}),3)
        first=next(v for v in entries.values() if v['name']=='B');first.update(width=3.2,checked=False)
        entries=add_layers(entries,dict(source='a.txt',path='a_new.npz',names=['B','C']))
        layer=next(v for v in entries.values() if v['name']=='B')
        self.assertEqual(len(entries),3);self.assertEqual(layer['width'],3.2);self.assertFalse(layer['checked']);self.assertEqual(layer['path'],'a_new.npz')

    def test_multiple_line_widths_colors_and_missing_intervals(self):
        values=np.ones((3,3));section=Slice(values,np.array([20,21,22]),np.array([1000,1002,1004]),'Crossline','Time (ms)','Inline 10',True)
        layers=[dict(time=np.array([1001,np.nan,1002]),color='#e35050',width=3.5,label='A'),dict(time=np.array([1002,1003,1003]),color='#267cb5',width=.8,label='B')]
        figure=seismic_figure(section,(0,2),overlays=layers);ax=figure.axes[0]
        self.assertEqual([line.get_linewidth() for line in ax.lines],[3.5,.8]);self.assertEqual([line.get_label() for line in ax.lines],['A','B'])
        self.assertTrue(np.isnan(ax.lines[0].get_ydata()[1]));np.testing.assert_array_equal(section.values,values)


if __name__=='__main__':unittest.main()
